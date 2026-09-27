from __future__ import annotations

"""
LocalIA — Backend × CPU Threads Benchmark
==========================================

Objetivo:
    Encontrar a melhor combinação entre:
      - backend/runtime: CPU AVX2 vs Vulkan
      - CPU threads por inferência: 4 vs 6 vs 8

Desenho:
    2 backends × 3 contagens de threads × 3 prompts × 3 rodadas
    = 54 gerações medidas

Por que o sweep termina em 8?
    O LM Studio já foi observado inicializando o llama threadpool com n_threads=8.
    O script CONFIRMA esse valor no log de cada load e recusa testar qualquer
    cpuThreads maior do que o threadpool efetivamente carregado.

Instrumentação:
    - LibreHardwareMonitor via http://localhost:8085/data.json
    - CPU Package temperature amostrada a cada 1 s durante inferência
    - CPU Package power
    - CPU Total load
    - clocks médios dos P-cores e E-cores
    - logs do LM Studio source=server para auditar:
        * n_threads do threadpool
        * contexto
        * batch / ubatch
        * GPU layers offloaded
    - LM Studio Python SDK para variar cpuThreads POR INFERÊNCIA

Requisitos:
    1. LM Studio aberto.
    2. llama.cpp log level = Debug no LM Studio.
    3. LibreHardwareMonitor aberto, Remote Web Server ativo na porta 8085.
    4. Python SDK do LM Studio:
           python -m pip install -U lmstudio

O script:
    - não pede edição manual durante o benchmark;
    - randomiza a ordem das 6 combinações em cada rodada;
    - randomiza os 3 prompts dentro de cada bloco;
    - descarrega/recarrega o modelo para cada combinação;
    - faz warm-up não contabilizado;
    - aplica thermal gate antes de cada bloco;
    - salva resultados imediatamente;
    - permite retomar depois de Ctrl+C.
"""

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ============================================================
# Configuração experimental
# ============================================================

API_BASE = "http://127.0.0.1:1234"
LHM_URL = "http://127.0.0.1:8085/data.json"

MODEL_KEY = "qwen/qwen3.5-9b"
EXPECTED_VARIANT = "qwen/qwen3.5-9b@q4_k_m"
INSTANCE_ID = "localia-backend-thread-bench"

RUNTIMES = {
    "cpu": "llama.cpp-win-x86_64-avx2@2.46.0",
    "vulkan": "llama.cpp-win-x86_64-vulkan-avx2@2.46.0",
}

# Sweep metodologicamente válido com o threadpool atualmente verificado em 8.
THREAD_COUNTS = [4, 6, 8]

ROUNDS = 3
SEED = 20260927

CONTEXT_LENGTH = 4096
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 384

EXPECTED_LOADED_CONFIG = {
    "context_length": 4096,
    "eval_batch_size": 2048,
    "physical_batch_size": 512,
    "parallel": 1,
    "flash_attention": False,
    "speculative_draft_mtp": False,
    "speculative_draft_simple": False,
    "offload_kv_cache_to_gpu": False,
}

PROMPTS = {
    "A": (
        "Explique em aproximadamente 150 palavras o que é uma API REST "
        "e cite suas principais características."
    ),
    "B": (
        "Explique em aproximadamente 150 palavras como funciona a "
        "fotossíntese e qual é sua importância para os ecossistemas."
    ),
    "C": (
        "Explique em aproximadamente 150 palavras quais foram as principais "
        "causas da Revolução Francesa."
    ),
}

WARMUP_PROMPT = (
    "Em no máximo 50 palavras, explique por que a água congela "
    "quando sua temperatura cai suficientemente."
)
WARMUP_MAX_OUTPUT_TOKENS = 96

# Temperatura
TARGET_START_TEMP_C = 70.0
THERMAL_STABLE_READINGS = 2
THERMAL_WAIT_MAX_SECONDS = 180
THERMAL_GATE_POLL_SECONDS = 5
THERMAL_SAMPLE_INTERVAL_SECONDS = 1.0

# Tempos auxiliares
COOLDOWN_AFTER_UNLOAD_SECONDS = 8
SETTLE_AFTER_RUNTIME_SWITCH_SECONDS = 2
SETTLE_AFTER_LOAD_SECONDS = 3
SETTLE_AFTER_WARMUP_SECONDS = 2
PAUSE_BETWEEN_PROMPTS_SECONDS = 2

LMS_COMMAND_TIMEOUT_SECONDS = 120.0
LMS_LOAD_TIMEOUT_SECONDS = 240.0
HTTP_TIMEOUT_SECONDS = 900.0

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"


# ============================================================
# Sensores LibreHardwareMonitor
# ============================================================

CPU_PACKAGE_TEMP_ID = "/intelcpu/0/temperature/14"
CORE_MAX_TEMP_ID = "/intelcpu/0/temperature/0"
CPU_PACKAGE_POWER_ID = "/intelcpu/0/power/0"
CPU_TOTAL_LOAD_ID = "/intelcpu/0/load/0"

P_CORE_CLOCK_IDS = [
    "/intelcpu/0/clock/1",
    "/intelcpu/0/clock/2",
    "/intelcpu/0/clock/3",
    "/intelcpu/0/clock/4",
]

E_CORE_CLOCK_IDS = [
    "/intelcpu/0/clock/5",
    "/intelcpu/0/clock/6",
    "/intelcpu/0/clock/7",
    "/intelcpu/0/clock/8",
    "/intelcpu/0/clock/9",
    "/intelcpu/0/clock/10",
    "/intelcpu/0/clock/11",
    "/intelcpu/0/clock/12",
]


# ============================================================
# Utilidades
# ============================================================

def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def seconds_to_hms(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "n/d"
    value = int(round(seconds))
    h, rem = divmod(value, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"AVISO: linha inválida em {path.name}:{line_no}; ignorada."
                )
    return rows


def run_command_metadata(*args: str) -> str | None:
    try:
        result = subprocess.run(
            list(args),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def safe_stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def fmt_num(value: Any, decimals: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return "n/d"


# ============================================================
# LMS CLI / REST
# ============================================================

def run_lms(
    *args: str,
    allow_failure: bool = False,
    timeout: float = LMS_COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    cmd = ["lms", *args]
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("O comando `lms` não foi encontrado no PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timeout após {timeout:.0f}s executando:\n  {' '.join(cmd)}"
        ) from exc

    if result.returncode != 0 and not allow_failure:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Falha ao executar:\n  {' '.join(cmd)}\n"
            f"Código: {result.returncode}\n{details or '(sem detalhes)'}"
        )
    return result


def api_get(path: str, timeout: float = 15.0) -> dict[str, Any]:
    request = Request(
        f"{API_BASE}{path}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} em GET {path}: {body}") from exc


def ensure_server() -> None:
    try:
        api_get("/api/v1/models", timeout=5.0)
        return
    except Exception:
        pass

    result = run_lms(
        "server",
        "start",
        allow_failure=True,
        timeout=30.0,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "Não foi possível iniciar o servidor do LM Studio.\n"
            "Mantenha o LM Studio aberto e tente novamente.\n"
            f"{details}"
        )

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            api_get("/api/v1/models", timeout=2.0)
            return
        except Exception:
            time.sleep(0.5)

    raise RuntimeError(
        "Servidor iniciado, mas /api/v1/models não respondeu em 15 s."
    )


def verify_selected_variant() -> dict[str, Any]:
    result = run_lms("ls", "--llm", "--json")
    try:
        models = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Não foi possível interpretar `lms ls --llm --json`."
        ) from exc

    for model in models:
        if model.get("modelKey") != MODEL_KEY:
            continue
        selected = model.get("selectedVariant")
        if selected != EXPECTED_VARIANT:
            raise RuntimeError(
                "Variante incorreta selecionada.\n"
                f"Esperado: {EXPECTED_VARIANT}\n"
                f"Atual:    {selected}"
            )
        return model

    raise RuntimeError(f"Modelo não encontrado: {MODEL_KEY}")


def runtime_ls_text() -> str:
    return run_lms("runtime", "ls").stdout


def verify_runtime_installed(alias: str) -> None:
    if alias not in runtime_ls_text():
        raise RuntimeError(
            f"Runtime necessário não está instalado:\n  {alias}"
        )


def selected_gguf_runtime() -> str | None:
    text = runtime_ls_text()
    for line in text.splitlines():
        if "✓" in line and "GGUF" in line:
            return line.strip()
    return None


def unload_all() -> None:
    run_lms("unload", "--all", allow_failure=True, timeout=60.0)


def select_runtime(alias: str) -> tuple[str, float]:
    unload_all()
    start = time.perf_counter()
    result = run_lms("runtime", "select", alias, timeout=60.0)
    elapsed = time.perf_counter() - start

    selected = selected_gguf_runtime()
    if not selected or alias not in selected:
        raise RuntimeError(
            "O runtime ativo não pôde ser confirmado.\n"
            f"Esperado: {alias}\n"
            f"Selecionado: {selected!r}\n"
            f"Saída: {result.stdout.strip()}"
        )
    return selected, elapsed


def load_model() -> tuple[subprocess.CompletedProcess[str], float]:
    start = time.perf_counter()
    result = run_lms(
        "load",
        MODEL_KEY,
        "--gpu",
        "off",
        "--context-length",
        str(CONTEXT_LENGTH),
        "--identifier",
        INSTANCE_ID,
        timeout=LMS_LOAD_TIMEOUT_SECONDS,
    )
    return result, time.perf_counter() - start


def find_loaded_instance(
    response: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for model in response.get("models", []):
        if model.get("key") != MODEL_KEY:
            continue
        for instance in model.get("loaded_instances", []):
            if instance.get("id") == INSTANCE_ID:
                return model, instance
    return None, None


def audit_loaded_config(config: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for key, expected in EXPECTED_LOADED_CONFIG.items():
        if key not in config:
            issues.append(f"{key}: ausente; esperado={expected!r}")
            continue
        actual = config.get(key)
        if actual != expected:
            issues.append(f"{key}: atual={actual!r}; esperado={expected!r}")
    return issues


# ============================================================
# Logs do servidor LM Studio
# ============================================================

def start_server_log_capture(
    log_path: Path,
) -> tuple[subprocess.Popen[Any] | None, Any | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    try:
        handle = log_path.open("w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            ["lms", "log", "stream", "--source", "server", "--json"],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(1.0)
        return proc, handle
    except Exception:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        return None, None


def stop_log_capture(
    proc: subprocess.Popen[Any] | None,
    handle: Any | None,
) -> None:
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    if handle is not None:
        try:
            handle.flush()
            handle.close()
        except Exception:
            pass


def extract_server_log_content(log_path: Path) -> str:
    if not log_path.exists():
        return ""

    chunks: list[str] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                content = (
                    obj.get("data", {}).get("content")
                    if isinstance(obj, dict)
                    else None
                )
                if isinstance(content, str):
                    chunks.append(content)
            except json.JSONDecodeError:
                # O preâmbulo "Streaming logs..." não é JSON.
                continue

    return "\n".join(chunks)


def _last_int(pattern: str, text: str) -> int | None:
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not matches:
        return None
    try:
        return int(matches[-1].group(1))
    except Exception:
        return None


def _last_str(pattern: str, text: str) -> str | None:
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not matches:
        return None
    return matches[-1].group(1)


def parse_load_log(log_path: Path) -> dict[str, Any]:
    text = extract_server_log_content(log_path)

    offload_matches = list(
        re.finditer(
            r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU",
            text,
            flags=re.IGNORECASE,
        )
    )

    offloaded_layers = None
    total_layers = None
    if offload_matches:
        offloaded_layers = int(offload_matches[-1].group(1))
        total_layers = int(offload_matches[-1].group(2))

    return {
        "log_available": bool(text.strip()),
        "threadpool_n_threads": _last_int(
            r"llama\s+threadpool\s+init,\s*n_threads\s*=\s*(\d+)",
            text,
        ),
        "http_server_threads": _last_int(
            r"using\s+(\d+)\s+threads\s+for\s+HTTP\s+server",
            text,
        ),
        "n_ctx": _last_int(r"\bn_ctx\s*=\s*(\d+)", text),
        "n_batch": _last_int(r"\bn_batch\s*=\s*(\d+)", text),
        "n_ubatch": _last_int(r"\bn_ubatch\s*=\s*(\d+)", text),
        "flash_attn": _last_str(
            r"\bflash_attn\s*=\s*([A-Za-z]+)",
            text,
        ),
        "offloaded_layers": offloaded_layers,
        "total_layers": total_layers,
    }


# ============================================================
# LibreHardwareMonitor
# ============================================================

def parse_localized_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
    if not match:
        return None

    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def fetch_lhm_json(timeout: float = 3.0) -> dict[str, Any]:
    request = Request(
        LHM_URL,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Não foi possível ler o LibreHardwareMonitor em "
            f"{LHM_URL}: {exc}"
        ) from exc


def flatten_sensor_nodes(
    node: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}

    sensor_id = node.get("SensorId")
    if isinstance(sensor_id, str):
        found[sensor_id] = node

    for child in node.get("Children", []) or []:
        if isinstance(child, dict):
            found.update(flatten_sensor_nodes(child))

    return found


def mean_existing(values: list[float | None]) -> float | None:
    actual = [v for v in values if isinstance(v, (int, float))]
    return statistics.mean(actual) if actual else None


def read_hardware_sample() -> dict[str, Any]:
    data = fetch_lhm_json()
    sensors = flatten_sensor_nodes(data)

    def sensor_value(sensor_id: str) -> float | None:
        sensor = sensors.get(sensor_id)
        if not sensor:
            return None
        return parse_localized_number(
            sensor.get("RawValue", sensor.get("Value"))
        )

    p_clocks = [sensor_value(x) for x in P_CORE_CLOCK_IDS]
    e_clocks = [sensor_value(x) for x in E_CORE_CLOCK_IDS]

    return {
        "timestamp": now_iso(),
        "monotonic": time.perf_counter(),
        "cpu_package_temp_c": sensor_value(CPU_PACKAGE_TEMP_ID),
        "core_max_temp_c": sensor_value(CORE_MAX_TEMP_ID),
        "cpu_package_power_w": sensor_value(CPU_PACKAGE_POWER_ID),
        "cpu_total_load_percent": sensor_value(CPU_TOTAL_LOAD_ID),
        "p_core_clock_mean_mhz": mean_existing(p_clocks),
        "e_core_clock_mean_mhz": mean_existing(e_clocks),
    }


def verify_lhm() -> dict[str, Any]:
    sample = read_hardware_sample()
    temp = sample.get("cpu_package_temp_c")
    if not isinstance(temp, (int, float)):
        raise RuntimeError(
            "LibreHardwareMonitor respondeu, mas o sensor CPU Package "
            f"({CPU_PACKAGE_TEMP_ID}) não foi encontrado."
        )
    return sample


def thermal_gate(
    session_dir: Path,
    block_id: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + THERMAL_WAIT_MAX_SECONDS
    stable = 0
    history: list[dict[str, Any]] = []

    print(
        f"[TEMP] Aguardando CPU Package ≤ {TARGET_START_TEMP_C:.0f} °C "
        f"por {THERMAL_STABLE_READINGS} leituras consecutivas..."
    )

    while True:
        sample = read_hardware_sample()
        history.append(sample)
        temp = sample.get("cpu_package_temp_c")

        if isinstance(temp, (int, float)):
            print(
                f"       {temp:.1f} °C | "
                f"{fmt_num(sample.get('cpu_package_power_w'), 1)} W | "
                f"{fmt_num(sample.get('cpu_total_load_percent'), 1)}% CPU"
            )
            if temp <= TARGET_START_TEMP_C:
                stable += 1
            else:
                stable = 0
        else:
            stable = 0

        if stable >= THERMAL_STABLE_READINGS:
            record = {
                "timestamp": now_iso(),
                "block_id": block_id,
                "status": "stable",
                "target_temp_c": TARGET_START_TEMP_C,
                "history": history,
            }
            append_jsonl(session_dir / "thermal_gates.jsonl", record)
            print("[TEMP] Faixa térmica de início atingida.")
            return sample

        if time.monotonic() >= deadline:
            record = {
                "timestamp": now_iso(),
                "block_id": block_id,
                "status": "timeout",
                "target_temp_c": TARGET_START_TEMP_C,
                "history": history,
            }
            append_jsonl(session_dir / "thermal_gates.jsonl", record)
            print(
                "[TEMP] Timeout térmico atingido; o bloco continuará, "
                "mas isso ficará registrado."
            )
            return sample

        time.sleep(THERMAL_GATE_POLL_SECONDS)


class ThermalMonitor:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            self.samples.append(read_hardware_sample())
        except Exception as exc:
            self.errors.append(str(exc))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(
            target=self._loop,
            name="localia-thermal-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 2))
        self._sample()


def summarize_thermal_samples(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [
            float(s[key])
            for s in samples
            if isinstance(s.get(key), (int, float))
        ]

    temps = vals("cpu_package_temp_c")
    core_max = vals("core_max_temp_c")
    power = vals("cpu_package_power_w")
    load = vals("cpu_total_load_percent")
    p_clock = vals("p_core_clock_mean_mhz")
    e_clock = vals("e_core_clock_mean_mhz")

    start_temp = temps[0] if temps else None
    end_temp = temps[-1] if temps else None

    return {
        "sample_count": len(samples),
        "temp_start_c": start_temp,
        "temp_end_c": end_temp,
        "temp_peak_c": max(temps) if temps else None,
        "temp_mean_c": safe_mean(temps),
        "temp_delta_c": (
            end_temp - start_temp
            if start_temp is not None and end_temp is not None
            else None
        ),
        "core_max_peak_c": max(core_max) if core_max else None,
        "package_power_mean_w": safe_mean(power),
        "package_power_peak_w": max(power) if power else None,
        "cpu_load_mean_percent": safe_mean(load),
        "cpu_load_peak_percent": max(load) if load else None,
        "p_core_clock_mean_mhz": safe_mean(p_clock),
        "e_core_clock_mean_mhz": safe_mean(e_clock),
        "seconds_at_or_above_80c": (
            sum(1 for x in temps if x >= 80.0) * THERMAL_SAMPLE_INTERVAL_SECONDS
        ),
        "seconds_at_or_above_85c": (
            sum(1 for x in temps if x >= 85.0) * THERMAL_SAMPLE_INTERVAL_SECONDS
        ),
        "seconds_at_or_above_90c": (
            sum(1 for x in temps if x >= 90.0) * THERMAL_SAMPLE_INTERVAL_SECONDS
        ),
    }


# ============================================================
# LM Studio Python SDK
# ============================================================

def import_lmstudio_sdk():
    try:
        import lmstudio as lms  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "O pacote Python `lmstudio` não está instalado.\n"
            "Execute uma vez:\n"
            "  python -m pip install -U lmstudio"
        ) from exc

    # Evita timeout de 60 s em gerações longas nas versões recentes do SDK.
    setter = getattr(lms, "set_sync_api_timeout", None)
    if callable(setter):
        setter(None)

    return lms


def getattr_any(obj: Any, names: list[str]) -> Any:
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if value is not None:
            return value
    return None


def serialize_object(obj: Any) -> Any:
    if obj is None:
        return None

    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass

    result: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None), list, dict)):
            result[name] = value

    return result or str(obj)


def run_sdk_inference(
    lms: Any,
    prompt: str,
    cpu_threads: int,
    max_tokens: int,
) -> tuple[dict[str, Any], float]:
    # Há exatamente um modelo carregado porque o script usa unload --all
    # antes de cada bloco.
    model = lms.llm()

    config = {
        "temperature": TEMPERATURE,
        "maxTokens": max_tokens,
        "cpuThreads": cpu_threads,
    }

    start = time.perf_counter()
    result = model.respond(prompt, config=config)
    wall = time.perf_counter() - start

    stats = getattr(result, "stats", None)
    prediction_config = getattr(result, "prediction_config", None)

    content = getattr_any(
        result,
        ["content", "text"],
    )
    if content is None:
        content = str(result)

    tps = getattr_any(
        stats,
        ["tokens_per_second", "tokensPerSecond"],
    )
    ttft = getattr_any(
        stats,
        ["time_to_first_token_sec", "timeToFirstTokenSec"],
    )
    predicted = getattr_any(
        stats,
        ["predicted_tokens_count", "predictedTokensCount"],
    )
    stop_reason = getattr_any(
        stats,
        ["stop_reason", "stopReason"],
    )

    echoed_threads = getattr_any(
        prediction_config,
        ["cpu_threads", "cpuThreads"],
    )

    reasoning_content = getattr_any(
        result,
        ["reasoning_content", "reasoningContent"],
    )

    return {
        "content": content,
        "tokens_per_second": tps,
        "time_to_first_token_seconds": ttft,
        "predicted_tokens_count": predicted,
        "stop_reason": stop_reason,
        "requested_cpu_threads": cpu_threads,
        "echoed_cpu_threads": echoed_threads,
        "prediction_config": serialize_object(prediction_config),
        "stats_raw": serialize_object(stats),
        "reasoning_content": reasoning_content,
    }, wall


# ============================================================
# Sessão / schedule
# ============================================================

def build_schedule(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    configs = [
        {
            "backend": backend,
            "runtime_alias": alias,
            "threads": threads,
        }
        for backend, alias in RUNTIMES.items()
        for threads in THREAD_COUNTS
    ]

    blocks: list[dict[str, Any]] = []

    for round_number in range(1, ROUNDS + 1):
        order = [dict(x) for x in configs]
        rng.shuffle(order)

        for block_index, cfg in enumerate(order, start=1):
            prompt_order = list(PROMPTS.keys())
            rng.shuffle(prompt_order)

            block_id = (
                f"R{round_number}-"
                f"{cfg['backend'].upper()}-"
                f"T{cfg['threads']}"
            )

            blocks.append(
                {
                    "block_id": block_id,
                    "round": round_number,
                    "block_index": block_index,
                    "backend": cfg["backend"],
                    "runtime_alias": cfg["runtime_alias"],
                    "threads": cfg["threads"],
                    "prompt_order": prompt_order,
                    "run_ids": [
                        f"{block_id}-P{prompt_id}"
                        for prompt_id in prompt_order
                    ],
                }
            )

    return blocks


def find_latest_incomplete_session() -> Path | None:
    if not RESULTS_ROOT.exists():
        return None

    candidates: list[Path] = []
    for path in RESULTS_ROOT.glob("backend_thread_benchmark_*"):
        session_file = path / "session.json"
        if not path.is_dir() or not session_file.exists():
            continue
        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not data.get("completed", False):
            candidates.append(path)

    return max(candidates, key=lambda p: p.name) if candidates else None


def create_session(seed: int) -> tuple[Path, dict[str, Any]]:
    session_dir = RESULTS_ROOT / f"backend_thread_benchmark_{timestamp_slug()}"
    session_dir.mkdir(parents=True, exist_ok=False)

    schedule = build_schedule(seed)

    session = {
        "session_version": 1,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "completed": False,
        "completed_at": None,
        "seed": seed,
        "model_key": MODEL_KEY,
        "expected_variant": EXPECTED_VARIANT,
        "runtimes": RUNTIMES,
        "thread_counts": THREAD_COUNTS,
        "configuration": {
            "context_length": CONTEXT_LENGTH,
            "gpu_offload": "off",
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "rounds": ROUNDS,
            "thermal_sample_interval_seconds": THERMAL_SAMPLE_INTERVAL_SECONDS,
            "target_start_temp_c": TARGET_START_TEMP_C,
            "expected_loaded_config": EXPECTED_LOADED_CONFIG,
        },
        "prompts": PROMPTS,
        "warmup_prompt": WARMUP_PROMPT,
        "schedule": schedule,
        "planned_measured_runs": (
            ROUNDS
            * len(RUNTIMES)
            * len(THREAD_COUNTS)
            * len(PROMPTS)
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "lms_version": run_command_metadata("lms", "--version"),
            "git_commit": run_command_metadata("git", "rev-parse", "HEAD"),
        },
        "verified_thread_pool_size": None,
    }

    atomic_write_json(session_dir / "session.json", session)
    return session_dir, session


def load_session(session_dir: Path) -> dict[str, Any]:
    return json.loads(
        (session_dir / "session.json").read_text(encoding="utf-8")
    )


def update_session_file(
    session_dir: Path,
    session: dict[str, Any],
) -> None:
    session["updated_at"] = now_iso()
    atomic_write_json(session_dir / "session.json", session)


def successful_run_ids(runs_path: Path) -> set[str]:
    return {
        row["run_id"]
        for row in read_jsonl(runs_path)
        if row.get("status") == "success" and row.get("run_id")
    }


# ============================================================
# CSV / resumo
# ============================================================

CSV_FIELDS = [
    "run_id",
    "timestamp",
    "round",
    "backend",
    "runtime_alias",
    "threads_requested",
    "threads_echoed",
    "threadpool_n_threads",
    "prompt_id",
    "tokens_per_second",
    "time_to_first_token_seconds",
    "predicted_tokens_count",
    "wall_time_seconds",
    "stop_reason",
    "temp_start_c",
    "temp_peak_c",
    "temp_end_c",
    "temp_mean_c",
    "package_power_mean_w",
    "package_power_peak_w",
    "cpu_load_mean_percent",
    "p_core_clock_mean_mhz",
    "e_core_clock_mean_mhz",
    "seconds_at_or_above_85c",
    "reasoning_marker_detected",
    "status",
]


def reasoning_marker_detected(record: dict[str, Any]) -> bool:
    inference = record.get("inference") or {}
    content = str(inference.get("content") or "").lower()
    reasoning = inference.get("reasoning_content")
    return bool(
        reasoning
        or "<think>" in content
        or "</think>" in content
    )


def write_runs_csv(session_dir: Path) -> None:
    runs = [
        x
        for x in read_jsonl(session_dir / "runs.jsonl")
        if x.get("status") == "success"
    ]

    with (session_dir / "runs.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in runs:
            inf = row.get("inference") or {}
            therm = row.get("thermal_summary") or {}
            load_log = row.get("load_log_parsed") or {}

            writer.writerow(
                {
                    "run_id": row.get("run_id"),
                    "timestamp": row.get("timestamp"),
                    "round": row.get("round"),
                    "backend": row.get("backend"),
                    "runtime_alias": row.get("runtime_alias"),
                    "threads_requested": row.get("threads"),
                    "threads_echoed": inf.get("echoed_cpu_threads"),
                    "threadpool_n_threads": load_log.get(
                        "threadpool_n_threads"
                    ),
                    "prompt_id": row.get("prompt_id"),
                    "tokens_per_second": inf.get("tokens_per_second"),
                    "time_to_first_token_seconds": inf.get(
                        "time_to_first_token_seconds"
                    ),
                    "predicted_tokens_count": inf.get(
                        "predicted_tokens_count"
                    ),
                    "wall_time_seconds": row.get("wall_time_seconds"),
                    "stop_reason": inf.get("stop_reason"),
                    "temp_start_c": therm.get("temp_start_c"),
                    "temp_peak_c": therm.get("temp_peak_c"),
                    "temp_end_c": therm.get("temp_end_c"),
                    "temp_mean_c": therm.get("temp_mean_c"),
                    "package_power_mean_w": therm.get(
                        "package_power_mean_w"
                    ),
                    "package_power_peak_w": therm.get(
                        "package_power_peak_w"
                    ),
                    "cpu_load_mean_percent": therm.get(
                        "cpu_load_mean_percent"
                    ),
                    "p_core_clock_mean_mhz": therm.get(
                        "p_core_clock_mean_mhz"
                    ),
                    "e_core_clock_mean_mhz": therm.get(
                        "e_core_clock_mean_mhz"
                    ),
                    "seconds_at_or_above_85c": therm.get(
                        "seconds_at_or_above_85c"
                    ),
                    "reasoning_marker_detected": reasoning_marker_detected(row),
                    "status": row.get("status"),
                }
            )


def collect_numeric(
    rows: list[dict[str, Any]],
    getter,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = getter(row)
        except Exception:
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def build_summary(session_dir: Path) -> dict[str, Any]:
    runs = [
        x
        for x in read_jsonl(session_dir / "runs.jsonl")
        if x.get("status") == "success"
    ]

    by_config: list[dict[str, Any]] = []

    for backend, alias in RUNTIMES.items():
        for threads in THREAD_COUNTS:
            rows = [
                r
                for r in runs
                if r.get("backend") == backend
                and r.get("threads") == threads
            ]

            tps = collect_numeric(
                rows,
                lambda r: (r.get("inference") or {}).get(
                    "tokens_per_second"
                ),
            )
            ttft = collect_numeric(
                rows,
                lambda r: (r.get("inference") or {}).get(
                    "time_to_first_token_seconds"
                ),
            )
            wall = collect_numeric(
                rows,
                lambda r: r.get("wall_time_seconds"),
            )
            peak_temp = collect_numeric(
                rows,
                lambda r: (r.get("thermal_summary") or {}).get(
                    "temp_peak_c"
                ),
            )
            mean_temp = collect_numeric(
                rows,
                lambda r: (r.get("thermal_summary") or {}).get(
                    "temp_mean_c"
                ),
            )
            power = collect_numeric(
                rows,
                lambda r: (r.get("thermal_summary") or {}).get(
                    "package_power_mean_w"
                ),
            )
            cpu_load = collect_numeric(
                rows,
                lambda r: (r.get("thermal_summary") or {}).get(
                    "cpu_load_mean_percent"
                ),
            )
            p_clock = collect_numeric(
                rows,
                lambda r: (r.get("thermal_summary") or {}).get(
                    "p_core_clock_mean_mhz"
                ),
            )
            e_clock = collect_numeric(
                rows,
                lambda r: (r.get("thermal_summary") or {}).get(
                    "e_core_clock_mean_mhz"
                ),
            )

            by_config.append(
                {
                    "backend": backend,
                    "runtime_alias": alias,
                    "threads": threads,
                    "n": len(rows),
                    "tokens_per_second_mean": safe_mean(tps),
                    "tokens_per_second_median": safe_median(tps),
                    "tokens_per_second_stdev": safe_stdev(tps),
                    "ttft_mean_seconds": safe_mean(ttft),
                    "wall_time_mean_seconds": safe_mean(wall),
                    "temp_peak_mean_c": safe_mean(peak_temp),
                    "temp_peak_max_c": max(peak_temp) if peak_temp else None,
                    "temp_mean_c": safe_mean(mean_temp),
                    "package_power_mean_w": safe_mean(power),
                    "cpu_load_mean_percent": safe_mean(cpu_load),
                    "p_core_clock_mean_mhz": safe_mean(p_clock),
                    "e_core_clock_mean_mhz": safe_mean(e_clock),
                    "reasoning_marker_runs": sum(
                        1 for row in rows if reasoning_marker_detected(row)
                    ),
                }
            )

    valid = [
        x
        for x in by_config
        if isinstance(x.get("tokens_per_second_mean"), (int, float))
    ]
    best_throughput = (
        max(valid, key=lambda x: x["tokens_per_second_mean"])
        if valid
        else None
    )

    valid_ttft = [
        x
        for x in by_config
        if isinstance(x.get("ttft_mean_seconds"), (int, float))
    ]
    best_ttft = (
        min(valid_ttft, key=lambda x: x["ttft_mean_seconds"])
        if valid_ttft
        else None
    )

    return {
        "generated_at": now_iso(),
        "successful_runs": len(runs),
        "planned_runs": (
            ROUNDS
            * len(RUNTIMES)
            * len(THREAD_COUNTS)
            * len(PROMPTS)
        ),
        "by_config": by_config,
        "best_throughput_config": best_throughput,
        "best_ttft_config": best_ttft,
    }


def write_summary_files(session_dir: Path) -> dict[str, Any]:
    summary = build_summary(session_dir)
    atomic_write_json(session_dir / "summary.json", summary)

    fields = [
        "backend",
        "runtime_alias",
        "threads",
        "n",
        "tokens_per_second_mean",
        "tokens_per_second_median",
        "tokens_per_second_stdev",
        "ttft_mean_seconds",
        "wall_time_mean_seconds",
        "temp_peak_mean_c",
        "temp_peak_max_c",
        "temp_mean_c",
        "package_power_mean_w",
        "cpu_load_mean_percent",
        "p_core_clock_mean_mhz",
        "e_core_clock_mean_mhz",
        "reasoning_marker_runs",
    ]

    with (session_dir / "summary.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["by_config"]:
            writer.writerow(row)

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 122)
    print(" RESUMO FINAL — BACKEND × CPU THREADS")
    print("=" * 122)
    print(
        f"{'backend':<8} {'thr':>3} {'n':>3} "
        f"{'tok/s':>8} {'TTFT':>8} {'tempo':>9} "
        f"{'T pico':>8} {'W méd':>8} {'CPU%':>8} "
        f"{'P MHz':>9} {'E MHz':>9}"
    )
    print("-" * 122)

    for row in summary["by_config"]:
        print(
            f"{row['backend']:<8} "
            f"{row['threads']:>3} "
            f"{row['n']:>3} "
            f"{fmt_num(row['tokens_per_second_mean']):>8} "
            f"{(fmt_num(row['ttft_mean_seconds']) + 's'):>8} "
            f"{(fmt_num(row['wall_time_mean_seconds'], 1) + 's'):>9} "
            f"{(fmt_num(row['temp_peak_mean_c'], 1) + '°C'):>8} "
            f"{(fmt_num(row['package_power_mean_w'], 1) + 'W'):>8} "
            f"{(fmt_num(row['cpu_load_mean_percent'], 1) + '%'):>8} "
            f"{fmt_num(row['p_core_clock_mean_mhz'], 0):>9} "
            f"{fmt_num(row['e_core_clock_mean_mhz'], 0):>9}"
        )

    best = summary.get("best_throughput_config")
    if best:
        print(
            "\nMaior throughput médio: "
            f"{best['backend'].upper()} / {best['threads']} threads "
            f"= {best['tokens_per_second_mean']:.3f} tok/s"
        )

    best_ttft = summary.get("best_ttft_config")
    if best_ttft:
        print(
            "Menor TTFT médio: "
            f"{best_ttft['backend'].upper()} / {best_ttft['threads']} threads "
            f"= {best_ttft['ttft_mean_seconds']:.3f} s"
        )

    print("=" * 122)


# ============================================================
# Benchmark
# ============================================================

def record_error(
    session_dir: Path,
    *,
    phase: str,
    message: str,
    block: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> None:
    record: dict[str, Any] = {
        "timestamp": now_iso(),
        "phase": phase,
        "message": message,
        "run_id": run_id,
    }

    if block:
        record.update(
            {
                "block_id": block.get("block_id"),
                "round": block.get("round"),
                "backend": block.get("backend"),
                "threads": block.get("threads"),
                "runtime_alias": block.get("runtime_alias"),
            }
        )

    append_jsonl(session_dir / "errors.jsonl", record)


def calculate_eta(
    process_started: float,
    newly_completed: int,
    remaining_runs: int,
) -> float | None:
    if newly_completed <= 0:
        return None
    return (
        (time.perf_counter() - process_started)
        / newly_completed
    ) * remaining_runs


def verify_prediction_threads(
    requested: int,
    echoed: Any,
) -> None:
    if echoed is None:
        raise RuntimeError(
            "O SDK não devolveu `cpu_threads` na configuração efetiva da "
            "predição. Para manter o experimento auditável, o bloco será "
            "interrompido."
        )

    try:
        echoed_int = int(echoed)
    except Exception as exc:
        raise RuntimeError(
            f"Valor de cpu_threads devolvido pelo SDK é inválido: {echoed!r}"
        ) from exc

    if echoed_int != requested:
        raise RuntimeError(
            "CPU threads solicitado e aplicado não coincidem.\n"
            f"Solicitado: {requested}\n"
            f"Aplicado:   {echoed_int}"
        )


def execute_benchmark(
    session_dir: Path,
    session: dict[str, Any],
    lms: Any,
) -> int:
    runs_path = session_dir / "runs.jsonl"
    loads_path = session_dir / "loads.jsonl"
    warmups_path = session_dir / "warmups.jsonl"
    switches_path = session_dir / "runtime_switches.jsonl"

    successful_ids = successful_run_ids(runs_path)
    total_runs = int(session["planned_measured_runs"])

    newly_completed = 0
    process_started = time.perf_counter()

    for block_number, block in enumerate(session["schedule"], start=1):
        pending = [
            run_id
            for run_id in block["run_ids"]
            if run_id not in successful_ids
        ]

        if not pending:
            print(
                f"\n[Bloco {block_number}/{len(session['schedule'])}] "
                f"{block['block_id']} já concluído — pulando."
            )
            continue

        backend = block["backend"]
        threads = int(block["threads"])
        runtime_alias = block["runtime_alias"]

        print("\n" + "=" * 88)
        print(
            f" Bloco {block_number}/{len(session['schedule'])} | "
            f"Rodada {block['round']}/{ROUNDS} | "
            f"{backend.upper()} | {threads} threads"
        )
        print(f" {runtime_alias}")
        print("=" * 88)

        loaded = False
        load_wall_time: float | None = None
        load_log_parsed: dict[str, Any] = {}

        try:
            ensure_server()

            print("[SWITCH] Descarregando modelo anterior...")
            unload_all()

            if COOLDOWN_AFTER_UNLOAD_SECONDS:
                time.sleep(COOLDOWN_AFTER_UNLOAD_SECONDS)

            print(f"[SWITCH] Selecionando {backend.upper()}...")
            selected_line, switch_seconds = select_runtime(runtime_alias)
            print(f"[SWITCH] OK — {selected_line}")

            append_jsonl(
                switches_path,
                {
                    "timestamp": now_iso(),
                    "block_id": block["block_id"],
                    "round": block["round"],
                    "backend": backend,
                    "threads": threads,
                    "runtime_alias": runtime_alias,
                    "switch_wall_time_seconds": switch_seconds,
                    "selected_line": selected_line,
                },
            )

            time.sleep(SETTLE_AFTER_RUNTIME_SWITCH_SECONDS)

            thermal_gate(session_dir, block["block_id"])

            log_path = (
                session_dir
                / "load_logs"
                / f"{block['block_id']}.jsonl"
            )
            log_proc, log_handle = start_server_log_capture(log_path)

            print("[LOAD] Carregando modelo com GPU offload OFF...")
            try:
                load_result, load_wall_time = load_model()
                loaded = True
                time.sleep(1.0)
            finally:
                stop_log_capture(log_proc, log_handle)

            load_log_parsed = parse_load_log(log_path)

            print(f"[LOAD] OK em {load_wall_time:.2f}s")
            print(
                "[LOAD-AUDIT] "
                f"threadpool={load_log_parsed.get('threadpool_n_threads')} | "
                f"offload={load_log_parsed.get('offloaded_layers')}/"
                f"{load_log_parsed.get('total_layers')} | "
                f"ctx={load_log_parsed.get('n_ctx')} | "
                f"batch={load_log_parsed.get('n_batch')} | "
                f"ubatch={load_log_parsed.get('n_ubatch')}"
            )

            pool_size = load_log_parsed.get("threadpool_n_threads")
            if not isinstance(pool_size, int):
                raise RuntimeError(
                    "Não consegui confirmar `llama threadpool init, "
                    "n_threads = ...` no server log.\n"
                    "Confirme no LM Studio: Settings → Developer → "
                    "llama.cpp log level = Debug."
                )

            if pool_size < max(THREAD_COUNTS):
                raise RuntimeError(
                    "O threadpool carregado é menor que o maior valor do sweep.\n"
                    f"Threadpool confirmado: {pool_size}\n"
                    f"Maior cpuThreads solicitado: {max(THREAD_COUNTS)}\n"
                    "O teste foi interrompido para evitar resultado inválido."
                )

            known_pool = session.get("verified_thread_pool_size")
            if known_pool is None:
                session["verified_thread_pool_size"] = pool_size
                update_session_file(session_dir, session)
            elif int(known_pool) != pool_size:
                raise RuntimeError(
                    "O tamanho do threadpool mudou entre blocos.\n"
                    f"Primeiro valor: {known_pool}\n"
                    f"Valor atual:    {pool_size}\n"
                    "O teste foi interrompido para preservar comparabilidade."
                )

            offloaded = load_log_parsed.get("offloaded_layers")
            if backend == "vulkan":
                if offloaded is None:
                    raise RuntimeError(
                        "Não consegui confirmar no log quantas camadas foram "
                        "offloaded para GPU."
                    )
                if int(offloaded) != 0:
                    raise RuntimeError(
                        f"Vulkan carregou {offloaded} camada(s) na GPU. "
                        "Esperado: 0."
                    )
            elif offloaded is not None and int(offloaded) != 0:
                raise RuntimeError(
                    f"Backend CPU registrou {offloaded} camada(s) na GPU."
                )

            models_response = api_get("/api/v1/models")
            _, loaded_instance = find_loaded_instance(models_response)
            if loaded_instance is None:
                raise RuntimeError(
                    f"Instância `{INSTANCE_ID}` não apareceu na API."
                )

            loaded_config = loaded_instance.get("config") or {}
            config_issues = audit_loaded_config(loaded_config)

            append_jsonl(
                loads_path,
                {
                    "timestamp": now_iso(),
                    "block_id": block["block_id"],
                    "round": block["round"],
                    "backend": backend,
                    "threads_for_predictions": threads,
                    "runtime_alias": runtime_alias,
                    "load_wall_time_seconds": load_wall_time,
                    "loaded_config": loaded_config,
                    "config_audit_issues": config_issues,
                    "load_log_path": str(log_path),
                    "load_log_parsed": load_log_parsed,
                    "lms_stdout": load_result.stdout.strip(),
                    "lms_stderr": load_result.stderr.strip(),
                },
            )

            if config_issues:
                raise RuntimeError(
                    "Auditoria da configuração falhou: "
                    + "; ".join(config_issues)
                )

            print("[AUDIT] Configuração de load confere.")

            time.sleep(SETTLE_AFTER_LOAD_SECONDS)

            # Warm-up com a MESMA contagem de threads do bloco.
            warm_monitor = ThermalMonitor(THERMAL_SAMPLE_INTERVAL_SECONDS)
            warm_monitor.start()
            print(
                f"[WARMUP] {backend.upper()} / {threads} threads | "
                f"CPU Package inicial: "
                f"{fmt_num(warm_monitor.samples[0].get('cpu_package_temp_c'), 1)} °C"
            )

            try:
                warm_inf, warm_wall = run_sdk_inference(
                    lms,
                    WARMUP_PROMPT,
                    threads,
                    WARMUP_MAX_OUTPUT_TOKENS,
                )
            finally:
                warm_monitor.stop()

            verify_prediction_threads(
                threads,
                warm_inf.get("echoed_cpu_threads"),
            )

            warm_thermal = summarize_thermal_samples(
                warm_monitor.samples
            )

            append_jsonl(
                warmups_path,
                {
                    "timestamp": now_iso(),
                    "block_id": block["block_id"],
                    "round": block["round"],
                    "backend": backend,
                    "threads": threads,
                    "runtime_alias": runtime_alias,
                    "wall_time_seconds": warm_wall,
                    "inference": warm_inf,
                    "thermal_summary": warm_thermal,
                    "thermal_samples": warm_monitor.samples,
                    "thermal_errors": warm_monitor.errors,
                    "load_log_parsed": load_log_parsed,
                },
            )

            print(
                f"[WARMUP] Fim | "
                f"{fmt_num(warm_inf.get('tokens_per_second'))} tok/s | "
                f"T pico {fmt_num(warm_thermal.get('temp_peak_c'), 1)} °C | "
                f"T fim {fmt_num(warm_thermal.get('temp_end_c'), 1)} °C"
            )

            time.sleep(SETTLE_AFTER_WARMUP_SECONDS)

            for prompt_position, prompt_id in enumerate(
                block["prompt_order"],
                start=1,
            ):
                run_id = f"{block['block_id']}-P{prompt_id}"

                if run_id in successful_ids:
                    continue

                completed_before = len(successful_ids)
                global_number = completed_before + 1
                remaining = total_runs - completed_before
                eta = calculate_eta(
                    process_started,
                    newly_completed,
                    remaining,
                )

                print(
                    f"\n[RUN {global_number}/{total_runs}] "
                    f"{run_id} | {backend.upper()} | "
                    f"{threads} threads | Prompt {prompt_id}"
                )

                if eta is not None:
                    print(f"      ETA: {seconds_to_hms(eta)}")

                monitor = ThermalMonitor(THERMAL_SAMPLE_INTERVAL_SECONDS)
                monitor.start()

                start_temp = monitor.samples[0].get(
                    "cpu_package_temp_c"
                )
                print(
                    f"      CPU Package antes: "
                    f"{fmt_num(start_temp, 1)} °C"
                )
                print(
                    "      Inferindo + amostrando sensores a cada 1 s..."
                )

                try:
                    inference, wall_time = run_sdk_inference(
                        lms,
                        PROMPTS[prompt_id],
                        threads,
                        MAX_OUTPUT_TOKENS,
                    )
                finally:
                    monitor.stop()

                verify_prediction_threads(
                    threads,
                    inference.get("echoed_cpu_threads"),
                )

                thermal_summary = summarize_thermal_samples(
                    monitor.samples
                )

                record = {
                    "status": "success",
                    "timestamp": now_iso(),
                    "run_id": run_id,
                    "block_id": block["block_id"],
                    "round": block["round"],
                    "prompt_position": prompt_position,
                    "backend": backend,
                    "runtime_alias": runtime_alias,
                    "threads": threads,
                    "prompt_id": prompt_id,
                    "prompt": PROMPTS[prompt_id],
                    "load_wall_time_seconds": load_wall_time,
                    "wall_time_seconds": wall_time,
                    "inference": inference,
                    "thermal_summary": thermal_summary,
                    "thermal_samples": monitor.samples,
                    "thermal_errors": monitor.errors,
                    "load_log_parsed": load_log_parsed,
                }

                append_jsonl(runs_path, record)
                successful_ids.add(run_id)
                newly_completed += 1

                write_runs_csv(session_dir)
                write_summary_files(session_dir)

                print(
                    f"      ✓ "
                    f"{fmt_num(inference.get('tokens_per_second'))} tok/s | "
                    f"TTFT "
                    f"{fmt_num(inference.get('time_to_first_token_seconds'))}s | "
                    f"{inference.get('predicted_tokens_count')} tokens | "
                    f"{wall_time:.1f}s"
                )
                print(
                    f"      CPU Package: "
                    f"{fmt_num(thermal_summary.get('temp_start_c'), 1)} → "
                    f"{fmt_num(thermal_summary.get('temp_end_c'), 1)} °C | "
                    f"pico {fmt_num(thermal_summary.get('temp_peak_c'), 1)} °C | "
                    f"média {fmt_num(thermal_summary.get('temp_mean_c'), 1)} °C"
                )
                print(
                    f"      CPU: "
                    f"{fmt_num(thermal_summary.get('cpu_load_mean_percent'), 1)}% | "
                    f"{fmt_num(thermal_summary.get('package_power_mean_w'), 1)} W | "
                    f"P {fmt_num(thermal_summary.get('p_core_clock_mean_mhz'), 0)} MHz | "
                    f"E {fmt_num(thermal_summary.get('e_core_clock_mean_mhz'), 0)} MHz"
                )

                if reasoning_marker_detected(record):
                    print(
                        "      ⚠ Marcador de reasoning detectado no output. "
                        "Isso ficará registrado para auditoria."
                    )

                time.sleep(PAUSE_BETWEEN_PROMPTS_SECONDS)

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            print(f"[BLOCO] ERRO: {exc}")
            record_error(
                session_dir,
                phase="block",
                message=str(exc),
                block=block,
            )

            # Para falhas metodológicas, parar em vez de seguir gerando
            # dados potencialmente inválidos.
            text = str(exc).lower()
            fatal_markers = (
                "threadpool",
                "cpu threads solicitado",
                "configuração falhou",
                "offloaded",
                "log level",
            )
            if any(marker in text for marker in fatal_markers):
                raise

        finally:
            if loaded:
                print("[CLEANUP] Descarregando modelo...")
                unload_all()

            session["successful_runs"] = len(successful_ids)
            session["remaining_runs"] = total_runs - len(successful_ids)
            update_session_file(session_dir, session)

    write_runs_csv(session_dir)
    summary = write_summary_files(session_dir)

    all_done = len(successful_ids) == total_runs

    session["successful_runs"] = len(successful_ids)
    session["remaining_runs"] = total_runs - len(successful_ids)
    session["completed"] = all_done
    if all_done:
        session["completed_at"] = now_iso()
    update_session_file(session_dir, session)

    print_summary(summary)
    print(f"\nResultados:\n  {session_dir}")

    if all_done:
        print("\n✓ Todas as 54 gerações medidas foram concluídas.")
        return 0

    print(
        f"\n⚠ Benchmark incompleto: "
        f"{len(successful_ids)}/{total_runs}."
    )
    print("Rode o mesmo comando para retomar.")
    return 2


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fatorial CPU AVX2 vs Vulkan × 4/6/8 CPU threads, "
            "com telemetria térmica do LibreHardwareMonitor."
        )
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Força uma nova sessão em vez de retomar uma incompleta.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Inicia sem pedir confirmação.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Seed de randomização (padrão: {SEED}).",
    )
    return parser.parse_args()


def choose_session(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], bool]:
    if not args.new:
        incomplete = find_latest_incomplete_session()
        if incomplete is not None:
            return incomplete, load_session(incomplete), True

    session_dir, session = create_session(args.seed)
    return session_dir, session, False


def main() -> int:
    args = parse_args()

    print("\n[PRECHECK] Validando ambiente...")

    ensure_server()

    lms = import_lmstudio_sdk()
    sdk_version = getattr(lms, "__version__", None)
    print(f"  ✓ lmstudio-python: {sdk_version or 'instalado'}")

    model_inventory = verify_selected_variant()
    print(f"  ✓ modelo: {EXPECTED_VARIANT}")

    for backend, alias in RUNTIMES.items():
        verify_runtime_installed(alias)
        print(f"  ✓ {backend}: {alias}")

    sensor_probe = verify_lhm()
    print(
        "  ✓ LibreHardwareMonitor: "
        f"CPU Package = "
        f"{sensor_probe['cpu_package_temp_c']:.1f} °C"
    )
    print(
        "    Sensor: "
        f"{CPU_PACKAGE_TEMP_ID}"
    )

    session_dir, session, resumed = choose_session(args)

    session["model_inventory_at_start"] = model_inventory
    session["runtime_ls_at_start"] = runtime_ls_text()
    session["hardware_probe_at_start"] = sensor_probe
    session["lmstudio_python_version"] = sdk_version
    update_session_file(session_dir, session)

    print("\n" + "=" * 88)
    print(" LocalIA — Backend × CPU Threads Benchmark")
    print("=" * 88)
    print(f" Modelo          : {MODEL_KEY} / Q4_K_M")
    print(f" Backends        : CPU AVX2 2.46.0 + Vulkan 2.46.0")
    print(f" CPU threads     : {THREAD_COUNTS}")
    print(f" GPU offload     : OFF em todos os blocos")
    print(f" Contexto        : {CONTEXT_LENGTH}")
    print(f" Temperatura     : {TEMPERATURE}")
    print(f" Rodadas         : {ROUNDS}")
    print(f" Prompts         : {len(PROMPTS)}")
    print(f" Gerações medidas: {session['planned_measured_runs']}")
    print(
        f" Sensor térmico  : CPU Package "
        f"({THERMAL_SAMPLE_INTERVAL_SECONDS:.0f} amostra/s)"
    )
    print(f" Thermal gate    : ≤ {TARGET_START_TEMP_C:.0f} °C")
    print(f" Sessão          : {session_dir.name}")
    print("=" * 88)

    print(
        "\nIMPORTANTE: o script exigirá que o server log confirme "
        "o threadpool real e que ele seja ≥ 8."
    )
    print(
        "Também exigirá que o SDK devolva cpu_threads igual ao valor "
        "solicitado em cada inferência."
    )

    if resumed:
        print("\nSessão incompleta encontrada; será retomada.")

    if not args.yes:
        answer = input(
            "\nPressione ENTER para começar ou digite 'cancelar': "
        ).strip().lower()
        if answer in {"cancelar", "cancel", "n", "nao", "não"}:
            print("Benchmark cancelado.")
            return 0

    try:
        return execute_benchmark(
            session_dir,
            session,
            lms,
        )
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário.")
        try:
            unload_all()
        except Exception:
            pass
        session["interrupted_at"] = now_iso()
        update_session_file(session_dir, session)
        print(
            "Resultados preservados. "
            "Rode o mesmo comando para retomar."
        )
        return 130
    except Exception as exc:
        print(f"\nERRO FATAL: {exc}")
        try:
            unload_all()
        except Exception:
            pass
        session["fatal_error_at"] = now_iso()
        session["fatal_error"] = str(exc)
        update_session_file(session_dir, session)
        print(
            "\nNenhum resultado já concluído foi perdido. "
            "Corrija o preflight indicado e rode novamente para retomar."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
