from __future__ import annotations

"""
LocalIA — Backend × Threads Benchmark v2
=========================================

Objetivo
--------
Repetir o benchmark fatorial de forma metodologicamente mais forte:

    2 backends × 3 cpuThreads × 3 prompts × 3 rodadas
    = 54 gerações medidas

Backends:
    - CPU AVX2 llama.cpp 2.46.0
    - Vulkan llama.cpp 2.46.0 (GPU offload = OFF)

Threads solicitadas por inferência:
    - 4
    - 6
    - 8

Correções em relação ao benchmark anterior
-------------------------------------------
1. THINKING DESATIVADO de forma explícita:
   - O Python SDK atual não expõe o custom field enableThinking do model.yaml
     na camada pública de config.
   - Portanto, o script obtém o próprio Jinja prompt template do Qwen e
     prefixa:
         {% set enable_thinking = false %}
   - O resultado é validado em TODA geração.
   - Se aparecer reasoning_content, <think>, "Thinking Process:" ou o
     marcador interno de reasoning do LM Studio, o benchmark ABORTA.

2. SEM LIMITE FIXO DE 384 TOKENS:
   - maxTokens=False
   - A geração termina naturalmente por EOS.
   - Se aparecer maxPredictedTokensReached ou contextLengthReached,
     o benchmark ABORTA.

3. VERIFICAÇÃO FORTE DAS THREADS:
   Há três camadas independentes:
   A) solicitado ao SDK: cpuThreads = 4 / 6 / 8
   B) ecoado pelo LM Studio no prediction_config
   C) uso REAL do processo llama-server medido pelo Windows/psutil:
      - CPU seconds do processo
      - "effective CPU cores"
      - CPU % normalizada da máquina
      - CPU time por thread do processo
      - participation ratio (número efetivo de threads CPU-ativas)
      - número mínimo de threads que explica 80% e 90% do CPU time
      - threads com >=25% do CPU time da thread mais ativa

   Além disso, o LibreHardwareMonitor registra:
      - CPU Package / Core Max
      - distância mínima até TjMax
      - Package Power
      - CPU Total
      - clocks médios de P-cores/E-cores
      - carga dos 16 processadores lógicos

4. PREFLIGHT OBRIGATÓRIO:
   Antes das 54 gerações, faz 1 geração por combinação (6 no total).
   Só libera a bateria completa se houver evidência no nível do SO de que
   4, 6 e 8 threads realmente produzem padrões diferentes de execução.
   Se cpuThreads=4/6/8 forem apenas "ecoados" mas o llama-server continuar
   usando essencialmente o mesmo conjunto de workers, o script para.

Requisitos
----------
- LM Studio aberto
- llama.cpp log level = Debug
- LibreHardwareMonitor aberto
- Remote Web Server do LibreHardwareMonitor = ON (porta 8085)
- lmstudio-python
- psutil

Instalação das dependências:
    python -m pip install -U lmstudio psutil
"""

import argparse
import copy
import csv
import hashlib
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
# Config experimental
# ============================================================

API_BASE = "http://127.0.0.1:1234"
LHM_URL = "http://127.0.0.1:8085/data.json"

MODEL_KEY = "qwen/qwen3.5-9b"
EXPECTED_VARIANT = "qwen/qwen3.5-9b@q4_k_m"
INSTANCE_ID = "localia-backend-thread-v2"

RUNTIMES = {
    "cpu": "llama.cpp-win-x86_64-avx2@2.46.0",
    "vulkan": "llama.cpp-win-x86_64-vulkan-avx2@2.46.0",
}

THREAD_COUNTS = [4, 6, 8]

ROUNDS = 3
SEED = 20260927

CONTEXT_LENGTH = 4096
TEMPERATURE = 0.0

# IMPORTANTE:
# False = sem cap explícito de tokens no LM Studio SDK.
MAX_TOKENS: int | bool = False

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

PREFLIGHT_PROMPT = PROMPTS["A"]

WARMUP_PROMPT = (
    "Em aproximadamente 80 palavras, explique por que a água congela "
    "quando sua temperatura cai suficientemente."
)

# Temperatura / telemetria
TARGET_START_TEMP_C = 70.0
THERMAL_STABLE_READINGS = 2
THERMAL_WAIT_MAX_SECONDS = 180
THERMAL_GATE_POLL_SECONDS = 5
MONITOR_INTERVAL_SECONDS = 1.0

# Tempos auxiliares
COOLDOWN_AFTER_UNLOAD_SECONDS = 8
SETTLE_AFTER_RUNTIME_SWITCH_SECONDS = 2
SETTLE_AFTER_LOAD_SECONDS = 3
SETTLE_AFTER_WARMUP_SECONDS = 2
PAUSE_BETWEEN_PROMPTS_SECONDS = 2

LMS_COMMAND_TIMEOUT_SECONDS = 120.0
LMS_LOAD_TIMEOUT_SECONDS = 240.0

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"

NO_THINK_PREFIX = "{%- set enable_thinking = false -%}\n"

THINKING_MARKERS = (
    "<think>",
    "</think>",
    "thinking process:",
    "__lm_studio_internal_lsep_synthetic_reasoning",
)

INVALID_STOP_REASONS = {
    "maxPredictedTokensReached",
    "contextLengthReached",
}


# ============================================================
# Sensores LibreHardwareMonitor observados neste notebook
# ============================================================

CPU_PACKAGE_TEMP_ID = "/intelcpu/0/temperature/14"
CORE_MAX_TEMP_ID = "/intelcpu/0/temperature/0"

DISTANCE_TO_TJMAX_IDS = [
    f"/intelcpu/0/temperature/{i}"
    for i in range(15, 27)
]

CPU_PACKAGE_POWER_ID = "/intelcpu/0/power/0"
CPU_TOTAL_LOAD_ID = "/intelcpu/0/load/0"

# /load/2..9 = 4 P-cores × 2 threads
P_LOGICAL_LOAD_IDS = [
    f"/intelcpu/0/load/{i}"
    for i in range(2, 10)
]

# /load/10..17 = 8 E-cores
E_LOGICAL_LOAD_IDS = [
    f"/intelcpu/0/load/{i}"
    for i in range(10, 18)
]

LOGICAL_LOAD_IDS = P_LOGICAL_LOAD_IDS + E_LOGICAL_LOAD_IDS

P_CORE_CLOCK_IDS = [
    "/intelcpu/0/clock/1",
    "/intelcpu/0/clock/2",
    "/intelcpu/0/clock/3",
    "/intelcpu/0/clock/4",
]

E_CORE_CLOCK_IDS = [
    f"/intelcpu/0/clock/{i}"
    for i in range(5, 13)
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


def fmt_num(value: Any, decimals: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return "n/d"


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
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"AVISO: linha inválida em {path.name}:"
                    f"{line_number}; ignorada."
                )

    return rows


def safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def safe_stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def mean_existing(values: list[float | None]) -> float | None:
    actual = [
        float(v)
        for v in values
        if isinstance(v, (int, float))
    ]
    return safe_mean(actual)


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8", errors="replace")
    ).hexdigest()


# ============================================================
# Dependências
# ============================================================

def import_dependencies():
    try:
        import lmstudio as lms  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Pacote `lmstudio` não instalado.\n"
            "Execute:\n"
            "  python -m pip install -U lmstudio psutil"
        ) from exc

    try:
        import psutil  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Pacote `psutil` não instalado.\n"
            "Execute:\n"
            "  python -m pip install -U lmstudio psutil"
        ) from exc

    setter = getattr(lms, "set_sync_api_timeout", None)

    if callable(setter):
        setter(None)

    return lms, psutil


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
        raise RuntimeError(
            "O comando `lms` não foi encontrado no PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timeout após {timeout:.0f}s executando:\n"
            f"  {' '.join(cmd)}"
        ) from exc

    if result.returncode != 0 and not allow_failure:
        details = (result.stderr or result.stdout or "").strip()

        raise RuntimeError(
            f"Falha ao executar:\n"
            f"  {' '.join(cmd)}\n"
            f"Código: {result.returncode}\n"
            f"{details or '(sem detalhes)'}"
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
        raise RuntimeError(
            f"HTTP {exc.code} em GET {path}: {body}"
        ) from exc


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

    raise RuntimeError(
        f"Modelo não encontrado: {MODEL_KEY}"
    )


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
    run_lms(
        "unload",
        "--all",
        allow_failure=True,
        timeout=60.0,
    )


def select_runtime(alias: str) -> tuple[str, float]:
    unload_all()

    start = time.perf_counter()

    result = run_lms(
        "runtime",
        "select",
        alias,
        timeout=60.0,
    )

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


def audit_loaded_config(
    config: dict[str, Any],
) -> list[str]:
    issues: list[str] = []

    for key, expected in EXPECTED_LOADED_CONFIG.items():
        if key not in config:
            issues.append(
                f"{key}: ausente; esperado={expected!r}"
            )
            continue

        actual = config.get(key)

        if actual != expected:
            issues.append(
                f"{key}: atual={actual!r}; esperado={expected!r}"
            )

    return issues


# ============================================================
# Server log
# ============================================================

def start_server_log_capture(
    log_path: Path,
) -> tuple[subprocess.Popen[Any] | None, Any | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handle = None

    try:
        handle = log_path.open(
            "w",
            encoding="utf-8",
            errors="replace",
        )

        proc = subprocess.Popen(
            [
                "lms",
                "log",
                "stream",
                "--source",
                "server",
                "--json",
            ],
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


def extract_server_log_content(
    log_path: Path,
) -> str:
    if not log_path.exists():
        return ""

    chunks: list[str] = []

    with log_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(obj, dict):
                continue

            content = obj.get("data", {}).get("content")

            if isinstance(content, str):
                chunks.append(content)

    return "\n".join(chunks)


def _last_int(
    pattern: str,
    text: str,
) -> int | None:
    matches = list(
        re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )

    if not matches:
        return None

    try:
        return int(matches[-1].group(1))
    except Exception:
        return None


def _last_str(
    pattern: str,
    text: str,
) -> str | None:
    matches = list(
        re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )

    if not matches:
        return None

    return matches[-1].group(1)


def parse_load_log(
    log_path: Path,
) -> dict[str, Any]:
    text = extract_server_log_content(
        log_path
    )

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
        offloaded_layers = int(
            offload_matches[-1].group(1)
        )
        total_layers = int(
            offload_matches[-1].group(2)
        )

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
        "n_ctx": _last_int(
            r"\bn_ctx\s*=\s*(\d+)",
            text,
        ),
        "n_batch": _last_int(
            r"\bn_batch\s*=\s*(\d+)",
            text,
        ),
        "n_ubatch": _last_int(
            r"\bn_ubatch\s*=\s*(\d+)",
            text,
        ),
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

def parse_localized_number(
    value: Any,
) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    if not text:
        return None

    match = re.search(
        r"[-+]?\d+(?:[.,]\d+)?",
        text,
    )

    if not match:
        return None

    try:
        return float(
            match.group(0).replace(",", ".")
        )
    except ValueError:
        return None


def fetch_lhm_json(
    timeout: float = 3.0,
) -> dict[str, Any]:
    request = Request(
        LHM_URL,
        headers={"Accept": "application/json"},
        method="GET",
    )

    try:
        with urlopen(
            request,
            timeout=timeout,
        ) as response:
            return json.loads(
                response.read().decode("utf-8")
            )

    except (
        HTTPError,
        URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
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
            found.update(
                flatten_sensor_nodes(child)
            )

    return found


def read_hardware_sample() -> dict[str, Any]:
    data = fetch_lhm_json()
    sensors = flatten_sensor_nodes(data)

    def value(sensor_id: str) -> float | None:
        sensor = sensors.get(sensor_id)

        if not sensor:
            return None

        return parse_localized_number(
            sensor.get(
                "RawValue",
                sensor.get("Value"),
            )
        )

    p_clocks = [
        value(sensor_id)
        for sensor_id in P_CORE_CLOCK_IDS
    ]

    e_clocks = [
        value(sensor_id)
        for sensor_id in E_CORE_CLOCK_IDS
    ]

    distances = [
        value(sensor_id)
        for sensor_id in DISTANCE_TO_TJMAX_IDS
    ]

    logical_loads = {
        sensor_id: value(sensor_id)
        for sensor_id in LOGICAL_LOAD_IDS
    }

    valid_distances = [
        x
        for x in distances
        if isinstance(x, (int, float))
    ]

    return {
        "timestamp": now_iso(),
        "monotonic": time.perf_counter(),
        "cpu_package_temp_c": value(
            CPU_PACKAGE_TEMP_ID
        ),
        "core_max_temp_c": value(
            CORE_MAX_TEMP_ID
        ),
        "distance_to_tjmax_min_c": (
            min(valid_distances)
            if valid_distances
            else None
        ),
        "cpu_package_power_w": value(
            CPU_PACKAGE_POWER_ID
        ),
        "cpu_total_load_percent": value(
            CPU_TOTAL_LOAD_ID
        ),
        "p_core_clock_mean_mhz": mean_existing(
            p_clocks
        ),
        "e_core_clock_mean_mhz": mean_existing(
            e_clocks
        ),
        "logical_loads_percent": logical_loads,
    }


def verify_lhm() -> dict[str, Any]:
    sample = read_hardware_sample()

    temp = sample.get(
        "cpu_package_temp_c"
    )

    if not isinstance(temp, (int, float)):
        raise RuntimeError(
            "LibreHardwareMonitor respondeu, mas o sensor "
            f"CPU Package ({CPU_PACKAGE_TEMP_ID}) não foi encontrado."
        )

    return sample


def thermal_gate(
    session_dir: Path,
    block_id: str,
) -> dict[str, Any]:
    deadline = (
        time.monotonic()
        + THERMAL_WAIT_MAX_SECONDS
    )

    stable = 0
    history: list[dict[str, Any]] = []

    print(
        f"[TEMP] Aguardando CPU Package ≤ "
        f"{TARGET_START_TEMP_C:.0f} °C por "
        f"{THERMAL_STABLE_READINGS} leituras consecutivas..."
    )

    while True:
        sample = read_hardware_sample()
        history.append(sample)

        temp = sample.get(
            "cpu_package_temp_c"
        )

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
            append_jsonl(
                session_dir
                / "thermal_gates.jsonl",
                {
                    "timestamp": now_iso(),
                    "block_id": block_id,
                    "status": "stable",
                    "target_temp_c": TARGET_START_TEMP_C,
                    "history": history,
                },
            )

            print(
                "[TEMP] Faixa térmica de início atingida."
            )

            return sample

        if time.monotonic() >= deadline:
            append_jsonl(
                session_dir
                / "thermal_gates.jsonl",
                {
                    "timestamp": now_iso(),
                    "block_id": block_id,
                    "status": "timeout",
                    "target_temp_c": TARGET_START_TEMP_C,
                    "history": history,
                },
            )

            print(
                "[TEMP] Timeout térmico atingido; "
                "o bloco continuará e isso ficará registrado."
            )

            return sample

        time.sleep(
            THERMAL_GATE_POLL_SECONDS
        )


# ============================================================
# Processo llama-server e CPU time por thread
# ============================================================

def process_text(
    proc: Any,
) -> str:
    parts: list[str] = []

    try:
        parts.append(
            str(proc.name())
        )
    except Exception:
        pass

    try:
        cmdline = proc.cmdline()

        if cmdline:
            parts.append(
                " ".join(
                    str(x)
                    for x in cmdline
                )
            )
    except Exception:
        pass

    try:
        parts.append(
            str(proc.exe())
        )
    except Exception:
        pass

    return " ".join(parts)


def snapshot_pids(
    psutil: Any,
) -> set[int]:
    return set(
        psutil.pids()
    )


def process_info(
    proc: Any,
) -> dict[str, Any]:
    try:
        rss = proc.memory_info().rss
    except Exception:
        rss = None

    try:
        create_time = proc.create_time()
    except Exception:
        create_time = None

    return {
        "pid": proc.pid,
        "name": (
            proc.name()
            if proc.is_running()
            else None
        ),
        "rss_bytes": rss,
        "create_time": create_time,
        "text": process_text(proc),
    }


def find_llama_server_process(
    psutil: Any,
    pids_before_load: set[int],
) -> tuple[Any, dict[str, Any]]:
    candidates: list[tuple[float, Any, dict[str, Any]]] = []

    for proc in psutil.process_iter():
        try:
            info = process_info(proc)
            text = str(info.get("text") or "").lower()
            rss = info.get("rss_bytes") or 0

            # O processo do runtime contém "llama" no nome/command line
            # nas builds atuais. O peso de "new PID" evita escolher
            # algum utilitário residual.
            contains_llama = "llama" in text
            contains_qwen = (
                "qwen3.5-9b" in text
                or "qwen3.5" in text
            )
            is_new = proc.pid not in pids_before_load

            if not contains_llama and not contains_qwen:
                continue

            score = 0.0

            if is_new:
                score += 1000.0

            if "llama-server" in text:
                score += 500.0
            elif "llama" in text:
                score += 250.0

            if contains_qwen:
                score += 200.0

            # Modelo ~6 GB: RSS grande é um sinal forte.
            score += min(
                float(rss) / (1024 ** 3),
                20.0,
            ) * 10.0

            candidates.append(
                (
                    score,
                    proc,
                    info,
                )
            )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            continue

    if not candidates:
        raise RuntimeError(
            "Não consegui identificar o processo llama-server "
            "depois do load. Rode o terminal como Administrador "
            "ou envie a saída de `Get-Process`/Task Manager."
        )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    score, proc, info = candidates[0]

    info["selection_score"] = score
    info["candidate_count"] = len(candidates)
    info["all_candidates"] = [
        {
            "score": item[0],
            **item[2],
        }
        for item in candidates[:10]
    ]

    return proc, info


def thread_cpu_total(
    thread_info: Any,
) -> float:
    return float(
        thread_info.user_time
        + thread_info.system_time
    )


def process_cpu_total(
    proc: Any,
) -> float:
    times = proc.cpu_times()

    return float(
        times.user
        + times.system
    )


def minimum_threads_for_fraction(
    deltas: list[float],
    fraction: float,
) -> int | None:
    positive = sorted(
        [
            x
            for x in deltas
            if x > 0
        ],
        reverse=True,
    )

    total = sum(positive)

    if not positive or total <= 0:
        return None

    target = total * fraction
    running = 0.0

    for index, value in enumerate(
        positive,
        start=1,
    ):
        running += value

        if running >= target:
            return index

    return len(positive)


def participation_ratio(
    deltas: list[float],
) -> float | None:
    positive = [
        x
        for x in deltas
        if x > 0
    ]

    if not positive:
        return None

    numerator = sum(positive) ** 2
    denominator = sum(
        x ** 2
        for x in positive
    )

    if denominator <= 0:
        return None

    return numerator / denominator


class RunMonitor:
    """
    Monitora simultaneamente:
      - sensores LHM;
      - CPU time TOTAL do processo llama-server;
      - CPU time individual das threads do llama-server.

    Isso mede uso real no nível do SO, não apenas a configuração
    solicitada ao SDK.
    """

    def __init__(
        self,
        process: Any,
        psutil: Any,
        interval: float,
    ) -> None:
        self.process = process
        self.psutil = psutil
        self.interval = interval

        self.hardware_samples: list[
            dict[str, Any]
        ] = []

        self.errors: list[str] = []

        self.thread_first: dict[int, float] = {}
        self.thread_last: dict[int, float] = {}

        self.process_cpu_start: float | None = None
        self.process_cpu_end: float | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _capture_process(self) -> None:
        try:
            if self.process_cpu_start is None:
                self.process_cpu_start = process_cpu_total(
                    self.process
                )

            self.process_cpu_end = process_cpu_total(
                self.process
            )

            for info in self.process.threads():
                total = thread_cpu_total(info)

                if info.id not in self.thread_first:
                    self.thread_first[
                        info.id
                    ] = total

                self.thread_last[
                    info.id
                ] = total

        except Exception as exc:
            self.errors.append(
                f"process-monitor: {exc}"
            )

    def _capture_hardware(self) -> None:
        try:
            self.hardware_samples.append(
                read_hardware_sample()
            )
        except Exception as exc:
            self.errors.append(
                f"lhm-monitor: {exc}"
            )

    def _capture(self) -> None:
        self._capture_process()
        self._capture_hardware()

    def _loop(self) -> None:
        while not self._stop.wait(
            self.interval
        ):
            self._capture()

    def start(self) -> None:
        self._capture()

        self._thread = threading.Thread(
            target=self._loop,
            name="localia-run-monitor",
            daemon=True,
        )

        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(
                timeout=max(
                    2.0,
                    self.interval * 2,
                )
            )

        self._capture()

    def summarize(
        self,
        wall_time_seconds: float,
    ) -> dict[str, Any]:
        hardware = summarize_hardware_samples(
            self.hardware_samples
        )

        thread_deltas: list[
            tuple[int, float]
        ] = []

        for tid, end_value in self.thread_last.items():
            start_value = self.thread_first.get(
                tid,
                end_value,
            )

            delta = max(
                0.0,
                end_value - start_value,
            )

            if delta > 0:
                thread_deltas.append(
                    (
                        tid,
                        delta,
                    )
                )

        thread_deltas.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        delta_values = [
            delta
            for _, delta in thread_deltas
        ]

        max_thread_delta = (
            max(delta_values)
            if delta_values
            else None
        )

        process_cpu_seconds = None

        if (
            self.process_cpu_start is not None
            and self.process_cpu_end is not None
        ):
            process_cpu_seconds = max(
                0.0,
                self.process_cpu_end
                - self.process_cpu_start,
            )

        logical_cpus = (
            self.psutil.cpu_count(
                logical=True
            )
            or 1
        )

        effective_cores = None
        process_machine_cpu_percent = None

        if (
            process_cpu_seconds is not None
            and wall_time_seconds > 0
        ):
            effective_cores = (
                process_cpu_seconds
                / wall_time_seconds
            )

            process_machine_cpu_percent = (
                effective_cores
                / logical_cpus
                * 100.0
            )

        active_25pct = None

        if (
            max_thread_delta is not None
            and max_thread_delta > 0
        ):
            threshold = (
                max_thread_delta * 0.25
            )

            active_25pct = sum(
                1
                for value in delta_values
                if value >= threshold
            )

        thread_summary = {
            "process_pid": self.process.pid,
            "logical_cpu_count": logical_cpus,
            "process_cpu_seconds": process_cpu_seconds,
            "effective_cpu_cores": effective_cores,
            "process_machine_cpu_percent": (
                process_machine_cpu_percent
            ),
            "thread_cpu_seconds_sum": sum(
                delta_values
            ),
            "thread_count_with_cpu_delta": len(
                delta_values
            ),
            "active_threads_ge_25pct_of_top": (
                active_25pct
            ),
            "dominant_threads_80pct": (
                minimum_threads_for_fraction(
                    delta_values,
                    0.80,
                )
            ),
            "dominant_threads_90pct": (
                minimum_threads_for_fraction(
                    delta_values,
                    0.90,
                )
            ),
            "effective_active_thread_count": (
                participation_ratio(
                    delta_values
                )
            ),
            "top_threads_cpu_seconds": [
                {
                    "tid": tid,
                    "cpu_seconds": delta,
                }
                for tid, delta in thread_deltas[:16]
            ],
        }

        return {
            "hardware": hardware,
            "process_threads": thread_summary,
            "monitor_errors": self.errors,
            "hardware_samples": self.hardware_samples,
        }


def summarize_hardware_samples(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [
            float(sample[key])
            for sample in samples
            if isinstance(
                sample.get(key),
                (int, float),
            )
        ]

    package_temp = vals(
        "cpu_package_temp_c"
    )
    core_max_temp = vals(
        "core_max_temp_c"
    )
    distance = vals(
        "distance_to_tjmax_min_c"
    )
    power = vals(
        "cpu_package_power_w"
    )
    cpu_total = vals(
        "cpu_total_load_percent"
    )
    p_clock = vals(
        "p_core_clock_mean_mhz"
    )
    e_clock = vals(
        "e_core_clock_mean_mhz"
    )

    logical_means: dict[
        str,
        float | None,
    ] = {}

    for sensor_id in LOGICAL_LOAD_IDS:
        values = []

        for sample in samples:
            loads = (
                sample.get(
                    "logical_loads_percent"
                )
                or {}
            )

            value = loads.get(
                sensor_id
            )

            if isinstance(
                value,
                (int, float),
            ):
                values.append(
                    float(value)
                )

        logical_means[
            sensor_id
        ] = safe_mean(values)

    p_logical = [
        value
        for sensor_id, value in logical_means.items()
        if sensor_id in P_LOGICAL_LOAD_IDS
        and isinstance(
            value,
            (int, float),
        )
    ]

    e_logical = [
        value
        for sensor_id, value in logical_means.items()
        if sensor_id in E_LOGICAL_LOAD_IDS
        and isinstance(
            value,
            (int, float),
        )
    ]

    return {
        "sample_count": len(samples),
        "temp_start_c": (
            package_temp[0]
            if package_temp
            else None
        ),
        "temp_end_c": (
            package_temp[-1]
            if package_temp
            else None
        ),
        "temp_peak_c": (
            max(package_temp)
            if package_temp
            else None
        ),
        "temp_mean_c": safe_mean(
            package_temp
        ),
        "core_max_peak_c": (
            max(core_max_temp)
            if core_max_temp
            else None
        ),
        "distance_to_tjmax_min_c": (
            min(distance)
            if distance
            else None
        ),
        "package_power_mean_w": safe_mean(
            power
        ),
        "package_power_peak_w": (
            max(power)
            if power
            else None
        ),
        "cpu_total_load_mean_percent": (
            safe_mean(cpu_total)
        ),
        "cpu_total_load_peak_percent": (
            max(cpu_total)
            if cpu_total
            else None
        ),
        "p_core_clock_mean_mhz": safe_mean(
            p_clock
        ),
        "e_core_clock_mean_mhz": safe_mean(
            e_clock
        ),
        "p_logical_load_mean_percent": (
            safe_mean(p_logical)
        ),
        "e_logical_load_mean_percent": (
            safe_mean(e_logical)
        ),
        "logical_load_means_percent": (
            logical_means
        ),
        "seconds_at_or_above_80c": (
            sum(
                1
                for x in package_temp
                if x >= 80.0
            )
            * MONITOR_INTERVAL_SECONDS
        ),
        "seconds_at_or_above_85c": (
            sum(
                1
                for x in package_temp
                if x >= 85.0
            )
            * MONITOR_INTERVAL_SECONDS
        ),
        "seconds_at_or_above_90c": (
            sum(
                1
                for x in package_temp
                if x >= 90.0
            )
            * MONITOR_INTERVAL_SECONDS
        ),
    }


# ============================================================
# SDK / prompt template / thinking
# ============================================================

def getattr_any(
    obj: Any,
    names: list[str],
) -> Any:
    for name in names:
        try:
            value = getattr(
                obj,
                name,
            )
        except Exception:
            continue

        if value is not None:
            return value

    return None


def serialize_object(
    obj: Any,
) -> Any:
    if obj is None:
        return None

    for method_name in (
        "to_dict",
        "model_dump",
        "dict",
    ):
        method = getattr(
            obj,
            method_name,
            None,
        )

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
            value = getattr(
                obj,
                name,
            )
        except Exception:
            continue

        if callable(value):
            continue

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                type(None),
                list,
                dict,
            ),
        ):
            result[
                name
            ] = value

    return result or str(obj)


def find_key(
    obj: dict[str, Any],
    *names: str,
) -> tuple[str | None, Any]:
    for name in names:
        if name in obj:
            return name, obj[name]

    lower_map = {
        str(key).lower(): key
        for key in obj
    }

    for name in names:
        actual = lower_map.get(
            name.lower()
        )

        if actual is not None:
            return actual, obj[actual]

    return None, None


def discover_default_prompt_template(
    model: Any,
) -> dict[str, Any]:
    """
    Faz uma geração mínima de 1 token apenas para obter o promptTemplate
    efetivo do próprio modelo/LM Studio.

    Essa geração NÃO faz parte do benchmark.
    """

    result = model.respond(
        "Responda apenas com OK.",
        config={
            "temperature": TEMPERATURE,
            "maxTokens": 1,
            "cpuThreads": max(
                THREAD_COUNTS
            ),
        },
    )

    prediction_config = serialize_object(
        getattr(
            result,
            "prediction_config",
            None,
        )
    )

    if not isinstance(
        prediction_config,
        dict,
    ):
        raise RuntimeError(
            "Não consegui ler prediction_config durante "
            "a descoberta do prompt template."
        )

    _, prompt_template = find_key(
        prediction_config,
        "promptTemplate",
        "prompt_template",
    )

    if not isinstance(
        prompt_template,
        dict,
    ):
        raise RuntimeError(
            "O LM Studio não devolveu um promptTemplate "
            "estruturado para o modelo."
        )

    return copy.deepcopy(
        prompt_template
    )


def build_no_think_prompt_template(
    prompt_template: dict[str, Any],
) -> dict[str, Any]:
    patched = copy.deepcopy(
        prompt_template
    )

    _, jinja_container = find_key(
        patched,
        "jinjaPromptTemplate",
        "jinja_prompt_template",
    )

    target_dict = None

    if isinstance(
        jinja_container,
        dict,
    ):
        target_dict = jinja_container
    else:
        target_dict = patched

    template_key, template = find_key(
        target_dict,
        "template",
    )

    if (
        template_key is None
        or not isinstance(
            template,
            str,
        )
    ):
        raise RuntimeError(
            "Não consegui localizar o texto Jinja "
            "dentro do promptTemplate."
        )

    if "enable_thinking" not in template:
        raise RuntimeError(
            "O prompt template atual não referencia `enable_thinking`. "
            "Não vou assumir que o patch de Think OFF funcionaria."
        )

    if not template.startswith(
        NO_THINK_PREFIX
    ):
        template = (
            NO_THINK_PREFIX
            + template
        )

    target_dict[
        template_key
    ] = template

    return patched


def detect_thinking(
    content: str,
    reasoning_content: Any,
) -> list[str]:
    problems: list[str] = []

    if reasoning_content:
        problems.append(
            "reasoning_content não está vazio"
        )

    lower = content.lower()

    for marker in THINKING_MARKERS:
        if marker.lower() in lower:
            problems.append(
                f"marcador detectado: {marker}"
            )

    return problems


def run_sdk_inference(
    model: Any,
    prompt: str,
    cpu_threads: int,
    no_think_prompt_template: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    config = {
        "temperature": TEMPERATURE,
        "maxTokens": MAX_TOKENS,
        "cpuThreads": cpu_threads,
        "promptTemplate": no_think_prompt_template,
    }

    start = time.perf_counter()

    result = model.respond(
        prompt,
        config=config,
    )

    wall = (
        time.perf_counter()
        - start
    )

    stats = getattr(
        result,
        "stats",
        None,
    )

    prediction_config_obj = getattr(
        result,
        "prediction_config",
        None,
    )

    prediction_config = serialize_object(
        prediction_config_obj
    )

    content = getattr_any(
        result,
        [
            "content",
            "text",
        ],
    )

    if content is None:
        content = str(result)

    content = str(content)

    reasoning_content = getattr_any(
        result,
        [
            "reasoning_content",
            "reasoningContent",
        ],
    )

    tps = getattr_any(
        stats,
        [
            "tokens_per_second",
            "tokensPerSecond",
        ],
    )

    ttft = getattr_any(
        stats,
        [
            "time_to_first_token_sec",
            "timeToFirstTokenSec",
        ],
    )

    predicted = getattr_any(
        stats,
        [
            "predicted_tokens_count",
            "predictedTokensCount",
        ],
    )

    stop_reason = getattr_any(
        stats,
        [
            "stop_reason",
            "stopReason",
        ],
    )

    echoed_threads = None

    if isinstance(
        prediction_config,
        dict,
    ):
        _, echoed_threads = find_key(
            prediction_config,
            "cpuThreads",
            "cpu_threads",
        )

    thinking_problems = detect_thinking(
        content,
        reasoning_content,
    )

    return {
        "content": content,
        "response_sha256": sha256_text(
            content
        ),
        "tokens_per_second": tps,
        "time_to_first_token_seconds": ttft,
        "predicted_tokens_count": predicted,
        "stop_reason": stop_reason,
        "requested_cpu_threads": cpu_threads,
        "echoed_cpu_threads": echoed_threads,
        "max_tokens_requested": MAX_TOKENS,
        "thinking_control_method": (
            "patched_model_jinja_enable_thinking_false"
        ),
        "thinking_problems": thinking_problems,
        "thinking_off_verified": (
            len(thinking_problems) == 0
        ),
        "reasoning_content": reasoning_content,
        "prediction_config": prediction_config,
        "stats_raw": serialize_object(
            stats
        ),
    }, wall


def verify_inference_result(
    inference: dict[str, Any],
    requested_threads: int,
) -> None:
    echoed = inference.get(
        "echoed_cpu_threads"
    )

    if echoed is None:
        raise RuntimeError(
            "O SDK não devolveu cpuThreads na configuração "
            "efetiva da predição."
        )

    try:
        echoed_int = int(
            echoed
        )
    except Exception as exc:
        raise RuntimeError(
            f"cpuThreads ecoado é inválido: {echoed!r}"
        ) from exc

    if echoed_int != requested_threads:
        raise RuntimeError(
            "cpuThreads solicitado e ecoado não coincidem.\n"
            f"Solicitado: {requested_threads}\n"
            f"Ecoado:     {echoed_int}"
        )

    if not inference.get(
        "thinking_off_verified"
    ):
        raise RuntimeError(
            "THINKING NÃO FOI DESATIVADO.\n"
            + "\n".join(
                str(x)
                for x in inference.get(
                    "thinking_problems",
                    []
                )
            )
        )

    stop_reason = inference.get(
        "stop_reason"
    )

    if stop_reason in INVALID_STOP_REASONS:
        raise RuntimeError(
            "A geração terminou por limite artificial/contexto, "
            "não por conclusão natural.\n"
            f"stop_reason = {stop_reason}"
        )


# ============================================================
# Load de um bloco
# ============================================================

def load_block(
    *,
    session_dir: Path,
    block_label: str,
    backend: str,
    runtime_alias: str,
    psutil: Any,
) -> tuple[
    Any,
    dict[str, Any],
    float,
    dict[str, Any],
    dict[str, Any],
]:
    """
    Retorna:
      process
      process_info
      load_wall_time
      load_log_parsed
      loaded_config
    """

    ensure_server()

    print(
        "[SWITCH] Descarregando modelo anterior..."
    )

    unload_all()

    if COOLDOWN_AFTER_UNLOAD_SECONDS:
        time.sleep(
            COOLDOWN_AFTER_UNLOAD_SECONDS
        )

    print(
        f"[SWITCH] Selecionando "
        f"{backend.upper()}..."
    )

    selected_line, switch_seconds = (
        select_runtime(
            runtime_alias
        )
    )

    print(
        f"[SWITCH] OK — {selected_line}"
    )

    append_jsonl(
        session_dir
        / "runtime_switches.jsonl",
        {
            "timestamp": now_iso(),
            "block_id": block_label,
            "backend": backend,
            "runtime_alias": runtime_alias,
            "switch_wall_time_seconds": switch_seconds,
            "selected_line": selected_line,
        },
    )

    time.sleep(
        SETTLE_AFTER_RUNTIME_SWITCH_SECONDS
    )

    thermal_gate(
        session_dir,
        block_label,
    )

    pids_before = snapshot_pids(
        psutil
    )

    log_path = (
        session_dir
        / "load_logs"
        / f"{block_label}.jsonl"
    )

    log_proc, log_handle = (
        start_server_log_capture(
            log_path
        )
    )

    print(
        "[LOAD] Carregando modelo com "
        "GPU offload OFF..."
    )

    try:
        load_result, load_wall_time = (
            load_model()
        )

        time.sleep(1.0)

    finally:
        stop_log_capture(
            log_proc,
            log_handle,
        )

    load_log_parsed = parse_load_log(
        log_path
    )

    models_response = api_get(
        "/api/v1/models"
    )

    _, loaded_instance = (
        find_loaded_instance(
            models_response
        )
    )

    if loaded_instance is None:
        raise RuntimeError(
            f"Instância `{INSTANCE_ID}` "
            "não apareceu na API."
        )

    loaded_config = (
        loaded_instance.get("config")
        or {}
    )

    config_issues = audit_loaded_config(
        loaded_config
    )

    if config_issues:
        raise RuntimeError(
            "Auditoria da configuração de load falhou: "
            + "; ".join(
                config_issues
            )
        )

    pool_size = load_log_parsed.get(
        "threadpool_n_threads"
    )

    if not isinstance(
        pool_size,
        int,
    ):
        raise RuntimeError(
            "Não consegui confirmar "
            "`llama threadpool init, n_threads = ...`.\n"
            "Confirme llama.cpp log level = Debug."
        )

    if pool_size < max(
        THREAD_COUNTS
    ):
        raise RuntimeError(
            "O threadpool carregado é menor do que o sweep.\n"
            f"threadpool={pool_size}, "
            f"máximo solicitado={max(THREAD_COUNTS)}"
        )

    offloaded = load_log_parsed.get(
        "offloaded_layers"
    )

    if backend == "vulkan":
        if offloaded is None:
            raise RuntimeError(
                "Não consegui auditar o número "
                "de camadas offloaded."
            )

        if int(offloaded) != 0:
            raise RuntimeError(
                f"Vulkan offloaded {offloaded} camada(s); "
                "esperado: 0."
            )

    elif (
        offloaded is not None
        and int(offloaded) != 0
    ):
        raise RuntimeError(
            f"CPU registrou {offloaded} camada(s) "
            "offloaded; esperado: 0."
        )

    model_process, model_process_info = (
        find_llama_server_process(
            psutil,
            pids_before,
        )
    )

    print(
        f"[LOAD] OK em {load_wall_time:.2f}s | "
        f"PID llama-server={model_process.pid}"
    )

    print(
        "[LOAD-AUDIT] "
        f"threadpool={pool_size} | "
        f"offload={offloaded}/"
        f"{load_log_parsed.get('total_layers')} | "
        f"ctx={load_log_parsed.get('n_ctx')} | "
        f"batch={load_log_parsed.get('n_batch')} | "
        f"ubatch={load_log_parsed.get('n_ubatch')}"
    )

    append_jsonl(
        session_dir
        / "loads.jsonl",
        {
            "timestamp": now_iso(),
            "block_id": block_label,
            "backend": backend,
            "runtime_alias": runtime_alias,
            "load_wall_time_seconds": (
                load_wall_time
            ),
            "loaded_config": loaded_config,
            "load_log_parsed": (
                load_log_parsed
            ),
            "model_process": (
                model_process_info
            ),
            "load_log_path": str(
                log_path
            ),
            "lms_stdout": (
                load_result.stdout.strip()
            ),
            "lms_stderr": (
                load_result.stderr.strip()
            ),
        },
    )

    time.sleep(
        SETTLE_AFTER_LOAD_SECONDS
    )

    return (
        model_process,
        model_process_info,
        load_wall_time,
        load_log_parsed,
        loaded_config,
    )


# ============================================================
# Uma inferência instrumentada
# ============================================================

def instrumented_inference(
    *,
    model: Any,
    psutil: Any,
    process: Any,
    prompt: str,
    threads: int,
    no_think_template: dict[str, Any],
) -> tuple[
    dict[str, Any],
    float,
    dict[str, Any],
]:
    monitor = RunMonitor(
        process=process,
        psutil=psutil,
        interval=MONITOR_INTERVAL_SECONDS,
    )

    monitor.start()

    try:
        inference, wall_time = (
            run_sdk_inference(
                model,
                prompt,
                threads,
                no_think_template,
            )
        )
    finally:
        monitor.stop()

    verify_inference_result(
        inference,
        threads,
    )

    telemetry = monitor.summarize(
        wall_time
    )

    return (
        inference,
        wall_time,
        telemetry,
    )


# ============================================================
# Verificação das threads
# ============================================================

def thread_metrics(
    record: dict[str, Any],
) -> dict[str, Any]:
    return (
        record.get("telemetry", {})
        .get("process_threads", {})
    )


def estimate_workers(
    record: dict[str, Any],
) -> float | None:
    """
    Usa três visões do CPU time por thread.

    O participation ratio é contínuo; dominant90 e active25 são discretos.
    Retornamos a mediana das métricas disponíveis como estimativa prática.
    """

    metrics = thread_metrics(
        record
    )

    values: list[float] = []

    effective = metrics.get(
        "effective_active_thread_count"
    )

    dominant90 = metrics.get(
        "dominant_threads_90pct"
    )

    active25 = metrics.get(
        "active_threads_ge_25pct_of_top"
    )

    for value in (
        effective,
        dominant90,
        active25,
    ):
        if isinstance(
            value,
            (int, float),
        ):
            values.append(
                float(value)
            )

    return (
        statistics.median(values)
        if values
        else None
    )


def validate_thread_preflight(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Não assume que o worker count precisa ser exatamente igual a cpuThreads.
    O llama-server possui helpers/threads de serviço.

    Mas exige duas coisas:
      1) cada valor precisa ser plausível (não 4 -> ~8 workers);
      2) 4 -> 6 -> 8 deve produzir crescimento observável no uso REAL
         do processo, por backend.

    Se 4/6/8 ficarem essencialmente iguais, o sweep NÃO é considerado válido.
    """

    report: dict[str, Any] = {
        "valid": True,
        "problems": [],
        "by_backend": {},
    }

    for backend in RUNTIMES:
        backend_records = [
            record
            for record in records
            if record.get("backend") == backend
        ]

        by_threads: dict[
            int,
            dict[str, Any],
        ] = {}

        for threads in THREAD_COUNTS:
            matches = [
                record
                for record in backend_records
                if int(
                    record.get(
                        "threads",
                        -1,
                    )
                )
                == threads
            ]

            if not matches:
                report["valid"] = False
                report["problems"].append(
                    f"{backend}: sem preflight para {threads} threads"
                )
                continue

            record = matches[0]
            metrics = thread_metrics(
                record
            )

            estimate = estimate_workers(
                record
            )

            by_threads[
                threads
            ] = {
                "worker_estimate": estimate,
                "effective_active_thread_count": metrics.get(
                    "effective_active_thread_count"
                ),
                "dominant_threads_80pct": metrics.get(
                    "dominant_threads_80pct"
                ),
                "dominant_threads_90pct": metrics.get(
                    "dominant_threads_90pct"
                ),
                "active_threads_ge_25pct_of_top": metrics.get(
                    "active_threads_ge_25pct_of_top"
                ),
                "effective_cpu_cores": metrics.get(
                    "effective_cpu_cores"
                ),
                "process_machine_cpu_percent": metrics.get(
                    "process_machine_cpu_percent"
                ),
            }

            if estimate is None:
                report["valid"] = False
                report["problems"].append(
                    f"{backend}/{threads}: "
                    "sem estimativa OS-level de workers"
                )
                continue

            # Margem para main/helper thread.
            if estimate > (
                threads + 2.0
            ):
                report["valid"] = False
                report["problems"].append(
                    f"{backend}/{threads}: "
                    f"estimativa de workers={estimate:.2f}, "
                    "muito acima do solicitado"
                )

        report["by_backend"][
            backend
        ] = by_threads

        estimates = {
            threads: (
                by_threads.get(
                    threads,
                    {}
                ).get(
                    "worker_estimate"
                )
            )
            for threads in THREAD_COUNTS
        }

        e4 = estimates.get(4)
        e6 = estimates.get(6)
        e8 = estimates.get(8)

        if all(
            isinstance(
                value,
                (int, float),
            )
            for value in (
                e4,
                e6,
                e8,
            )
        ):
            # O principal teste: se 4 e 8 dão quase o mesmo conjunto
            # efetivo de workers, cpuThreads não está realmente variando
            # o executor de forma suficiente para um benchmark válido.
            if float(e8) - float(e4) < 1.5:
                report["valid"] = False
                report["problems"].append(
                    f"{backend}: uso real de threads não escalou "
                    f"de 4→8 (estimativa {e4:.2f}→{e8:.2f})"
                )

            # Permite algum ruído, mas rejeita inversão grande.
            if float(e6) + 0.75 < float(e4):
                report["valid"] = False
                report["problems"].append(
                    f"{backend}: 6 threads aparentam usar "
                    "menos workers que 4 de forma incompatível"
                )

            if float(e8) + 0.75 < float(e6):
                report["valid"] = False
                report["problems"].append(
                    f"{backend}: 8 threads aparentam usar "
                    "menos workers que 6 de forma incompatível"
                )

    return report


# ============================================================
# Preflight
# ============================================================

def run_preflight(
    *,
    session_dir: Path,
    lms: Any,
    psutil: Any,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    print("\n" + "=" * 94)
    print(" PREFLIGHT — VERIFICANDO THINK OFF + THREADS REAIS NO SO")
    print("=" * 94)

    records: list[dict[str, Any]] = []
    no_think_template: dict[
        str,
        Any,
    ] | None = None

    # Ordem balanceada/randomizada, mas pequena.
    configs = [
        (
            backend,
            runtime_alias,
            threads,
        )
        for backend, runtime_alias in RUNTIMES.items()
        for threads in THREAD_COUNTS
    ]

    rng = random.Random(
        SEED + 1001
    )

    rng.shuffle(
        configs
    )

    for index, (
        backend,
        runtime_alias,
        threads,
    ) in enumerate(
        configs,
        start=1,
    ):
        label = (
            f"PREFLIGHT-{index}-"
            f"{backend.upper()}-T{threads}"
        )

        print("\n" + "-" * 94)
        print(
            f"[PREFLIGHT {index}/{len(configs)}] "
            f"{backend.upper()} / {threads} threads"
        )

        process = None

        try:
            (
                process,
                process_info_data,
                load_wall,
                load_log,
                _loaded_config,
            ) = load_block(
                session_dir=session_dir,
                block_label=label,
                backend=backend,
                runtime_alias=runtime_alias,
                psutil=psutil,
            )

            model = lms.llm()

            if no_think_template is None:
                print(
                    "[THINK] Descobrindo o Jinja oficial do modelo..."
                )

                default_template = (
                    discover_default_prompt_template(
                        model
                    )
                )

                no_think_template = (
                    build_no_think_prompt_template(
                        default_template
                    )
                )

                atomic_write_json(
                    session_dir
                    / "no_think_prompt_template.json",
                    no_think_template,
                )

                print(
                    "[THINK] Jinja patchado com "
                    "`enable_thinking = false`."
                )

            print(
                "[TEST] Gerando sem limite fixo de tokens..."
            )

            inference, wall, telemetry = (
                instrumented_inference(
                    model=model,
                    psutil=psutil,
                    process=process,
                    prompt=PREFLIGHT_PROMPT,
                    threads=threads,
                    no_think_template=no_think_template,
                )
            )

            record = {
                "status": "success",
                "timestamp": now_iso(),
                "label": label,
                "backend": backend,
                "runtime_alias": runtime_alias,
                "threads": threads,
                "load_wall_time_seconds": load_wall,
                "load_log_parsed": load_log,
                "model_process": process_info_data,
                "wall_time_seconds": wall,
                "inference": inference,
                "telemetry": telemetry,
            }

            append_jsonl(
                session_dir
                / "preflight.jsonl",
                record,
            )

            records.append(
                record
            )

            metrics = (
                telemetry[
                    "process_threads"
                ]
            )

            hardware = (
                telemetry[
                    "hardware"
                ]
            )

            print(
                f"[OK] {fmt_num(inference.get('tokens_per_second'))} tok/s | "
                f"{inference.get('predicted_tokens_count')} tokens | "
                f"stop={inference.get('stop_reason')}"
            )

            print(
                "[THREAD-OS] "
                f"effective cores="
                f"{fmt_num(metrics.get('effective_cpu_cores'), 2)} | "
                f"machine CPU="
                f"{fmt_num(metrics.get('process_machine_cpu_percent'), 1)}% | "
                f"participation="
                f"{fmt_num(metrics.get('effective_active_thread_count'), 2)} | "
                f"80%CPU={metrics.get('dominant_threads_80pct')} thr | "
                f"90%CPU={metrics.get('dominant_threads_90pct')} thr | "
                f">=25%top={metrics.get('active_threads_ge_25pct_of_top')} thr"
            )

            print(
                "[THERMAL] "
                f"Package pico="
                f"{fmt_num(hardware.get('temp_peak_c'), 1)} °C | "
                f"Core Max pico="
                f"{fmt_num(hardware.get('core_max_peak_c'), 1)} °C | "
                f"dist. TjMax mínima="
                f"{fmt_num(hardware.get('distance_to_tjmax_min_c'), 1)} °C"
            )

        finally:
            unload_all()

    if no_think_template is None:
        raise RuntimeError(
            "Preflight não conseguiu gerar o prompt template Think OFF."
        )

    report = validate_thread_preflight(
        records
    )

    atomic_write_json(
        session_dir
        / "preflight_report.json",
        report,
    )

    print("\n" + "=" * 94)
    print(" RESULTADO DO PREFLIGHT")
    print("=" * 94)

    for backend, values in report[
        "by_backend"
    ].items():
        print(
            f"{backend.upper()}:"
        )

        for threads in THREAD_COUNTS:
            item = values.get(
                threads,
                {}
            )

            print(
                f"  {threads} threads -> "
                f"workers~{fmt_num(item.get('worker_estimate'), 2)} | "
                f"effective cores "
                f"{fmt_num(item.get('effective_cpu_cores'), 2)} | "
                f"process CPU "
                f"{fmt_num(item.get('process_machine_cpu_percent'), 1)}%"
            )

    if not report["valid"]:
        print("\n✗ PREFLIGHT REPROVADO.")

        for problem in report[
            "problems"
        ]:
            print(
                f"  - {problem}"
            )

        print(
            "\nA bateria de 54 execuções NÃO será iniciada. "
            "Isso evita repetir um benchmark em que 4/6/8 "
            "são apenas rótulos e não mudanças reais de execução."
        )

        raise RuntimeError(
            "Preflight de controle real de threads falhou. "
            "Veja preflight_report.json."
        )

    print(
        "\n✓ PREFLIGHT APROVADO: "
        "Think OFF e variação OS-level de threads confirmados."
    )

    return (
        records,
        no_think_template,
    )


# ============================================================
# Schedule / sessão
# ============================================================

def build_schedule(
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(
        seed
    )

    configs = [
        {
            "backend": backend,
            "runtime_alias": alias,
            "threads": threads,
        }
        for backend, alias in RUNTIMES.items()
        for threads in THREAD_COUNTS
    ]

    blocks: list[
        dict[str, Any]
    ] = []

    for round_number in range(
        1,
        ROUNDS + 1,
    ):
        order = [
            dict(item)
            for item in configs
        ]

        rng.shuffle(
            order
        )

        for block_index, cfg in enumerate(
            order,
            start=1,
        ):
            prompt_order = list(
                PROMPTS.keys()
            )

            rng.shuffle(
                prompt_order
            )

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
                    "runtime_alias": cfg[
                        "runtime_alias"
                    ],
                    "threads": cfg[
                        "threads"
                    ],
                    "prompt_order": (
                        prompt_order
                    ),
                    "run_ids": [
                        f"{block_id}-"
                        f"P{prompt_id}"
                        for prompt_id
                        in prompt_order
                    ],
                }
            )

    return blocks


def find_latest_incomplete_session() -> Path | None:
    if not RESULTS_ROOT.exists():
        return None

    candidates: list[
        Path
    ] = []

    for path in RESULTS_ROOT.glob(
        "backend_thread_benchmark_v2_*"
    ):
        session_file = (
            path / "session.json"
        )

        if (
            not path.is_dir()
            or not session_file.exists()
        ):
            continue

        try:
            data = json.loads(
                session_file.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if not data.get(
            "completed",
            False,
        ):
            candidates.append(
                path
            )

    return (
        max(
            candidates,
            key=lambda p: p.name,
        )
        if candidates
        else None
    )


def create_session(
    seed: int,
) -> tuple[
    Path,
    dict[str, Any],
]:
    session_dir = (
        RESULTS_ROOT
        / f"backend_thread_benchmark_v2_{timestamp_slug()}"
    )

    session_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    schedule = build_schedule(
        seed
    )

    session = {
        "session_version": 2,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "completed": False,
        "completed_at": None,
        "preflight_completed": False,
        "seed": seed,
        "model_key": MODEL_KEY,
        "expected_variant": EXPECTED_VARIANT,
        "runtimes": RUNTIMES,
        "thread_counts": THREAD_COUNTS,
        "configuration": {
            "context_length": CONTEXT_LENGTH,
            "gpu_offload": "off",
            "temperature": TEMPERATURE,
            "max_tokens": False,
            "thinking": "explicitly_disabled_via_jinja",
            "thinking_jinja_prefix": NO_THINK_PREFIX,
            "rounds": ROUNDS,
            "monitor_interval_seconds": (
                MONITOR_INTERVAL_SECONDS
            ),
            "target_start_temp_c": (
                TARGET_START_TEMP_C
            ),
            "expected_loaded_config": (
                EXPECTED_LOADED_CONFIG
            ),
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
            "lms_version": (
                run_command_metadata(
                    "lms",
                    "--version",
                )
            ),
            "git_commit": (
                run_command_metadata(
                    "git",
                    "rev-parse",
                    "HEAD",
                )
            ),
        },
    }

    atomic_write_json(
        session_dir
        / "session.json",
        session,
    )

    return (
        session_dir,
        session,
    )


def load_session(
    session_dir: Path,
) -> dict[str, Any]:
    return json.loads(
        (
            session_dir
            / "session.json"
        ).read_text(
            encoding="utf-8"
        )
    )


def update_session_file(
    session_dir: Path,
    session: dict[str, Any],
) -> None:
    session[
        "updated_at"
    ] = now_iso()

    atomic_write_json(
        session_dir
        / "session.json",
        session,
    )


def successful_run_ids(
    runs_path: Path,
) -> set[str]:
    return {
        row["run_id"]
        for row in read_jsonl(
            runs_path
        )
        if row.get("status")
        == "success"
        and row.get("run_id")
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
    "thinking_off_verified",
    "response_sha256",
    "process_pid",
    "effective_cpu_cores",
    "process_machine_cpu_percent",
    "effective_active_thread_count",
    "dominant_threads_80pct",
    "dominant_threads_90pct",
    "active_threads_ge_25pct_of_top",
    "lhm_cpu_total_mean_percent",
    "temp_start_c",
    "temp_peak_c",
    "temp_end_c",
    "temp_mean_c",
    "core_max_peak_c",
    "distance_to_tjmax_min_c",
    "package_power_mean_w",
    "p_core_clock_mean_mhz",
    "e_core_clock_mean_mhz",
    "status",
]


def write_runs_csv(
    session_dir: Path,
) -> None:
    runs = [
        row
        for row in read_jsonl(
            session_dir
            / "runs.jsonl"
        )
        if row.get("status")
        == "success"
    ]

    with (
        session_dir
        / "runs.csv"
    ).open(
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
            inference = (
                row.get(
                    "inference"
                )
                or {}
            )

            process_threads = (
                row.get(
                    "telemetry",
                    {},
                ).get(
                    "process_threads",
                    {},
                )
            )

            hardware = (
                row.get(
                    "telemetry",
                    {},
                ).get(
                    "hardware",
                    {},
                )
            )

            load_log = (
                row.get(
                    "load_log_parsed"
                )
                or {}
            )

            writer.writerow(
                {
                    "run_id": row.get(
                        "run_id"
                    ),
                    "timestamp": row.get(
                        "timestamp"
                    ),
                    "round": row.get(
                        "round"
                    ),
                    "backend": row.get(
                        "backend"
                    ),
                    "runtime_alias": row.get(
                        "runtime_alias"
                    ),
                    "threads_requested": row.get(
                        "threads"
                    ),
                    "threads_echoed": inference.get(
                        "echoed_cpu_threads"
                    ),
                    "threadpool_n_threads": load_log.get(
                        "threadpool_n_threads"
                    ),
                    "prompt_id": row.get(
                        "prompt_id"
                    ),
                    "tokens_per_second": inference.get(
                        "tokens_per_second"
                    ),
                    "time_to_first_token_seconds": inference.get(
                        "time_to_first_token_seconds"
                    ),
                    "predicted_tokens_count": inference.get(
                        "predicted_tokens_count"
                    ),
                    "wall_time_seconds": row.get(
                        "wall_time_seconds"
                    ),
                    "stop_reason": inference.get(
                        "stop_reason"
                    ),
                    "thinking_off_verified": inference.get(
                        "thinking_off_verified"
                    ),
                    "response_sha256": inference.get(
                        "response_sha256"
                    ),
                    "process_pid": process_threads.get(
                        "process_pid"
                    ),
                    "effective_cpu_cores": process_threads.get(
                        "effective_cpu_cores"
                    ),
                    "process_machine_cpu_percent": process_threads.get(
                        "process_machine_cpu_percent"
                    ),
                    "effective_active_thread_count": process_threads.get(
                        "effective_active_thread_count"
                    ),
                    "dominant_threads_80pct": process_threads.get(
                        "dominant_threads_80pct"
                    ),
                    "dominant_threads_90pct": process_threads.get(
                        "dominant_threads_90pct"
                    ),
                    "active_threads_ge_25pct_of_top": process_threads.get(
                        "active_threads_ge_25pct_of_top"
                    ),
                    "lhm_cpu_total_mean_percent": hardware.get(
                        "cpu_total_load_mean_percent"
                    ),
                    "temp_start_c": hardware.get(
                        "temp_start_c"
                    ),
                    "temp_peak_c": hardware.get(
                        "temp_peak_c"
                    ),
                    "temp_end_c": hardware.get(
                        "temp_end_c"
                    ),
                    "temp_mean_c": hardware.get(
                        "temp_mean_c"
                    ),
                    "core_max_peak_c": hardware.get(
                        "core_max_peak_c"
                    ),
                    "distance_to_tjmax_min_c": hardware.get(
                        "distance_to_tjmax_min_c"
                    ),
                    "package_power_mean_w": hardware.get(
                        "package_power_mean_w"
                    ),
                    "p_core_clock_mean_mhz": hardware.get(
                        "p_core_clock_mean_mhz"
                    ),
                    "e_core_clock_mean_mhz": hardware.get(
                        "e_core_clock_mean_mhz"
                    ),
                    "status": row.get(
                        "status"
                    ),
                }
            )


def collect_numeric(
    rows: list[dict[str, Any]],
    getter,
) -> list[float]:
    values: list[float] = []

    for row in rows:
        try:
            value = getter(
                row
            )
        except Exception:
            continue

        if isinstance(
            value,
            (int, float),
        ):
            values.append(
                float(value)
            )

    return values


def build_summary(
    session_dir: Path,
) -> dict[str, Any]:
    runs = [
        row
        for row in read_jsonl(
            session_dir
            / "runs.jsonl"
        )
        if row.get("status")
        == "success"
    ]

    by_config: list[
        dict[str, Any]
    ] = []

    for backend, alias in RUNTIMES.items():
        for threads in THREAD_COUNTS:
            rows = [
                row
                for row in runs
                if row.get(
                    "backend"
                )
                == backend
                and int(
                    row.get(
                        "threads",
                        -1,
                    )
                )
                == threads
            ]

            def vals(path: tuple[str, ...]) -> list[float]:
                def getter(row: dict[str, Any]):
                    current: Any = row

                    for key in path:
                        if not isinstance(
                            current,
                            dict,
                        ):
                            return None

                        current = current.get(
                            key
                        )

                    return current

                return collect_numeric(
                    rows,
                    getter,
                )

            tps = vals(
                (
                    "inference",
                    "tokens_per_second",
                )
            )

            ttft = vals(
                (
                    "inference",
                    "time_to_first_token_seconds",
                )
            )

            wall = vals(
                (
                    "wall_time_seconds",
                )
            )

            effective_cores = vals(
                (
                    "telemetry",
                    "process_threads",
                    "effective_cpu_cores",
                )
            )

            process_cpu_percent = vals(
                (
                    "telemetry",
                    "process_threads",
                    "process_machine_cpu_percent",
                )
            )

            active_thread_count = vals(
                (
                    "telemetry",
                    "process_threads",
                    "effective_active_thread_count",
                )
            )

            dominant90 = vals(
                (
                    "telemetry",
                    "process_threads",
                    "dominant_threads_90pct",
                )
            )

            lhm_cpu = vals(
                (
                    "telemetry",
                    "hardware",
                    "cpu_total_load_mean_percent",
                )
            )

            temp_peak = vals(
                (
                    "telemetry",
                    "hardware",
                    "temp_peak_c",
                )
            )

            temp_mean = vals(
                (
                    "telemetry",
                    "hardware",
                    "temp_mean_c",
                )
            )

            distance = vals(
                (
                    "telemetry",
                    "hardware",
                    "distance_to_tjmax_min_c",
                )
            )

            power = vals(
                (
                    "telemetry",
                    "hardware",
                    "package_power_mean_w",
                )
            )

            p_clock = vals(
                (
                    "telemetry",
                    "hardware",
                    "p_core_clock_mean_mhz",
                )
            )

            e_clock = vals(
                (
                    "telemetry",
                    "hardware",
                    "e_core_clock_mean_mhz",
                )
            )

            hashes_by_prompt: dict[
                str,
                list[str],
            ] = {}

            for row in rows:
                prompt_id = str(
                    row.get(
                        "prompt_id"
                    )
                )

                response_hash = (
                    row.get(
                        "inference",
                        {}
                    ).get(
                        "response_sha256"
                    )
                )

                if isinstance(
                    response_hash,
                    str,
                ):
                    hashes_by_prompt.setdefault(
                        prompt_id,
                        [],
                    ).append(
                        response_hash
                    )

            unique_outputs = {
                prompt_id: len(
                    set(hashes)
                )
                for prompt_id, hashes in hashes_by_prompt.items()
            }

            by_config.append(
                {
                    "backend": backend,
                    "runtime_alias": alias,
                    "threads": threads,
                    "n": len(rows),
                    "tokens_per_second_mean": safe_mean(
                        tps
                    ),
                    "tokens_per_second_median": safe_median(
                        tps
                    ),
                    "tokens_per_second_stdev": safe_stdev(
                        tps
                    ),
                    "ttft_mean_seconds": safe_mean(
                        ttft
                    ),
                    "wall_time_mean_seconds": safe_mean(
                        wall
                    ),
                    "effective_cpu_cores_mean": safe_mean(
                        effective_cores
                    ),
                    "process_machine_cpu_percent_mean": safe_mean(
                        process_cpu_percent
                    ),
                    "effective_active_thread_count_mean": safe_mean(
                        active_thread_count
                    ),
                    "dominant_threads_90pct_mean": safe_mean(
                        dominant90
                    ),
                    "lhm_cpu_total_mean_percent": safe_mean(
                        lhm_cpu
                    ),
                    "temp_peak_mean_c": safe_mean(
                        temp_peak
                    ),
                    "temp_peak_max_c": (
                        max(temp_peak)
                        if temp_peak
                        else None
                    ),
                    "temp_mean_c": safe_mean(
                        temp_mean
                    ),
                    "distance_to_tjmax_min_c": (
                        min(distance)
                        if distance
                        else None
                    ),
                    "package_power_mean_w": safe_mean(
                        power
                    ),
                    "p_core_clock_mean_mhz": safe_mean(
                        p_clock
                    ),
                    "e_core_clock_mean_mhz": safe_mean(
                        e_clock
                    ),
                    "thinking_failures": sum(
                        1
                        for row in rows
                        if not (
                            row.get(
                                "inference",
                                {}
                            ).get(
                                "thinking_off_verified"
                            )
                        )
                    ),
                    "unique_outputs_per_prompt": unique_outputs,
                }
            )

    return {
        "generated_at": now_iso(),
        "successful_runs": len(
            runs
        ),
        "planned_runs": (
            ROUNDS
            * len(RUNTIMES)
            * len(THREAD_COUNTS)
            * len(PROMPTS)
        ),
        "by_config": by_config,
    }


def write_summary_files(
    session_dir: Path,
) -> dict[str, Any]:
    summary = build_summary(
        session_dir
    )

    atomic_write_json(
        session_dir
        / "summary.json",
        summary,
    )

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
        "effective_cpu_cores_mean",
        "process_machine_cpu_percent_mean",
        "effective_active_thread_count_mean",
        "dominant_threads_90pct_mean",
        "lhm_cpu_total_mean_percent",
        "temp_peak_mean_c",
        "temp_peak_max_c",
        "temp_mean_c",
        "distance_to_tjmax_min_c",
        "package_power_mean_w",
        "p_core_clock_mean_mhz",
        "e_core_clock_mean_mhz",
        "thinking_failures",
        "unique_outputs_per_prompt",
    ]

    with (
        session_dir
        / "summary.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in summary[
            "by_config"
        ]:
            writer.writerow(
                row
            )

    return summary


# ============================================================
# Execução principal
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
                "block_id": block.get(
                    "block_id"
                ),
                "round": block.get(
                    "round"
                ),
                "backend": block.get(
                    "backend"
                ),
                "threads": block.get(
                    "threads"
                ),
                "runtime_alias": block.get(
                    "runtime_alias"
                ),
            }
        )

    append_jsonl(
        session_dir
        / "errors.jsonl",
        record,
    )


def calculate_eta(
    process_started: float,
    newly_completed: int,
    remaining_runs: int,
) -> float | None:
    if newly_completed <= 0:
        return None

    elapsed = (
        time.perf_counter()
        - process_started
    )

    return (
        elapsed
        / newly_completed
    ) * remaining_runs


def execute_benchmark(
    *,
    session_dir: Path,
    session: dict[str, Any],
    lms: Any,
    psutil: Any,
    no_think_template: dict[str, Any],
) -> int:
    runs_path = (
        session_dir
        / "runs.jsonl"
    )

    successful_ids = successful_run_ids(
        runs_path
    )

    total_runs = int(
        session[
            "planned_measured_runs"
        ]
    )

    newly_completed = 0
    process_started = time.perf_counter()

    for block_number, block in enumerate(
        session[
            "schedule"
        ],
        start=1,
    ):
        pending = [
            run_id
            for run_id in block[
                "run_ids"
            ]
            if run_id
            not in successful_ids
        ]

        if not pending:
            print(
                f"\n[Bloco {block_number}/"
                f"{len(session['schedule'])}] "
                f"{block['block_id']} já concluído — pulando."
            )

            continue

        backend = str(
            block[
                "backend"
            ]
        )

        threads = int(
            block[
                "threads"
            ]
        )

        runtime_alias = str(
            block[
                "runtime_alias"
            ]
        )

        print("\n" + "=" * 94)
        print(
            f" Bloco {block_number}/"
            f"{len(session['schedule'])} | "
            f"Rodada {block['round']}/{ROUNDS} | "
            f"{backend.upper()} | "
            f"{threads} threads"
        )
        print(
            f" {runtime_alias}"
        )
        print("=" * 94)

        loaded = False

        try:
            (
                process,
                process_info_data,
                load_wall_time,
                load_log_parsed,
                _loaded_config,
            ) = load_block(
                session_dir=session_dir,
                block_label=block[
                    "block_id"
                ],
                backend=backend,
                runtime_alias=runtime_alias,
                psutil=psutil,
            )

            loaded = True

            model = lms.llm()

            # Warm-up também sem thinking e sem cap fixo.
            print(
                f"[WARMUP] "
                f"{backend.upper()} / "
                f"{threads} threads | "
                "Think OFF | sem cap de tokens"
            )

            warm_inf, warm_wall, warm_tel = (
                instrumented_inference(
                    model=model,
                    psutil=psutil,
                    process=process,
                    prompt=WARMUP_PROMPT,
                    threads=threads,
                    no_think_template=no_think_template,
                )
            )

            append_jsonl(
                session_dir
                / "warmups.jsonl",
                {
                    "timestamp": now_iso(),
                    "block_id": block[
                        "block_id"
                    ],
                    "round": block[
                        "round"
                    ],
                    "backend": backend,
                    "threads": threads,
                    "runtime_alias": runtime_alias,
                    "wall_time_seconds": warm_wall,
                    "inference": warm_inf,
                    "telemetry": warm_tel,
                    "load_log_parsed": (
                        load_log_parsed
                    ),
                    "model_process": (
                        process_info_data
                    ),
                },
            )

            warm_threads = warm_tel[
                "process_threads"
            ]

            warm_hw = warm_tel[
                "hardware"
            ]

            print(
                f"[WARMUP] "
                f"{fmt_num(warm_inf.get('tokens_per_second'))} tok/s | "
                f"{warm_inf.get('predicted_tokens_count')} tokens | "
                f"workers~"
                f"{fmt_num(estimate_workers({'telemetry': warm_tel}), 2)} | "
                f"procCPU "
                f"{fmt_num(warm_threads.get('process_machine_cpu_percent'), 1)}% | "
                f"T pico "
                f"{fmt_num(warm_hw.get('temp_peak_c'), 1)} °C"
            )

            time.sleep(
                SETTLE_AFTER_WARMUP_SECONDS
            )

            for prompt_position, prompt_id in enumerate(
                block[
                    "prompt_order"
                ],
                start=1,
            ):
                run_id = (
                    f"{block['block_id']}-"
                    f"P{prompt_id}"
                )

                if run_id in successful_ids:
                    continue

                completed_before = len(
                    successful_ids
                )

                global_number = (
                    completed_before + 1
                )

                remaining = (
                    total_runs
                    - completed_before
                )

                eta = calculate_eta(
                    process_started,
                    newly_completed,
                    remaining,
                )

                print(
                    f"\n[RUN {global_number}/{total_runs}] "
                    f"{run_id} | "
                    f"{backend.upper()} | "
                    f"{threads} threads | "
                    f"Prompt {prompt_id}"
                )

                if eta is not None:
                    print(
                        f"      ETA: "
                        f"{seconds_to_hms(eta)}"
                    )

                print(
                    "      Think OFF | sem cap fixo | "
                    "monitorando threads reais do llama-server..."
                )

                inference, wall_time, telemetry = (
                    instrumented_inference(
                        model=model,
                        psutil=psutil,
                        process=process,
                        prompt=PROMPTS[
                            prompt_id
                        ],
                        threads=threads,
                        no_think_template=no_think_template,
                    )
                )

                process_threads = telemetry[
                    "process_threads"
                ]

                hardware = telemetry[
                    "hardware"
                ]

                record = {
                    "status": "success",
                    "timestamp": now_iso(),
                    "run_id": run_id,
                    "block_id": block[
                        "block_id"
                    ],
                    "round": block[
                        "round"
                    ],
                    "prompt_position": (
                        prompt_position
                    ),
                    "backend": backend,
                    "runtime_alias": (
                        runtime_alias
                    ),
                    "threads": threads,
                    "prompt_id": prompt_id,
                    "prompt": PROMPTS[
                        prompt_id
                    ],
                    "load_wall_time_seconds": (
                        load_wall_time
                    ),
                    "wall_time_seconds": (
                        wall_time
                    ),
                    "inference": inference,
                    "telemetry": telemetry,
                    "load_log_parsed": (
                        load_log_parsed
                    ),
                    "model_process": (
                        process_info_data
                    ),
                }

                append_jsonl(
                    runs_path,
                    record,
                )

                successful_ids.add(
                    run_id
                )

                newly_completed += 1

                write_runs_csv(
                    session_dir
                )

                write_summary_files(
                    session_dir
                )

                print(
                    f"      ✓ "
                    f"{fmt_num(inference.get('tokens_per_second'))} tok/s | "
                    f"TTFT "
                    f"{fmt_num(inference.get('time_to_first_token_seconds'))}s | "
                    f"{inference.get('predicted_tokens_count')} tokens | "
                    f"stop={inference.get('stop_reason')} | "
                    f"{wall_time:.1f}s"
                )

                print(
                    "      THREADS reais: "
                    f"effective cores="
                    f"{fmt_num(process_threads.get('effective_cpu_cores'), 2)} | "
                    f"procCPU="
                    f"{fmt_num(process_threads.get('process_machine_cpu_percent'), 1)}% | "
                    f"participation="
                    f"{fmt_num(process_threads.get('effective_active_thread_count'), 2)} | "
                    f"90%CPU="
                    f"{process_threads.get('dominant_threads_90pct')} thr | "
                    f">=25%top="
                    f"{process_threads.get('active_threads_ge_25pct_of_top')} thr"
                )

                print(
                    "      LHM: "
                    f"CPU Total="
                    f"{fmt_num(hardware.get('cpu_total_load_mean_percent'), 1)}% | "
                    f"Package "
                    f"{fmt_num(hardware.get('temp_start_c'), 1)}→"
                    f"{fmt_num(hardware.get('temp_end_c'), 1)} °C | "
                    f"pico "
                    f"{fmt_num(hardware.get('temp_peak_c'), 1)} °C | "
                    f"CoreMax "
                    f"{fmt_num(hardware.get('core_max_peak_c'), 1)} °C | "
                    f"TjMax dist min "
                    f"{fmt_num(hardware.get('distance_to_tjmax_min_c'), 1)} °C"
                )

                time.sleep(
                    PAUSE_BETWEEN_PROMPTS_SECONDS
                )

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            print(
                f"[BLOCO] ERRO FATAL: {exc}"
            )

            record_error(
                session_dir,
                phase="block",
                message=str(exc),
                block=block,
            )

            # Qualquer violação metodológica encerra.
            raise

        finally:
            if loaded:
                print(
                    "[CLEANUP] Descarregando modelo..."
                )

                unload_all()

            session[
                "successful_runs"
            ] = len(
                successful_ids
            )

            session[
                "remaining_runs"
            ] = (
                total_runs
                - len(
                    successful_ids
                )
            )

            update_session_file(
                session_dir,
                session,
            )

    write_runs_csv(
        session_dir
    )

    summary = write_summary_files(
        session_dir
    )

    all_done = (
        len(
            successful_ids
        )
        == total_runs
    )

    session[
        "successful_runs"
    ] = len(
        successful_ids
    )

    session[
        "remaining_runs"
    ] = (
        total_runs
        - len(
            successful_ids
        )
    )

    session[
        "completed"
    ] = all_done

    if all_done:
        session[
            "completed_at"
        ] = now_iso()

    update_session_file(
        session_dir,
        session,
    )

    print("\n" + "=" * 118)
    print(
        " RESUMO FINAL — CPU/VULKAN × 4/6/8 THREADS — THINK OFF"
    )
    print("=" * 118)

    for row in summary[
        "by_config"
    ]:
        print(
            f"{row['backend']:<7} "
            f"T{row['threads']:<2} | "
            f"n={row['n']:<2} | "
            f"{fmt_num(row['tokens_per_second_mean'])} tok/s | "
            f"TTFT {fmt_num(row['ttft_mean_seconds'])}s | "
            f"OS workers~"
            f"{fmt_num(row['effective_active_thread_count_mean'], 2)} | "
            f"cores efetivos "
            f"{fmt_num(row['effective_cpu_cores_mean'], 2)} | "
            f"procCPU "
            f"{fmt_num(row['process_machine_cpu_percent_mean'], 1)}% | "
            f"T pico "
            f"{fmt_num(row['temp_peak_mean_c'], 1)} °C"
        )

    print("=" * 118)

    print(
        f"\nResultados:\n  {session_dir}"
    )

    if all_done:
        print(
            "\n✓ Todas as 54 gerações medidas foram concluídas."
        )
        return 0

    return 2


# ============================================================
# CLI / main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark v2: CPU AVX2 vs Vulkan × 4/6/8 cpuThreads, "
            "Think OFF, sem cap fixo, com auditoria OS-level das threads."
        )
    )

    parser.add_argument(
        "--new",
        action="store_true",
        help="Força uma nova sessão.",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Não pede confirmação.",
    )

    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help=(
            "Pula o preflight OS-level. "
            "NÃO recomendado."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=(
            f"Seed de randomização "
            f"(padrão: {SEED})."
        ),
    )

    return parser.parse_args()


def choose_session(
    args: argparse.Namespace,
) -> tuple[
    Path,
    dict[str, Any],
    bool,
]:
    if not args.new:
        incomplete = (
            find_latest_incomplete_session()
        )

        if incomplete is not None:
            return (
                incomplete,
                load_session(
                    incomplete
                ),
                True,
            )

    session_dir, session = (
        create_session(
            args.seed
        )
    )

    return (
        session_dir,
        session,
        False,
    )


def main() -> int:
    args = parse_args()

    print(
        "\n[PRECHECK] Validando ambiente..."
    )

    lms, psutil = (
        import_dependencies()
    )

    ensure_server()

    model_inventory = (
        verify_selected_variant()
    )

    print(
        f"  ✓ modelo: "
        f"{EXPECTED_VARIANT}"
    )

    for backend, alias in RUNTIMES.items():
        verify_runtime_installed(
            alias
        )

        print(
            f"  ✓ {backend}: {alias}"
        )

    hardware_probe = (
        verify_lhm()
    )

    print(
        "  ✓ LibreHardwareMonitor: "
        f"CPU Package = "
        f"{hardware_probe['cpu_package_temp_c']:.1f} °C"
    )

    print(
        f"  ✓ processadores lógicos: "
        f"{psutil.cpu_count(logical=True)}"
    )

    session_dir, session, resumed = (
        choose_session(
            args
        )
    )

    session[
        "model_inventory_at_start"
    ] = model_inventory

    session[
        "runtime_ls_at_start"
    ] = runtime_ls_text()

    session[
        "hardware_probe_at_start"
    ] = hardware_probe

    session[
        "psutil_version"
    ] = getattr(
        psutil,
        "__version__",
        None,
    )

    session[
        "lmstudio_python_version"
    ] = getattr(
        lms,
        "__version__",
        None,
    )

    update_session_file(
        session_dir,
        session,
    )

    print("\n" + "=" * 94)
    print(
        " LocalIA — Backend × Threads Benchmark v2"
    )
    print("=" * 94)
    print(
        f" Modelo            : "
        f"{MODEL_KEY} / Q4_K_M"
    )
    print(
        " Backends          : "
        "CPU AVX2 2.46.0 + Vulkan 2.46.0"
    )
    print(
        f" cpuThreads        : "
        f"{THREAD_COUNTS}"
    )
    print(
        " GPU offload       : OFF"
    )
    print(
        f" Contexto          : "
        f"{CONTEXT_LENGTH}"
    )
    print(
        " Thinking          : "
        "OFF via Jinja enable_thinking=false"
    )
    print(
        " Limite de tokens  : "
        "NENHUM cap fixo (maxTokens=False)"
    )
    print(
        f" Monitor           : "
        f"{MONITOR_INTERVAL_SECONDS:.1f}s"
    )
    print(
        " Auditoria threads : "
        "SDK + process CPU + per-thread CPU time"
    )
    print(
        f" Gerações medidas  : "
        f"{session['planned_measured_runs']}"
    )
    print(
        " Preflight         : "
        "6 gerações, obrigatório por padrão"
    )
    print(
        f" Sessão            : "
        f"{session_dir.name}"
    )
    print("=" * 94)

    if resumed:
        print(
            "\nSessão incompleta encontrada; "
            "será retomada."
        )

    if not args.yes:
        answer = input(
            "\nPressione ENTER para começar "
            "ou digite 'cancelar': "
        ).strip().lower()

        if answer in {
            "cancelar",
            "cancel",
            "n",
            "nao",
            "não",
        }:
            print(
                "Benchmark cancelado."
            )
            return 0

    try:
        no_think_path = (
            session_dir
            / "no_think_prompt_template.json"
        )

        if (
            session.get(
                "preflight_completed"
            )
            and no_think_path.exists()
        ):
            no_think_template = json.loads(
                no_think_path.read_text(
                    encoding="utf-8"
                )
            )

            print(
                "\n[PREFLIGHT] Já concluído nesta sessão."
            )

        elif args.skip_preflight:
            raise RuntimeError(
                "--skip-preflight só é permitido ao retomar "
                "uma sessão cujo preflight já tenha sido concluído."
            )

        else:
            (
                _preflight_records,
                no_think_template,
            ) = run_preflight(
                session_dir=session_dir,
                lms=lms,
                psutil=psutil,
            )

            session[
                "preflight_completed"
            ] = True

            session[
                "preflight_completed_at"
            ] = now_iso()

            update_session_file(
                session_dir,
                session,
            )

        return execute_benchmark(
            session_dir=session_dir,
            session=session,
            lms=lms,
            psutil=psutil,
            no_think_template=no_think_template,
        )

    except KeyboardInterrupt:
        print(
            "\n\nInterrompido pelo usuário."
        )

        try:
            unload_all()
        except Exception:
            pass

        session[
            "interrupted_at"
        ] = now_iso()

        update_session_file(
            session_dir,
            session,
        )

        print(
            "Resultados preservados. "
            "Rode o mesmo comando para retomar."
        )

        return 130

    except Exception as exc:
        print(
            f"\nERRO FATAL: {exc}"
        )

        try:
            unload_all()
        except Exception:
            pass

        session[
            "fatal_error_at"
        ] = now_iso()

        session[
            "fatal_error"
        ] = str(exc)

        update_session_file(
            session_dir,
            session,
        )

        print(
            "\nNenhum resultado já concluído foi perdido."
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
