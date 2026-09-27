from __future__ import annotations

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
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


# ============================================================
# LocalIA — Vulkan GPU Mode Diagnostic Benchmark
#
# Objetivo:
#   descobrir se existe diferença real entre:
#
#       --gpu off
#       --gpu 0.0
#
# mantendo TODO o restante igual.
#
# 2 modos × 3 prompts × 3 rodadas = 18 gerações medidas
#
# Extras:
# - força Vulkan llama.cpp 2.46.0;
# - captura logs do modelo DURANTE O LOAD;
# - tenta extrair n_threads / n_threads_batch / GPU layers dos logs;
# - mede temperatura antes/depois do warm-up e de cada inferência;
# - espera resfriamento antes de cada bloco quando necessário;
# - não faz polling de temperatura durante a inferência, para não
#   contaminar o benchmark com atividade de monitoramento;
# - salva tudo incrementalmente e permite retomar.
#
# Temperatura:
# 1) LibreHardwareMonitor WMI (preferido)
# 2) OpenHardwareMonitor WMI
# 3) ACPI Thermal Zone (fallback; NÃO é garantidamente CPU Package)
#
# Para leitura real de "CPU Package", deixe o LibreHardwareMonitor
# aberto durante o benchmark (idealmente como Administrador).
# ============================================================


API_BASE = "http://127.0.0.1:1234"

MODEL_KEY = "qwen/qwen3.5-9b"
EXPECTED_VARIANT = "qwen/qwen3.5-9b@q4_k_m"
INSTANCE_ID = "localia-gpu-mode-bench"

VULKAN_RUNTIME = "llama.cpp-win-x86_64-vulkan-avx2@2.46.0"

CONTEXT_LENGTH = 4096

GPU_MODES = {
    "off": "off",
    "zero": "0.0",
}

ROUNDS = 3
SEED = 20260927

CPU_THREADS_TARGET = 8  # metadata; tentaremos descobrir o real pelos logs

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

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 384
REASONING = "off"

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

COOLDOWN_AFTER_UNLOAD_SECONDS = 10
SETTLE_AFTER_LOAD_SECONDS = 5
SETTLE_AFTER_WARMUP_SECONDS = 2
PAUSE_BETWEEN_PROMPTS_SECONDS = 2

# Controle térmico antes de cada bloco.
TARGET_START_TEMP_C = 70.0
THERMAL_WAIT_MAX_SECONDS = 120
THERMAL_POLL_SECONDS = 5

LMS_LOAD_TIMEOUT_SECONDS = 180.0
LMS_COMMAND_TIMEOUT_SECONDS = 90.0
HTTP_TIMEOUT_SECONDS = 900.0

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def seconds_to_hms(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "n/d"

    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)

    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def fmt_temp(sample: dict[str, Any] | None) -> str:
    if not sample or not isinstance(sample.get("celsius"), (int, float)):
        return "n/d"

    source = sample.get("source", "sensor")
    sensor = sample.get("sensor", "")
    value = float(sample["celsius"])

    if sensor:
        return f"{value:.1f} °C ({source}: {sensor})"
    return f"{value:.1f} °C ({source})"


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


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
                    f"AVISO: linha inválida em {path.name}: "
                    f"{line_number}; ignorada."
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


# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------

def find_powershell() -> str | None:
    for exe in ("powershell.exe", "pwsh.exe", "powershell", "pwsh"):
        try:
            result = subprocess.run(
                [exe, "-NoProfile", "-NonInteractive", "-Command", "$PSVersionTable.PSVersion.Major"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return exe
        except Exception:
            continue

    return None


POWERSHELL = find_powershell()


def run_powershell_json(script: str) -> Any | None:
    if not POWERSHELL:
        return None

    try:
        result = subprocess.run(
            [
                POWERSHELL,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )

        if result.returncode != 0:
            return None

        text = result.stdout.strip()
        if not text:
            return None

        return json.loads(text)

    except Exception:
        return None


def _normalize_sensor_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _choose_cpu_sensor(
    items: list[dict[str, Any]],
    source: str,
    reliability: str,
) -> dict[str, Any] | None:
    valid: list[dict[str, Any]] = []

    for item in items:
        name = str(item.get("Name", "")).strip()
        value = item.get("Value")

        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue

        if not (-20.0 <= value_f <= 125.0):
            continue

        valid.append(
            {
                "name": name,
                "value": value_f,
            }
        )

    if not valid:
        return None

    priorities = [
        "cpu package",
        "cpu die",
        "core max",
        "cpu tctl",
        "cpu",
        "core",
    ]

    chosen: dict[str, Any] | None = None

    for keyword in priorities:
        matches = [
            item
            for item in valid
            if keyword in item["name"].lower()
        ]

        if matches:
            # Para múltiplos cores, usamos o mais quente.
            chosen = max(matches, key=lambda item: item["value"])
            break

    if chosen is None:
        chosen = max(valid, key=lambda item: item["value"])

    return {
        "timestamp": now_iso(),
        "celsius": chosen["value"],
        "source": source,
        "sensor": chosen["name"],
        "reliability": reliability,
    }


def read_cpu_temperature() -> dict[str, Any] | None:
    """
    Tenta obter CPU Package/Die via WMI de ferramentas de hardware.
    Se não houver, usa Thermal Zone ACPI como fallback explicitamente
    marcado como aproximado.
    """

    # LibreHardwareMonitor
    lhm_script = r"""
$ErrorActionPreference = 'Stop'
$x = Get-CimInstance -Namespace 'root/LibreHardwareMonitor' -ClassName Sensor |
    Where-Object {
        $_.SensorType -eq 'Temperature' -and
        ($_.Name -match 'CPU|Package|Core|Die|Tctl')
    } |
    Select-Object Name, Value, Identifier
$x | ConvertTo-Json -Compress
"""
    data = run_powershell_json(lhm_script)
    items = _normalize_sensor_items(data)

    sample = _choose_cpu_sensor(
        items,
        source="LibreHardwareMonitor",
        reliability="cpu_sensor",
    )
    if sample:
        return sample

    # OpenHardwareMonitor
    ohm_script = r"""
$ErrorActionPreference = 'Stop'
$x = Get-CimInstance -Namespace 'root/OpenHardwareMonitor' -ClassName Sensor |
    Where-Object {
        $_.SensorType -eq 'Temperature' -and
        ($_.Name -match 'CPU|Package|Core|Die|Tctl')
    } |
    Select-Object Name, Value, Identifier
$x | ConvertTo-Json -Compress
"""
    data = run_powershell_json(ohm_script)
    items = _normalize_sensor_items(data)

    sample = _choose_cpu_sensor(
        items,
        source="OpenHardwareMonitor",
        reliability="cpu_sensor",
    )
    if sample:
        return sample

    # ACPI fallback — NÃO é garantidamente CPU Package.
    acpi_script = r"""
$ErrorActionPreference = 'Stop'
$x = Get-CimInstance -Namespace 'root/wmi' -ClassName MSAcpi_ThermalZoneTemperature |
    ForEach-Object {
        [pscustomobject]@{
            Name = $_.InstanceName
            Value = [math]::Round(($_.CurrentTemperature / 10.0) - 273.15, 1)
        }
    }
$x | ConvertTo-Json -Compress
"""
    data = run_powershell_json(acpi_script)
    items = _normalize_sensor_items(data)

    if items:
        valid = []
        for item in items:
            try:
                value = float(item.get("Value"))
            except (TypeError, ValueError):
                continue

            if -20 <= value <= 125:
                valid.append(
                    {
                        "name": str(item.get("Name", "ACPI Thermal Zone")),
                        "value": value,
                    }
                )

        if valid:
            chosen = max(valid, key=lambda item: item["value"])
            return {
                "timestamp": now_iso(),
                "celsius": chosen["value"],
                "source": "Windows ACPI Thermal Zone",
                "sensor": chosen["name"],
                "reliability": "approximate_not_cpu_package",
            }

    return None


def record_temperature(
    session_dir: Path,
    phase: str,
    *,
    block_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    sample = read_cpu_temperature()

    record = {
        "timestamp": now_iso(),
        "phase": phase,
        "block_id": block_id,
        "run_id": run_id,
        "sample": sample,
    }

    append_jsonl(session_dir / "temperatures.jsonl", record)
    return sample


def thermal_gate(
    session_dir: Path,
    block_id: str,
) -> dict[str, Any] | None:
    """
    Espera até TARGET_START_TEMP_C quando um sensor está disponível.
    Limite de espera evita travar indefinidamente.
    """

    sample = record_temperature(
        session_dir,
        "thermal_gate_start",
        block_id=block_id,
    )

    if not sample or not isinstance(sample.get("celsius"), (int, float)):
        print("[TEMP] Temperatura indisponível; seguindo sem gate térmico.")
        return sample

    current = float(sample["celsius"])

    print(f"[TEMP] Antes do bloco: {fmt_temp(sample)}")

    if current <= TARGET_START_TEMP_C:
        return sample

    print(
        f"[TEMP] Acima de {TARGET_START_TEMP_C:.0f} °C. "
        "Aguardando resfriamento..."
    )

    deadline = time.monotonic() + THERMAL_WAIT_MAX_SECONDS

    while time.monotonic() < deadline:
        time.sleep(THERMAL_POLL_SECONDS)

        sample = record_temperature(
            session_dir,
            "thermal_gate_wait",
            block_id=block_id,
        )

        if not sample or not isinstance(sample.get("celsius"), (int, float)):
            print("[TEMP] Sensor ficou indisponível; continuando.")
            return sample

        current = float(sample["celsius"])
        print(f"       {fmt_temp(sample)}")

        if current <= TARGET_START_TEMP_C:
            print("[TEMP] Faixa térmica de início atingida.")
            return sample

    print(
        "[TEMP] Tempo máximo de resfriamento atingido; "
        "o benchmark seguirá e registrará a temperatura."
    )
    return sample


# ------------------------------------------------------------
# LMS / API
# ------------------------------------------------------------

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
            f"Timeout após {timeout:.0f}s executando:\n  {' '.join(cmd)}"
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


def api_post(
    path: str,
    payload: dict[str, Any],
    timeout: float = HTTP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")

    request = Request(
        f"{API_BASE}{path}",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} em POST {path}: {body}"
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
        "Servidor iniciado, mas /api/v1/models não respondeu em 15s."
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


def select_vulkan_runtime() -> None:
    unload_all()

    run_lms(
        "runtime",
        "select",
        VULKAN_RUNTIME,
        timeout=60.0,
    )

    selected = selected_gguf_runtime()

    if not selected or VULKAN_RUNTIME not in selected:
        raise RuntimeError(
            "Não foi possível confirmar o runtime Vulkan.\n"
            f"Esperado: {VULKAN_RUNTIME}\n"
            f"Selecionado: {selected!r}"
        )


def unload_all() -> None:
    run_lms(
        "unload",
        "--all",
        allow_failure=True,
        timeout=60.0,
    )


def load_model(
    gpu_argument: str,
) -> tuple[subprocess.CompletedProcess[str], float]:
    start = time.perf_counter()

    result = run_lms(
        "load",
        MODEL_KEY,
        "--gpu",
        gpu_argument,
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


def extract_message(response: dict[str, Any]) -> str:
    chunks: list[str] = []

    for item in response.get("output", []):
        if item.get("type") != "message":
            continue

        content = item.get("content")

        if isinstance(content, str):
            chunks.append(content)

    return "\n".join(chunks)


def run_inference(
    prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": INSTANCE_ID,
        "input": prompt,
        "temperature": TEMPERATURE,
        "max_output_tokens": max_output_tokens,
        "reasoning": REASONING,
        "context_length": CONTEXT_LENGTH,
        "store": False,
        "stream": False,
    }

    start = time.perf_counter()
    response = api_post("/api/v1/chat", payload)
    return response, time.perf_counter() - start


# ------------------------------------------------------------
# Model load log capture
# ------------------------------------------------------------

def start_model_log_capture(
    log_path: Path,
) -> tuple[subprocess.Popen[Any] | None, Any | None]:
    """
    Captura o stream de logs direto em arquivo para não manter
    uma thread Python nem encher um PIPE.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        handle = log_path.open("w", encoding="utf-8", errors="replace")

        proc = subprocess.Popen(
            ["lms", "log", "stream", "--source", "model"],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        time.sleep(1.0)
        return proc, handle

    except Exception:
        try:
            handle.close()  # type: ignore[name-defined]
        except Exception:
            pass
        return None, None


def stop_model_log_capture(
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


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def parse_load_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "log_available": False,
            "n_threads": None,
            "n_threads_batch": None,
            "n_gpu_layers": None,
            "offload_mentions": [],
        }

    raw = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    text = strip_ansi(raw)

    def last_int(patterns: list[str]) -> int | None:
        values: list[int] = []

        for pattern in patterns:
            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                try:
                    values.append(int(match.group(1)))
                except Exception:
                    continue

        return values[-1] if values else None

    n_threads = last_int(
        [
            r"\bn_threads\s*[:=]\s*(\d+)",
            r"\bthreads\s*[:=]\s*(\d+)",
        ]
    )

    n_threads_batch = last_int(
        [
            r"\bn_threads_batch\s*[:=]\s*(\d+)",
            r"\bthreads[_\s-]*batch\s*[:=]\s*(\d+)",
        ]
    )

    n_gpu_layers = last_int(
        [
            r"\bn_gpu_layers\s*[:=]\s*(\d+)",
            r"\bgpu[_\s-]*layers\s*[:=]\s*(\d+)",
        ]
    )

    interesting: list[str] = []

    keywords = (
        "thread",
        "gpu",
        "offload",
        "layer",
        "batch",
        "flash",
        "kv",
    )

    for line in text.splitlines():
        low = line.lower()

        if any(keyword in low for keyword in keywords):
            line = line.strip()

            if line and line not in interesting:
                interesting.append(line)

        if len(interesting) >= 80:
            break

    return {
        "log_available": bool(text.strip()),
        "n_threads": n_threads,
        "n_threads_batch": n_threads_batch,
        "n_gpu_layers": n_gpu_layers,
        "interesting_lines": interesting,
    }


# ------------------------------------------------------------
# Schedule / session
# ------------------------------------------------------------

def build_schedule(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    modes = [
        {
            "mode": name,
            "gpu_argument": arg,
        }
        for name, arg in GPU_MODES.items()
    ]

    blocks: list[dict[str, Any]] = []

    for round_number in range(1, ROUNDS + 1):
        mode_order = [dict(item) for item in modes]
        rng.shuffle(mode_order)

        for block_index, item in enumerate(mode_order, start=1):
            prompt_order = list(PROMPTS.keys())
            rng.shuffle(prompt_order)

            block_id = (
                f"R{round_number}-"
                f"GPU-{item['mode'].upper()}"
            )

            blocks.append(
                {
                    "block_id": block_id,
                    "round": round_number,
                    "block_index": block_index,
                    "mode": item["mode"],
                    "gpu_argument": item["gpu_argument"],
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

    for path in RESULTS_ROOT.glob("gpu_mode_benchmark_*"):
        if not path.is_dir():
            continue

        session_file = path / "session.json"

        if not session_file.exists():
            continue

        try:
            data = json.loads(
                session_file.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        if not data.get("completed", False):
            candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.name)


def create_session(seed: int) -> tuple[Path, dict[str, Any]]:
    session_dir = (
        RESULTS_ROOT
        / f"gpu_mode_benchmark_{timestamp_slug()}"
    )
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
        "runtime": VULKAN_RUNTIME,
        "gpu_modes": GPU_MODES,
        "configuration": {
            "context_length": CONTEXT_LENGTH,
            "cpu_threads_target_metadata_only": CPU_THREADS_TARGET,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "reasoning": REASONING,
            "rounds": ROUNDS,
            "expected_loaded_config": EXPECTED_LOADED_CONFIG,
            "target_start_temp_c": TARGET_START_TEMP_C,
            "thermal_wait_max_seconds": THERMAL_WAIT_MAX_SECONDS,
        },
        "prompts": PROMPTS,
        "warmup_prompt": WARMUP_PROMPT,
        "schedule": schedule,
        "planned_measured_runs": (
            ROUNDS
            * len(GPU_MODES)
            * len(PROMPTS)
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "lms_version": run_command_metadata("lms", "--version"),
            "git_commit": run_command_metadata("git", "rev-parse", "HEAD"),
            "powershell": POWERSHELL,
        },
    }

    atomic_write_json(
        session_dir / "session.json",
        session,
    )

    return session_dir, session


def load_session(session_dir: Path) -> dict[str, Any]:
    return json.loads(
        (session_dir / "session.json").read_text(
            encoding="utf-8"
        )
    )


def update_session_file(
    session_dir: Path,
    session: dict[str, Any],
) -> None:
    session["updated_at"] = now_iso()

    atomic_write_json(
        session_dir / "session.json",
        session,
    )


def successful_run_ids(
    runs_path: Path,
) -> set[str]:
    return {
        row["run_id"]
        for row in read_jsonl(runs_path)
        if row.get("status") == "success"
        and row.get("run_id")
    }


# ------------------------------------------------------------
# CSV / summary
# ------------------------------------------------------------

CSV_FIELDS = [
    "run_id",
    "timestamp",
    "round",
    "mode",
    "gpu_argument",
    "prompt_id",
    "tokens_per_second",
    "time_to_first_token_seconds",
    "input_tokens",
    "total_output_tokens",
    "reasoning_output_tokens",
    "wall_time_seconds",
    "load_wall_time_seconds",
    "temp_before_c",
    "temp_after_c",
    "temp_delta_c",
    "temp_source",
    "n_threads_from_log",
    "n_threads_batch_from_log",
    "n_gpu_layers_from_log",
    "status",
]


def temp_value(sample: dict[str, Any] | None) -> float | None:
    if not sample:
        return None

    value = sample.get("celsius")

    if isinstance(value, (int, float)):
        return float(value)

    return None


def write_runs_csv(session_dir: Path) -> None:
    runs = [
        row
        for row in read_jsonl(
            session_dir / "runs.jsonl"
        )
        if row.get("status") == "success"
    ]

    with (
        session_dir / "runs.csv"
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
            stats = row.get("stats") or {}
            before = row.get("temp_before") or {}
            after = row.get("temp_after") or {}
            parsed_log = row.get("load_log_parsed") or {}

            before_c = temp_value(before)
            after_c = temp_value(after)

            delta = (
                after_c - before_c
                if before_c is not None
                and after_c is not None
                else None
            )

            writer.writerow(
                {
                    "run_id": row.get("run_id"),
                    "timestamp": row.get("timestamp"),
                    "round": row.get("round"),
                    "mode": row.get("mode"),
                    "gpu_argument": row.get("gpu_argument"),
                    "prompt_id": row.get("prompt_id"),
                    "tokens_per_second": stats.get(
                        "tokens_per_second"
                    ),
                    "time_to_first_token_seconds": stats.get(
                        "time_to_first_token_seconds"
                    ),
                    "input_tokens": stats.get("input_tokens"),
                    "total_output_tokens": stats.get(
                        "total_output_tokens"
                    ),
                    "reasoning_output_tokens": stats.get(
                        "reasoning_output_tokens"
                    ),
                    "wall_time_seconds": row.get(
                        "wall_time_seconds"
                    ),
                    "load_wall_time_seconds": row.get(
                        "load_wall_time_seconds"
                    ),
                    "temp_before_c": before_c,
                    "temp_after_c": after_c,
                    "temp_delta_c": delta,
                    "temp_source": after.get("source")
                    or before.get("source"),
                    "n_threads_from_log": parsed_log.get(
                        "n_threads"
                    ),
                    "n_threads_batch_from_log": parsed_log.get(
                        "n_threads_batch"
                    ),
                    "n_gpu_layers_from_log": parsed_log.get(
                        "n_gpu_layers"
                    ),
                    "status": row.get("status"),
                }
            )


def safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def safe_stdev(values: list[float]) -> float | None:
    return (
        statistics.stdev(values)
        if len(values) >= 2
        else None
    )


def stat_values(
    rows: list[dict[str, Any]],
    key: str,
) -> list[float]:
    values: list[float] = []

    for row in rows:
        value = (row.get("stats") or {}).get(key)

        if isinstance(value, (int, float)):
            values.append(float(value))

    return values


def temperature_values(
    rows: list[dict[str, Any]],
    field: str,
) -> list[float]:
    values: list[float] = []

    for row in rows:
        sample = row.get(field)
        value = temp_value(sample)

        if value is not None:
            values.append(value)

    return values


def build_summary(session_dir: Path) -> dict[str, Any]:
    runs = [
        row
        for row in read_jsonl(
            session_dir / "runs.jsonl"
        )
        if row.get("status") == "success"
    ]

    by_mode: list[dict[str, Any]] = []

    for mode, gpu_argument in GPU_MODES.items():
        rows = [
            row
            for row in runs
            if row.get("mode") == mode
        ]

        tps = stat_values(
            rows,
            "tokens_per_second",
        )
        ttft = stat_values(
            rows,
            "time_to_first_token_seconds",
        )

        wall = [
            float(row["wall_time_seconds"])
            for row in rows
            if isinstance(
                row.get("wall_time_seconds"),
                (int, float),
            )
        ]

        temp_before = temperature_values(
            rows,
            "temp_before",
        )
        temp_after = temperature_values(
            rows,
            "temp_after",
        )

        thread_values = [
            int((row.get("load_log_parsed") or {}).get("n_threads"))
            for row in rows
            if isinstance(
                (row.get("load_log_parsed") or {}).get("n_threads"),
                int,
            )
        ]

        gpu_layer_values = [
            int((row.get("load_log_parsed") or {}).get("n_gpu_layers"))
            for row in rows
            if isinstance(
                (row.get("load_log_parsed") or {}).get("n_gpu_layers"),
                int,
            )
        ]

        by_mode.append(
            {
                "mode": mode,
                "gpu_argument": gpu_argument,
                "n": len(rows),
                "tokens_per_second_mean": safe_mean(tps),
                "tokens_per_second_median": safe_median(tps),
                "tokens_per_second_stdev": safe_stdev(tps),
                "ttft_mean_seconds": safe_mean(ttft),
                "ttft_median_seconds": safe_median(ttft),
                "ttft_stdev_seconds": safe_stdev(ttft),
                "wall_time_mean_seconds": safe_mean(wall),
                "temp_before_mean_c": safe_mean(temp_before),
                "temp_after_mean_c": safe_mean(temp_after),
                "n_threads_values_seen": sorted(set(thread_values)),
                "n_gpu_layers_values_seen": sorted(
                    set(gpu_layer_values)
                ),
            }
        )

    off = next(
        (x for x in by_mode if x["mode"] == "off"),
        None,
    )
    zero = next(
        (x for x in by_mode if x["mode"] == "zero"),
        None,
    )

    comparison: dict[str, Any] = {}

    if off and zero:
        off_tps = off.get("tokens_per_second_mean")
        zero_tps = zero.get("tokens_per_second_mean")
        off_ttft = off.get("ttft_mean_seconds")
        zero_ttft = zero.get("ttft_mean_seconds")

        if off_tps and zero_tps:
            comparison[
                "gpu_off_vs_gpu_zero_throughput_percent"
            ] = (
                (off_tps / zero_tps) - 1.0
            ) * 100.0

        if off_ttft and zero_ttft:
            comparison[
                "gpu_off_vs_gpu_zero_ttft_percent"
            ] = (
                (off_ttft / zero_ttft) - 1.0
            ) * 100.0

    return {
        "generated_at": now_iso(),
        "successful_runs": len(runs),
        "planned_runs": (
            ROUNDS
            * len(GPU_MODES)
            * len(PROMPTS)
        ),
        "by_mode": by_mode,
        "comparison": comparison,
    }


def write_summary_files(
    session_dir: Path,
) -> dict[str, Any]:
    summary = build_summary(session_dir)

    atomic_write_json(
        session_dir / "summary.json",
        summary,
    )

    fields = [
        "mode",
        "gpu_argument",
        "n",
        "tokens_per_second_mean",
        "tokens_per_second_median",
        "tokens_per_second_stdev",
        "ttft_mean_seconds",
        "ttft_median_seconds",
        "ttft_stdev_seconds",
        "wall_time_mean_seconds",
        "temp_before_mean_c",
        "temp_after_mean_c",
        "n_threads_values_seen",
        "n_gpu_layers_values_seen",
    ]

    with (
        session_dir / "summary.csv"
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

        for row in summary["by_mode"]:
            writer.writerow(row)

    return summary


def print_summary(
    summary: dict[str, Any],
) -> None:
    print("\n" + "=" * 96)
    print(" RESUMO FINAL — Vulkan: --gpu off vs --gpu 0.0")
    print("=" * 96)

    def f(value: Any, decimals: int = 3) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.{decimals}f}"
        return "n/d"

    print(
        f"{'modo':<8} "
        f"{'n':>3} "
        f"{'tok/s':>9} "
        f"{'TTFT':>9} "
        f"{'tempo':>10} "
        f"{'T antes':>10} "
        f"{'T depois':>10} "
        f"{'threads':>10}"
    )
    print("-" * 96)

    for row in summary["by_mode"]:
        threads = row.get("n_threads_values_seen") or []

        print(
            f"{row['mode']:<8} "
            f"{row['n']:>3} "
            f"{f(row['tokens_per_second_mean']):>9} "
            f"{(f(row['ttft_mean_seconds']) + 's'):>9} "
            f"{(f(row['wall_time_mean_seconds'], 1) + 's'):>10} "
            f"{(f(row['temp_before_mean_c'], 1) + '°C'):>10} "
            f"{(f(row['temp_after_mean_c'], 1) + '°C'):>10} "
            f"{str(threads):>10}"
        )

    delta = summary.get("comparison", {}).get(
        "gpu_off_vs_gpu_zero_throughput_percent"
    )

    if isinstance(delta, (int, float)):
        if delta >= 0:
            print(
                f"\n--gpu off: +{delta:.1f}% throughput "
                "vs --gpu 0.0"
            )
        else:
            print(
                f"\n--gpu off: {delta:.1f}% throughput "
                "vs --gpu 0.0"
            )

    print("=" * 96)


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------

def record_error(
    session_dir: Path,
    *,
    phase: str,
    message: str,
    block: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> None:
    record = {
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
                "mode": block.get("mode"),
                "gpu_argument": block.get("gpu_argument"),
            }
        )

    append_jsonl(
        session_dir / "errors.jsonl",
        record,
    )


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


def execute_benchmark(
    session_dir: Path,
    session: dict[str, Any],
) -> int:
    runs_path = session_dir / "runs.jsonl"
    loads_path = session_dir / "loads.jsonl"
    warmups_path = session_dir / "warmups.jsonl"

    successful_ids = successful_run_ids(
        runs_path
    )

    total_runs = int(
        session["planned_measured_runs"]
    )

    newly_completed = 0
    process_started = time.perf_counter()

    if successful_ids:
        print(
            f"\nRetomando sessão: "
            f"{len(successful_ids)}/{total_runs} concluídas."
        )

    for block_number, block in enumerate(
        session["schedule"],
        start=1,
    ):
        pending = [
            run_id
            for run_id in block["run_ids"]
            if run_id not in successful_ids
        ]

        if not pending:
            print(
                f"\n[Bloco {block_number}/"
                f"{len(session['schedule'])}] "
                f"{block['block_id']} já concluído — pulando."
            )
            continue

        mode = block["mode"]
        gpu_argument = block["gpu_argument"]

        print("\n" + "=" * 82)
        print(
            f" Bloco {block_number}/{len(session['schedule'])} | "
            f"Rodada {block['round']}/{ROUNDS} | "
            f"--gpu {gpu_argument}"
        )
        print(
            f" Prompts: {', '.join(block['prompt_order'])}"
        )
        print("=" * 82)

        loaded = False
        load_wall_time: float | None = None
        parsed_log: dict[str, Any] = {}

        try:
            ensure_server()

            print("[LOAD] Descarregando modelo anterior...")
            unload_all()

            if COOLDOWN_AFTER_UNLOAD_SECONDS > 0:
                print(
                    f"[WAIT] Cooldown "
                    f"{COOLDOWN_AFTER_UNLOAD_SECONDS}s..."
                )
                time.sleep(
                    COOLDOWN_AFTER_UNLOAD_SECONDS
                )

            thermal_gate(
                session_dir,
                block["block_id"],
            )

            log_path = (
                session_dir
                / "load_logs"
                / f"{block['block_id']}.log"
            )

            log_proc, log_handle = start_model_log_capture(
                log_path
            )

            print(
                f"[LOAD] Carregando modelo com "
                f"`--gpu {gpu_argument}`..."
            )

            try:
                load_result, load_wall_time = load_model(
                    gpu_argument
                )
                loaded = True
                time.sleep(1.0)
            finally:
                stop_model_log_capture(
                    log_proc,
                    log_handle,
                )

            parsed_log = parse_load_log(
                log_path
            )

            print(
                f"[LOAD] OK em {load_wall_time:.2f}s"
            )

            print(
                "[LOG] n_threads="
                f"{parsed_log.get('n_threads', 'n/d')} | "
                "n_threads_batch="
                f"{parsed_log.get('n_threads_batch', 'n/d')} | "
                "n_gpu_layers="
                f"{parsed_log.get('n_gpu_layers', 'n/d')}"
            )

            models_response = api_get(
                "/api/v1/models"
            )

            _, loaded_instance = find_loaded_instance(
                models_response
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

            issues = audit_loaded_config(
                loaded_config
            )

            append_jsonl(
                loads_path,
                {
                    "timestamp": now_iso(),
                    "block_id": block["block_id"],
                    "round": block["round"],
                    "mode": mode,
                    "gpu_argument": gpu_argument,
                    "load_wall_time_seconds": load_wall_time,
                    "loaded_config": loaded_config,
                    "config_audit_issues": issues,
                    "load_log_path": str(log_path),
                    "load_log_parsed": parsed_log,
                    "lms_stdout": load_result.stdout.strip(),
                    "lms_stderr": load_result.stderr.strip(),
                },
            )

            if issues:
                raise RuntimeError(
                    "Auditoria da configuração falhou: "
                    + "; ".join(issues)
                )

            print("[AUDIT] Configuração confere.")

            if SETTLE_AFTER_LOAD_SECONDS > 0:
                time.sleep(
                    SETTLE_AFTER_LOAD_SECONDS
                )

            warm_before = record_temperature(
                session_dir,
                "warmup_before",
                block_id=block["block_id"],
            )

            print(
                f"[WARMUP] Início: "
                f"{fmt_temp(warm_before)}"
            )

            warm_response, warm_wall = run_inference(
                WARMUP_PROMPT,
                WARMUP_MAX_OUTPUT_TOKENS,
            )

            warm_after = record_temperature(
                session_dir,
                "warmup_after",
                block_id=block["block_id"],
            )

            warm_stats = (
                warm_response.get("stats")
                or {}
            )

            append_jsonl(
                warmups_path,
                {
                    "timestamp": now_iso(),
                    "block_id": block["block_id"],
                    "round": block["round"],
                    "mode": mode,
                    "gpu_argument": gpu_argument,
                    "wall_time_seconds": warm_wall,
                    "stats": warm_stats,
                    "temp_before": warm_before,
                    "temp_after": warm_after,
                    "load_log_parsed": parsed_log,
                    "response_text": extract_message(
                        warm_response
                    ),
                },
            )

            print(
                f"[WARMUP] Fim:    "
                f"{fmt_temp(warm_after)} | "
                f"{warm_stats.get('tokens_per_second', 'n/d')} "
                "tok/s"
            )

            if SETTLE_AFTER_WARMUP_SECONDS > 0:
                time.sleep(
                    SETTLE_AFTER_WARMUP_SECONDS
                )

            for prompt_position, prompt_id in enumerate(
                block["prompt_order"],
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

                before = record_temperature(
                    session_dir,
                    "run_before",
                    block_id=block["block_id"],
                    run_id=run_id,
                )

                print(
                    f"\n[RUN {global_number}/{total_runs}] "
                    f"{run_id} | "
                    f"--gpu {gpu_argument} | "
                    f"Prompt {prompt_id}"
                )

                print(
                    f"      CPU antes: "
                    f"{fmt_temp(before)}"
                )

                if eta is not None:
                    print(
                        f"      ETA: "
                        f"{seconds_to_hms(eta)}"
                    )

                print(
                    "      Inferindo... "
                    "(sem monitoramento em background)"
                )

                try:
                    response, wall_time = run_inference(
                        PROMPTS[prompt_id],
                        MAX_OUTPUT_TOKENS,
                    )

                    after = record_temperature(
                        session_dir,
                        "run_after",
                        block_id=block["block_id"],
                        run_id=run_id,
                    )

                    stats = (
                        response.get("stats")
                        or {}
                    )

                    reasoning_tokens = stats.get(
                        "reasoning_output_tokens"
                    )

                    record = {
                        "status": "success",
                        "timestamp": now_iso(),
                        "run_id": run_id,
                        "block_id": block["block_id"],
                        "round": block["round"],
                        "prompt_position": prompt_position,
                        "mode": mode,
                        "gpu_argument": gpu_argument,
                        "prompt_id": prompt_id,
                        "prompt": PROMPTS[prompt_id],
                        "load_wall_time_seconds": load_wall_time,
                        "wall_time_seconds": wall_time,
                        "stats": stats,
                        "reasoning_off_verified": (
                            reasoning_tokens == 0
                        ),
                        "temp_before": before,
                        "temp_after": after,
                        "load_log_parsed": parsed_log,
                        "response_text": extract_message(
                            response
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

                    tps = stats.get(
                        "tokens_per_second"
                    )
                    ttft = stats.get(
                        "time_to_first_token_seconds"
                    )
                    out_tokens = stats.get(
                        "total_output_tokens"
                    )

                    tps_s = (
                        f"{tps:.3f}"
                        if isinstance(
                            tps,
                            (int, float),
                        )
                        else "n/d"
                    )

                    ttft_s = (
                        f"{ttft:.3f}s"
                        if isinstance(
                            ttft,
                            (int, float),
                        )
                        else "n/d"
                    )

                    before_c = temp_value(before)
                    after_c = temp_value(after)

                    if (
                        before_c is not None
                        and after_c is not None
                    ):
                        temp_text = (
                            f"{before_c:.1f} → "
                            f"{after_c:.1f} °C "
                            f"(Δ {after_c - before_c:+.1f})"
                        )
                    else:
                        temp_text = "n/d"

                    print(
                        f"      ✓ {tps_s} tok/s | "
                        f"TTFT {ttft_s} | "
                        f"{out_tokens} tokens | "
                        f"{wall_time:.1f}s"
                    )

                    print(
                        f"      CPU temp: {temp_text}"
                    )

                except Exception as exc:
                    print(
                        f"      ✗ ERRO: {exc}"
                    )

                    record_error(
                        session_dir,
                        phase="inference",
                        message=str(exc),
                        block=block,
                        run_id=run_id,
                    )

                if PAUSE_BETWEEN_PROMPTS_SECONDS > 0:
                    time.sleep(
                        PAUSE_BETWEEN_PROMPTS_SECONDS
                    )

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            print(
                f"[BLOCO] ERRO: {exc}"
            )

            record_error(
                session_dir,
                phase="block",
                message=str(exc),
                block=block,
            )

        finally:
            if loaded:
                print(
                    "[CLEANUP] Descarregando modelo..."
                )
                unload_all()

            session[
                "successful_runs"
            ] = len(successful_ids)

            session[
                "remaining_runs"
            ] = (
                total_runs
                - len(successful_ids)
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
        len(successful_ids)
        == total_runs
    )

    session[
        "successful_runs"
    ] = len(successful_ids)

    session[
        "remaining_runs"
    ] = (
        total_runs
        - len(successful_ids)
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

    print_summary(summary)

    print(
        f"\nResultados:\n  {session_dir}"
    )

    if all_done:
        print(
            "\n✓ Todas as 18 gerações "
            "foram concluídas."
        )
        return 0

    print(
        f"\n⚠ Benchmark incompleto: "
        f"{len(successful_ids)}/{total_runs}."
    )
    print(
        "Rode o mesmo comando novamente "
        "para retomar."
    )

    return 2


# ------------------------------------------------------------
# CLI / main
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara Vulkan `--gpu off` "
            "contra Vulkan `--gpu 0.0`."
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
        help="Não pede confirmação para iniciar.",
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
        incomplete = (
            find_latest_incomplete_session()
        )

        if incomplete is not None:
            return (
                incomplete,
                load_session(incomplete),
                True,
            )

    session_dir, session = (
        create_session(args.seed)
    )

    return (
        session_dir,
        session,
        False,
    )


def main() -> int:
    args = parse_args()

    print(
        "\n[PRECHECK] Validando "
        "LM Studio, Vulkan, modelo e temperatura..."
    )

    ensure_server()
    verify_runtime_installed(
        VULKAN_RUNTIME
    )
    select_vulkan_runtime()

    selected = selected_gguf_runtime()

    print(
        f"  ✓ runtime: {selected}"
    )

    model_inventory = (
        verify_selected_variant()
    )

    print(
        f"  ✓ modelo: "
        f"{EXPECTED_VARIANT}"
    )

    temp_probe = (
        read_cpu_temperature()
    )

    if temp_probe:
        print(
            f"  ✓ temperatura: "
            f"{fmt_temp(temp_probe)}"
        )

        if (
            temp_probe.get("reliability")
            == "approximate_not_cpu_package"
        ):
            print(
                "    ⚠ É uma Thermal Zone ACPI; "
                "não é garantia de CPU Package."
            )
            print(
                "    Para CPU Package real, "
                "deixe o LibreHardwareMonitor aberto."
            )
    else:
        print(
            "  ⚠ temperatura: sensor não encontrado."
        )
        print(
            "    O benchmark continuará, mas as "
            "temperaturas aparecerão como n/d."
        )
        print(
            "    Para CPU Package real, abra o "
            "LibreHardwareMonitor durante o teste."
        )

    session_dir, session, resumed = (
        choose_session(args)
    )

    session[
        "model_inventory_at_start"
    ] = model_inventory

    session[
        "runtime_ls_at_start"
    ] = runtime_ls_text()

    session[
        "temperature_probe_at_start"
    ] = temp_probe

    update_session_file(
        session_dir,
        session,
    )

    print("\n" + "=" * 82)
    print(" LocalIA — Vulkan GPU Mode Diagnostic")
    print("=" * 82)
    print(
        f" Modelo           : "
        f"{MODEL_KEY} / Q4_K_M"
    )
    print(
        f" Runtime          : "
        f"{VULKAN_RUNTIME}"
    )
    print(
        f" Comparação       : "
        f"--gpu off  VS  --gpu 0.0"
    )
    print(
        f" Contexto         : "
        f"{CONTEXT_LENGTH}"
    )
    print(
        f" CPU threads alvo : "
        f"{CPU_THREADS_TARGET} "
        "(tentaremos ler do log)"
    )
    print(
        f" Temp. início alvo: "
        f"≤ {TARGET_START_TEMP_C:.0f} °C"
    )
    print(
        f" Rodadas          : "
        f"{ROUNDS}"
    )
    print(
        f" Gerações medidas : "
        f"{session['planned_measured_runs']}"
    )
    print(
        f" Sessão           : "
        f"{session_dir.name}"
    )
    print("=" * 82)

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
        return execute_benchmark(
            session_dir,
            session,
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


if __name__ == "__main__":
    raise SystemExit(main())
