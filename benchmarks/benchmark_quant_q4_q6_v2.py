from __future__ import annotations

"""
LocalIA — Screening Backend × Threads v5
=========================================

Objetivo
--------
Repetir o benchmark fatorial de forma metodologicamente mais forte:

    2 backends × 7 thread pools × 3 prompts × 1 rodada
    = 42 gerações medidas

Backends:
    - CPU AVX2 llama.cpp 2.46.0
    - Vulkan llama.cpp 2.46.0 (GPU offload = OFF)

Thread pools e cpuThreads testados:
    - 2 / 4 / 6 / 8 / 10 / 12 / 16

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
   - maxTokens não é enviado nas gerações medidas.
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

4. CONTROLE LOAD-TIME CONFIRMADO EM CADA BLOCO:
   Antes das gerações de cada condição, o script altera o Per-model Default,
   faz reload e exige que o server log confirme exatamente n_threads=X.
   O uso real do processo também é medido, mas não é usado como critério
   artificial de aprovação/reprovação: queremos observar o comportamento real.

5. POWER BEHAVIOR COLETADO PARA ANÁLISE POSTERIOR:
   A série temporal de potência, clocks e temperatura é salva para investigar
   PL1/PL2 depois. O script não atribui automaticamente a causa de throttling.

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
import tempfile
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
INSTANCE_ID = "localia-screening-v5"

RUNTIMES = {
    "cpu": "llama.cpp-win-x86_64-avx2@2.46.0",
    "vulkan": "llama.cpp-win-x86_64-vulkan-avx2@2.46.0",
}

THREAD_COUNTS = [2, 4, 6, 8, 10, 12, 16]

ROUNDS = 1
SEED = 20260927

CONTEXT_LENGTH = 4096
TEMPERATURE = 0.0

# IMPORTANTE:
# False = sem cap explícito de tokens no LM Studio SDK.
MAX_TOKENS: int | bool | None = None

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
MONITOR_INTERVAL_SECONDS = 0.5

# Tempos auxiliares
COOLDOWN_AFTER_UNLOAD_SECONDS = 6
SETTLE_AFTER_RUNTIME_SWITCH_SECONDS = 2
SETTLE_AFTER_LOAD_SECONDS = 3
SETTLE_AFTER_WARMUP_SECONDS = 2
PAUSE_BETWEEN_PROMPTS_SECONDS = 2
PAUSE_BETWEEN_QUALITY_TASKS_SECONDS = 0.5

LMS_COMMAND_TIMEOUT_SECONDS = 120.0
LMS_LOAD_TIMEOUT_SECONDS = 240.0

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"

CONFIG_ROOT = (
    Path.home()
    / ".lmstudio"
    / ".internal"
    / "user-concrete-model-default-config"
)
CONFIG_WATCH_SETTLE_SECONDS = 2.0

REASONING_ENABLE_KEY = "llm.prediction.reasoning.enableThinking"

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
# Per-model Default / CPU Thread Pool Size REAL
# ============================================================

def collect_threadpool_values(
    obj: Any,
    path: str = "$",
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            current = f"{path}.{key}"

            if key in {
                "cpuThreadPoolSize",
                "llm.load.llama.cpuThreadPoolSize",
            }:
                found.append({
                    "path": current,
                    "value": value,
                    "style": "direct-key",
                })

            if (
                key == "key"
                and value == "llm.load.llama.cpuThreadPoolSize"
            ):
                found.append({
                    "path": path,
                    "value": obj.get("value"),
                    "style": "field-entry",
                })

            found.extend(
                collect_threadpool_values(value, current)
            )

    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            found.extend(
                collect_threadpool_values(
                    item,
                    f"{path}[{index}]",
                )
            )

    return found


def set_threadpool_value(
    obj: Any,
    new_value: int,
) -> int:
    changed = 0

    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in {
                "cpuThreadPoolSize",
                "llm.load.llama.cpuThreadPoolSize",
            }:
                obj[key] = int(new_value)
                changed += 1

        if (
            obj.get("key")
            == "llm.load.llama.cpuThreadPoolSize"
        ):
            obj["value"] = int(new_value)
            changed += 1

        for key, value in list(obj.items()):
            if key in {
                "cpuThreadPoolSize",
                "llm.load.llama.cpuThreadPoolSize",
                "value",
            }:
                continue
            changed += set_threadpool_value(
                value,
                new_value,
            )

    elif isinstance(obj, list):
        for item in obj:
            changed += set_threadpool_value(
                item,
                new_value,
            )

    return changed


def find_model_config_file() -> Path:
    if not CONFIG_ROOT.exists():
        raise RuntimeError(
            "Diretório de Per-model Defaults não encontrado:\n"
            f"  {CONFIG_ROOT}"
        )

    candidates: list[tuple[int, Path]] = []

    for path in CONFIG_ROOT.rglob("*.json"):
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception:
            continue

        values = collect_threadpool_values(data)
        if not values:
            continue

        normalized = str(path).lower().replace("\\", "/")
        raw_lower = raw.lower()
        score = 0

        if "qwen" in normalized:
            score += 20
        if "qwen3.5-9b" in normalized:
            score += 100
        if "qwen3.5-9b" in raw_lower:
            score += 50
        if "qwen/qwen3.5-9b" in raw_lower:
            score += 100

        candidates.append((score, path))

    if not candidates:
        raise RuntimeError(
            "Não encontrei o Per-model Default do Qwen "
            "contendo cpuThreadPoolSize."
        )

    candidates.sort(
        key=lambda item: (item[0], str(item[1])),
        reverse=True,
    )

    best_score = candidates[0][0]
    best = [x for x in candidates if x[0] == best_score]

    if len(best) != 1:
        details = "\n".join(
            f"  score={score:3d} {path}"
            for score, path in candidates[:10]
        )
        raise RuntimeError(
            "Há mais de um arquivo candidato; "
            "não vou escolher automaticamente.\n"
            + details
        )

    return best[0][1]


def write_threadpool_to_config(
    config_path: Path,
    value: int,
) -> dict[str, Any]:
    data = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    before = collect_threadpool_values(data)
    changed = set_threadpool_value(data, value)

    if changed <= 0:
        raise RuntimeError(
            "cpuThreadPoolSize não pôde ser alterado."
        )

    atomic_write_json(config_path, data)

    reloaded = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    after = collect_threadpool_values(reloaded)
    values_after = {item.get("value") for item in after}

    if values_after != {int(value)}:
        raise RuntimeError(
            "Config inconsistente após escrever CPU Thread Pool Size.\n"
            f"Valores encontrados: {sorted(values_after, key=str)}"
        )

    return {
        "changed_occurrences": changed,
        "before": before,
        "after": after,
    }



def collect_operation_field_values(
    data: dict[str, Any],
    key: str,
) -> list[dict[str, Any]]:
    operation = data.get("operation")

    if not isinstance(operation, dict):
        return []

    fields = operation.get("fields")

    if not isinstance(fields, list):
        return []

    found: list[dict[str, Any]] = []

    for index, item in enumerate(fields):
        if (
            isinstance(item, dict)
            and item.get("key") == key
        ):
            found.append(
                {
                    "index": index,
                    "key": key,
                    "value": item.get("value"),
                }
            )

    return found


def set_operation_field(
    data: dict[str, Any],
    key: str,
    value: Any,
) -> dict[str, Any]:
    operation = data.setdefault(
        "operation",
        {},
    )

    if not isinstance(operation, dict):
        raise RuntimeError(
            "`operation` existe no Per-model Default, "
            "mas não é um objeto JSON."
        )

    fields = operation.setdefault(
        "fields",
        [],
    )

    if not isinstance(fields, list):
        raise RuntimeError(
            "`operation.fields` existe, mas não é uma lista."
        )

    matches = [
        item
        for item in fields
        if (
            isinstance(item, dict)
            and item.get("key") == key
        )
    ]

    before = [
        {
            "key": item.get("key"),
            "value": item.get("value"),
        }
        for item in matches
    ]

    if len(matches) > 1:
        raise RuntimeError(
            f"Há {len(matches)} entradas duplicadas para {key}; "
            "não vou escolher silenciosamente."
        )

    if matches:
        matches[0]["value"] = value
        action = "updated"
    else:
        fields.append(
            {
                "key": key,
                "value": value,
            }
        )
        action = "added"

    return {
        "action": action,
        "before": before,
        "after_value": value,
    }


def write_block_overrides_to_config(
    config_path: Path,
    thread_pool_size: int,
) -> dict[str, Any]:
    """
    Aplica no mesmo write atômico:
      - load-time CPU Thread Pool Size;
      - prediction-time Enable Thinking = false.

    O segundo campo usa o próprio schema nativo do LM Studio:
      llm.prediction.reasoning.enableThinking
    """
    data = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    threadpool_before = collect_threadpool_values(
        data
    )

    changed = set_threadpool_value(
        data,
        thread_pool_size,
    )

    if changed <= 0:
        raise RuntimeError(
            "cpuThreadPoolSize não pôde ser alterado."
        )

    reasoning_edit = set_operation_field(
        data,
        REASONING_ENABLE_KEY,
        False,
    )

    atomic_write_json(
        config_path,
        data,
    )

    reloaded = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    threadpool_after = collect_threadpool_values(
        reloaded
    )

    pool_values = {
        item.get("value")
        for item in threadpool_after
    }

    if pool_values != {
        int(thread_pool_size)
    }:
        raise RuntimeError(
            "CPU Thread Pool Size inconsistente após escrita.\n"
            f"Valores encontrados: {sorted(pool_values, key=str)}"
        )

    reasoning_after = collect_operation_field_values(
        reloaded,
        REASONING_ENABLE_KEY,
    )

    if len(reasoning_after) != 1:
        raise RuntimeError(
            "A configuração Think OFF não ficou única no "
            "Per-model Default."
        )

    if reasoning_after[0].get("value") is not False:
        raise RuntimeError(
            "Think OFF não persistiu como false no "
            "Per-model Default."
        )

    return {
        "threadpool": {
            "changed_occurrences": changed,
            "before": threadpool_before,
            "after": threadpool_after,
        },
        "reasoning_enable_thinking": {
            **reasoning_edit,
            "key": REASONING_ENABLE_KEY,
            "verified_after": reasoning_after,
        },
    }


def restore_config_bytes(
    config_path: Path,
    original_bytes: bytes,
) -> None:
    temp = config_path.with_suffix(
        config_path.suffix + ".localia.restore.tmp"
    )
    temp.write_bytes(original_bytes)
    os.replace(temp, config_path)


def json_equivalent_bytes(
    left: bytes,
    right: bytes,
) -> bool:
    try:
        return (
            json.loads(left.decode("utf-8"))
            == json.loads(right.decode("utf-8"))
        )
    except Exception:
        return left == right


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


def parse_process_cli_threads(text: str) -> int | None:
    match = re.search(
        r"(?:^|\s)--threads\s+(\d+)(?:\s|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def audit_process_command_line(
    process_info_data: dict[str, Any],
    requested_threads: int,
) -> dict[str, Any]:
    text = str(process_info_data.get("text") or "")
    cli_threads = parse_process_cli_threads(text)

    if cli_threads != requested_threads:
        raise RuntimeError(
            "O command line real do llama-server não confirma o número "
            "de threads solicitado.\n"
            f"Solicitado: {requested_threads}\n"
            f"--threads observado: {cli_threads}"
        )

    checks = {
        "threads": cli_threads,
        "ctx_size_4096": bool(re.search(r"(?:^|\s)--ctx-size\s+4096(?:\s|$)", text)),
        "n_gpu_layers_0": bool(re.search(r"(?:^|\s)--n-gpu-layers\s+0(?:\s|$)", text)),
        "batch_size_2048": bool(re.search(r"(?:^|\s)--batch-size\s+2048(?:\s|$)", text)),
        "ubatch_size_512": bool(re.search(r"(?:^|\s)--ubatch-size\s+512(?:\s|$)", text)),
        "parallel_1": bool(re.search(r"(?:^|\s)--parallel\s+1(?:\s|$)", text)),
        "flash_attn_off": bool(re.search(r"(?:^|\s)--flash-attn\s+off(?:\s|$)", text)),
        "no_kv_offload": "--no-kv-offload" in text,
    }
    failed = [
        key for key, value in checks.items()
        if key != "threads" and value is not True
    ]
    if failed:
        raise RuntimeError(
            "O command line real do llama-server divergiu da configuração "
            "experimental esperada: " + ", ".join(failed)
        )
    return checks


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


def _series(
    samples: list[dict[str, Any]],
    key: str,
) -> list[tuple[float, float]]:
    if not samples:
        return []

    t0 = samples[0].get("monotonic")
    if not isinstance(t0, (int, float)):
        return []

    out: list[tuple[float, float]] = []
    for sample in samples:
        t = sample.get("monotonic")
        value = sample.get(key)
        if isinstance(t, (int, float)) and isinstance(value, (int, float)):
            out.append((float(t) - float(t0), float(value)))
    return out


def _window_mean(
    series: list[tuple[float, float]],
    start_s: float | None,
    end_s: float | None,
) -> float | None:
    values = []
    for elapsed, value in series:
        if start_s is not None and elapsed < start_s:
            continue
        if end_s is not None and elapsed >= end_s:
            continue
        values.append(value)
    return safe_mean(values)


def _last_window_mean(
    series: list[tuple[float, float]],
    seconds: float,
) -> float | None:
    if not series:
        return None
    end = max(t for t, _ in series)
    return _window_mean(series, max(0.0, end - seconds), None)


def _percent_drop(
    initial: float | None,
    final: float | None,
) -> float | None:
    if initial is None or final is None or initial == 0:
        return None
    return (initial - final) / initial * 100.0


def summarize_hardware_samples(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [
            float(sample[key])
            for sample in samples
            if isinstance(sample.get(key), (int, float))
        ]

    package_temp = vals("cpu_package_temp_c")
    core_max_temp = vals("core_max_temp_c")
    distance = vals("distance_to_tjmax_min_c")
    power = vals("cpu_package_power_w")
    cpu_total = vals("cpu_total_load_percent")
    p_clock = vals("p_core_clock_mean_mhz")
    e_clock = vals("e_core_clock_mean_mhz")

    logical_means: dict[str, float | None] = {}
    for sensor_id in LOGICAL_LOAD_IDS:
        values = []
        for sample in samples:
            loads = sample.get("logical_loads_percent") or {}
            value = loads.get(sensor_id)
            if isinstance(value, (int, float)):
                values.append(float(value))
        logical_means[sensor_id] = safe_mean(values)

    p_logical = [
        value
        for sensor_id, value in logical_means.items()
        if sensor_id in P_LOGICAL_LOAD_IDS
        and isinstance(value, (int, float))
    ]
    e_logical = [
        value
        for sensor_id, value in logical_means.items()
        if sensor_id in E_LOGICAL_LOAD_IDS
        and isinstance(value, (int, float))
    ]

    power_series = _series(samples, "cpu_package_power_w")
    p_clock_series = _series(samples, "p_core_clock_mean_mhz")
    e_clock_series = _series(samples, "e_core_clock_mean_mhz")

    power_first5 = _window_mean(power_series, 0.0, 5.0)
    power_first10 = _window_mean(power_series, 0.0, 10.0)
    power_10_20 = _window_mean(power_series, 10.0, 20.0)
    power_last10 = _last_window_mean(power_series, 10.0)
    p_first5 = _window_mean(p_clock_series, 0.0, 5.0)
    p_last10 = _last_window_mean(p_clock_series, 10.0)
    e_first5 = _window_mean(e_clock_series, 0.0, 5.0)
    e_last10 = _last_window_mean(e_clock_series, 10.0)

    power_peak = max(power) if power else None
    time_to_power_peak = None
    if power_peak is not None:
        for elapsed, value in power_series:
            if value == power_peak:
                time_to_power_peak = elapsed
                break

    duration = None
    if len(samples) >= 2:
        t0 = samples[0].get("monotonic")
        t1 = samples[-1].get("monotonic")
        if isinstance(t0, (int, float)) and isinstance(t1, (int, float)):
            duration = float(t1) - float(t0)

    return {
        "sample_count": len(samples),
        "sample_interval_target_seconds": MONITOR_INTERVAL_SECONDS,
        "duration_sampled_seconds": duration,
        "temp_start_c": package_temp[0] if package_temp else None,
        "temp_end_c": package_temp[-1] if package_temp else None,
        "temp_peak_c": max(package_temp) if package_temp else None,
        "temp_mean_c": safe_mean(package_temp),
        "core_max_peak_c": max(core_max_temp) if core_max_temp else None,
        "distance_to_tjmax_min_c": min(distance) if distance else None,
        "package_power_mean_w": safe_mean(power),
        "package_power_peak_w": power_peak,
        "time_to_package_power_peak_seconds": time_to_power_peak,
        "package_power_first_5s_mean_w": power_first5,
        "package_power_first_10s_mean_w": power_first10,
        "package_power_10_20s_mean_w": power_10_20,
        "package_power_last_10s_mean_w": power_last10,
        "package_power_drop_first5_to_last10_percent": _percent_drop(
            power_first5, power_last10
        ),
        "cpu_total_load_mean_percent": safe_mean(cpu_total),
        "cpu_total_load_peak_percent": max(cpu_total) if cpu_total else None,
        "p_core_clock_mean_mhz": safe_mean(p_clock),
        "p_core_clock_first_5s_mean_mhz": p_first5,
        "p_core_clock_last_10s_mean_mhz": p_last10,
        "p_core_clock_drop_first5_to_last10_percent": _percent_drop(
            p_first5, p_last10
        ),
        "e_core_clock_mean_mhz": safe_mean(e_clock),
        "e_core_clock_first_5s_mean_mhz": e_first5,
        "e_core_clock_last_10s_mean_mhz": e_last10,
        "e_core_clock_drop_first5_to_last10_percent": _percent_drop(
            e_first5, e_last10
        ),
        "p_logical_load_mean_percent": safe_mean(p_logical),
        "e_logical_load_mean_percent": safe_mean(e_logical),
        "logical_load_means_percent": logical_means,
        "seconds_at_or_above_80c": sum(x >= 80.0 for x in package_temp) * MONITOR_INTERVAL_SECONDS,
        "seconds_at_or_above_85c": sum(x >= 85.0 for x in package_temp) * MONITOR_INTERVAL_SECONDS,
        "seconds_at_or_above_90c": sum(x >= 90.0 for x in package_temp) * MONITOR_INTERVAL_SECONDS,
        "power_behavior_interpretation": "descriptive_only_no_pl1_pl2_claim",
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

    if isinstance(obj, dict):
        return dict(obj)

    if isinstance(obj, list):
        return list(obj)

    if isinstance(obj, (str, int, float, bool)):
        return obj

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
) -> tuple[dict[str, Any], float]:
    """
    Não envia promptTemplate nem enableThinking pela API.

    O Think OFF é aplicado pelo Per-model Default do próprio LM Studio:
      llm.prediction.reasoning.enableThinking = false

    Isso evita o caminho que falhou no v3, em que um promptTemplate
    customizado por requisição ainda resultava em reasoning no Qwen3.5.
    """
    config = {
        "temperature": TEMPERATURE,
        "cpuThreads": cpu_threads,
        # Deliberadamente sem maxTokens nas medições.
        # Deliberadamente sem promptTemplate customizado.
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
        "max_tokens_requested": None,
        "thinking_control_method": (
            "lmstudio_per_model_default_"
            "llm.prediction.reasoning.enableThinking_false"
        ),
        "thinking_config_key": REASONING_ENABLE_KEY,
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
    threads: int,
    psutil: Any,
    config_path: Path,
) -> tuple[
    Any,
    dict[str, Any],
    float,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Carrega uma condição e só aceita o bloco se n_threads == solicitado."""

    ensure_server()

    print(f"[CONFIG] CPU Thread Pool Size → {threads}")
    edit_info = write_block_overrides_to_config(
        config_path,
        threads,
    )
    changed_occurrences = (
        edit_info
        .get("threadpool", {})
        .get("changed_occurrences")
    )
    if not isinstance(changed_occurrences, int) or changed_occurrences <= 0:
        raise RuntimeError(
            "Contrato interno inválido: não consegui confirmar "
            "quantas ocorrências de cpuThreadPoolSize foram alteradas."
        )
    print(
        f"[CONFIG] Threadpool alterado em "
        f"{changed_occurrences} ocorrência(s) | "
        "Thinking=false confirmado"
    )
    time.sleep(CONFIG_WATCH_SETTLE_SECONDS)

    print("[SWITCH] Descarregando modelo anterior...")
    unload_all()
    if COOLDOWN_AFTER_UNLOAD_SECONDS:
        time.sleep(COOLDOWN_AFTER_UNLOAD_SECONDS)

    print(f"[SWITCH] Selecionando {backend.upper()}...")
    selected_line, switch_seconds = select_runtime(runtime_alias)
    print(f"[SWITCH] OK — {selected_line}")

    append_jsonl(
        session_dir / "runtime_switches.jsonl",
        {
            "timestamp": now_iso(),
            "block_id": block_label,
            "backend": backend,
            "threads": threads,
            "runtime_alias": runtime_alias,
            "switch_wall_time_seconds": switch_seconds,
            "selected_line": selected_line,
        },
    )

    time.sleep(SETTLE_AFTER_RUNTIME_SWITCH_SECONDS)
    thermal_gate(session_dir, block_label)

    pids_before = snapshot_pids(psutil)
    log_path = session_dir / "load_logs" / f"{block_label}.jsonl"
    log_proc, log_handle = start_server_log_capture(log_path)

    print("[LOAD] Carregando modelo com GPU offload OFF...")
    try:
        load_result, load_wall_time = load_model()
        time.sleep(1.0)
    finally:
        stop_log_capture(log_proc, log_handle)

    load_log_parsed = parse_load_log(log_path)
    pool_size = load_log_parsed.get("threadpool_n_threads")

    if pool_size != threads:
        raise RuntimeError(
            "THREADPOOL REAL NÃO CORRESPONDE AO SOLICITADO.\n"
            f"Solicitado: {threads}\n"
            f"llama.cpp n_threads: {pool_size}"
        )

    models_response = api_get("/api/v1/models")
    _, loaded_instance = find_loaded_instance(models_response)
    if loaded_instance is None:
        raise RuntimeError(
            f"Instância `{INSTANCE_ID}` não apareceu na API."
        )

    loaded_config = loaded_instance.get("config") or {}
    config_issues = audit_loaded_config(loaded_config)
    if config_issues:
        raise RuntimeError(
            "Auditoria da configuração de load falhou: "
            + "; ".join(config_issues)
        )

    offloaded = load_log_parsed.get("offloaded_layers")
    if backend == "vulkan":
        if offloaded is None:
            raise RuntimeError(
                "Não consegui auditar o número de camadas offloaded."
            )
        if int(offloaded) != 0:
            raise RuntimeError(
                f"Vulkan offloaded {offloaded} camada(s); esperado: 0."
            )
    elif offloaded is not None and int(offloaded) != 0:
        raise RuntimeError(
            f"CPU registrou {offloaded} camada(s) offloaded; esperado: 0."
        )

    model_process, model_process_info = find_llama_server_process(
        psutil,
        pids_before,
    )
    process_cli_audit = audit_process_command_line(
        model_process_info,
        threads,
    )

    print(
        f"[LOAD] OK em {load_wall_time:.2f}s | "
        f"PID llama-server={model_process.pid} | "
        f"CLI --threads={process_cli_audit['threads']}"
    )
    print(
        "[LOAD-AUDIT] "
        f"threadpool={pool_size} | "
        f"offload={offloaded}/{load_log_parsed.get('total_layers')} | "
        f"ctx={load_log_parsed.get('n_ctx')} | "
        f"batch={load_log_parsed.get('n_batch')} | "
        f"ubatch={load_log_parsed.get('n_ubatch')}"
    )

    append_jsonl(
        session_dir / "loads.jsonl",
        {
            "timestamp": now_iso(),
            "block_id": block_label,
            "backend": backend,
            "threads": threads,
            "runtime_alias": runtime_alias,
            "load_wall_time_seconds": load_wall_time,
            "config_edit": edit_info,
            "loaded_config": loaded_config,
            "load_log_parsed": load_log_parsed,
            "model_process": model_process_info,
            "process_cli_audit": process_cli_audit,
            "load_log_path": str(log_path),
            "lms_stdout": load_result.stdout.strip(),
            "lms_stderr": load_result.stderr.strip(),
        },
    )

    time.sleep(SETTLE_AFTER_LOAD_SECONDS)

    return (
        model_process,
        model_process_info,
        load_wall_time,
        load_log_parsed,
        loaded_config,
        edit_info,
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

def run_preflight(*args: Any, **kwargs: Any):
    raise RuntimeError(
        "Preflight legado desativado no screening v3. "
        "O threadpool agora é controlado e verificado em cada bloco."
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
        "screening_backend_threads_v5_*"
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
        / f"screening_backend_threads_v5_{timestamp_slug()}"
    )

    session_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    schedule = build_schedule(
        seed
    )

    session = {
        "session_version": 5,
        "phase": "screening",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "completed": False,
        "completed_at": None,
        "power_behavior_note": "descriptive_only_no_pl1_pl2_claim",
        "seed": seed,
        "model_key": MODEL_KEY,
        "expected_variant": EXPECTED_VARIANT,
        "runtimes": RUNTIMES,
        "thread_counts": THREAD_COUNTS,
        "configuration": {
            "context_length": CONTEXT_LENGTH,
            "gpu_offload": "off",
            "temperature": TEMPERATURE,
            "max_tokens": None,
            "thinking": "lmstudio_native_reasoning_enableThinking_false",
            "thinking_config_key": REASONING_ENABLE_KEY,
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
    "run_id", "timestamp", "round", "backend", "runtime_alias",
    "threads_requested", "threads_echoed", "threadpool_n_threads",
    "prompt_id", "tokens_per_second", "time_to_first_token_seconds",
    "predicted_tokens_count", "wall_time_seconds", "stop_reason",
    "thinking_off_verified", "response_sha256",
    "process_pid", "effective_cpu_cores", "process_machine_cpu_percent",
    "effective_active_thread_count", "dominant_threads_80pct",
    "dominant_threads_90pct", "active_threads_ge_25pct_of_top",
    "lhm_cpu_total_mean_percent", "p_logical_load_mean_percent",
    "e_logical_load_mean_percent",
    "temp_start_c", "temp_peak_c", "temp_end_c", "temp_mean_c",
    "core_max_peak_c", "distance_to_tjmax_min_c",
    "package_power_mean_w", "package_power_peak_w",
    "package_power_first_5s_mean_w", "package_power_first_10s_mean_w",
    "package_power_10_20s_mean_w", "package_power_last_10s_mean_w",
    "package_power_drop_first5_to_last10_percent",
    "time_to_package_power_peak_seconds",
    "p_core_clock_mean_mhz", "p_core_clock_first_5s_mean_mhz",
    "p_core_clock_last_10s_mean_mhz",
    "p_core_clock_drop_first5_to_last10_percent",
    "e_core_clock_mean_mhz", "e_core_clock_first_5s_mean_mhz",
    "e_core_clock_last_10s_mean_mhz",
    "e_core_clock_drop_first5_to_last10_percent",
    "tokens_per_second_per_package_watt",
    "status",
]



def write_runs_csv(
    session_dir: Path,
) -> None:
    runs = [
        row
        for row in read_jsonl(session_dir / "runs.jsonl")
        if row.get("status") == "success"
    ]

    with (session_dir / "runs.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        csv_writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        csv_writer.writeheader()

        for row in runs:
            inference = row.get("inference") or {}
            telemetry = row.get("telemetry") or {}
            process_threads = telemetry.get("process_threads") or {}
            hardware = telemetry.get("hardware") or {}
            load_log = row.get("load_log_parsed") or {}

            tps = inference.get("tokens_per_second")
            avg_w = hardware.get("package_power_mean_w")
            tok_per_w = None
            if (
                isinstance(tps, (int, float))
                and isinstance(avg_w, (int, float))
                and avg_w > 0
            ):
                tok_per_w = float(tps) / float(avg_w)

            csv_writer.writerow({
                "run_id": row.get("run_id"),
                "timestamp": row.get("timestamp"),
                "round": row.get("round"),
                "backend": row.get("backend"),
                "runtime_alias": row.get("runtime_alias"),
                "threads_requested": row.get("threads"),
                "threads_echoed": inference.get("echoed_cpu_threads"),
                "threadpool_n_threads": load_log.get("threadpool_n_threads"),
                "prompt_id": row.get("prompt_id"),
                "tokens_per_second": tps,
                "time_to_first_token_seconds": inference.get("time_to_first_token_seconds"),
                "predicted_tokens_count": inference.get("predicted_tokens_count"),
                "wall_time_seconds": row.get("wall_time_seconds"),
                "stop_reason": inference.get("stop_reason"),
                "thinking_off_verified": inference.get("thinking_off_verified"),
                "response_sha256": inference.get("response_sha256"),
                "process_pid": process_threads.get("process_pid"),
                "effective_cpu_cores": process_threads.get("effective_cpu_cores"),
                "process_machine_cpu_percent": process_threads.get("process_machine_cpu_percent"),
                "effective_active_thread_count": process_threads.get("effective_active_thread_count"),
                "dominant_threads_80pct": process_threads.get("dominant_threads_80pct"),
                "dominant_threads_90pct": process_threads.get("dominant_threads_90pct"),
                "active_threads_ge_25pct_of_top": process_threads.get("active_threads_ge_25pct_of_top"),
                "lhm_cpu_total_mean_percent": hardware.get("cpu_total_load_mean_percent"),
                "p_logical_load_mean_percent": hardware.get("p_logical_load_mean_percent"),
                "e_logical_load_mean_percent": hardware.get("e_logical_load_mean_percent"),
                "temp_start_c": hardware.get("temp_start_c"),
                "temp_peak_c": hardware.get("temp_peak_c"),
                "temp_end_c": hardware.get("temp_end_c"),
                "temp_mean_c": hardware.get("temp_mean_c"),
                "core_max_peak_c": hardware.get("core_max_peak_c"),
                "distance_to_tjmax_min_c": hardware.get("distance_to_tjmax_min_c"),
                "package_power_mean_w": avg_w,
                "package_power_peak_w": hardware.get("package_power_peak_w"),
                "package_power_first_5s_mean_w": hardware.get("package_power_first_5s_mean_w"),
                "package_power_first_10s_mean_w": hardware.get("package_power_first_10s_mean_w"),
                "package_power_10_20s_mean_w": hardware.get("package_power_10_20s_mean_w"),
                "package_power_last_10s_mean_w": hardware.get("package_power_last_10s_mean_w"),
                "package_power_drop_first5_to_last10_percent": hardware.get("package_power_drop_first5_to_last10_percent"),
                "time_to_package_power_peak_seconds": hardware.get("time_to_package_power_peak_seconds"),
                "p_core_clock_mean_mhz": hardware.get("p_core_clock_mean_mhz"),
                "p_core_clock_first_5s_mean_mhz": hardware.get("p_core_clock_first_5s_mean_mhz"),
                "p_core_clock_last_10s_mean_mhz": hardware.get("p_core_clock_last_10s_mean_mhz"),
                "p_core_clock_drop_first5_to_last10_percent": hardware.get("p_core_clock_drop_first5_to_last10_percent"),
                "e_core_clock_mean_mhz": hardware.get("e_core_clock_mean_mhz"),
                "e_core_clock_first_5s_mean_mhz": hardware.get("e_core_clock_first_5s_mean_mhz"),
                "e_core_clock_last_10s_mean_mhz": hardware.get("e_core_clock_last_10s_mean_mhz"),
                "e_core_clock_drop_first5_to_last10_percent": hardware.get("e_core_clock_drop_first5_to_last10_percent"),
                "tokens_per_second_per_package_watt": tok_per_w,
                "status": row.get("status"),
            })


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

            power_peak = vals(("telemetry", "hardware", "package_power_peak_w"))
            power_first5 = vals(("telemetry", "hardware", "package_power_first_5s_mean_w"))
            power_first10 = vals(("telemetry", "hardware", "package_power_first_10s_mean_w"))
            power_10_20 = vals(("telemetry", "hardware", "package_power_10_20s_mean_w"))
            power_last10 = vals(("telemetry", "hardware", "package_power_last_10s_mean_w"))
            power_drop = vals(("telemetry", "hardware", "package_power_drop_first5_to_last10_percent"))
            p_clock_first5 = vals(("telemetry", "hardware", "p_core_clock_first_5s_mean_mhz"))
            p_clock_last10 = vals(("telemetry", "hardware", "p_core_clock_last_10s_mean_mhz"))

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

            efficiency_values: list[float] = []
            for item in rows:
                item_tps = (item.get("inference") or {}).get("tokens_per_second")
                item_power = ((item.get("telemetry") or {}).get("hardware") or {}).get("package_power_mean_w")
                if (
                    isinstance(item_tps, (int, float))
                    and isinstance(item_power, (int, float))
                    and item_power > 0
                ):
                    efficiency_values.append(float(item_tps) / float(item_power))

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
                    "package_power_peak_mean_w": safe_mean(power_peak),
                    "package_power_first_5s_mean_w": safe_mean(power_first5),
                    "package_power_first_10s_mean_w": safe_mean(power_first10),
                    "package_power_10_20s_mean_w": safe_mean(power_10_20),
                    "package_power_last_10s_mean_w": safe_mean(power_last10),
                    "package_power_drop_first5_to_last10_percent_mean": safe_mean(power_drop),
                    "p_core_clock_first_5s_mean_mhz": safe_mean(p_clock_first5),
                    "p_core_clock_last_10s_mean_mhz": safe_mean(p_clock_last10),
                    "p_core_clock_mean_mhz": safe_mean(
                        p_clock
                    ),
                    "e_core_clock_mean_mhz": safe_mean(
                        e_clock
                    ),
                    "tokens_per_second_per_package_watt_mean": safe_mean(efficiency_values),
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

    valid_configs = [
        row for row in by_config
        if isinstance(row.get("tokens_per_second_mean"), (int, float))
    ]
    top3_by_backend: dict[str, list[dict[str, Any]]] = {}
    for backend_name in RUNTIMES:
        backend_rows = [
            row for row in valid_configs
            if row.get("backend") == backend_name
        ]
        backend_rows.sort(
            key=lambda row: float(row["tokens_per_second_mean"]),
            reverse=True,
        )
        top3_by_backend[backend_name] = [
            {
                "threads": row["threads"],
                "tokens_per_second_mean": row["tokens_per_second_mean"],
                "ttft_mean_seconds": row["ttft_mean_seconds"],
                "package_power_mean_w": row["package_power_mean_w"],
                "temp_peak_mean_c": row["temp_peak_mean_c"],
            }
            for row in backend_rows[:3]
        ]

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
        "top3_by_backend_by_throughput": top3_by_backend,
        "screening_note": (
            "Top-3 is exploratory only; final choice requires a "
            "confirmatory benchmark with repeated rounds."
        ),
        "pl1_pl2_note": (
            "Power behavior is descriptive only; no PL1/PL2 or "
            "throttling cause is inferred automatically."
        ),
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
        "backend", "runtime_alias", "threads", "n",
        "tokens_per_second_mean", "tokens_per_second_median",
        "tokens_per_second_stdev", "ttft_mean_seconds",
        "wall_time_mean_seconds", "effective_cpu_cores_mean",
        "process_machine_cpu_percent_mean",
        "effective_active_thread_count_mean",
        "dominant_threads_90pct_mean", "lhm_cpu_total_mean_percent",
        "temp_peak_mean_c", "temp_peak_max_c", "temp_mean_c",
        "distance_to_tjmax_min_c",
        "package_power_mean_w", "package_power_peak_mean_w",
        "package_power_first_5s_mean_w", "package_power_first_10s_mean_w",
        "package_power_10_20s_mean_w", "package_power_last_10s_mean_w",
        "package_power_drop_first5_to_last10_percent_mean",
        "p_core_clock_mean_mhz", "p_core_clock_first_5s_mean_mhz",
        "p_core_clock_last_10s_mean_mhz", "e_core_clock_mean_mhz",
        "tokens_per_second_per_package_watt_mean",
        "thinking_failures", "unique_outputs_per_prompt",
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



def run_internal_self_tests(
    original_config_bytes: bytes,
) -> dict[str, Any]:
    """Pure software tests. No LM Studio calls and no real config mutation."""
    checks: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="localia-screening-selftest-") as temp_dir:
        config_copy = Path(temp_dir) / "qwen3.5-9b.json"
        config_copy.write_bytes(original_config_bytes)

        for threads in THREAD_COUNTS:
            restore_config_bytes(config_copy, original_config_bytes)
            edit_info = write_block_overrides_to_config(config_copy, threads)
            changed = edit_info.get("threadpool", {}).get("changed_occurrences")
            if not isinstance(changed, int) or changed <= 0:
                raise RuntimeError(
                    f"SELFTEST: config_edit inválido em T{threads}: {edit_info!r}"
                )
            current = json.loads(config_copy.read_text(encoding="utf-8"))
            pool_values = {
                item.get("value") for item in collect_threadpool_values(current)
            }
            if pool_values != {threads}:
                raise RuntimeError(
                    f"SELFTEST: threadpool incorreto em T{threads}: {pool_values}"
                )
            thinking_values = collect_operation_field_values(
                current, REASONING_ENABLE_KEY
            )
            if (
                len(thinking_values) != 1
                or thinking_values[0].get("value") is not False
            ):
                raise RuntimeError(
                    f"SELFTEST: Thinking=false não persistiu em T{threads}."
                )
        checks["config_matrix"] = True

        restore_config_bytes(config_copy, original_config_bytes)
        if config_copy.read_bytes() != original_config_bytes:
            raise RuntimeError("SELFTEST: restauração byte-a-byte falhou.")
        checks["byte_restore"] = True

        synthetic_cli = (
            "llama-server.exe --ctx-size 4096 --n-gpu-layers 0 "
            "--batch-size 2048 --ubatch-size 512 --threads 12 "
            "--parallel 1 --flash-attn off --no-kv-offload"
        )
        cli_audit = audit_process_command_line({"text": synthetic_cli}, 12)
        if cli_audit.get("threads") != 12:
            raise RuntimeError("SELFTEST: parser de --threads falhou.")
        checks["cli_audit"] = True

        if detect_thinking("OK", None):
            raise RuntimeError("SELFTEST: falso positivo de thinking.")
        if not detect_thinking("Thinking Process: teste", None):
            raise RuntimeError("SELFTEST: Thinking Process não detectado.")
        if not detect_thinking("Resposta", "reasoning interno"):
            raise RuntimeError("SELFTEST: reasoning_content não detectado.")
        checks["thinking_detector"] = True

        schedule = build_schedule(SEED)
        expected_blocks = len(RUNTIMES) * len(THREAD_COUNTS) * ROUNDS
        if len(schedule) != expected_blocks:
            raise RuntimeError(
                f"SELFTEST: {len(schedule)} blocos; esperado {expected_blocks}."
            )
        run_ids = [rid for block in schedule for rid in block["run_ids"]]
        if len(run_ids) != 42 or len(set(run_ids)) != 42:
            raise RuntimeError("SELFTEST: run_ids não são 42 e únicos.")
        combos = {(b["backend"], int(b["threads"])) for b in schedule}
        expected_combos = {
            (backend, threads)
            for backend in RUNTIMES
            for threads in THREAD_COUNTS
        }
        if combos != expected_combos:
            raise RuntimeError("SELFTEST: matriz backend × threads incompleta.")
        checks["schedule"] = True

        samples = []
        for i in range(41):
            elapsed = i * 0.5
            samples.append({
                "monotonic": 1000.0 + elapsed,
                "cpu_package_temp_c": 60.0 + min(elapsed, 20.0),
                "core_max_temp_c": 62.0 + min(elapsed, 20.0),
                "distance_to_tjmax_min_c": max(5.0, 40.0 - elapsed),
                "cpu_package_power_w": 30.0 if elapsed < 5.0 else 20.0,
                "cpu_total_load_percent": 50.0,
                "p_core_clock_mean_mhz": 3500.0 if elapsed < 5.0 else 2800.0,
                "e_core_clock_mean_mhz": 2800.0 if elapsed < 5.0 else 2200.0,
                "logical_loads_percent": {
                    sensor_id: 50.0 for sensor_id in LOGICAL_LOAD_IDS
                },
            })
        power_summary = summarize_hardware_samples(samples)
        if power_summary.get("package_power_first_5s_mean_w") != 30.0:
            raise RuntimeError("SELFTEST: janela inicial de potência incorreta.")
        if power_summary.get("package_power_last_10s_mean_w") != 20.0:
            raise RuntimeError("SELFTEST: janela final de potência incorreta.")
        checks["power_summary"] = True

    return checks


def run_thinking_preflight(
    *,
    session_dir: Path,
    lms: Any,
    psutil: Any,
    config_path: Path,
    original_config_bytes: bytes,
) -> dict[str, Any]:
    """
    Teste curto e LIMITADO antes das 42 medições.

    Usa o caminho nativo do LM Studio:
      operation.fields:
        llm.prediction.reasoning.enableThinking = false

    O maxTokens=96 existe SOMENTE no preflight para que uma falha de
    thinking não prenda o usuário por vários minutos.
    """
    print("\n" + "=" * 94)
    print(" PREFLIGHT THINK OFF — CONTROLE NATIVO DO LM STUDIO")
    print("=" * 94)

    test_threads = 2
    backend = "cpu"
    runtime_alias = RUNTIMES[backend]
    label = "PREFLIGHT-THINK-OFF-CPU-T2"

    restore_config_bytes(
        config_path,
        original_config_bytes,
    )

    time.sleep(
        CONFIG_WATCH_SETTLE_SECONDS
    )

    config_edit = write_block_overrides_to_config(
        config_path,
        test_threads,
    )

    time.sleep(
        CONFIG_WATCH_SETTLE_SECONDS
    )

    loaded = False

    try:
        unload_all()
        selected_line, switch_seconds = select_runtime(
            runtime_alias
        )

        pids_before = snapshot_pids(
            psutil
        )

        log_path = (
            session_dir
            / "load_logs"
            / f"{label}.jsonl"
        )

        log_proc, log_handle = start_server_log_capture(
            log_path
        )

        try:
            load_result, load_wall = load_model()
            loaded = True
            time.sleep(1.0)
        finally:
            stop_log_capture(
                log_proc,
                log_handle,
            )

        load_log = parse_load_log(
            log_path
        )

        observed_pool = load_log.get(
            "threadpool_n_threads"
        )

        if observed_pool != test_threads:
            raise RuntimeError(
                "Preflight: threadpool não corresponde ao solicitado.\n"
                f"Solicitado={test_threads}; observado={observed_pool}"
            )

        process, process_info_data = find_llama_server_process(
            psutil,
            pids_before,
        )
        process_cli_audit = audit_process_command_line(
            process_info_data,
            test_threads,
        )

        model = lms.llm()

        # Limite SOMENTE para diagnóstico de falha.
        start = time.perf_counter()
        result = model.respond(
            "Responda somente com a palavra OK.",
            config={
                "temperature": 0.0,
                "cpuThreads": test_threads,
                "maxTokens": 96,
            },
        )
        wall = time.perf_counter() - start

        stats = getattr(
            result,
            "stats",
            None,
        )
        prediction_config = serialize_object(
            getattr(result, "prediction_config", None)
        )
        echoed_threads = None
        if isinstance(prediction_config, dict):
            _, echoed_threads = find_key(
                prediction_config,
                "cpuThreads",
                "cpu_threads",
            )
        if echoed_threads is None or int(echoed_threads) != test_threads:
            raise RuntimeError(
                "Preflight: cpuThreads da predição não foi confirmado.\n"
                f"Solicitado={test_threads}; ecoado={echoed_threads!r}"
            )

        content = getattr_any(
            result,
            [
                "content",
                "text",
            ],
        )

        if content is None:
            content = str(
                result
            )

        content = str(
            content
        )

        reasoning_content = getattr_any(
            result,
            [
                "reasoning_content",
                "reasoningContent",
            ],
        )

        stop_reason = getattr_any(
            stats,
            [
                "stop_reason",
                "stopReason",
            ],
        )

        predicted = getattr_any(
            stats,
            [
                "predicted_tokens_count",
                "predictedTokensCount",
            ],
        )

        problems = detect_thinking(
            content,
            reasoning_content,
        )

        passed = (
            len(problems) == 0
            and stop_reason
            not in INVALID_STOP_REASONS
        )

        record = {
            "timestamp": now_iso(),
            "passed": passed,
            "backend": backend,
            "runtime_alias": runtime_alias,
            "selected_runtime_line": selected_line,
            "runtime_switch_seconds": switch_seconds,
            "threads": test_threads,
            "config_edit": config_edit,
            "load_wall_time_seconds": load_wall,
            "load_log_parsed": load_log,
            "model_process": process_info_data,
            "process_cli_audit": process_cli_audit,
            "prediction_config": prediction_config,
            "echoed_cpu_threads": echoed_threads,
            "content": content,
            "reasoning_content": reasoning_content,
            "thinking_problems": problems,
            "stop_reason": stop_reason,
            "predicted_tokens_count": predicted,
            "wall_time_seconds": wall,
            "preflight_max_tokens": 96,
            "lms_stdout": load_result.stdout.strip(),
            "lms_stderr": load_result.stderr.strip(),
        }

        atomic_write_json(
            session_dir
            / "thinking_preflight.json",
            record,
        )

        if problems:
            raise RuntimeError(
                "O CONTROLE NATIVO THINK OFF TAMBÉM FALHOU.\n"
                + "\n".join(
                    str(x)
                    for x in problems
                )
            )

        if stop_reason in INVALID_STOP_REASONS:
            raise RuntimeError(
                "Preflight Think OFF não encerrou naturalmente.\n"
                f"stop_reason={stop_reason}"
            )

        print(
            "[THINK] ✓ sem reasoning detectável | "
            f"tokens={predicted} | "
            f"stop={stop_reason} | "
            f"{wall:.2f}s"
        )

        print(
            "[THINK] ✓ método validado: "
            f"{REASONING_ENABLE_KEY}=false"
        )

        return record

    finally:
        if loaded:
            unload_all()

        restore_config_bytes(
            config_path,
            original_config_bytes,
        )

        time.sleep(
            CONFIG_WATCH_SETTLE_SECONDS
        )

        print(
            "[PREFLIGHT] Per-model Default original restaurado."
        )



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
    config_path: Path,
    original_config_bytes: bytes,
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
                _config_edit,
            ) = load_block(
                session_dir=session_dir,
                block_label=block[
                    "block_id"
                ],
                backend=backend,
                runtime_alias=runtime_alias,
                threads=threads,
                psutil=psutil,
                config_path=config_path,
            )

            loaded = True

            model = lms.llm()

            # Warm-up também sem thinking e sem cap fixo.
            print(
                f"[WARMUP] "
                f"{backend.upper()} / "
                f"{threads} threads | "
                "Think OFF | fim natural (sem maxTokens)"
            )

            warm_inf, warm_wall, warm_tel = (
                instrumented_inference(
                    model=model,
                    psutil=psutil,
                    process=process,
                    prompt=WARMUP_PROMPT,
                    threads=threads,
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
                    "      Think OFF | fim natural | "
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

                print(
                    "      POWER: "
                    f"avg={fmt_num(hardware.get('package_power_mean_w'), 1)} W | "
                    f"0-5s={fmt_num(hardware.get('package_power_first_5s_mean_w'), 1)} W | "
                    f"last10={fmt_num(hardware.get('package_power_last_10s_mean_w'), 1)} W | "
                    f"drop={fmt_num(hardware.get('package_power_drop_first5_to_last10_percent'), 1)}%"
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

            print("[RESTORE] Restaurando Per-model Default original...")
            restore_config_bytes(
                config_path,
                original_config_bytes,
            )
            time.sleep(CONFIG_WATCH_SETTLE_SECONDS)
            if not json_equivalent_bytes(
                config_path.read_bytes(),
                original_config_bytes,
            ):
                raise RuntimeError(
                    "Falha ao restaurar o Per-model Default original."
                )
            print("[RESTORE] ✓ original restaurado.")

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
        " SCREENING FINAL — CPU/VULKAN × 2/4/6/8/10/12/16 — THINK OFF"
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
            "\n✓ Screening concluído: todas as 42 gerações medidas foram concluídas."
        )
        return 0

    return 2


# ============================================================
# CLI / main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screening v5: CPU AVX2 vs Vulkan × "
            "2/4/6/8/10/12/16 thread pools, Think OFF, "
            "sem limite artificial de tokens, threadpool load-time "
            "verificado e telemetria de potência/temperatura."
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
        "--seed",
        type=int,
        default=SEED,
        help=f"Seed de randomização (padrão: {SEED}).",
    )
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help=(
            "Executa somente os testes internos de software e sai; "
            "não carrega o modelo e não altera o config real."
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

    print("\n[PRECHECK] Validando ambiente...")
    lms, psutil = import_dependencies()
    ensure_server()

    model_inventory = verify_selected_variant()
    print(f"  ✓ modelo: {EXPECTED_VARIANT}")

    for backend, alias in RUNTIMES.items():
        verify_runtime_installed(alias)
        print(f"  ✓ {backend}: {alias}")

    hardware_probe = verify_lhm()
    print(
        "  ✓ LibreHardwareMonitor: "
        f"CPU Package={hardware_probe['cpu_package_temp_c']:.1f} °C | "
        f"Package Power={fmt_num(hardware_probe.get('cpu_package_power_w'), 1)} W"
    )
    print(
        f"  ✓ processadores lógicos: "
        f"{psutil.cpu_count(logical=True)}"
    )

    config_path = find_model_config_file()
    current_config_bytes = config_path.read_bytes()
    current_config_json = json.loads(
        current_config_bytes.decode("utf-8")
    )
    current_pool_values = collect_threadpool_values(
        current_config_json
    )

    print("  ✓ Per-model Default:")
    print(f"    {config_path}")
    for item in current_pool_values:
        print(
            f"    {item['path']} = {item['value']!r}"
        )

    print("\n[SELFTEST] Validando contratos internos do script...")
    selftest_results = run_internal_self_tests(current_config_bytes)
    print(
        "[SELFTEST] ✓ "
        + ", ".join(
            key for key, ok in selftest_results.items() if ok
        )
    )
    if args.self_test_only:
        print(
            "[SELFTEST] Concluído. Nenhuma sessão de benchmark foi criada."
        )
        return 0

    session_dir, session, resumed = choose_session(args)
    baseline_path = session_dir / "original_model_default_config.json"

    if resumed:
        if not baseline_path.exists():
            raise RuntimeError(
                "Sessão incompleta sem backup do config original."
            )
        original_config_bytes = baseline_path.read_bytes()
        if not json_equivalent_bytes(
            current_config_bytes,
            original_config_bytes,
        ):
            raise RuntimeError(
                "Existe uma sessão incompleta, mas o Per-model Default "
                "atual mudou desde sua criação. Para proteger suas "
                "configurações, o resume foi bloqueado. Use --new."
            )
    else:
        original_config_bytes = current_config_bytes
        baseline_path.write_bytes(original_config_bytes)

    session["model_default_config_path"] = str(config_path)
    session["model_inventory_at_start"] = model_inventory
    session["runtime_ls_at_start"] = runtime_ls_text()
    session["hardware_probe_at_start"] = hardware_probe
    session["psutil_version"] = getattr(psutil, "__version__", None)
    session["lmstudio_python_version"] = getattr(lms, "__version__", None)
    session["screening_note"] = (
        "Power behavior is collected descriptively; PL1/PL2 analysis "
        "is deferred and must not be inferred automatically."
    )
    session["internal_selftests"] = selftest_results
    update_session_file(session_dir, session)

    print("\n" + "=" * 100)
    print(" LocalIA — Screening Backend × Threads v5")
    print("=" * 100)
    print(f" Modelo             : {MODEL_KEY} / Q4_K_M")
    print(" Backends           : CPU AVX2 2.46.0 + Vulkan 2.46.0")
    print(f" Thread pools       : {THREAD_COUNTS}")
    print(" Load-time control  : cpuThreadPoolSize + n_threads no server log")
    print(" Prediction threads : cpuThreads = mesmo valor do pool")
    print(" GPU offload        : OFF")
    print(f" Contexto           : {CONTEXT_LENGTH}")
    print(" Thinking           : OFF via Per-model Default nativo + validação")
    print(" Limite de tokens   : nenhum limite artificial nas medições")
    print(f" Telemetria         : {MONITOR_INTERVAL_SECONDS:.1f}s")
    print(" Power behavior     : coletado; PL1/PL2 analisado depois")
    print(f" Gerações medidas   : {session['planned_measured_runs']}")
    print(f" Warm-ups           : {len(RUNTIMES) * len(THREAD_COUNTS)}")
    print(f" Sessão             : {session_dir.name}")
    print("=" * 100)

    if resumed:
        print("\nSessão incompleta encontrada; será retomada.")

    if not args.yes:
        answer = input(
            "\nPressione ENTER para começar ou digite 'cancelar': "
        ).strip().lower()
        if answer in {"cancelar", "cancel", "n", "nao", "não"}:
            print("Screening cancelado.")
            return 0

    try:
        preflight_path = (
            session_dir
            / "thinking_preflight.json"
        )

        preflight_ok = False

        if preflight_path.exists():
            try:
                previous_preflight = json.loads(
                    preflight_path.read_text(
                        encoding="utf-8"
                    )
                )
                preflight_ok = bool(
                    previous_preflight.get(
                        "passed"
                    )
                )
            except Exception:
                preflight_ok = False

        if not preflight_ok:
            run_thinking_preflight(
                session_dir=session_dir,
                lms=lms,
                psutil=psutil,
                config_path=config_path,
                original_config_bytes=original_config_bytes,
            )
        else:
            print(
                "\n[THINK] Preflight Think OFF já aprovado "
                "nesta sessão."
            )

        return execute_benchmark(
            session_dir=session_dir,
            session=session,
            lms=lms,
            psutil=psutil,
            config_path=config_path,
            original_config_bytes=original_config_bytes,
        )

    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário.")
        try:
            unload_all()
        except Exception:
            pass
        try:
            restore_config_bytes(config_path, original_config_bytes)
            time.sleep(CONFIG_WATCH_SETTLE_SECONDS)
            print("✓ Per-model Default original restaurado.")
        except Exception as exc:
            print(f"ERRO ao restaurar config original: {exc}")

        session["interrupted_at"] = now_iso()
        update_session_file(session_dir, session)
        print(
            "Resultados preservados. Rode o mesmo comando para retomar."
        )
        return 130

    except Exception as exc:
        print(f"\nERRO FATAL: {exc}")
        try:
            unload_all()
        except Exception:
            pass
        try:
            restore_config_bytes(config_path, original_config_bytes)
            time.sleep(CONFIG_WATCH_SETTLE_SECONDS)
            print("✓ Per-model Default original restaurado.")
        except Exception as restore_exc:
            print(
                f"ERRO CRÍTICO ao restaurar config: {restore_exc}"
            )
            print(f"Backup: {baseline_path}")

        session["fatal_error_at"] = now_iso()
        session["fatal_error"] = str(exc)
        update_session_file(session_dir, session)
        print("\nNenhum resultado já concluído foi perdido.")
        return 1

    finally:
        try:
            restore_config_bytes(config_path, original_config_bytes)
        except Exception:
            pass



# ===== QUANTIZATION BENCHMARK APPENDIX =====

# =============================================================================
# LocalIA — Qwen3.5-9B Q4_K_M × Q6_K Benchmark v2
# =============================================================================

QUANT_INSTANCE_ID = "localia-quant-bench-v2"
INSTANCE_ID = QUANT_INSTANCE_ID
Q_RUNTIME = "llama.cpp-win-x86_64-avx2@2.46.0"
Q_THREADS = 8
Q_ROUNDS = 3
Q_SEED = 20260927

Q_QUANTS = {
    "q4_k_m": {"display": "Q4_K_M", "tokens": ("q4_k_m", "q4-k-m", "q4km")},
    "q6_k": {"display": "Q6_K", "tokens": ("q6_k", "q6-k", "q6k")},
}

Q_PERF_PROMPTS = {
    "A": "Explique em aproximadamente 150 palavras o que é uma API REST e cite suas principais características.",
    "B": "Explique em aproximadamente 150 palavras como funciona a fotossíntese e qual é sua importância para os ecossistemas.",
    "C": "Explique em aproximadamente 150 palavras quais foram as principais causas da Revolução Francesa.",
}

Q_ROUTER_POLICY = """
Você é Jev, a camada de decisão de um harness local.
Regras de roteamento:
1. local_fast: rota padrão para tarefas curtas de classificação, extração,
   transformação, resumo e perguntas gerais. Se cloud_allowed=false e
   local_fast estiver disponível, é obrigatório usar local_fast.
2. codex: edição, depuração ou refatoração de código/repositório quando
   cloud_allowed=true.
3. claude: síntese/análise de contexto muito longo (>20.000 tokens) quando
   cloud_allowed=true.
4. api_fallback: somente quando local_fast estiver indisponível E as rotas
   de assinatura codex/claude também estiverem indisponíveis.
5. Ações externas destrutivas ou irreversíveis exigem approval_required=true.
6. Responda SOMENTE JSON válido, sem Markdown e sem texto extra.
""".strip()

Q_QUALITY_TASKS = {
    "route_private": {
        "prompt": Q_ROUTER_POLICY + """

Cenário:
- tarefa: resumir notas clínicas privadas em 6 tópicos
- cloud_allowed: false
- local_fast_available: true
- codex_available: true
- claude_available: true
- ação externa: nenhuma

Retorne exatamente as chaves:
{"route": "...", "approval_required": false}
""",
        "expected": {"route": "local_fast", "approval_required": False},
    },
    "route_code": {
        "prompt": Q_ROUTER_POLICY + """

Cenário:
- tarefa: editar três arquivos Python de um repositório, corrigir testes e gerar o patch
- cloud_allowed: true
- local_fast_available: true
- codex_available: true
- claude_available: true
- ação externa: nenhuma

Retorne exatamente as chaves:
{"route": "...", "approval_required": false}
""",
        "expected": {"route": "codex", "approval_required": False},
    },
    "route_long_context": {
        "prompt": Q_ROUTER_POLICY + """

Cenário:
- tarefa: sintetizar 45 artigos, com contexto estimado em 78.000 tokens
- cloud_allowed: true
- local_fast_available: true
- codex_available: true
- claude_available: true
- ação externa: nenhuma

Retorne exatamente as chaves:
{"route": "...", "approval_required": false}
""",
        "expected": {"route": "claude", "approval_required": False},
    },
    "route_fallback": {
        "prompt": Q_ROUTER_POLICY + """

Cenário:
- tarefa: classificar uma frase curta
- cloud_allowed: true
- local_fast_available: false
- codex_available: false
- claude_available: false
- api_fallback_available: true
- ação externa: nenhuma

Retorne exatamente as chaves:
{"route": "...", "approval_required": false}
""",
        "expected": {"route": "api_fallback", "approval_required": False},
    },
    "privacy_overrides_code": {
        "prompt": Q_ROUTER_POLICY + """

Cenário:
- tarefa: revisar um pequeno trecho de código que contém dados privados do usuário
- cloud_allowed: false
- local_fast_available: true
- codex_available: true
- claude_available: true
- ação externa: nenhuma

Retorne exatamente as chaves:
{"route": "...", "approval_required": false}
""",
        "expected": {"route": "local_fast", "approval_required": False},
    },
    "dangerous_action_gate": {
        "prompt": Q_ROUTER_POLICY + """

Cenário:
- tarefa: apagar uma branch remota do GitHub depois de modificar o repositório
- cloud_allowed: true
- local_fast_available: true
- codex_available: true
- claude_available: true
- ação externa: destrutiva/irreversível

Retorne exatamente as chaves:
{"route": "...", "approval_required": true}
""",
        "expected": {"route": "codex", "approval_required": True},
    },
    "context_selection": {
        "prompt": """
Selecione o MENOR conjunto de documentos necessário para responder:
"Qual foi o desempenho térmico e de throughput do Qwen local?"

Documentos:
A = atas administrativas da liga acadêmica
B = benchmark de throughput do Qwen local
C = planejamento de dieta
D = telemetria térmica do notebook durante inferência
E = calendário de provas

Responda SOMENTE JSON válido, sem Markdown ou texto extra, exatamente com:
{"selected": ["ID", "..."]}
Ordene os IDs alfabeticamente.
""".strip(),
        "expected": {"selected": ["B", "D"]},
    },
    "constraint_extraction": {
        "prompt": """
Extraia as restrições do pedido abaixo.

Pedido:
"Quero que os dados nunca saiam do notebook. A resposta precisa começar em até 2500 ms. Não há edição de código. Como nuvem é proibida, somente a rota local_fast pode ser usada."

Responda SOMENTE JSON válido, sem Markdown e com exatamente estas chaves:
{"privacy":"local_only","max_latency_ms":2500,"requires_code":false,"allowed_routes":["local_fast"]}
""".strip(),
        "expected": {
            "privacy": "local_only",
            "max_latency_ms": 2500,
            "requires_code": False,
            "allowed_routes": ["local_fast"],
        },
    },
    "escalation": {
        "prompt": """
Política de escalonamento:
- comece em local_fast;
- se local_fast falhar DUAS vezes em produzir o JSON exigido, escale;
- se a tarefa não for edição de código e cloud_allowed=true, o alvo é claude;
- se cloud_allowed=false, não escale para nuvem.

Estado:
- local_fast já falhou 2 vezes por JSON inválido;
- tarefa não é edição de código;
- cloud_allowed=true;
- claude_available=true.

Responda SOMENTE JSON válido:
{"escalate":true,"target":"claude"}
""".strip(),
        "expected": {"escalate": True, "target": "claude"},
    },
    "dependency_order": {
        "prompt": """
Há quatro etapas A, B, C e D.
Restrições:
- B precisa acontecer antes de D.
- D precisa acontecer antes de A.
- A precisa acontecer antes de C.

Retorne SOMENTE JSON válido com a única ordem que satisfaz todas:
{"order":["B","D","A","C"]}
""".strip(),
        "expected": {"order": ["B", "D", "A", "C"]},
    },
}


def q_norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def q_variant_blob(candidate: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("modelKey", "path", "displayName", "indexedModelIdentifier"):
        value = candidate.get(key)
        if value:
            pieces.append(str(value))
    quant = candidate.get("quantization")
    if isinstance(quant, dict):
        for key in ("name", "type", "quantization"):
            if quant.get(key):
                pieces.append(str(quant[key]))
    elif quant:
        pieces.append(str(quant))
    return " ".join(pieces)


def q_candidate_matches(candidate: dict[str, Any], quant_key: str) -> bool:
    raw = q_variant_blob(candidate).lower()
    normalized = q_norm(raw)
    return any(
        token.lower() in raw or q_norm(token) in normalized
        for token in Q_QUANTS[quant_key]["tokens"]
    )


def q_parse_variant_listing(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    if payload and any(
        isinstance(item, dict) and "model" in item and "variants" in item
        for item in payload
    ):
        for group in payload:
            if not isinstance(group, dict):
                continue
            model = group.get("model")
            if isinstance(model, dict) and model.get("modelKey") == MODEL_KEY:
                variants = group.get("variants")
                if isinstance(variants, list):
                    return [x for x in variants if isinstance(x, dict)]
        return []
    return [x for x in payload if isinstance(x, dict)]


def q_list_variants() -> list[dict[str, Any]]:
    attempts = [
        ("lms ls MODEL --json", ("ls", MODEL_KEY, "--json")),
        ("lms ls --llm --json --variants", ("ls", "--llm", "--json", "--variants")),
    ]
    errors = []
    for label, args in attempts:
        result = run_lms(*args, allow_failure=True, timeout=30.0)
        if result.returncode != 0:
            errors.append(f"{label}: {result.stderr.strip() or result.stdout.strip()}")
            continue
        try:
            variants = q_parse_variant_listing(json.loads(result.stdout))
        except Exception as exc:
            errors.append(f"{label}: JSON inválido: {exc}")
            continue
        if variants:
            return variants
        errors.append(f"{label}: nenhuma variante retornada")
    raise RuntimeError("Não consegui listar variantes exatas.\n" + "\n".join(errors))


def q_resolve_variants() -> dict[str, dict[str, Any]]:
    variants = q_list_variants()
    resolved: dict[str, dict[str, Any]] = {}
    for quant_key in Q_QUANTS:
        matches = [x for x in variants if q_candidate_matches(x, quant_key)]
        if len(matches) != 1:
            compact = [
                {
                    "modelKey": x.get("modelKey"),
                    "path": x.get("path"),
                    "quantization": x.get("quantization"),
                    "sizeBytes": x.get("sizeBytes"),
                }
                for x in variants
            ]
            raise RuntimeError(
                f"Esperava 1 {Q_QUANTS[quant_key]['display']}, encontrei {len(matches)}. "
                f"Inventário={json.dumps(compact, ensure_ascii=False)}"
            )
        resolved[quant_key] = copy.deepcopy(matches[0])
    return resolved


def q_load_candidates(variant: dict[str, Any]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    path = variant.get("path")
    key = variant.get("modelKey")
    if isinstance(path, str) and path.strip():
        result.append(("variant_path", path.strip()))
    if isinstance(key, str) and key.strip() and key.strip() != path:
        result.append(("variant_model_key", key.strip()))
    if not result:
        raise RuntimeError("Variante sem path/modelKey utilizável.")
    return result


def q_get_cli_arg(cmdline: list[str], flag: str) -> str | None:
    for idx, item in enumerate(cmdline):
        if item == flag and idx + 1 < len(cmdline):
            return cmdline[idx + 1]
        prefix = flag + "="
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def q_process_info(proc: Any) -> dict[str, Any]:
    try:
        cmdline = [str(x) for x in proc.cmdline()]
    except Exception:
        cmdline = []
    try:
        rss = proc.memory_info().rss
    except Exception:
        rss = None
    try:
        name = proc.name()
    except Exception:
        name = None
    try:
        exe = proc.exe()
    except Exception:
        exe = None
    return {
        "pid": proc.pid,
        "name": name,
        "rss_bytes": rss,
        "rss_gib": (rss / 1024**3) if isinstance(rss, (int, float)) else None,
        "cmdline": cmdline,
        "exe": exe,
        "text": " ".join([str(name or ""), str(exe or ""), *cmdline]),
    }


def q_find_llama_process(psutil: Any, pids_before: set[int]) -> tuple[Any, dict[str, Any]]:
    candidates = []
    for proc in psutil.process_iter():
        try:
            info = q_process_info(proc)
            blob = info["text"].lower()
            if "llama" not in blob:
                continue
            score = 0.0
            if proc.pid not in pids_before:
                score += 1000
            if "llama-server" in blob:
                score += 500
            if "qwen3.5-9b" in blob:
                score += 200
            rss = info.get("rss_bytes") or 0
            score += min(float(rss) / 1024**3, 20.0) * 10
            candidates.append((score, proc, info))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates:
        raise RuntimeError("Não consegui identificar llama-server.exe.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    score, proc, info = candidates[0]
    info["selection_score"] = score
    info["candidate_count"] = len(candidates)
    return proc, info


def q_audit_cli(process_data: dict[str, Any], quant_key: str) -> dict[str, Any]:
    cmdline = process_data.get("cmdline") or []
    threads_raw = q_get_cli_arg(cmdline, "--threads")
    model_path = q_get_cli_arg(cmdline, "--model")
    gpu_layers_raw = q_get_cli_arg(cmdline, "--n-gpu-layers")
    ctx_raw = q_get_cli_arg(cmdline, "--ctx-size")

    if threads_raw is None or int(threads_raw) != Q_THREADS:
        raise RuntimeError(f"CLI --threads incorreto: {threads_raw!r}; esperado={Q_THREADS}")

    gpu_layers = int(gpu_layers_raw) if gpu_layers_raw is not None else None
    if gpu_layers is not None and gpu_layers != 0:
        raise RuntimeError(f"CLI --n-gpu-layers incorreto: {gpu_layers!r}; esperado=0")

    ctx = int(ctx_raw) if ctx_raw is not None else None
    if ctx is not None and ctx != CONTEXT_LENGTH:
        raise RuntimeError(f"CLI --ctx-size incorreto: {ctx!r}; esperado={CONTEXT_LENGTH}")

    if not model_path:
        raise RuntimeError("CLI real não contém --model.")

    model_norm = q_norm(model_path)
    if not any(q_norm(tok) in model_norm for tok in Q_QUANTS[quant_key]["tokens"]):
        raise RuntimeError(
            f"GGUF real não corresponde a {Q_QUANTS[quant_key]['display']}: {model_path}"
        )

    return {
        "threads": int(threads_raw),
        "gpu_layers": gpu_layers,
        "context_length": ctx,
        "model_path": model_path,
        "quant_verified": True,
    }


def q_find_loaded_instance_by_id(response: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for model in response.get("models", []):
        if not isinstance(model, dict):
            continue
        for instance in model.get("loaded_instances", []):
            if isinstance(instance, dict) and instance.get("id") == INSTANCE_ID:
                return model, instance
    return None, None


def q_memory_snapshot(psutil: Any, process: Any | None = None) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    rss = None
    if process is not None:
        try:
            rss = process.memory_info().rss
        except Exception:
            pass
    return {
        "timestamp": now_iso(),
        "monotonic": time.perf_counter(),
        "system_total_bytes": vm.total,
        "system_available_bytes": vm.available,
        "system_used_percent": vm.percent,
        "swap_total_bytes": swap.total,
        "swap_used_bytes": swap.used,
        "swap_used_percent": swap.percent,
        "process_rss_bytes": rss,
    }


def q_bytes_gib(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value) / 1024**3


class QuantRunMonitor(RunMonitor):
    def __init__(self, process: Any, psutil: Any, interval: float) -> None:
        super().__init__(process, psutil, interval)
        self.memory_samples: list[dict[str, Any]] = []

    def _capture(self) -> None:
        super()._capture()
        try:
            self.memory_samples.append(q_memory_snapshot(self.psutil, self.process))
        except Exception as exc:
            self.errors.append(f"memory-monitor: {exc}")

    def summarize(self, wall_time_seconds: float) -> dict[str, Any]:
        base = super().summarize(wall_time_seconds)
        base["memory"] = q_summarize_memory(self.memory_samples)
        base["memory_samples"] = self.memory_samples
        return base


def q_summarize_memory(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(x[key]) for x in samples if isinstance(x.get(key), (int, float))]

    available = vals("system_available_bytes")
    used = vals("system_used_percent")
    swap = vals("swap_used_bytes")
    swap_pct = vals("swap_used_percent")
    rss = vals("process_rss_bytes")
    total = samples[0].get("system_total_bytes") if samples else None
    swap_total = samples[0].get("swap_total_bytes") if samples else None

    return {
        "sample_count": len(samples),
        "system_total_gib": q_bytes_gib(total),
        "system_available_start_gib": q_bytes_gib(available[0]) if available else None,
        "system_available_min_gib": q_bytes_gib(min(available)) if available else None,
        "system_available_mean_gib": q_bytes_gib(safe_mean(available)) if available else None,
        "system_available_end_gib": q_bytes_gib(available[-1]) if available else None,
        "system_used_percent_mean": safe_mean(used),
        "system_used_percent_peak": max(used) if used else None,
        "swap_total_gib": q_bytes_gib(swap_total),
        "swap_used_start_gib": q_bytes_gib(swap[0]) if swap else None,
        "swap_used_peak_gib": q_bytes_gib(max(swap)) if swap else None,
        "swap_used_end_gib": q_bytes_gib(swap[-1]) if swap else None,
        "swap_used_percent_peak": max(swap_pct) if swap_pct else None,
        "process_rss_start_gib": q_bytes_gib(rss[0]) if rss else None,
        "process_rss_peak_gib": q_bytes_gib(max(rss)) if rss else None,
        "process_rss_mean_gib": q_bytes_gib(safe_mean(rss)) if rss else None,
        "process_rss_end_gib": q_bytes_gib(rss[-1]) if rss else None,
    }


def q_instrumented_inference(model: Any, psutil: Any, process: Any, prompt: str) -> tuple[dict[str, Any], float, dict[str, Any]]:
    monitor = QuantRunMonitor(process, psutil, MONITOR_INTERVAL_SECONDS)
    monitor.start()
    try:
        inference, wall = run_sdk_inference(model, prompt, Q_THREADS)
    finally:
        monitor.stop()
    verify_inference_result(inference, Q_THREADS)
    telemetry = monitor.summarize(wall)
    tps = inference.get("tokens_per_second")
    power = telemetry.get("hardware", {}).get("package_power_mean_w")
    if isinstance(tps, (int, float)) and isinstance(power, (int, float)) and power > 0:
        telemetry["tokens_per_second_per_package_watt"] = float(tps) / float(power)
    return inference, wall, telemetry


class QLoadMonitor:
    """Poll de RAM/pagefile/RSS durante o load bloqueante do SDK."""

    def __init__(self, psutil: Any, pids_before: set[int], interval: float = 0.5) -> None:
        self.psutil = psutil
        self.pids_before = set(pids_before)
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.discovered_pid: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _discover(self) -> Any | None:
        if self.discovered_pid is not None:
            try:
                return self.psutil.Process(self.discovered_pid)
            except Exception:
                self.discovered_pid = None

        candidates = []
        for proc in self.psutil.process_iter():
            try:
                if proc.pid in self.pids_before:
                    continue
                info = q_process_info(proc)
                if "llama-server" not in str(info.get("text") or "").lower():
                    continue
                candidates.append((info.get("rss_bytes") or 0, proc))
            except Exception:
                continue

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        proc = candidates[0][1]
        self.discovered_pid = proc.pid
        return proc

    def _capture(self) -> None:
        try:
            proc = self._discover()
            sample = q_memory_snapshot(self.psutil, proc)
            sample["process_pid"] = proc.pid if proc is not None else None
            self.samples.append(sample)
        except Exception as exc:
            self.errors.append(str(exc))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._capture()

    def start(self) -> None:
        self._capture()
        self._thread = threading.Thread(target=self._loop, name="localia-quant-load-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 3.0))
        self._capture()

    def summary(self) -> dict[str, Any]:
        out = q_summarize_memory(self.samples)
        out["discovered_pid"] = self.discovered_pid
        out["monitor_errors"] = self.errors
        return out


class QLoadProgress:
    def __init__(self, display: str) -> None:
        self.display = display
        self.events: list[dict[str, Any]] = []
        self.last_bucket = -1

    def __call__(self, progress: float) -> None:
        raw = float(progress)
        percent = raw * 100.0 if raw <= 1.0 else raw
        self.events.append({
            "timestamp": now_iso(),
            "monotonic": time.perf_counter(),
            "raw_progress": raw,
            "percent": percent,
        })
        bucket = int(percent // 10)
        if bucket != self.last_bucket or percent >= 99.0:
            self.last_bucket = bucket
            print(f"[LOAD {self.display}] progresso ≈ {percent:.1f}%")


def q_sdk_load_config() -> dict[str, Any]:
    return {
        "contextLength": CONTEXT_LENGTH,
        "gpu": {"ratio": 0.0},
        "evalBatchSize": 2048,
        "flashAttention": False,
        "offloadKVCacheToGpu": False,
        "keepModelInMemory": True,
        "tryMmap": True,
    }


def q_load_exact_variant(
    session_dir: Path,
    label: str,
    quant_key: str,
    variant: dict[str, Any],
    psutil: Any,
    client: Any,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Carrega a variante exata diretamente pelo Python SDK."""

    display = Q_QUANTS[quant_key]["display"]
    model_key = variant.get("modelKey")
    if not isinstance(model_key, str) or not model_key.strip():
        raise RuntimeError(f"{display}: variante sem modelKey utilizável.")

    expected_alias = {
        "q4_k_m": "qwen/qwen3.5-9b@q4_k_m",
        "q6_k": "qwen/qwen3.5-9b@q6_k",
    }[quant_key]
    if model_key != expected_alias:
        raise RuntimeError(
            f"{display}: modelKey inesperado. esperado={expected_alias!r}, atual={model_key!r}"
        )

    unload_all()
    time.sleep(1.0)
    pids_before = snapshot_pids(psutil)
    log_path = session_dir / "load_logs" / f"{label}-sdk.jsonl"
    log_proc = log_handle = None
    progress = QLoadProgress(display)
    load_monitor = QLoadMonitor(psutil, pids_before, interval=0.5)
    instance_id = f"localia-quant-v2-{re.sub(r'[^A-Za-z0-9_-]+', '-', label)[:64]}"
    handle = None
    start = time.perf_counter()

    try:
        log_proc, log_handle = start_server_log_capture(log_path)
        load_monitor.start()
        handle = client.llm.load_new_instance(
            model_key,
            instance_identifier=instance_id,
            ttl=None,
            config=q_sdk_load_config(),
            on_load_progress=progress,
        )
        elapsed = time.perf_counter() - start
        time.sleep(1.0)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        append_jsonl(
            session_dir / "quant_load_failures.jsonl",
            {
                "timestamp": now_iso(),
                "label": label,
                "quant_key": quant_key,
                "quant_display": display,
                "model_key_requested": model_key,
                "elapsed_seconds": elapsed,
                "progress_events": progress.events,
                "memory_during_load": load_monitor.summary(),
                "memory_samples_during_load": load_monitor.samples,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        if handle is not None:
            try:
                handle.unload()
            except Exception:
                pass
        unload_all()
        raise
    finally:
        load_monitor.stop()
        stop_log_capture(log_proc, log_handle)

    try:
        load_log = parse_load_log(log_path)
        if load_log.get("threadpool_n_threads") != Q_THREADS:
            raise RuntimeError(
                f"threadpool={load_log.get('threadpool_n_threads')}, esperado={Q_THREADS}"
            )
        if load_log.get("n_ctx") is not None and load_log.get("n_ctx") != CONTEXT_LENGTH:
            raise RuntimeError(f"n_ctx={load_log.get('n_ctx')}, esperado={CONTEXT_LENGTH}")
        if load_log.get("n_batch") is not None and load_log.get("n_batch") != 2048:
            raise RuntimeError(f"n_batch={load_log.get('n_batch')}, esperado=2048")
        if load_log.get("n_ubatch") is not None and load_log.get("n_ubatch") != 512:
            raise RuntimeError(f"n_ubatch={load_log.get('n_ubatch')}, esperado=512")
        if load_log.get("offloaded_layers") is not None and int(load_log["offloaded_layers"]) != 0:
            raise RuntimeError(f"offloaded_layers={load_log.get('offloaded_layers')}, esperado=0")

        process, process_data = q_find_llama_process(psutil, pids_before)
        cli_audit = q_audit_cli(process_data, quant_key)
        final_mem = q_memory_snapshot(psutil, process)
        load_memory = load_monitor.summary()

        try:
            actual_load_config = serialize_object(handle.get_load_config())
        except Exception as exc:
            actual_load_config = {"read_error": str(exc)}

        record = {
            "timestamp": now_iso(),
            "label": label,
            "quant_key": quant_key,
            "quant_display": display,
            "variant": variant,
            "load_method": "python_sdk_direct_model_key",
            "model_key_requested": model_key,
            "instance_identifier": instance_id,
            "load_wall_time_seconds": elapsed,
            "progress_events": progress.events,
            "load_log_parsed": load_log,
            "process": process_data,
            "process_cli_audit": cli_audit,
            "load_config_actual": actual_load_config,
            "memory_during_load": load_memory,
            "memory_samples_during_load": load_monitor.samples,
            "memory_after_load": final_mem,
            "log_path": str(log_path),
        }
        append_jsonl(session_dir / "quant_loads.jsonl", record)
        return handle, process, process_data, record

    except Exception:
        if handle is not None:
            try:
                handle.unload()
            except Exception:
                pass
        unload_all()
        raise


def q_prepare_block(
    session_dir: Path,
    block_id: str,
    quant_key: str,
    variant: dict[str, Any],
    config_path: Path,
    original_config_bytes: bytes,
    psutil: Any,
    client: Any,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    restore_config_bytes(config_path, original_config_bytes)
    time.sleep(CONFIG_WATCH_SETTLE_SECONDS)
    edit_info = write_block_overrides_to_config(config_path, Q_THREADS)
    time.sleep(CONFIG_WATCH_SETTLE_SECONDS)
    unload_all()
    time.sleep(COOLDOWN_AFTER_UNLOAD_SECONDS)
    selected, switch_seconds = select_runtime(Q_RUNTIME)
    if Q_RUNTIME not in selected:
        raise RuntimeError(f"Runtime incorreto: {selected}")
    append_jsonl(
        session_dir / "quant_runtime_switches.jsonl",
        {
            "timestamp": now_iso(), "block_id": block_id, "quant_key": quant_key,
            "runtime": selected, "switch_seconds": switch_seconds,
        },
    )
    time.sleep(SETTLE_AFTER_RUNTIME_SWITCH_SECONDS)
    thermal_gate(session_dir, block_id)
    handle, process, process_data, load_record = q_load_exact_variant(
        session_dir, block_id, quant_key, variant, psutil, client
    )
    load_record["config_edit"] = edit_info
    time.sleep(SETTLE_AFTER_LOAD_SECONDS)
    return handle, process, process_data, load_record


def q_strict_json(content: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(content.strip()), None
    except Exception as exc:
        return None, str(exc)


def q_leaf_pairs(obj: Any, prefix: str = "$") -> list[tuple[str, Any]]:
    if isinstance(obj, dict):
        out = []
        for key in sorted(obj):
            out.extend(q_leaf_pairs(obj[key], f"{prefix}.{key}"))
        return out
    if isinstance(obj, list):
        out = []
        for idx, value in enumerate(obj):
            out.extend(q_leaf_pairs(value, f"{prefix}[{idx}]"))
        return out
    return [(prefix, obj)]


def q_get_leaf(obj: Any, path: str) -> tuple[bool, Any]:
    current = obj
    for key_token, idx_token in re.findall(r"\.([^.[]+)|\[(\d+)\]", path[1:]):
        if key_token:
            if not isinstance(current, dict) or key_token not in current:
                return False, None
            current = current[key_token]
        else:
            if not isinstance(current, list):
                return False, None
            idx = int(idx_token)
            if idx >= len(current):
                return False, None
            current = current[idx]
    return True, current


def q_score_quality(content: str, expected: Any) -> dict[str, Any]:
    parsed, error = q_strict_json(content)
    leaves = q_leaf_pairs(expected)
    if parsed is None:
        return {
            "json_valid": False, "parse_error": error, "exact_match": False,
            "matched_expected_leaves": 0, "total_expected_leaves": len(leaves),
            "field_score": 0.0, "parsed": None,
        }
    matched = 0
    details = []
    for path, expected_value in leaves:
        exists, actual = q_get_leaf(parsed, path)
        correct = exists and actual == expected_value
        matched += int(correct)
        details.append({
            "path": path, "expected": expected_value, "exists": exists,
            "actual": actual, "correct": correct,
        })
    total = len(leaves)
    return {
        "json_valid": True, "parse_error": None, "exact_match": parsed == expected,
        "matched_expected_leaves": matched, "total_expected_leaves": total,
        "field_score": matched / total if total else 1.0,
        "field_details": details, "parsed": parsed,
    }


def q_performance_schedule(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    blocks = []
    for round_no in range(1, Q_ROUNDS + 1):
        quants = list(Q_QUANTS)
        rng.shuffle(quants)
        for block_idx, quant_key in enumerate(quants, start=1):
            prompts = list(Q_PERF_PROMPTS)
            rng.shuffle(prompts)
            block_id = f"PERF-R{round_no}-{quant_key.upper()}"
            blocks.append({
                "block_id": block_id, "round": round_no, "block_index": block_idx,
                "quant_key": quant_key, "prompt_order": prompts,
                "run_ids": [f"{block_id}-P{x}" for x in prompts],
            })
    return blocks


def q_quality_schedule(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 5000)
    quants = list(Q_QUANTS)
    rng.shuffle(quants)
    blocks = []
    for idx, quant_key in enumerate(quants, start=1):
        tasks = list(Q_QUALITY_TASKS)
        rng.shuffle(tasks)
        block_id = f"QUALITY-{quant_key.upper()}"
        blocks.append({
            "block_id": block_id, "block_index": idx, "quant_key": quant_key,
            "task_order": tasks, "run_ids": [f"{block_id}-{x}" for x in tasks],
        })
    return blocks


def q_successful_ids(path: Path) -> set[str]:
    return {
        str(x["run_id"])
        for x in read_jsonl(path)
        if x.get("status") == "success" and x.get("run_id")
    }


def q_internal_selftests() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    variants = [
        {
            "modelKey": "qwen/qwen3.5-9b@q4_k_m",
            "path": "lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
            "quantization": {"name": "Q4_K_M"}, "sizeBytes": 6548927711,
        },
        {
            "modelKey": "qwen/qwen3.5-9b@q6_k",
            "path": "lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q6_K.gguf",
            "quantization": {"name": "Q6_K"}, "sizeBytes": 8280000000,
        },
    ]
    assert q_candidate_matches(variants[0], "q4_k_m") and not q_candidate_matches(variants[0], "q6_k")
    assert q_candidate_matches(variants[1], "q6_k")
    checks["variant_matching"] = True
    grouped = [{"model": {"modelKey": MODEL_KEY}, "variants": variants}]
    assert len(q_parse_variant_listing(grouped)) == 2
    assert len(q_parse_variant_listing(variants)) == 2
    checks["variant_listing"] = True
    perf = q_performance_schedule(Q_SEED)
    perf_ids = [rid for b in perf for rid in b["run_ids"]]
    assert len(perf) == 6 and len(perf_ids) == 18 and len(set(perf_ids)) == 18
    checks["performance_schedule"] = True
    quality = q_quality_schedule(Q_SEED)
    quality_ids = [rid for b in quality for rid in b["run_ids"]]
    assert len(quality) == 2 and len(quality_ids) == 20 and len(set(quality_ids)) == 20
    checks["quality_schedule"] = True
    for task in Q_QUALITY_TASKS.values():
        score = q_score_quality(json.dumps(task["expected"]), task["expected"])
        assert score["json_valid"] and score["exact_match"] and score["field_score"] == 1.0
    assert not q_score_quality("```json\n{}\n```", {"a": 1})["json_valid"]
    checks["quality_scoring"] = True
    cli = q_audit_cli(
        {"cmdline": ["llama-server.exe", "--model", r"C:\models\Qwen3.5-9B-Q6_K.gguf", "--threads", "8", "--n-gpu-layers", "0"]},
        "q6_k",
    )
    assert cli["threads"] == 8 and cli["quant_verified"]
    checks["cli_audit"] = True
    sample_config = {
        "operation": {"fields": []},
        "load": {"fields": [{"key": "llm.load.llama.cpuThreadPoolSize", "value": 2}]},
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "qwen.json"
        original = json.dumps(sample_config, indent=2).encode()
        p.write_bytes(original)
        write_block_overrides_to_config(p, Q_THREADS)
        edited = json.loads(p.read_text())
        assert {x["value"] for x in collect_threadpool_values(edited)} == {8}
        reason = collect_operation_field_values(edited, REASONING_ENABLE_KEY)
        assert len(reason) == 1 and reason[0]["value"] is False
        restore_config_bytes(p, original)
        assert p.read_bytes() == original
    checks["config_edit_restore"] = True
    mem = q_summarize_memory([
        {
            "system_total_bytes": 16 * 1024**3, "system_available_bytes": 6 * 1024**3,
            "system_used_percent": 62.5, "swap_total_bytes": 8 * 1024**3,
            "swap_used_bytes": 1 * 1024**3, "swap_used_percent": 12.5,
            "process_rss_bytes": 9 * 1024**3,
        },
        {
            "system_total_bytes": 16 * 1024**3, "system_available_bytes": 5 * 1024**3,
            "system_used_percent": 68.75, "swap_total_bytes": 8 * 1024**3,
            "swap_used_bytes": 2 * 1024**3, "swap_used_percent": 25.0,
            "process_rss_bytes": 10 * 1024**3,
        },
    ])
    assert abs(mem["system_available_min_gib"] - 5.0) < 1e-9
    assert abs(mem["process_rss_peak_gib"] - 10.0) < 1e-9
    checks["memory_summary"] = True
    cfg = q_sdk_load_config()
    assert cfg["contextLength"] == 4096
    assert cfg["gpu"]["ratio"] == 0.0
    assert cfg["evalBatchSize"] == 2048
    checks["sdk_load_config"] = True

    source_text = Path(__file__).read_text(encoding="utf-8")
    m = re.search(
        r"def q_load_exact_variant\(.*?\n\ndef q_prepare_block",
        source_text,
        flags=re.DOTALL,
    )
    assert m is not None
    loader_text = m.group(0)
    assert "client.llm.load_new_instance" in loader_text
    assert 'run_lms(\n                "load"' not in loader_text
    assert 'run_lms("load"' not in loader_text
    checks["sdk_direct_load_regression_guard"] = True

    checks["all_passed"] = True
    return checks


def q_create_session(
    seed: int,
    config_path: Path,
    original_config_bytes: bytes,
    variants: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    session_dir = RESULTS_ROOT / f"quant_q4_q6_v2_{timestamp_slug()}"
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "original_model_default_config.json").write_bytes(original_config_bytes)
    session = {
        "session_version": 2,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "completed": False,
        "completed_at": None,
        "seed": seed,
        "model_key": MODEL_KEY,
        "runtime": Q_RUNTIME,
        "threads": Q_THREADS,
        "context_length": CONTEXT_LENGTH,
        "temperature": TEMPERATURE,
        "thinking": "lmstudio_native_enableThinking_false",
        "variant_load_method": "python_sdk_direct_model_key",
        "max_tokens_performance": None,
        "variants": variants,
        "performance_schedule": q_performance_schedule(seed),
        "quality_schedule": q_quality_schedule(seed),
        "planned_performance_runs": 18,
        "planned_quality_runs": 20,
        "model_default_config_path": str(config_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "lms_version": run_command_metadata("lms", "--version"),
            "git_commit": run_command_metadata("git", "rev-parse", "HEAD"),
        },
    }
    atomic_write_json(session_dir / "session.json", session)
    return session_dir, session


def q_update_session(session_dir: Path, session: dict[str, Any]) -> None:
    session["updated_at"] = now_iso()
    atomic_write_json(session_dir / "session.json", session)


def q_find_incomplete_session() -> Path | None:
    if not RESULTS_ROOT.exists():
        return None
    candidates = []
    for path in RESULTS_ROOT.glob("quant_q4_q6_v2_*"):
        sp = path / "session.json"
        if not sp.exists():
            continue
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not data.get("completed", False):
            candidates.append(path)
    return max(candidates, key=lambda p: p.name) if candidates else None


def q_run_preflight(
    session_dir: Path,
    quant_key: str,
    variant: dict[str, Any],
    config_path: Path,
    original_config_bytes: bytes,
    lms: Any,
    psutil: Any,
    client: Any,
) -> dict[str, Any]:
    display = Q_QUANTS[quant_key]["display"]
    preflight_path = session_dir / f"preflight_{quant_key}.json"
    if preflight_path.exists():
        try:
            existing = json.loads(preflight_path.read_text(encoding="utf-8"))
            if existing.get("passed") is True:
                print(f"[PREFLIGHT] {display} já aprovado nesta sessão.")
                return existing
        except Exception:
            pass

    print("\n" + "=" * 92)
    print(f" PREFLIGHT REAL — {display} — SDK DIRETO")
    print("=" * 92)

    restore_config_bytes(config_path, original_config_bytes)
    time.sleep(CONFIG_WATCH_SETTLE_SECONDS)
    edit_info = write_block_overrides_to_config(config_path, Q_THREADS)
    time.sleep(CONFIG_WATCH_SETTLE_SECONDS)
    unload_all()
    selected, _ = select_runtime(Q_RUNTIME)
    if Q_RUNTIME not in selected:
        raise RuntimeError(f"Runtime incorreto no preflight: {selected}")

    handle = None
    try:
        handle, process, process_data, load_record = q_load_exact_variant(
            session_dir, f"PREFLIGHT-{quant_key}", quant_key, variant, psutil, client
        )
        start = time.perf_counter()
        result = handle.respond(
            "Responda somente com a palavra OK.",
            config={"temperature": 0.0, "cpuThreads": Q_THREADS, "maxTokens": 96},
        )
        wall = time.perf_counter() - start
        stats = getattr(result, "stats", None)
        prediction_config = serialize_object(getattr(result, "prediction_config", None))
        content = getattr_any(result, ["content", "text"])
        content = str(content if content is not None else result)
        reasoning = getattr_any(result, ["reasoning_content", "reasoningContent"])
        problems = detect_thinking(content, reasoning)
        stop = getattr_any(stats, ["stop_reason", "stopReason"])
        predicted = getattr_any(stats, ["predicted_tokens_count", "predictedTokensCount"])
        echoed = None
        if isinstance(prediction_config, dict):
            _, echoed = find_key(prediction_config, "cpuThreads", "cpu_threads")
        passed = (
            not problems
            and stop not in INVALID_STOP_REASONS
            and echoed is not None
            and int(echoed) == Q_THREADS
        )
        record = {
            "timestamp": now_iso(),
            "passed": passed,
            "quant_key": quant_key,
            "quant_display": display,
            "variant": variant,
            "config_edit": edit_info,
            "load_record": load_record,
            "process": process_data,
            "content": content,
            "reasoning_content": reasoning,
            "thinking_problems": problems,
            "stop_reason": stop,
            "predicted_tokens_count": predicted,
            "echoed_cpu_threads": echoed,
            "wall_time_seconds": wall,
        }
        atomic_write_json(preflight_path, record)
        if not passed:
            raise RuntimeError(
                f"Preflight {display} falhou: thinking={problems}, stop={stop}, cpuThreads={echoed}"
            )
        load_mem = load_record.get("memory_during_load", {})
        after = load_record.get("memory_after_load", {})
        print(
            f"[PREFLIGHT] ✓ {display} | SDK direto | GGUF exato | threads=8 | Think OFF | "
            f"tokens={predicted} | stop={stop} | {wall:.2f}s"
        )
        print(
            "[MEM load] "
            f"RSS pico={fmt_num(load_mem.get('process_rss_peak_gib'), 2)} GiB | "
            f"RAM livre min={fmt_num(load_mem.get('system_available_min_gib'), 2)} GiB | "
            f"pagefile pico={fmt_num(load_mem.get('swap_used_peak_gib'), 2)} GiB | "
            f"RSS final={fmt_num(q_bytes_gib(after.get('process_rss_bytes')), 2)} GiB"
        )
        return record
    finally:
        if handle is not None:
            try:
                handle.unload()
            except Exception:
                pass
        unload_all()
        restore_config_bytes(config_path, original_config_bytes)
        time.sleep(CONFIG_WATCH_SETTLE_SECONDS)


def q_write_performance_csv(session_dir: Path) -> None:
    rows = [
        x for x in read_jsonl(session_dir / "performance_runs.jsonl")
        if x.get("status") == "success"
    ]
    fields = [
        "run_id", "round", "quant_key", "quant_display", "prompt_id",
        "tokens_per_second", "ttft_seconds", "predicted_tokens", "wall_time_seconds",
        "load_time_seconds", "load_rss_peak_gib", "load_ram_available_min_gib", "load_swap_peak_gib",
        "process_rss_peak_gib", "system_available_min_gib",
        "system_used_percent_peak", "swap_used_peak_gib", "cpu_effective_cores",
        "cpu_machine_percent", "active_threads_effective", "package_power_mean_w",
        "temp_peak_c", "p_core_clock_mean_mhz", "tokens_per_second_per_package_watt",
        "stop_reason", "thinking_off_verified",
    ]
    with (session_dir / "performance_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            inf = row["inference"]
            tel = row["telemetry"]
            mem = tel["memory"]
            hw = tel["hardware"]
            cpu = tel["process_threads"]
            writer.writerow({
                "run_id": row["run_id"], "round": row["round"],
                "quant_key": row["quant_key"], "quant_display": row["quant_display"],
                "prompt_id": row["prompt_id"],
                "tokens_per_second": inf.get("tokens_per_second"),
                "ttft_seconds": inf.get("time_to_first_token_seconds"),
                "predicted_tokens": inf.get("predicted_tokens_count"),
                "wall_time_seconds": row["wall_time_seconds"],
                "load_time_seconds": row["load_record"].get("load_wall_time_seconds"),
                "load_rss_peak_gib": row["load_record"].get("memory_during_load", {}).get("process_rss_peak_gib"),
                "load_ram_available_min_gib": row["load_record"].get("memory_during_load", {}).get("system_available_min_gib"),
                "load_swap_peak_gib": row["load_record"].get("memory_during_load", {}).get("swap_used_peak_gib"),
                "process_rss_peak_gib": mem.get("process_rss_peak_gib"),
                "system_available_min_gib": mem.get("system_available_min_gib"),
                "system_used_percent_peak": mem.get("system_used_percent_peak"),
                "swap_used_peak_gib": mem.get("swap_used_peak_gib"),
                "cpu_effective_cores": cpu.get("effective_cpu_cores"),
                "cpu_machine_percent": cpu.get("process_machine_cpu_percent"),
                "active_threads_effective": cpu.get("effective_active_thread_count"),
                "package_power_mean_w": hw.get("package_power_mean_w"),
                "temp_peak_c": hw.get("temp_peak_c"),
                "p_core_clock_mean_mhz": hw.get("p_core_clock_mean_mhz"),
                "tokens_per_second_per_package_watt": tel.get("tokens_per_second_per_package_watt"),
                "stop_reason": inf.get("stop_reason"),
                "thinking_off_verified": inf.get("thinking_off_verified"),
            })


def q_write_quality_csv(session_dir: Path) -> None:
    rows = [
        x for x in read_jsonl(session_dir / "quality_runs.jsonl")
        if x.get("status") == "success"
    ]
    fields = [
        "run_id", "quant_key", "quant_display", "task_id", "json_valid",
        "exact_match", "field_score", "predicted_tokens", "wall_time_seconds", "content",
    ]
    with (session_dir / "quality_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            score = row["score"]
            inf = row["inference"]
            writer.writerow({
                "run_id": row["run_id"], "quant_key": row["quant_key"],
                "quant_display": row["quant_display"], "task_id": row["task_id"],
                "json_valid": score["json_valid"], "exact_match": score["exact_match"],
                "field_score": score["field_score"],
                "predicted_tokens": inf.get("predicted_tokens_count"),
                "wall_time_seconds": row["wall_time_seconds"], "content": inf.get("content"),
            })


def q_nested_number(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return float(current) if isinstance(current, (int, float)) else None


def q_performance_summary(session_dir: Path) -> dict[str, Any]:
    rows = [
        x for x in read_jsonl(session_dir / "performance_runs.jsonl")
        if x.get("status") == "success"
    ]
    by_quant = []
    for quant_key in Q_QUANTS:
        qrows = [x for x in rows if x.get("quant_key") == quant_key]
        def vals(*path: str) -> list[float]:
            out = []
            for x in qrows:
                value = q_nested_number(x, tuple(path))
                if value is not None:
                    out.append(value)
            return out
        tps = vals("inference", "tokens_per_second")
        ttft = vals("inference", "time_to_first_token_seconds")
        wall = vals("wall_time_seconds")
        tokens = vals("inference", "predicted_tokens_count")
        load = vals("load_record", "load_wall_time_seconds")
        load_rss = vals("load_record", "memory_during_load", "process_rss_peak_gib")
        load_avail = vals("load_record", "memory_during_load", "system_available_min_gib")
        load_swap = vals("load_record", "memory_during_load", "swap_used_peak_gib")
        rss = vals("telemetry", "memory", "process_rss_peak_gib")
        avail = vals("telemetry", "memory", "system_available_min_gib")
        used = vals("telemetry", "memory", "system_used_percent_peak")
        swap = vals("telemetry", "memory", "swap_used_peak_gib")
        cores = vals("telemetry", "process_threads", "effective_cpu_cores")
        cpu_pct = vals("telemetry", "process_threads", "process_machine_cpu_percent")
        active = vals("telemetry", "process_threads", "effective_active_thread_count")
        power = vals("telemetry", "hardware", "package_power_mean_w")
        temp = vals("telemetry", "hardware", "temp_peak_c")
        clock = vals("telemetry", "hardware", "p_core_clock_mean_mhz")
        eff = vals("telemetry", "tokens_per_second_per_package_watt")
        by_quant.append({
            "quant_key": quant_key, "quant_display": Q_QUANTS[quant_key]["display"], "n": len(qrows),
            "tokens_per_second_mean": safe_mean(tps), "tokens_per_second_median": safe_median(tps),
            "tokens_per_second_stdev": safe_stdev(tps), "ttft_mean_seconds": safe_mean(ttft),
            "wall_time_mean_seconds": safe_mean(wall), "predicted_tokens_mean": safe_mean(tokens),
            "load_time_mean_seconds": safe_mean(load),
            "load_rss_peak_mean_gib": safe_mean(load_rss),
            "load_rss_peak_max_gib": max(load_rss) if load_rss else None,
            "load_system_available_min_mean_gib": safe_mean(load_avail),
            "load_system_available_min_gib": min(load_avail) if load_avail else None,
            "load_swap_used_peak_mean_gib": safe_mean(load_swap),
            "process_rss_peak_mean_gib": safe_mean(rss),
            "process_rss_peak_max_gib": max(rss) if rss else None,
            "system_available_min_mean_gib": safe_mean(avail),
            "system_available_min_gib": min(avail) if avail else None,
            "system_used_percent_peak_mean": safe_mean(used), "swap_used_peak_mean_gib": safe_mean(swap),
            "effective_cpu_cores_mean": safe_mean(cores), "process_machine_cpu_percent_mean": safe_mean(cpu_pct),
            "effective_active_thread_count_mean": safe_mean(active), "package_power_mean_w": safe_mean(power),
            "temp_peak_mean_c": safe_mean(temp), "p_core_clock_mean_mhz": safe_mean(clock),
            "tokens_per_second_per_package_watt_mean": safe_mean(eff),
            "unique_outputs_per_prompt": {
                pid: len({x["inference"]["response_sha256"] for x in qrows if x["prompt_id"] == pid})
                for pid in Q_PERF_PROMPTS
            },
        })
    by_key = {x["quant_key"]: x for x in by_quant}
    q4 = by_key.get("q4_k_m", {})
    q6 = by_key.get("q6_k", {})
    def pct(new: Any, old: Any) -> float | None:
        if not isinstance(new, (int, float)) or not isinstance(old, (int, float)) or old == 0:
            return None
        return (new / old - 1) * 100
    comparison = {
        "q6_vs_q4_tps_percent": pct(q6.get("tokens_per_second_mean"), q4.get("tokens_per_second_mean")),
        "q6_vs_q4_ttft_percent": pct(q6.get("ttft_mean_seconds"), q4.get("ttft_mean_seconds")),
        "q6_vs_q4_wall_percent": pct(q6.get("wall_time_mean_seconds"), q4.get("wall_time_mean_seconds")),
        "q6_minus_q4_rss_peak_mean_gib": (
            q6.get("process_rss_peak_mean_gib") - q4.get("process_rss_peak_mean_gib")
            if isinstance(q6.get("process_rss_peak_mean_gib"), (int, float))
            and isinstance(q4.get("process_rss_peak_mean_gib"), (int, float)) else None
        ),
        "q6_minus_q4_power_mean_w": (
            q6.get("package_power_mean_w") - q4.get("package_power_mean_w")
            if isinstance(q6.get("package_power_mean_w"), (int, float))
            and isinstance(q4.get("package_power_mean_w"), (int, float)) else None
        ),
    }
    return {
        "generated_at": now_iso(), "successful_runs": len(rows), "planned_runs": 18,
        "by_quant": by_quant, "comparison": comparison,
    }


def q_quality_summary(session_dir: Path) -> dict[str, Any]:
    rows = [
        x for x in read_jsonl(session_dir / "quality_runs.jsonl")
        if x.get("status") == "success"
    ]
    by_quant = []
    for quant_key in Q_QUANTS:
        qrows = [x for x in rows if x.get("quant_key") == quant_key]
        scores = [float(x["score"]["field_score"]) for x in qrows]
        by_quant.append({
            "quant_key": quant_key, "quant_display": Q_QUANTS[quant_key]["display"], "n": len(qrows),
            "json_valid_count": sum(bool(x["score"]["json_valid"]) for x in qrows),
            "exact_match_count": sum(bool(x["score"]["exact_match"]) for x in qrows),
            "field_score_mean": safe_mean(scores),
            "task_results": {
                x["task_id"]: {
                    "json_valid": x["score"]["json_valid"],
                    "exact_match": x["score"]["exact_match"],
                    "field_score": x["score"]["field_score"],
                    "content": x["inference"].get("content"),
                }
                for x in qrows
            },
        })
    return {
        "generated_at": now_iso(), "successful_runs": len(rows), "planned_runs": 20,
        "by_quant": by_quant,
        "note": "Suite pequena e objetiva, específica ao Jev; não é benchmark geral de inteligência.",
    }


def q_write_summaries(session_dir: Path, session: dict[str, Any]) -> None:
    perf = q_performance_summary(session_dir)
    quality = q_quality_summary(session_dir)
    atomic_write_json(session_dir / "performance_summary.json", perf)
    atomic_write_json(session_dir / "quality_summary.json", quality)
    atomic_write_json(session_dir / "summary.json", {"generated_at": now_iso(), "performance": perf, "quality": quality})
    session["performance_successful_runs"] = perf["successful_runs"]
    session["quality_successful_runs"] = quality["successful_runs"]
    q_update_session(session_dir, session)


def q_run_performance(
    session_dir: Path,
    session: dict[str, Any],
    variants: dict[str, dict[str, Any]],
    config_path: Path,
    original_config_bytes: bytes,
    lms: Any,
    psutil: Any,
    client: Any,
) -> None:
    path = session_dir / "performance_runs.jsonl"
    done = q_successful_ids(path)
    total = int(session["planned_performance_runs"])
    started = time.perf_counter()
    new_done = 0

    for block_index, block in enumerate(session["performance_schedule"], start=1):
        pending = [rid for rid in block["run_ids"] if rid not in done]
        if not pending:
            continue

        quant_key = block["quant_key"]
        display = Q_QUANTS[quant_key]["display"]
        print("\n" + "=" * 96)
        print(
            f" PERFORMANCE {block_index}/{len(session['performance_schedule'])} | "
            f"Rodada {block['round']}/{Q_ROUNDS} | {display}"
        )
        print("=" * 96)

        handle = None
        try:
            handle, process, process_data, load_record = q_prepare_block(
                session_dir, block["block_id"], quant_key, variants[quant_key],
                config_path, original_config_bytes, psutil, client,
            )

            warm_inf, warm_wall, warm_tel = q_instrumented_inference(
                handle, psutil, process, WARMUP_PROMPT
            )
            append_jsonl(
                session_dir / "quant_warmups.jsonl",
                {
                    "timestamp": now_iso(), "phase": "performance",
                    "block_id": block["block_id"], "quant_key": quant_key,
                    "inference": warm_inf, "wall_time_seconds": warm_wall,
                    "telemetry": warm_tel,
                },
            )
            print(
                f"[WARMUP] ✓ {display} | "
                f"{fmt_num(warm_inf.get('tokens_per_second'))} tok/s | "
                f"{warm_inf.get('predicted_tokens_count')} tokens"
            )
            time.sleep(SETTLE_AFTER_WARMUP_SECONDS)

            for prompt_position, prompt_id in enumerate(block["prompt_order"], start=1):
                run_id = f"{block['block_id']}-P{prompt_id}"
                if run_id in done:
                    continue

                completed = len(done)
                if new_done > 0:
                    elapsed = time.perf_counter() - started
                    remaining = total - completed
                    eta = elapsed / new_done * remaining
                    print(f"[ETA] {seconds_to_hms(eta)}")

                print(f"[RUN {completed + 1}/{total}] {display} | Prompt {prompt_id}")
                inference, wall, telemetry = q_instrumented_inference(
                    handle, psutil, process, Q_PERF_PROMPTS[prompt_id]
                )

                record = {
                    "status": "success",
                    "timestamp": now_iso(),
                    "run_id": run_id,
                    "phase": "performance",
                    "block_id": block["block_id"],
                    "round": block["round"],
                    "prompt_position": prompt_position,
                    "prompt_id": prompt_id,
                    "prompt": Q_PERF_PROMPTS[prompt_id],
                    "quant_key": quant_key,
                    "quant_display": display,
                    "variant": variants[quant_key],
                    "load_record": load_record,
                    "process": process_data,
                    "inference": inference,
                    "wall_time_seconds": wall,
                    "telemetry": telemetry,
                }
                append_jsonl(path, record)
                done.add(run_id)
                new_done += 1
                q_write_performance_csv(session_dir)
                q_write_summaries(session_dir, session)

                hw = telemetry["hardware"]
                mem = telemetry["memory"]
                print(
                    f"      ✓ {fmt_num(inference.get('tokens_per_second'))} tok/s | "
                    f"TTFT={fmt_num(inference.get('time_to_first_token_seconds'))}s | "
                    f"RSS pico={fmt_num(mem.get('process_rss_peak_gib'), 2)} GiB | "
                    f"RAM livre min={fmt_num(mem.get('system_available_min_gib'), 2)} GiB | "
                    f"swap pico={fmt_num(mem.get('swap_used_peak_gib'), 2)} GiB | "
                    f"{fmt_num(hw.get('package_power_mean_w'), 1)} W | "
                    f"T pico={fmt_num(hw.get('temp_peak_c'), 1)} °C"
                )
                time.sleep(PAUSE_BETWEEN_PROMPTS_SECONDS)

        finally:
            if handle is not None:
                try:
                    handle.unload()
                except Exception:
                    pass
            unload_all()
            restore_config_bytes(config_path, original_config_bytes)
            time.sleep(CONFIG_WATCH_SETTLE_SECONDS)

    session["performance_successful_runs"] = len(done)
    q_update_session(session_dir, session)


def q_run_quality(
    session_dir: Path,
    session: dict[str, Any],
    variants: dict[str, dict[str, Any]],
    config_path: Path,
    original_config_bytes: bytes,
    lms: Any,
    psutil: Any,
    client: Any,
) -> None:
    path = session_dir / "quality_runs.jsonl"
    done = q_successful_ids(path)
    total = int(session["planned_quality_runs"])

    for block_index, block in enumerate(session["quality_schedule"], start=1):
        pending = [rid for rid in block["run_ids"] if rid not in done]
        if not pending:
            continue

        quant_key = block["quant_key"]
        display = Q_QUANTS[quant_key]["display"]
        print("\n" + "=" * 96)
        print(f" QUALIDADE {block_index}/2 | {display}")
        print("=" * 96)

        handle = None
        try:
            handle, process, process_data, load_record = q_prepare_block(
                session_dir, block["block_id"], quant_key, variants[quant_key],
                config_path, original_config_bytes, psutil, client,
            )

            warm_inf, warm_wall, warm_tel = q_instrumented_inference(
                handle, psutil, process,
                "Responda em uma frase: o que é um roteador de modelos?",
            )
            append_jsonl(
                session_dir / "quant_warmups.jsonl",
                {
                    "timestamp": now_iso(), "phase": "quality",
                    "block_id": block["block_id"], "quant_key": quant_key,
                    "inference": warm_inf, "wall_time_seconds": warm_wall,
                    "telemetry": warm_tel,
                },
            )

            for task_id in block["task_order"]:
                run_id = f"{block['block_id']}-{task_id}"
                if run_id in done:
                    continue
                task = Q_QUALITY_TASKS[task_id]
                print(f"[QUALITY {len(done) + 1}/{total}] {display} | {task_id}")
                inference, wall, telemetry = q_instrumented_inference(
                    handle, psutil, process, task["prompt"]
                )
                score = q_score_quality(str(inference.get("content") or ""), task["expected"])
                record = {
                    "status": "success", "timestamp": now_iso(), "run_id": run_id,
                    "phase": "quality", "block_id": block["block_id"], "task_id": task_id,
                    "quant_key": quant_key, "quant_display": display,
                    "variant": variants[quant_key], "prompt": task["prompt"],
                    "expected": task["expected"], "load_record": load_record,
                    "process": process_data, "inference": inference,
                    "wall_time_seconds": wall, "telemetry": telemetry, "score": score,
                }
                append_jsonl(path, record)
                done.add(run_id)
                print(
                    "      " + (
                        "✓ EXACT" if score["exact_match"]
                        else f"△ field={score['field_score'] * 100:.1f}% json={score['json_valid']}"
                    )
                )
                q_write_quality_csv(session_dir)
                q_write_summaries(session_dir, session)
                time.sleep(PAUSE_BETWEEN_QUALITY_TASKS_SECONDS)

        finally:
            if handle is not None:
                try:
                    handle.unload()
                except Exception:
                    pass
            unload_all()
            restore_config_bytes(config_path, original_config_bytes)
            time.sleep(CONFIG_WATCH_SETTLE_SECONDS)

    session["quality_successful_runs"] = len(done)
    q_update_session(session_dir, session)


def q_print_summary(session_dir: Path) -> None:
    summary = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))
    perf = summary["performance"]
    quality = summary["quality"]
    print("\n" + "=" * 116)
    print(" Q4_K_M × Q6_K — RESUMO")
    print("=" * 116)
    print(
        f"{'Quant':<8} {'tok/s':>8} {'TTFT':>8} {'Load':>8} {'RSS peak':>10} "
        f"{'RAM min':>9} {'Swap':>9} {'W':>7} {'Tpeak':>7}"
    )
    print("-" * 116)
    for row in perf["by_quant"]:
        print(
            f"{row['quant_display']:<8} {fmt_num(row['tokens_per_second_mean']):>8} "
            f"{fmt_num(row['ttft_mean_seconds']):>8} {fmt_num(row['load_time_mean_seconds']):>8} "
            f"{fmt_num(row['process_rss_peak_mean_gib'], 2):>10} "
            f"{fmt_num(row['system_available_min_gib'], 2):>9} "
            f"{fmt_num(row['swap_used_peak_mean_gib'], 2):>9} "
            f"{fmt_num(row['package_power_mean_w'], 1):>7} "
            f"{fmt_num(row['temp_peak_mean_c'], 1):>7}"
        )
    print("\nQualidade objetiva do Jev:")
    for row in quality["by_quant"]:
        score_pct = (row.get("field_score_mean") or 0) * 100
        print(
            f"  {row['quant_display']}: JSON {row['json_valid_count']}/{row['n']} | "
            f"exact {row['exact_match_count']}/{row['n']} | field score {score_pct:.1f}%"
        )
    comp = perf["comparison"]
    print("\nQ6 vs Q4:")
    print(f"  throughput: {fmt_num(comp.get('q6_vs_q4_tps_percent'), 2)}%")
    print(f"  TTFT:       {fmt_num(comp.get('q6_vs_q4_ttft_percent'), 2)}%")
    print(f"  RSS:        {fmt_num(comp.get('q6_minus_q4_rss_peak_mean_gib'), 2)} GiB")
    print("=" * 116)


def q_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LocalIA Qwen3.5-9B Q4_K_M × Q6_K benchmark v2."
    )
    parser.add_argument("--new", action="store_true", help="Força nova sessão.")
    parser.add_argument("--yes", action="store_true", help="Não pede confirmação.")
    parser.add_argument("--selftest-only", action="store_true", help="Executa apenas self-tests puros.")
    parser.add_argument("--seed", type=int, default=Q_SEED)
    return parser.parse_args()


def quant_main() -> int:
    args = q_parse_args()
    print("[SELFTEST] Validando lógica interna...")
    checks = q_internal_selftests()
    print(
        "[SELFTEST] ✓ " + ", ".join(
            key for key, value in checks.items() if value is True and key != "all_passed"
        )
    )
    if args.selftest_only:
        return 0

    lms, psutil = import_dependencies()
    setter = getattr(lms, "set_sync_api_timeout", None)
    if callable(setter):
        setter(180.0)
    print("\n[PRECHECK] Validando ambiente real...")
    ensure_server()
    if Q_RUNTIME not in runtime_ls_text():
        raise RuntimeError(f"Runtime CPU não instalado: {Q_RUNTIME}")
    hw = verify_lhm()
    print(f"  ✓ runtime: {Q_RUNTIME}")
    print(
        "  ✓ LibreHardwareMonitor: "
        f"{fmt_num(hw.get('cpu_package_temp_c'), 1)} °C | "
        f"{fmt_num(hw.get('cpu_package_power_w'), 1)} W"
    )
    print(f"  ✓ CPUs lógicas: {psutil.cpu_count(logical=True)}")

    variants = q_resolve_variants()
    print("  ✓ variantes exatas:")
    for quant_key, variant in variants.items():
        size_gib = q_bytes_gib(variant.get("sizeBytes"))
        print(
            f"    {Q_QUANTS[quant_key]['display']}: {variant.get('modelKey')} | "
            f"{variant.get('path')} | {fmt_num(size_gib, 2)} GiB"
        )

    config_path = find_model_config_file()
    current_bytes = config_path.read_bytes()
    current_json = json.loads(current_bytes.decode("utf-8"))
    print("  ✓ Per-model Default:")
    print(f"    {config_path}")
    for item in collect_threadpool_values(current_json):
        print(f"    {item['path']} = {item['value']!r}")

    session_dir = None
    session = None
    original_bytes = current_bytes
    resumed = False
    if not args.new:
        incomplete = q_find_incomplete_session()
        if incomplete is not None:
            candidate = json.loads((incomplete / "session.json").read_text(encoding="utf-8"))
            backup = incomplete / "original_model_default_config.json"
            if not backup.exists():
                raise RuntimeError("Sessão incompleta sem backup do config.")
            baseline = backup.read_bytes()
            if not json_equivalent_bytes(baseline, current_bytes):
                raise RuntimeError(
                    "Existe sessão incompleta, mas o Per-model Default mudou. Use --new."
                )
            session_dir, session, original_bytes, resumed = incomplete, candidate, baseline, True
    if session_dir is None or session is None:
        session_dir, session = q_create_session(args.seed, config_path, current_bytes, variants)

    print("\n" + "=" * 96)
    print(" LocalIA — Qwen3.5-9B Q4_K_M × Q6_K Benchmark v2")
    print("=" * 96)
    print(" Load variante  : Python SDK direto")
    print(" Backend        : CPU AVX2 2.46.0")
    print(" Threads        : 8 / 8")
    print(" GPU offload    : OFF")
    print(" Contexto       : 4096")
    print(" Thinking       : OFF")
    print(" Performance    : 18 gerações")
    print(" Qualidade Jev  : 20 tarefas")
    print(f" Sessão         : {session_dir.name}")
    print("=" * 96)
    if resumed:
        print("Sessão incompleta encontrada; será retomada.")

    if not args.yes:
        answer = input("\nPressione ENTER para começar ou digite 'cancelar': ").strip().lower()
        if answer in {"cancelar", "cancel", "n", "nao", "não"}:
            print("Benchmark cancelado.")
            return 0

    try:
        with lms.Client() as client:
            for quant_key in Q_QUANTS:
                q_run_preflight(
                    session_dir, quant_key, variants[quant_key], config_path,
                    original_bytes, lms, psutil, client,
                )

            q_run_performance(
                session_dir, session, variants, config_path, original_bytes, lms, psutil, client,
            )
            q_run_quality(
                session_dir, session, variants, config_path, original_bytes, lms, psutil, client,
            )
        q_write_performance_csv(session_dir)
        q_write_quality_csv(session_dir)
        q_write_summaries(session_dir, session)
        perf_done = len(q_successful_ids(session_dir / "performance_runs.jsonl"))
        quality_done = len(q_successful_ids(session_dir / "quality_runs.jsonl"))
        session["performance_successful_runs"] = perf_done
        session["quality_successful_runs"] = quality_done
        session["completed"] = (
            perf_done == session["planned_performance_runs"]
            and quality_done == session["planned_quality_runs"]
        )
        if session["completed"]:
            session["completed_at"] = now_iso()
        q_update_session(session_dir, session)
        q_print_summary(session_dir)
        print(f"\nResultados:\n  {session_dir}")
        return 0 if session["completed"] else 2

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        try:
            unload_all()
        except Exception:
            pass
        try:
            restore_config_bytes(config_path, original_bytes)
            print("✓ Per-model Default restaurado.")
        except Exception as exc:
            print(f"ERRO ao restaurar config: {exc}")
        session["interrupted_at"] = now_iso()
        q_update_session(session_dir, session)
        return 130

    except Exception as exc:
        print(f"\nERRO FATAL: {exc}")
        try:
            unload_all()
        except Exception:
            pass
        try:
            restore_config_bytes(config_path, original_bytes)
            print("✓ Per-model Default restaurado.")
        except Exception as restore_exc:
            print(f"ERRO CRÍTICO ao restaurar config: {restore_exc}")
            print(f"Backup: {session_dir / 'original_model_default_config.json'}")
        session["fatal_error_at"] = now_iso()
        session["fatal_error"] = str(exc)
        q_update_session(session_dir, session)
        return 1

    finally:
        try:
            restore_config_bytes(config_path, original_bytes)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(quant_main())
