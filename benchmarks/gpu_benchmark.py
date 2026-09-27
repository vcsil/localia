from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ============================================================
# LocalIA — GPU Benchmark
#
# 54 gerações medidas:
#   6 níveis de GPU offload
# × 3 prompts
# × 3 rodadas
#
# Objetivos de desenho:
# - mínimo overhead do Python durante a inferência;
# - sem dependências externas;
# - sem streaming de tokens;
# - resultados persistidos imediatamente;
# - retomada após interrupção;
# - ordem randomizada, mas reproduzível;
# - auditoria da configuração carregada;
# - warm-up por configuração (não contabilizado).
#
# Observação:
# CPU_THREADS_TARGET=8 é registrado como metadata. O caminho REST/CLI
# usado por este script não confirma esse parâmetro diretamente.
# A comparação interna continua válida porque TODAS as configurações
# são carregadas pela mesma rota. O benchmark GPU=0/32 é a baseline
# interna desta bateria.
# ============================================================


# ----------------------------
# Configuração geral
# ----------------------------

API_BASE = "http://127.0.0.1:1234"

MODEL_KEY = "qwen/qwen3.5-9b"
EXPECTED_VARIANT = "qwen/qwen3.5-9b@q4_k_m"
INSTANCE_ID = "localia-bench"

CONTEXT_LENGTH = 4096
TOTAL_MODEL_LAYERS = 32

GPU_CONFIGS = [
    {"layers": 0,  "ratio": 0.000},
    {"layers": 4,  "ratio": 0.125},
    {"layers": 8,  "ratio": 0.250},
    {"layers": 16, "ratio": 0.500},
    {"layers": 24, "ratio": 0.750},
    {"layers": 32, "ratio": 1.000},
]

ROUNDS = 3
SEED = 20260926

# Metadata apenas; veja comentário no cabeçalho.
CPU_THREADS_TARGET = 8

# Configuração que será auditada pela API após cada load.
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

# Pequenos intervalos para reduzir efeitos de transição e permitir
# estabilização do runtime/hardware.
COOLDOWN_AFTER_UNLOAD_SECONDS = 15
SETTLE_AFTER_LOAD_SECONDS = 5
SETTLE_AFTER_WARMUP_SECONDS = 2
PAUSE_BETWEEN_PROMPTS_SECONDS = 2

LMS_LOAD_TIMEOUT_SECONDS = 180.0
LMS_COMMAND_TIMEOUT_SECONDS = 60.0
HTTP_TIMEOUT_SECONDS = 900.0

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"


# ----------------------------
# Utilidades
# ----------------------------

def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def seconds_to_hms(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "n/d"

    seconds_i = int(round(seconds))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def set_python_below_normal_priority() -> bool:
    """
    Reduz a prioridade apenas do processo Python para minimizar interferência.
    O processo de inferência do LM Studio não é alterado.
    """
    if os.name != "nt":
        return False

    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        return bool(
            kernel32.SetPriorityClass(
                handle,
                BELOW_NORMAL_PRIORITY_CLASS,
            )
        )
    except Exception:
        return False


def get_ac_power_status() -> str:
    if os.name != "nt":
        return "unknown"

    class SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = SYSTEM_POWER_STATUS()

    try:
        ok = ctypes.windll.kernel32.GetSystemPowerStatus(
            ctypes.byref(status)
        )
        if not ok:
            return "unknown"

        if status.ACLineStatus == 1:
            return "AC"
        if status.ACLineStatus == 0:
            return "battery"
        return "unknown"
    except Exception:
        return "unknown"


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
        handle.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
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


# ----------------------------
# LM Studio CLI / API
# ----------------------------

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
        details = (
            result.stderr
            or result.stdout
            or ""
        ).strip()

        raise RuntimeError(
            f"Falha ao executar:\n"
            f"  {' '.join(cmd)}\n"
            f"Código: {result.returncode}\n"
            f"{details or '(sem detalhes)'}"
        )

    return result


def api_get(
    path: str,
    timeout: float = 15.0,
) -> dict[str, Any]:
    request = Request(
        f"{API_BASE}{path}",
        headers={"Accept": "application/json"},
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(
                response.read().decode("utf-8")
            )
    except HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )
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
            return json.loads(
                response.read().decode("utf-8")
            )
    except HTTPError as exc:
        response_body = exc.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"HTTP {exc.code} em POST {path}: "
            f"{response_body}"
        ) from exc


def ensure_server() -> None:
    try:
        api_get(
            "/api/v1/models",
            timeout=5.0,
        )
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
        details = (
            result.stderr
            or result.stdout
            or ""
        ).strip()

        raise RuntimeError(
            "Não foi possível iniciar o servidor do LM Studio.\n"
            "Mantenha o LM Studio aberto e tente novamente.\n"
            f"{details}"
        )

    deadline = time.monotonic() + 15.0

    while time.monotonic() < deadline:
        try:
            api_get(
                "/api/v1/models",
                timeout=2.0,
            )
            return
        except Exception:
            time.sleep(0.5)

    raise RuntimeError(
        "O servidor foi iniciado, mas /api/v1/models "
        "não respondeu em 15 segundos."
    )


def verify_selected_variant() -> dict[str, Any]:
    result = run_lms(
        "ls",
        "--llm",
        "--json",
    )

    try:
        models = json.loads(
            result.stdout
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Não foi possível interpretar "
            "`lms ls --llm --json`."
        ) from exc

    for model in models:
        if model.get("modelKey") != MODEL_KEY:
            continue

        selected = model.get(
            "selectedVariant"
        )

        if selected != EXPECTED_VARIANT:
            raise RuntimeError(
                "Variante incorreta selecionada.\n"
                f"  Esperado: {EXPECTED_VARIANT}\n"
                f"  Atual:    {selected}"
            )

        return model

    raise RuntimeError(
        f"Modelo não encontrado: {MODEL_KEY}"
    )


def unload_all() -> None:
    run_lms(
        "unload",
        "--all",
        allow_failure=True,
        timeout=60.0,
    )


def load_model(
    gpu_ratio: float,
) -> tuple[
    subprocess.CompletedProcess[str],
    float,
]:
    start = time.perf_counter()

    result = run_lms(
        "load",
        MODEL_KEY,
        "--gpu",
        str(gpu_ratio),
        "--context-length",
        str(CONTEXT_LENGTH),
        "--identifier",
        INSTANCE_ID,
        timeout=LMS_LOAD_TIMEOUT_SECONDS,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return result, elapsed


def find_loaded_instance(
    response: dict[str, Any],
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    for model in response.get(
        "models",
        [],
    ):
        if model.get("key") != MODEL_KEY:
            continue

        for instance in model.get(
            "loaded_instances",
            [],
        ):
            if (
                instance.get("id")
                == INSTANCE_ID
            ):
                return model, instance

    return None, None


def audit_loaded_config(
    config: dict[str, Any],
) -> list[str]:
    issues: list[str] = []

    for key, expected in (
        EXPECTED_LOADED_CONFIG.items()
    ):
        if key not in config:
            issues.append(
                f"{key}: ausente; "
                f"esperado={expected!r}"
            )
            continue

        actual = config.get(key)

        if actual != expected:
            issues.append(
                f"{key}: "
                f"atual={actual!r}; "
                f"esperado={expected!r}"
            )

    return issues


def extract_message(
    response: dict[str, Any],
) -> str:
    chunks: list[str] = []

    for item in response.get(
        "output",
        [],
    ):
        if item.get("type") != "message":
            continue

        content = item.get("content")

        if isinstance(content, str):
            chunks.append(content)

    return "\n".join(chunks)


def make_chat_payload(
    prompt: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "model": INSTANCE_ID,
        "input": prompt,
        "temperature": TEMPERATURE,
        "max_output_tokens": (
            max_output_tokens
        ),
        "reasoning": REASONING,
        "context_length": (
            CONTEXT_LENGTH
        ),
        "store": False,
        "stream": False,
    }


def run_inference(
    prompt: str,
    max_output_tokens: int,
) -> tuple[
    dict[str, Any],
    float,
]:
    payload = make_chat_payload(
        prompt,
        max_output_tokens,
    )

    start = time.perf_counter()

    response = api_post(
        "/api/v1/chat",
        payload,
    )

    wall_time = (
        time.perf_counter()
        - start
    )

    return response, wall_time


# ----------------------------
# Planejamento / sessão
# ----------------------------

def build_schedule(
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    blocks: list[dict[str, Any]] = []

    for round_number in range(
        1,
        ROUNDS + 1,
    ):
        gpu_order = [
            dict(item)
            for item in GPU_CONFIGS
        ]
        rng.shuffle(gpu_order)

        for config_index, gpu in enumerate(
            gpu_order,
            start=1,
        ):
            prompt_order = list(
                PROMPTS.keys()
            )
            rng.shuffle(prompt_order)

            block_id = (
                f"R{round_number}-"
                f"G{gpu['layers']:02d}"
            )

            blocks.append(
                {
                    "block_id": block_id,
                    "round": round_number,
                    "config_index": (
                        config_index
                    ),
                    "gpu_layers": (
                        gpu["layers"]
                    ),
                    "gpu_ratio": (
                        gpu["ratio"]
                    ),
                    "prompt_order": (
                        prompt_order
                    ),
                    "run_ids": [
                        (
                            f"{block_id}-"
                            f"P{prompt_id}"
                        )
                        for prompt_id
                        in prompt_order
                    ],
                }
            )

    return blocks


def planned_run_count(
    schedule: list[dict[str, Any]],
) -> int:
    return sum(
        len(block["run_ids"])
        for block in schedule
    )


def find_latest_incomplete_session() -> Path | None:
    if not RESULTS_ROOT.exists():
        return None

    candidates: list[Path] = []

    for path in RESULTS_ROOT.glob(
        "gpu_benchmark_*"
    ):
        if not path.is_dir():
            continue

        session_file = (
            path / "session.json"
        )

        if not session_file.exists():
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
            candidates.append(path)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda p: p.name,
    )


def create_session(
    seed: int,
) -> tuple[
    Path,
    dict[str, Any],
]:
    session_dir = (
        RESULTS_ROOT
        / f"gpu_benchmark_{timestamp_slug()}"
    )
    session_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    schedule = build_schedule(seed)

    lms_version = (
        run_command_metadata(
            "lms",
            "--version",
        )
    )

    git_commit = (
        run_command_metadata(
            "git",
            "rev-parse",
            "HEAD",
        )
    )

    session = {
        "session_version": 1,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "completed": False,
        "completed_at": None,
        "seed": seed,
        "model_key": MODEL_KEY,
        "expected_variant": (
            EXPECTED_VARIANT
        ),
        "configuration": {
            "context_length": (
                CONTEXT_LENGTH
            ),
            "cpu_threads_target_metadata_only": (
                CPU_THREADS_TARGET
            ),
            "temperature": (
                TEMPERATURE
            ),
            "max_output_tokens": (
                MAX_OUTPUT_TOKENS
            ),
            "reasoning": REASONING,
            "rounds": ROUNDS,
            "gpu_configs": (
                GPU_CONFIGS
            ),
            "expected_loaded_config": (
                EXPECTED_LOADED_CONFIG
            ),
            "warmup_max_output_tokens": (
                WARMUP_MAX_OUTPUT_TOKENS
            ),
            "cooldown_after_unload_seconds": (
                COOLDOWN_AFTER_UNLOAD_SECONDS
            ),
            "settle_after_load_seconds": (
                SETTLE_AFTER_LOAD_SECONDS
            ),
            "settle_after_warmup_seconds": (
                SETTLE_AFTER_WARMUP_SECONDS
            ),
            "pause_between_prompts_seconds": (
                PAUSE_BETWEEN_PROMPTS_SECONDS
            ),
        },
        "prompts": PROMPTS,
        "warmup_prompt": (
            WARMUP_PROMPT
        ),
        "schedule": schedule,
        "planned_measured_runs": (
            planned_run_count(
                schedule
            )
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "lms_version": lms_version,
            "git_commit": git_commit,
        },
    }

    atomic_write_json(
        session_dir / "session.json",
        session,
    )

    return session_dir, session


def load_session(
    session_dir: Path,
) -> dict[str, Any]:
    path = session_dir / "session.json"

    if not path.exists():
        raise RuntimeError(
            f"session.json não encontrado em "
            f"{session_dir}"
        )

    return json.loads(
        path.read_text(
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
    ids: set[str] = set()

    for row in read_jsonl(
        runs_path
    ):
        if (
            row.get("status")
            == "success"
            and row.get("run_id")
        ):
            ids.add(
                row["run_id"]
            )

    return ids


# ----------------------------
# CSV / resumo
# ----------------------------

CSV_FIELDS = [
    "run_id",
    "timestamp",
    "round",
    "gpu_layers",
    "gpu_ratio",
    "prompt_id",
    "tokens_per_second",
    "time_to_first_token_seconds",
    "input_tokens",
    "total_output_tokens",
    "reasoning_output_tokens",
    "wall_time_seconds",
    "load_wall_time_seconds",
    "stop_reason",
    "response_length_chars",
    "status",
]


def write_runs_csv(
    session_dir: Path,
) -> None:
    runs = [
        row
        for row in read_jsonl(
            session_dir / "runs.jsonl"
        )
        if row.get("status")
        == "success"
    ]

    path = (
        session_dir / "runs.csv"
    )

    with path.open(
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
            stats = (
                row.get("stats")
                or {}
            )

            writer.writerow(
                {
                    "run_id": (
                        row.get("run_id")
                    ),
                    "timestamp": (
                        row.get("timestamp")
                    ),
                    "round": (
                        row.get("round")
                    ),
                    "gpu_layers": (
                        row.get(
                            "gpu_layers"
                        )
                    ),
                    "gpu_ratio": (
                        row.get(
                            "gpu_ratio"
                        )
                    ),
                    "prompt_id": (
                        row.get(
                            "prompt_id"
                        )
                    ),
                    "tokens_per_second": (
                        stats.get(
                            "tokens_per_second"
                        )
                    ),
                    "time_to_first_token_seconds": (
                        stats.get(
                            "time_to_first_token_seconds"
                        )
                    ),
                    "input_tokens": (
                        stats.get(
                            "input_tokens"
                        )
                    ),
                    "total_output_tokens": (
                        stats.get(
                            "total_output_tokens"
                        )
                    ),
                    "reasoning_output_tokens": (
                        stats.get(
                            "reasoning_output_tokens"
                        )
                    ),
                    "wall_time_seconds": (
                        row.get(
                            "wall_time_seconds"
                        )
                    ),
                    "load_wall_time_seconds": (
                        row.get(
                            "load_wall_time_seconds"
                        )
                    ),
                    "stop_reason": (
                        row.get(
                            "stop_reason"
                        )
                    ),
                    "response_length_chars": (
                        len(
                            row.get(
                                "response_text",
                                "",
                            )
                        )
                    ),
                    "status": (
                        row.get("status")
                    ),
                }
            )


def safe_mean(
    values: list[float],
) -> float | None:
    return (
        statistics.mean(values)
        if values
        else None
    )


def safe_median(
    values: list[float],
) -> float | None:
    return (
        statistics.median(values)
        if values
        else None
    )


def safe_stdev(
    values: list[float],
) -> float | None:
    return (
        statistics.stdev(values)
        if len(values) >= 2
        else None
    )


def numeric_values(
    rows: list[dict[str, Any]],
    stat_key: str,
) -> list[float]:
    values: list[float] = []

    for row in rows:
        value = (
            row.get("stats")
            or {}
        ).get(stat_key)

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
            session_dir / "runs.jsonl"
        )
        if row.get("status")
        == "success"
    ]

    grouped: dict[
        int,
        list[dict[str, Any]],
    ] = {}

    for row in runs:
        layers = int(
            row["gpu_layers"]
        )
        grouped.setdefault(
            layers,
            [],
        ).append(row)

    by_gpu: list[dict[str, Any]] = []

    for config in GPU_CONFIGS:
        layers = int(
            config["layers"]
        )
        rows = grouped.get(
            layers,
            [],
        )

        tps = numeric_values(
            rows,
            "tokens_per_second",
        )
        ttft = numeric_values(
            rows,
            "time_to_first_token_seconds",
        )
        output_tokens = numeric_values(
            rows,
            "total_output_tokens",
        )

        wall_times = [
            float(
                row["wall_time_seconds"]
            )
            for row in rows
            if isinstance(
                row.get(
                    "wall_time_seconds"
                ),
                (int, float),
            )
        ]

        by_gpu.append(
            {
                "gpu_layers": layers,
                "gpu_ratio": (
                    config["ratio"]
                ),
                "n": len(rows),
                "tokens_per_second_mean": (
                    safe_mean(tps)
                ),
                "tokens_per_second_median": (
                    safe_median(tps)
                ),
                "tokens_per_second_stdev": (
                    safe_stdev(tps)
                ),
                "ttft_mean_seconds": (
                    safe_mean(ttft)
                ),
                "ttft_median_seconds": (
                    safe_median(ttft)
                ),
                "ttft_stdev_seconds": (
                    safe_stdev(ttft)
                ),
                "wall_time_mean_seconds": (
                    safe_mean(
                        wall_times
                    )
                ),
                "output_tokens_mean": (
                    safe_mean(
                        output_tokens
                    )
                ),
            }
        )

    baseline = next(
        (
            row
            for row in by_gpu
            if row["gpu_layers"] == 0
        ),
        None,
    )

    baseline_tps = (
        baseline.get(
            "tokens_per_second_mean"
        )
        if baseline
        else None
    )
    baseline_ttft = (
        baseline.get(
            "ttft_mean_seconds"
        )
        if baseline
        else None
    )

    for row in by_gpu:
        tps_mean = row.get(
            "tokens_per_second_mean"
        )
        ttft_mean = row.get(
            "ttft_mean_seconds"
        )

        if (
            baseline_tps
            and tps_mean is not None
        ):
            row[
                "throughput_vs_gpu0_percent"
            ] = (
                (
                    tps_mean
                    / baseline_tps
                )
                - 1.0
            ) * 100.0
        else:
            row[
                "throughput_vs_gpu0_percent"
            ] = None

        if (
            baseline_ttft
            and ttft_mean is not None
        ):
            row[
                "ttft_vs_gpu0_percent"
            ] = (
                (
                    ttft_mean
                    / baseline_ttft
                )
                - 1.0
            ) * 100.0
        else:
            row[
                "ttft_vs_gpu0_percent"
            ] = None

    valid_tps_rows = [
        row
        for row in by_gpu
        if row.get(
            "tokens_per_second_mean"
        ) is not None
    ]

    valid_ttft_rows = [
        row
        for row in by_gpu
        if row.get(
            "ttft_mean_seconds"
        ) is not None
    ]

    best_tps = (
        max(
            valid_tps_rows,
            key=lambda row: row[
                "tokens_per_second_mean"
            ],
        )
        if valid_tps_rows
        else None
    )

    best_ttft = (
        min(
            valid_ttft_rows,
            key=lambda row: row[
                "ttft_mean_seconds"
            ],
        )
        if valid_ttft_rows
        else None
    )

    return {
        "generated_at": now_iso(),
        "successful_runs": len(runs),
        "planned_runs": (
            ROUNDS
            * len(GPU_CONFIGS)
            * len(PROMPTS)
        ),
        "by_gpu": by_gpu,
        "highest_mean_throughput": (
            {
                "gpu_layers": (
                    best_tps[
                        "gpu_layers"
                    ]
                ),
                "tokens_per_second_mean": (
                    best_tps[
                        "tokens_per_second_mean"
                    ]
                ),
            }
            if best_tps
            else None
        ),
        "lowest_mean_ttft": (
            {
                "gpu_layers": (
                    best_ttft[
                        "gpu_layers"
                    ]
                ),
                "ttft_mean_seconds": (
                    best_ttft[
                        "ttft_mean_seconds"
                    ]
                ),
            }
            if best_ttft
            else None
        ),
    }


def write_summary_files(
    session_dir: Path,
) -> dict[str, Any]:
    summary = build_summary(
        session_dir
    )

    atomic_write_json(
        session_dir / "summary.json",
        summary,
    )

    csv_path = (
        session_dir / "summary.csv"
    )

    fields = [
        "gpu_layers",
        "gpu_ratio",
        "n",
        "tokens_per_second_mean",
        "tokens_per_second_median",
        "tokens_per_second_stdev",
        "ttft_mean_seconds",
        "ttft_median_seconds",
        "ttft_stdev_seconds",
        "wall_time_mean_seconds",
        "output_tokens_mean",
        "throughput_vs_gpu0_percent",
        "ttft_vs_gpu0_percent",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in summary["by_gpu"]:
            writer.writerow(row)

    return summary


def print_summary(
    summary: dict[str, Any],
) -> None:
    print("\n" + "=" * 86)
    print(" RESUMO FINAL")
    print("=" * 86)

    print(
        f"{'GPU':>8}  "
        f"{'n':>3}  "
        f"{'tok/s média':>12}  "
        f"{'desvio':>9}  "
        f"{'TTFT média':>11}  "
        f"{'Δ tok/s vs 0':>13}"
    )
    print("-" * 86)

    for row in summary["by_gpu"]:
        layers = row["gpu_layers"]
        n = row["n"]
        tps = row[
            "tokens_per_second_mean"
        ]
        stdev = row[
            "tokens_per_second_stdev"
        ]
        ttft = row[
            "ttft_mean_seconds"
        ]
        delta = row[
            "throughput_vs_gpu0_percent"
        ]

        tps_s = (
            f"{tps:.3f}"
            if tps is not None
            else "n/d"
        )
        std_s = (
            f"{stdev:.3f}"
            if stdev is not None
            else "n/d"
        )
        ttft_s = (
            f"{ttft:.3f}s"
            if ttft is not None
            else "n/d"
        )
        delta_s = (
            f"{delta:+.1f}%"
            if delta is not None
            else "n/d"
        )

        print(
            f"{layers:>2}/32   "
            f"{n:>3}  "
            f"{tps_s:>12}  "
            f"{std_s:>9}  "
            f"{ttft_s:>11}  "
            f"{delta_s:>13}"
        )

    print("=" * 86)

    best = summary.get(
        "highest_mean_throughput"
    )

    if best:
        print(
            "Maior throughput médio: "
            f"{best['gpu_layers']}/32 "
            f"({best['tokens_per_second_mean']:.3f} tok/s)"
        )

    best_ttft = summary.get(
        "lowest_mean_ttft"
    )

    if best_ttft:
        print(
            "Menor TTFT médio:        "
            f"{best_ttft['gpu_layers']}/32 "
            f"({best_ttft['ttft_mean_seconds']:.3f}s)"
        )


# ----------------------------
# Execução do benchmark
# ----------------------------

def print_preflight(
    session_dir: Path,
    session: dict[str, Any],
    ac_status: str,
    priority_lowered: bool,
) -> None:
    print("=" * 78)
    print(" LocalIA — GPU Benchmark completo")
    print("=" * 78)
    print(f" Sessão              : {session_dir.name}")
    print(f" Modelo              : {MODEL_KEY}")
    print(f" Variante            : {EXPECTED_VARIANT}")
    print(f" Contexto            : {CONTEXT_LENGTH}")
    print(f" Rodadas             : {ROUNDS}")
    print(f" Prompts por config  : {len(PROMPTS)}")
    print(f" Configurações GPU   : {len(GPU_CONFIGS)}")
    print(
        f" Gerações medidas    : "
        f"{session['planned_measured_runs']}"
    )
    print(
        " GPU layers          : "
        + ", ".join(
            str(item["layers"])
            for item in GPU_CONFIGS
        )
    )
    print(
        f" CPU threads alvo    : "
        f"{CPU_THREADS_TARGET} "
        f"(metadata; não verificado)"
    )
    print(f" Think/Reasoning     : OFF")
    print(
        f" Alimentação         : "
        f"{ac_status}"
    )
    print(
        f" Prioridade Python   : "
        f"{'abaixo do normal' if priority_lowered else 'normal'}"
    )
    print(f" Seed                : {session['seed']}")
    print("=" * 78)


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

    seconds_per_run = (
        elapsed
        / newly_completed
    )

    return (
        seconds_per_run
        * remaining_runs
    )


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
                "block_id": block.get(
                    "block_id"
                ),
                "round": block.get(
                    "round"
                ),
                "gpu_layers": block.get(
                    "gpu_layers"
                ),
                "gpu_ratio": block.get(
                    "gpu_ratio"
                ),
            }
        )

    append_jsonl(
        session_dir / "errors.jsonl",
        record,
    )


def execute_benchmark(
    session_dir: Path,
    session: dict[str, Any],
) -> int:
    runs_path = (
        session_dir / "runs.jsonl"
    )
    warmups_path = (
        session_dir / "warmups.jsonl"
    )
    loads_path = (
        session_dir / "loads.jsonl"
    )

    successful_ids = (
        successful_run_ids(
            runs_path
        )
    )

    total_runs = int(
        session[
            "planned_measured_runs"
        ]
    )

    initially_completed = len(
        successful_ids
    )
    newly_completed = 0
    process_started = (
        time.perf_counter()
    )

    if initially_completed:
        print(
            f"\nRetomando sessão: "
            f"{initially_completed}/{total_runs} "
            f"gerações já concluídas."
        )

    schedule = session["schedule"]

    for block_number, block in enumerate(
        schedule,
        start=1,
    ):
        missing_run_ids = [
            run_id
            for run_id in block["run_ids"]
            if run_id not in successful_ids
        ]

        if not missing_run_ids:
            print(
                f"\n[Bloco {block_number}/{len(schedule)}] "
                f"{block['block_id']} já concluído — pulando."
            )
            continue

        gpu_layers = int(
            block["gpu_layers"]
        )
        gpu_ratio = float(
            block["gpu_ratio"]
        )

        print("\n" + "=" * 78)
        print(
            f" Bloco {block_number}/{len(schedule)} | "
            f"Rodada {block['round']}/{ROUNDS} | "
            f"GPU {gpu_layers}/32 "
            f"({gpu_ratio * 100:.1f}%)"
        )
        print(
            f" Prompts planejados: "
            f"{', '.join(block['prompt_order'])}"
        )
        print(
            f" Pendentes neste bloco: "
            f"{len(missing_run_ids)}"
        )
        print("=" * 78)

        loaded = False
        load_wall_time = None

        try:
            ensure_server()

            print(
                f"[LOAD] Descarregando modelo anterior..."
            )
            unload_all()

            if (
                COOLDOWN_AFTER_UNLOAD_SECONDS
                > 0
            ):
                print(
                    f"[WAIT] Cooldown "
                    f"{COOLDOWN_AFTER_UNLOAD_SECONDS}s..."
                )
                time.sleep(
                    COOLDOWN_AFTER_UNLOAD_SECONDS
                )

            print(
                f"[LOAD] Carregando GPU "
                f"{gpu_layers}/32..."
            )

            load_result, load_wall_time = (
                load_model(
                    gpu_ratio
                )
            )
            loaded = True

            print(
                f"[LOAD] OK em "
                f"{load_wall_time:.2f}s"
            )

            models_response = api_get(
                "/api/v1/models"
            )

            loaded_model, loaded_instance = (
                find_loaded_instance(
                    models_response
                )
            )

            if loaded_instance is None:
                raise RuntimeError(
                    f"Instância `{INSTANCE_ID}` "
                    f"não apareceu na API."
                )

            loaded_config = (
                loaded_instance.get(
                    "config"
                )
                or {}
            )

            issues = (
                audit_loaded_config(
                    loaded_config
                )
            )

            load_record = {
                "timestamp": now_iso(),
                "block_id": (
                    block["block_id"]
                ),
                "round": (
                    block["round"]
                ),
                "gpu_layers": (
                    gpu_layers
                ),
                "gpu_ratio": (
                    gpu_ratio
                ),
                "load_wall_time_seconds": (
                    load_wall_time
                ),
                "loaded_config": (
                    loaded_config
                ),
                "config_audit_issues": (
                    issues
                ),
                "lms_stdout": (
                    load_result.stdout.strip()
                ),
                "lms_stderr": (
                    load_result.stderr.strip()
                ),
            }

            append_jsonl(
                loads_path,
                load_record,
            )

            if issues:
                detail = "; ".join(
                    issues
                )
                raise RuntimeError(
                    "Auditoria de configuração "
                    f"falhou: {detail}"
                )

            print(
                "[AUDIT] Configuração carregada "
                "confere com o baseline."
            )

            if (
                SETTLE_AFTER_LOAD_SECONDS
                > 0
            ):
                print(
                    f"[WAIT] Estabilização "
                    f"{SETTLE_AFTER_LOAD_SECONDS}s..."
                )
                time.sleep(
                    SETTLE_AFTER_LOAD_SECONDS
                )

            print(
                "[WARMUP] Executando aquecimento "
                "(não contabilizado)..."
            )

            warmup_response, warmup_wall = (
                run_inference(
                    WARMUP_PROMPT,
                    WARMUP_MAX_OUTPUT_TOKENS,
                )
            )

            warmup_stats = (
                warmup_response.get(
                    "stats"
                )
                or {}
            )

            warmup_record = {
                "timestamp": now_iso(),
                "block_id": (
                    block["block_id"]
                ),
                "round": (
                    block["round"]
                ),
                "gpu_layers": (
                    gpu_layers
                ),
                "gpu_ratio": (
                    gpu_ratio
                ),
                "wall_time_seconds": (
                    warmup_wall
                ),
                "stats": warmup_stats,
                "response_text": (
                    extract_message(
                        warmup_response
                    )
                ),
            }

            append_jsonl(
                warmups_path,
                warmup_record,
            )

            print(
                f"[WARMUP] OK — "
                f"{warmup_stats.get('tokens_per_second', 'n/d')} "
                f"tok/s"
            )

            if (
                SETTLE_AFTER_WARMUP_SECONDS
                > 0
            ):
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
                    print(
                        f"[SKIP] {run_id} "
                        "já concluído."
                    )
                    continue

                completed_before = len(
                    successful_ids
                )
                global_number = (
                    completed_before + 1
                )
                remaining_before = (
                    total_runs
                    - completed_before
                )

                eta = calculate_eta(
                    process_started,
                    newly_completed,
                    remaining_before,
                )

                print(
                    "\n"
                    f"[RUN {global_number}/{total_runs}] "
                    f"{run_id} | "
                    f"GPU {gpu_layers}/32 | "
                    f"Prompt {prompt_id}"
                )

                if eta is not None:
                    print(
                        f"      ETA aproximada: "
                        f"{seconds_to_hms(eta)}"
                    )

                print(
                    "      Inferindo... "
                    "(sem streaming; Python ocioso)"
                )

                prompt = PROMPTS[
                    prompt_id
                ]

                try:
                    response, wall_time = (
                        run_inference(
                            prompt,
                            MAX_OUTPUT_TOKENS,
                        )
                    )

                    stats = (
                        response.get(
                            "stats"
                        )
                        or {}
                    )

                    reasoning_tokens = (
                        stats.get(
                            "reasoning_output_tokens"
                        )
                    )

                    response_text = (
                        extract_message(
                            response
                        )
                    )

                    stop_reason = (
                        response.get(
                            "stop_reason"
                        )
                        or stats.get(
                            "stop_reason"
                        )
                    )

                    record = {
                        "status": "success",
                        "timestamp": now_iso(),
                        "run_id": run_id,
                        "block_id": (
                            block["block_id"]
                        ),
                        "round": (
                            block["round"]
                        ),
                        "config_index": (
                            block["config_index"]
                        ),
                        "prompt_position": (
                            prompt_position
                        ),
                        "prompt_id": (
                            prompt_id
                        ),
                        "prompt": prompt,
                        "gpu_layers": (
                            gpu_layers
                        ),
                        "gpu_ratio": (
                            gpu_ratio
                        ),
                        "load_wall_time_seconds": (
                            load_wall_time
                        ),
                        "wall_time_seconds": (
                            wall_time
                        ),
                        "stats": stats,
                        "stop_reason": (
                            stop_reason
                        ),
                        "reasoning_off_verified": (
                            reasoning_tokens == 0
                        ),
                        "response_text": (
                            response_text
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

                    print(
                        f"      ✓ {tps_s} tok/s | "
                        f"TTFT {ttft_s} | "
                        f"{out_tokens} tokens | "
                        f"{wall_time:.1f}s"
                    )

                    if reasoning_tokens != 0:
                        print(
                            "      ⚠ reasoning tokens "
                            f"inesperados: "
                            f"{reasoning_tokens!r}"
                        )

                except Exception as exc:
                    message = str(exc)

                    print(
                        f"      ✗ ERRO: {message}"
                    )

                    record_error(
                        session_dir,
                        phase="inference",
                        message=message,
                        block=block,
                        run_id=run_id,
                    )

                if (
                    PAUSE_BETWEEN_PROMPTS_SECONDS
                    > 0
                ):
                    time.sleep(
                        PAUSE_BETWEEN_PROMPTS_SECONDS
                    )

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            message = str(exc)

            print(
                f"[BLOCO] ERRO: {message}"
            )

            record_error(
                session_dir,
                phase="block",
                message=message,
                block=block,
            )

        finally:
            if loaded:
                print(
                    "[CLEANUP] Descarregando modelo..."
                )
                unload_all()
                print(
                    "[CLEANUP] OK"
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
    session["completed"] = all_done

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
        f"\nResultados da sessão:\n"
        f"  {session_dir}"
    )

    if all_done:
        print(
            "\n✓ Todas as 54 gerações "
            "foram concluídas."
        )
        return 0

    print(
        f"\n⚠ Benchmark incompleto: "
        f"{len(successful_ids)}/{total_runs} "
        f"gerações concluídas."
    )
    print(
        "Execute o mesmo comando novamente "
        "para retomar as pendentes."
    )

    return 2


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark de GPU offload do "
            "LocalIA/LM Studio."
        )
    )

    parser.add_argument(
        "--new",
        action="store_true",
        help=(
            "Inicia uma nova sessão mesmo "
            "que exista uma incompleta."
        ),
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Não pede confirmação antes "
            "de iniciar/retomar."
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
            session = load_session(
                incomplete
            )
            return (
                incomplete,
                session,
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

    priority_lowered = (
        set_python_below_normal_priority()
    )

    print(
        "\n[PRECHECK] Verificando "
        "servidor e modelo..."
    )

    ensure_server()
    model_inventory = (
        verify_selected_variant()
    )

    ac_status = (
        get_ac_power_status()
    )

    session_dir, session, resumed = (
        choose_session(args)
    )

    session[
        "model_inventory_at_start"
    ] = model_inventory
    session[
        "ac_power_at_start"
    ] = ac_status
    session[
        "python_priority_lowered"
    ] = priority_lowered

    update_session_file(
        session_dir,
        session,
    )

    print_preflight(
        session_dir,
        session,
        ac_status,
        priority_lowered,
    )

    if resumed:
        print(
            "\nSessão incompleta encontrada; "
            "o benchmark será retomado."
        )

    if ac_status == "battery":
        print(
            "\n⚠ ATENÇÃO: o Windows reportou "
            "alimentação por bateria."
        )
        print(
            "Para resultados consistentes, "
            "conecte o carregador."
        )

        if not args.yes:
            answer = input(
                "Continuar mesmo assim? [s/N]: "
            ).strip().lower()

            if answer not in {
                "s",
                "sim",
                "y",
                "yes",
            }:
                print(
                    "Benchmark cancelado."
                )
                return 1

    if not args.yes:
        print(
            "\nDurante as inferências o terminal "
            "não fará streaming de tokens."
        )
        print(
            "Os resultados serão gravados após "
            "cada geração e podem ser retomados."
        )

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
        print(
            "Os resultados já concluídos "
            "foram preservados."
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
            "Execute o mesmo comando novamente "
            "para retomar."
        )

        return 130


if __name__ == "__main__":
    raise SystemExit(main())
