from __future__ import annotations

import argparse
import csv
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
from urllib.error import HTTPError
from urllib.request import Request, urlopen

API_BASE = "http://127.0.0.1:1234"

MODEL_KEY = "qwen/qwen3.5-9b"
EXPECTED_VARIANT = "qwen/qwen3.5-9b@q4_k_m"
INSTANCE_ID = "localia-runtime-bench"

CONTEXT_LENGTH = 4096

RUNTIMES = {
    "cpu": "llama.cpp-win-x86_64-avx2@2.46.0",
    "vulkan": "llama.cpp-win-x86_64-vulkan-avx2@2.46.0",
}

ROUNDS = 3
SEED = 20260927
CPU_THREADS_TARGET = 8

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
    "A": "Explique em aproximadamente 150 palavras o que é uma API REST e cite suas principais características.",
    "B": "Explique em aproximadamente 150 palavras como funciona a fotossíntese e qual é sua importância para os ecossistemas.",
    "C": "Explique em aproximadamente 150 palavras quais foram as principais causas da Revolução Francesa.",
}

WARMUP_PROMPT = (
    "Em no máximo 50 palavras, explique por que a água congela "
    "quando sua temperatura cai suficientemente."
)
WARMUP_MAX_OUTPUT_TOKENS = 96

COOLDOWN_AFTER_UNLOAD_SECONDS = 10
SETTLE_AFTER_RUNTIME_SWITCH_SECONDS = 3
SETTLE_AFTER_LOAD_SECONDS = 5
SETTLE_AFTER_WARMUP_SECONDS = 2
PAUSE_BETWEEN_PROMPTS_SECONDS = 2

LMS_LOAD_TIMEOUT_SECONDS = 180.0
LMS_COMMAND_TIMEOUT_SECONDS = 90.0
HTTP_TIMEOUT_SECONDS = 900.0

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"


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


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
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
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"AVISO: linha inválida em {path.name}: {line_number}; ignorada.")
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


def api_post(
    path: str,
    payload: dict[str, Any],
    timeout: float = HTTP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{API_BASE}{path}",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} em POST {path}: {body}") from exc


def ensure_server() -> None:
    try:
        api_get("/api/v1/models", timeout=5.0)
        return
    except Exception:
        pass

    result = run_lms("server", "start", allow_failure=True, timeout=30.0)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "Não foi possível iniciar o servidor do LM Studio. "
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

    raise RuntimeError("Servidor iniciado, mas /api/v1/models não respondeu em 15s.")


def verify_selected_variant() -> dict[str, Any]:
    result = run_lms("ls", "--llm", "--json")
    try:
        models = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Não foi possível interpretar `lms ls --llm --json`.") from exc

    for model in models:
        if model.get("modelKey") != MODEL_KEY:
            continue
        selected = model.get("selectedVariant")
        if selected != EXPECTED_VARIANT:
            raise RuntimeError(
                "Variante incorreta selecionada.\n"
                f"Esperado: {EXPECTED_VARIANT}\nAtual: {selected}"
            )
        return model

    raise RuntimeError(f"Modelo não encontrado: {MODEL_KEY}")


def runtime_ls_text() -> str:
    return run_lms("runtime", "ls").stdout


def verify_runtime_installed(alias: str) -> None:
    if alias not in runtime_ls_text():
        raise RuntimeError(f"Runtime necessário não está instalado:\n  {alias}")


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

    selected_line = selected_gguf_runtime()
    if not selected_line or alias not in selected_line:
        raise RuntimeError(
            "O comando de seleção terminou, mas o runtime ativo não pôde "
            "ser confirmado.\n"
            f"Esperado: {alias}\nSelecionado: {selected_line!r}\n"
            f"Saída: {result.stdout.strip()}"
        )
    return result.stdout.strip(), elapsed


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
    issues = []
    for key, expected in EXPECTED_LOADED_CONFIG.items():
        if key not in config:
            issues.append(f"{key}: ausente; esperado={expected!r}")
            continue
        actual = config.get(key)
        if actual != expected:
            issues.append(f"{key}: atual={actual!r}; esperado={expected!r}")
    return issues


def extract_message(response: dict[str, Any]) -> str:
    chunks = []
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


def build_schedule(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    blocks = []
    runtime_items = [
        {"runtime": name, "alias": alias}
        for name, alias in RUNTIMES.items()
    ]

    for round_number in range(1, ROUNDS + 1):
        runtime_order = [dict(item) for item in runtime_items]
        rng.shuffle(runtime_order)

        for block_index, item in enumerate(runtime_order, start=1):
            prompt_order = list(PROMPTS.keys())
            rng.shuffle(prompt_order)
            block_id = f"R{round_number}-{item['runtime'].upper()}"
            blocks.append(
                {
                    "block_id": block_id,
                    "round": round_number,
                    "block_index": block_index,
                    "runtime": item["runtime"],
                    "runtime_alias": item["alias"],
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

    candidates = []
    for path in RESULTS_ROOT.glob("runtime_benchmark_*"):
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
    session_dir = RESULTS_ROOT / f"runtime_benchmark_{timestamp_slug()}"
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
        "configuration": {
            "context_length": CONTEXT_LENGTH,
            "gpu_offload": "off",
            "cpu_threads_target_metadata_only": CPU_THREADS_TARGET,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "reasoning": REASONING,
            "rounds": ROUNDS,
            "expected_loaded_config": EXPECTED_LOADED_CONFIG,
        },
        "prompts": PROMPTS,
        "warmup_prompt": WARMUP_PROMPT,
        "schedule": schedule,
        "planned_measured_runs": ROUNDS * len(RUNTIMES) * len(PROMPTS),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "lms_version": run_command_metadata("lms", "--version"),
            "git_commit": run_command_metadata("git", "rev-parse", "HEAD"),
        },
    }

    atomic_write_json(session_dir / "session.json", session)
    return session_dir, session


def load_session(session_dir: Path) -> dict[str, Any]:
    return json.loads((session_dir / "session.json").read_text(encoding="utf-8"))


def update_session_file(session_dir: Path, session: dict[str, Any]) -> None:
    session["updated_at"] = now_iso()
    atomic_write_json(session_dir / "session.json", session)


def successful_run_ids(runs_path: Path) -> set[str]:
    return {
        row["run_id"]
        for row in read_jsonl(runs_path)
        if row.get("status") == "success" and row.get("run_id")
    }


CSV_FIELDS = [
    "run_id",
    "timestamp",
    "round",
    "runtime",
    "runtime_alias",
    "prompt_id",
    "tokens_per_second",
    "time_to_first_token_seconds",
    "input_tokens",
    "total_output_tokens",
    "reasoning_output_tokens",
    "wall_time_seconds",
    "load_wall_time_seconds",
    "runtime_switch_seconds",
    "response_length_chars",
    "status",
]


def write_runs_csv(session_dir: Path) -> None:
    runs = [
        row
        for row in read_jsonl(session_dir / "runs.jsonl")
        if row.get("status") == "success"
    ]

    with (session_dir / "runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()

        for row in runs:
            stats = row.get("stats") or {}
            writer.writerow(
                {
                    "run_id": row.get("run_id"),
                    "timestamp": row.get("timestamp"),
                    "round": row.get("round"),
                    "runtime": row.get("runtime"),
                    "runtime_alias": row.get("runtime_alias"),
                    "prompt_id": row.get("prompt_id"),
                    "tokens_per_second": stats.get("tokens_per_second"),
                    "time_to_first_token_seconds": stats.get(
                        "time_to_first_token_seconds"
                    ),
                    "input_tokens": stats.get("input_tokens"),
                    "total_output_tokens": stats.get("total_output_tokens"),
                    "reasoning_output_tokens": stats.get(
                        "reasoning_output_tokens"
                    ),
                    "wall_time_seconds": row.get("wall_time_seconds"),
                    "load_wall_time_seconds": row.get("load_wall_time_seconds"),
                    "runtime_switch_seconds": row.get("runtime_switch_seconds"),
                    "response_length_chars": len(row.get("response_text", "")),
                    "status": row.get("status"),
                }
            )


def safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def safe_stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def stat_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = (row.get("stats") or {}).get(key)
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def build_summary(session_dir: Path) -> dict[str, Any]:
    runs = [
        row
        for row in read_jsonl(session_dir / "runs.jsonl")
        if row.get("status") == "success"
    ]

    by_runtime = []

    for runtime_name, alias in RUNTIMES.items():
        rows = [r for r in runs if r.get("runtime") == runtime_name]
        tps = stat_values(rows, "tokens_per_second")
        ttft = stat_values(rows, "time_to_first_token_seconds")
        wall = [
            float(r["wall_time_seconds"])
            for r in rows
            if isinstance(r.get("wall_time_seconds"), (int, float))
        ]
        loads = [
            float(r["load_wall_time_seconds"])
            for r in rows
            if isinstance(r.get("load_wall_time_seconds"), (int, float))
        ]

        by_runtime.append(
            {
                "runtime": runtime_name,
                "runtime_alias": alias,
                "n": len(rows),
                "tokens_per_second_mean": safe_mean(tps),
                "tokens_per_second_median": safe_median(tps),
                "tokens_per_second_stdev": safe_stdev(tps),
                "ttft_mean_seconds": safe_mean(ttft),
                "ttft_median_seconds": safe_median(ttft),
                "ttft_stdev_seconds": safe_stdev(ttft),
                "wall_time_mean_seconds": safe_mean(wall),
                "load_time_mean_seconds": safe_mean(loads),
            }
        )

    cpu = next((x for x in by_runtime if x["runtime"] == "cpu"), None)
    vulkan = next((x for x in by_runtime if x["runtime"] == "vulkan"), None)

    comparison = {}
    if cpu and vulkan:
        cpu_tps = cpu.get("tokens_per_second_mean")
        vk_tps = vulkan.get("tokens_per_second_mean")
        cpu_ttft = cpu.get("ttft_mean_seconds")
        vk_ttft = vulkan.get("ttft_mean_seconds")

        if cpu_tps and vk_tps:
            comparison["cpu_vs_vulkan_throughput_percent"] = (
                (cpu_tps / vk_tps) - 1.0
            ) * 100.0
        if cpu_ttft and vk_ttft:
            comparison["cpu_vs_vulkan_ttft_percent"] = (
                (cpu_ttft / vk_ttft) - 1.0
            ) * 100.0

    return {
        "generated_at": now_iso(),
        "successful_runs": len(runs),
        "planned_runs": ROUNDS * len(RUNTIMES) * len(PROMPTS),
        "by_runtime": by_runtime,
        "comparison": comparison,
    }


def write_summary_files(session_dir: Path) -> dict[str, Any]:
    summary = build_summary(session_dir)
    atomic_write_json(session_dir / "summary.json", summary)

    fields = [
        "runtime",
        "runtime_alias",
        "n",
        "tokens_per_second_mean",
        "tokens_per_second_median",
        "tokens_per_second_stdev",
        "ttft_mean_seconds",
        "ttft_median_seconds",
        "ttft_stdev_seconds",
        "wall_time_mean_seconds",
        "load_time_mean_seconds",
    ]

    with (session_dir / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["by_runtime"]:
            writer.writerow(row)

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 92)
    print(" RESUMO FINAL — CPU llama.cpp vs Vulkan llama.cpp (GPU offload OFF)")
    print("=" * 92)

    for row in summary["by_runtime"]:
        tps = row["tokens_per_second_mean"]
        ttft = row["ttft_mean_seconds"]
        wall = row["wall_time_mean_seconds"]
        print(
            f"{row['runtime']:<8} | n={row['n']:>2} | "
            f"tok/s={tps:.3f if False else ''}"
        )
        # impressão robusta abaixo
        print(
            f"         throughput médio: "
            f"{tps:.3f if isinstance(tps, (int, float)) else 0}"
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
                "block_id": block.get("block_id"),
                "round": block.get("round"),
                "runtime": block.get("runtime"),
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
        (time.perf_counter() - process_started) / newly_completed
    ) * remaining_runs


def execute_benchmark(
    session_dir: Path,
    session: dict[str, Any],
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

        runtime_name = block["runtime"]
        runtime_alias = block["runtime_alias"]

        print("\n" + "=" * 80)
        print(
            f" Bloco {block_number}/{len(session['schedule'])} | "
            f"Rodada {block['round']}/{ROUNDS} | "
            f"{runtime_name.upper()}"
        )
        print(f" {runtime_alias}")
        print("=" * 80)

        loaded = False
        load_wall_time = None
        runtime_switch_seconds = None

        try:
            ensure_server()

            print("[SWITCH] Descarregando modelo anterior...")
            unload_all()
            time.sleep(COOLDOWN_AFTER_UNLOAD_SECONDS)

            print(f"[SWITCH] Selecionando {runtime_name.upper()}...")
            switch_stdout, runtime_switch_seconds = select_runtime(runtime_alias)
            selected_line = selected_gguf_runtime()

            append_jsonl(
                switches_path,
                {
                    "timestamp": now_iso(),
                    "block_id": block["block_id"],
                    "round": block["round"],
                    "runtime": runtime_name,
                    "runtime_alias": runtime_alias,
                    "switch_wall_time_seconds": runtime_switch_seconds,
                    "selected_line_after_switch": selected_line,
                    "lms_stdout": switch_stdout,
                },
            )

            print(f"[SWITCH] OK — {selected_line}")
            time.sleep(SETTLE_AFTER_RUNTIME_SWITCH_SECONDS)

            print("[LOAD] Carregando com GPU offload OFF...")
            load_result, load_wall_time = load_model()
            loaded = True
            print(f"[LOAD] OK em {load_wall_time:.2f}s")

            models_response = api_get("/api/v1/models")
            _, loaded_instance = find_loaded_instance(models_response)
            if loaded_instance is None:
                raise RuntimeError(f"Instância `{INSTANCE_ID}` não apareceu na API.")

            loaded_config = loaded_instance.get("config") or {}
            issues = audit_loaded_config(loaded_config)

            append_jsonl(
                loads_path,
                {
                    "timestamp": now_iso(),
                    "block_id": block["block_id"],
                    "round": block["round"],
                    "runtime": runtime_name,
                    "runtime_alias": runtime_alias,
                    "runtime_switch_seconds": runtime_switch_seconds,
                    "load_wall_time_seconds": load_wall_time,
                    "loaded_config": loaded_config,
                    "config_audit_issues": issues,
                    "lms_stdout": load_result.stdout.strip(),
                    "lms_stderr": load_result.stderr.strip(),
                },
            )

            if issues:
                raise RuntimeError(
                    "Auditoria da configuração falhou: " + "; ".join(issues)
                )

            print("[AUDIT] Configuração confere.")
            time.sleep(SETTLE_AFTER_LOAD_SECONDS)

            print("[WARMUP] Aquecimento não contabilizado...")
            warm_response, warm_wall = run_inference(
                WARMUP_PROMPT, WARMUP_MAX_OUTPUT_TOKENS
            )
            warm_stats = warm_response.get("stats") or {}

            append_jsonl(
                warmups_path,
                {
                    "timestamp": now_iso(),
                    "block_id": block["block_id"],
                    "round": block["round"],
                    "runtime": runtime_name,
                    "runtime_alias": runtime_alias,
                    "wall_time_seconds": warm_wall,
                    "stats": warm_stats,
                    "response_text": extract_message(warm_response),
                },
            )

            print(
                f"[WARMUP] OK — "
                f"{warm_stats.get('tokens_per_second', 'n/d')} tok/s"
            )
            time.sleep(SETTLE_AFTER_WARMUP_SECONDS)

            for prompt_position, prompt_id in enumerate(
                block["prompt_order"], start=1
            ):
                run_id = f"{block['block_id']}-P{prompt_id}"

                if run_id in successful_ids:
                    continue

                completed_before = len(successful_ids)
                global_number = completed_before + 1
                remaining = total_runs - completed_before
                eta = calculate_eta(
                    process_started, newly_completed, remaining
                )

                print(
                    f"\n[RUN {global_number}/{total_runs}] "
                    f"{run_id} | {runtime_name.upper()} | Prompt {prompt_id}"
                )
                if eta is not None:
                    print(f"      ETA: {seconds_to_hms(eta)}")
                print("      Inferindo...")

                try:
                    response, wall_time = run_inference(
                        PROMPTS[prompt_id],
                        MAX_OUTPUT_TOKENS,
                    )
                    stats = response.get("stats") or {}
                    reasoning_tokens = stats.get("reasoning_output_tokens")

                    record = {
                        "status": "success",
                        "timestamp": now_iso(),
                        "run_id": run_id,
                        "block_id": block["block_id"],
                        "round": block["round"],
                        "prompt_position": prompt_position,
                        "runtime": runtime_name,
                        "runtime_alias": runtime_alias,
                        "prompt_id": prompt_id,
                        "prompt": PROMPTS[prompt_id],
                        "runtime_switch_seconds": runtime_switch_seconds,
                        "load_wall_time_seconds": load_wall_time,
                        "wall_time_seconds": wall_time,
                        "stats": stats,
                        "reasoning_off_verified": reasoning_tokens == 0,
                        "response_text": extract_message(response),
                    }

                    append_jsonl(runs_path, record)
                    successful_ids.add(run_id)
                    newly_completed += 1

                    write_runs_csv(session_dir)
                    write_summary_files(session_dir)

                    tps = stats.get("tokens_per_second")
                    ttft = stats.get("time_to_first_token_seconds")
                    out_tokens = stats.get("total_output_tokens")

                    tps_s = f"{tps:.3f}" if isinstance(tps, (int, float)) else "n/d"
                    ttft_s = (
                        f"{ttft:.3f}s"
                        if isinstance(ttft, (int, float))
                        else "n/d"
                    )

                    print(
                        f"      ✓ {tps_s} tok/s | "
                        f"TTFT {ttft_s} | {out_tokens} tokens | "
                        f"{wall_time:.1f}s"
                    )

                except Exception as exc:
                    print(f"      ✗ ERRO: {exc}")
                    record_error(
                        session_dir,
                        phase="inference",
                        message=str(exc),
                        block=block,
                        run_id=run_id,
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

    print("\n" + "=" * 78)
    print("RESUMO FINAL")
    print("=" * 78)
    for row in summary["by_runtime"]:
        tps = row["tokens_per_second_mean"]
        med = row["tokens_per_second_median"]
        ttft = row["ttft_mean_seconds"]
        wall = row["wall_time_mean_seconds"]

        def f(v, n=3):
            return f"{v:.{n}f}" if isinstance(v, (int, float)) else "n/d"

        print(
            f"{row['runtime']:<8} | n={row['n']:>2} | "
            f"tok/s média={f(tps)} | mediana={f(med)} | "
            f"TTFT={f(ttft)}s | tempo={f(wall,1)}s"
        )

    delta = summary.get("comparison", {}).get(
        "cpu_vs_vulkan_throughput_percent"
    )
    if isinstance(delta, (int, float)):
        if delta >= 0:
            print(f"CPU: +{delta:.1f}% throughput vs Vulkan")
        else:
            print(f"CPU: {delta:.1f}% throughput vs Vulkan")

    print(f"\nResultados: {session_dir}")

    if all_done:
        print("✓ Todas as 18 gerações foram concluídas.")
        return 0

    print(
        f"⚠ Benchmark incompleto: {len(successful_ids)}/{total_runs}. "
        "Rode o mesmo comando para retomar."
    )
    return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara CPU llama.cpp vs Vulkan llama.cpp "
            "com GPU offload desligado."
        )
    )
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
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

    print("\n[PRECHECK] Validando LM Studio, modelo e runtimes...")
    ensure_server()
    model_inventory = verify_selected_variant()

    for runtime_name, alias in RUNTIMES.items():
        verify_runtime_installed(alias)
        print(f"  ✓ {runtime_name}: {alias}")

    session_dir, session, resumed = choose_session(args)
    session["model_inventory_at_start"] = model_inventory
    session["runtime_ls_at_start"] = runtime_ls_text()
    update_session_file(session_dir, session)

    print("\n" + "=" * 80)
    print(" LocalIA — Runtime Benchmark")
    print("=" * 80)
    print(f" Modelo           : {MODEL_KEY}")
    print(f" Variante         : {EXPECTED_VARIANT}")
    print(f" Contexto         : {CONTEXT_LENGTH}")
    print(" GPU offload      : OFF em ambos")
    print(f" CPU threads alvo : {CPU_THREADS_TARGET} (metadata; não verificado)")
    print(" Runtimes         : CPU 2.46.0 vs Vulkan 2.46.0")
    print(f" Rodadas          : {ROUNDS}")
    print(f" Gerações medidas : {session['planned_measured_runs']}")
    print(f" Sessão           : {session_dir.name}")
    print("=" * 80)

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
        return execute_benchmark(session_dir, session)
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário.")
        try:
            unload_all()
        except Exception:
            pass
        session["interrupted_at"] = now_iso()
        update_session_file(session_dir, session)
        print("Resultados preservados. Rode o mesmo comando para retomar.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
