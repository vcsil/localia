from __future__ import annotations

"""
LocalIA — Probe de carregamento de variantes via Python SDK
===========================================================

Por que este probe existe?
--------------------------
O `lms load` da CLI atual não resolve corretamente os aliases de variante
como `qwen/qwen3.5-9b@q6_k` em modo não-interativo. Este probe evita a CLI
para a seleção da variante e chama diretamente a API oficial do Python SDK:

    client.llm.load_new_instance("qwen/qwen3.5-9b@q6_k", ...)

Ele testa Q4_K_M e Q6_K separadamente, SEM fazer benchmark e SEM inferência.

O que ele verifica:
- runtime CPU AVX2 selecionado;
- load direto pelo modelKey da variante;
- progresso real do load;
- modelo GGUF efetivamente aberto pelo llama-server;
- quantização correta no caminho --model;
- RSS do processo;
- RAM disponível e pagefile antes/depois;
- unload ao final de cada tentativa.

Ele NÃO altera o Per-model Default.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

MODEL_VARIANTS = {
    "q4_k_m": {
        "display": "Q4_K_M",
        "model_key": "qwen/qwen3.5-9b@q4_k_m",
        "path_markers": ("q4_k_m", "q4-k-m", "q4km"),
    },
    "q6_k": {
        "display": "Q6_K",
        "model_key": "qwen/qwen3.5-9b@q6_k",
        "path_markers": ("q6_k", "q6-k", "q6k"),
    },
}

CPU_RUNTIME = "llama.cpp-win-x86_64-avx2@2.46.0"
CONTEXT_LENGTH = 4096
LOAD_TIMEOUT_SECONDS = 180.0

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def bytes_to_gib(value: int | float | None) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value) / (1024 ** 3)


def fmt(value: Any, decimals: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{decimals}f}"
    return "n/d"


def normalize(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value or "").lower(),
    )


def run_lms(
    *args: str,
    allow_failure: bool = False,
    timeout: float = 60.0,
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
            "`lms` não foi encontrado no PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timeout executando: {' '.join(cmd)}"
        ) from exc

    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"Falha em: {' '.join(cmd)}\n"
            f"{(result.stderr or result.stdout or '').strip()}"
        )

    return result


def select_cpu_runtime() -> None:
    run_lms(
        "unload",
        "--all",
        allow_failure=True,
        timeout=30.0,
    )

    run_lms(
        "runtime",
        "select",
        CPU_RUNTIME,
        timeout=30.0,
    )

    runtime_ls = run_lms(
        "runtime",
        "ls",
        timeout=30.0,
    ).stdout

    selected_line = None

    for line in runtime_ls.splitlines():
        if "✓" in line and "GGUF" in line:
            selected_line = line.strip()
            break

    if (
        selected_line is None
        or CPU_RUNTIME not in selected_line
    ):
        raise RuntimeError(
            "Runtime CPU ativo não foi confirmado.\n"
            f"Esperado: {CPU_RUNTIME}\n"
            f"Encontrado: {selected_line!r}"
        )

    print(f"[RUNTIME] ✓ {selected_line}")


def memory_snapshot(psutil: Any) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    return {
        "timestamp": now_iso(),
        "ram_total_bytes": vm.total,
        "ram_available_bytes": vm.available,
        "ram_used_percent": vm.percent,
        "swap_total_bytes": swap.total,
        "swap_used_bytes": swap.used,
        "swap_used_percent": swap.percent,
    }


def process_info(proc: Any) -> dict[str, Any]:
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

    return {
        "pid": proc.pid,
        "name": name,
        "rss_bytes": rss,
        "rss_gib": bytes_to_gib(rss),
        "cmdline": cmdline,
        "text": " ".join(cmdline),
    }


def find_llama_process(
    psutil: Any,
    pids_before: set[int],
) -> tuple[Any, dict[str, Any]]:
    candidates: list[tuple[float, Any, dict[str, Any]]] = []

    for proc in psutil.process_iter():
        try:
            info = process_info(proc)
            blob = (
                f"{info.get('name') or ''} "
                f"{info.get('text') or ''}"
            ).lower()

            if "llama-server" not in blob:
                continue

            score = 0.0

            if proc.pid not in pids_before:
                score += 1000.0

            rss = info.get("rss_bytes") or 0
            score += min(
                float(rss) / (1024 ** 3),
                20.0,
            ) * 10.0

            candidates.append(
                (score, proc, info)
            )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            continue

    if not candidates:
        raise RuntimeError(
            "llama-server não foi encontrado após o load."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return (
        candidates[0][1],
        candidates[0][2],
    )


def get_cli_arg(
    cmdline: list[str],
    flag: str,
) -> str | None:
    for index, item in enumerate(cmdline):
        if item == flag:
            if index + 1 < len(cmdline):
                return cmdline[index + 1]

        prefix = flag + "="
        if item.startswith(prefix):
            return item[len(prefix):]

    return None


def verify_loaded_quant(
    process_data: dict[str, Any],
    quant_key: str,
) -> dict[str, Any]:
    cmdline = process_data.get("cmdline") or []
    model_path = get_cli_arg(cmdline, "--model")

    if not model_path:
        raise RuntimeError(
            "O llama-server não expôs --model."
        )

    normalized_path = normalize(model_path)

    markers = MODEL_VARIANTS[
        quant_key
    ][
        "path_markers"
    ]

    ok = any(
        normalize(marker) in normalized_path
        for marker in markers
    )

    if not ok:
        raise RuntimeError(
            "VARIANTE INCORRETA CARREGADA.\n"
            f"Esperada: {MODEL_VARIANTS[quant_key]['display']}\n"
            f"--model:  {model_path}"
        )

    gpu_layers = get_cli_arg(
        cmdline,
        "--n-gpu-layers",
    )

    context = get_cli_arg(
        cmdline,
        "--ctx-size",
    )

    return {
        "model_path": model_path,
        "quant_verified": True,
        "gpu_layers": (
            int(gpu_layers)
            if gpu_layers is not None
            else None
        ),
        "context_length": (
            int(context)
            if context is not None
            else None
        ),
    }


class ProgressPrinter:
    def __init__(self, display: str) -> None:
        self.display = display
        self.last_bucket = -1
        self.last_progress = 0.0

    def __call__(self, progress: float) -> None:
        self.last_progress = float(progress)

        # SDK normalmente usa 0..1.
        percent = (
            float(progress) * 100.0
            if progress <= 1.0
            else float(progress)
        )

        bucket = int(percent // 10)

        if bucket != self.last_bucket or percent >= 99.0:
            self.last_bucket = bucket
            print(
                f"[LOAD {self.display}] progresso ≈ "
                f"{percent:.1f}%"
            )


def run_variant_probe(
    *,
    client: Any,
    lms: Any,
    psutil: Any,
    quant_key: str,
) -> dict[str, Any]:
    spec = MODEL_VARIANTS[quant_key]
    display = spec["display"]
    model_key = spec["model_key"]
    instance_id = (
        f"localia-variant-probe-{quant_key}"
    )

    print("\n" + "=" * 88)
    print(
        f" SDK DIRECT LOAD — {display}"
    )
    print("=" * 88)

    # Evita sobra de qualquer tentativa anterior.
    run_lms(
        "unload",
        "--all",
        allow_failure=True,
        timeout=30.0,
    )

    time.sleep(2.0)

    before = memory_snapshot(psutil)
    pids_before = set(psutil.pids())

    print(
        "[MEM antes] "
        f"RAM livre={fmt(bytes_to_gib(before['ram_available_bytes']))} GiB | "
        f"RAM usada={fmt(before['ram_used_percent'], 1)}% | "
        f"pagefile usado={fmt(bytes_to_gib(before['swap_used_bytes']))} GiB"
    )

    progress = ProgressPrinter(display)
    handle = None

    start = time.perf_counter()

    try:
        handle = client.llm.load_new_instance(
            model_key,
            instance_identifier=instance_id,
            ttl=None,
            config={
                "contextLength": CONTEXT_LENGTH,
                "gpu": {
                    "ratio": 0.0,
                },
            },
            on_load_progress=progress,
        )

        load_seconds = (
            time.perf_counter()
            - start
        )

        process, process_data = find_llama_process(
            psutil,
            pids_before,
        )

        cli_audit = verify_loaded_quant(
            process_data,
            quant_key,
        )

        after = memory_snapshot(psutil)

        print(
            f"[PASS] {display} carregado diretamente "
            f"via SDK em {load_seconds:.2f}s."
        )

        print(
            f"[MODEL] {cli_audit['model_path']}"
        )

        print(
            "[MEM depois] "
            f"RSS={fmt(process_data.get('rss_gib'))} GiB | "
            f"RAM livre={fmt(bytes_to_gib(after['ram_available_bytes']))} GiB | "
            f"RAM usada={fmt(after['ram_used_percent'], 1)}% | "
            f"pagefile usado={fmt(bytes_to_gib(after['swap_used_bytes']))} GiB"
        )

        return {
            "timestamp": now_iso(),
            "status": "success",
            "quant_key": quant_key,
            "quant_display": display,
            "model_key_requested": model_key,
            "load_seconds": load_seconds,
            "last_progress": progress.last_progress,
            "memory_before": before,
            "memory_after": after,
            "process": process_data,
            "cli_audit": cli_audit,
        }

    except Exception as exc:
        elapsed = (
            time.perf_counter()
            - start
        )

        after_failure = memory_snapshot(
            psutil
        )

        print(
            f"[FAIL] {display}: {type(exc).__name__}: {exc}"
        )

        print(
            "[MEM falha] "
            f"RAM livre={fmt(bytes_to_gib(after_failure['ram_available_bytes']))} GiB | "
            f"RAM usada={fmt(after_failure['ram_used_percent'], 1)}% | "
            f"pagefile usado={fmt(bytes_to_gib(after_failure['swap_used_bytes']))} GiB"
        )

        return {
            "timestamp": now_iso(),
            "status": "failed",
            "quant_key": quant_key,
            "quant_display": display,
            "model_key_requested": model_key,
            "elapsed_seconds": elapsed,
            "last_progress": progress.last_progress,
            "memory_before": before,
            "memory_after_failure": after_failure,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    finally:
        if handle is not None:
            try:
                handle.unload()
            except Exception:
                pass

        # Rede de segurança para qualquer load parcial.
        run_lms(
            "unload",
            "--all",
            allow_failure=True,
            timeout=30.0,
        )

        time.sleep(3.0)


def selftest() -> None:
    q4 = normalize(
        r"C:\models\Qwen3.5-9B-Q4_K_M.gguf"
    )

    q6 = normalize(
        r"C:\models\Qwen3.5-9B-Q6_K.gguf"
    )

    assert normalize("q4_k_m") in q4
    assert normalize("q6_k") not in q4
    assert normalize("q6_k") in q6

    sample = [
        "llama-server.exe",
        "--model",
        r"C:\models\Qwen3.5-9B-Q6_K.gguf",
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        "0",
    ]

    assert (
        get_cli_arg(
            sample,
            "--model",
        )
        == r"C:\models\Qwen3.5-9B-Q6_K.gguf"
    )

    assert (
        get_cli_arg(
            sample,
            "--ctx-size",
        )
        == "4096"
    )


def main() -> int:
    selftest()

    try:
        import lmstudio as lms  # type: ignore
        import psutil  # type: ignore
    except ImportError:
        print(
            "Instale dependências:\n"
            "  python -m pip install -U lmstudio psutil"
        )
        return 1

    setter = getattr(
        lms,
        "set_sync_api_timeout",
        None,
    )

    if callable(setter):
        setter(
            LOAD_TIMEOUT_SECONDS
        )

    print(
        "[SELFTEST] ✓ lógica local validada."
    )

    select_cpu_runtime()

    results_dir = (
        RESULTS_ROOT
        / (
            "variant_sdk_probe_"
            f"{timestamp_slug()}"
        )
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    report = {
        "created_at": now_iso(),
        "python": sys.version,
        "runtime": CPU_RUNTIME,
        "context_length": CONTEXT_LENGTH,
        "sdk_timeout_seconds": LOAD_TIMEOUT_SECONDS,
        "tests": [],
    }

    report_path = (
        results_dir
        / "report.json"
    )

    with lms.Client() as client:
        # Primeiro valida que o alias Q4 também funciona pela rota SDK;
        # só depois tenta Q6.
        for quant_key in (
            "q4_k_m",
            "q6_k",
        ):
            result = run_variant_probe(
                client=client,
                lms=lms,
                psutil=psutil,
                quant_key=quant_key,
            )

            report["tests"].append(
                result
            )

            report_path.write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            if (
                quant_key == "q4_k_m"
                and result["status"]
                != "success"
            ):
                print(
                    "\nQ4 falhou pela rota SDK. "
                    "Q6 não será tentado."
                )
                break

    report["finished_at"] = now_iso()

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\nRelatório:"
    )
    print(
        f"  {report_path}"
    )

    successes = [
        item
        for item in report["tests"]
        if item.get("status")
        == "success"
    ]

    if len(successes) == 2:
        print(
            "\n✓ Q4_K_M e Q6_K foram carregados "
            "corretamente por modelKey de variante via SDK."
        )
        return 0

    print(
        "\n⚠ O probe não confirmou as duas variantes. "
        "Envie o terminal e report.json."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
