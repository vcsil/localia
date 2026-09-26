from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE = "http://127.0.0.1:1234"

MODEL_KEY = "qwen/qwen3.5-9b@q4_k_m"
INSTANCE_ID = "localia-bench"

CONTEXT_LENGTH = 4096

GPU_RATIO = 0.125
GPU_LAYERS_EXPECTED = 4

CPU_THREADS_TARGET = 8

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 320

PROMPT_ID = "A"

PROMPT = (
    "Explique em aproximadamente 150 palavras o que é uma API REST "
    "e cite suas principais características."
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def banner() -> None:
    print("=" * 62)
    print(" LocalIA — GPU Smoke Benchmark")
    print("=" * 62)

    print(f" Modelo       : {MODEL_KEY}")
    print(f" Instância    : {INSTANCE_ID}")
    print(f" Contexto     : {CONTEXT_LENGTH}")
    print(
        f" GPU offload  : {GPU_RATIO:.3f} "
        f"(~{GPU_LAYERS_EXPECTED}/32 camadas)"
    )

    print(f" CPU alvo     : {CPU_THREADS_TARGET} threads")
    print(" Think        : OFF")
    print(f" Temperatura  : {TEMPERATURE}")

    print("=" * 62)


def run_lms(
    *args: str,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:

    cmd = ["lms", *args]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0 and not allow_failure:

        details = (
            result.stderr
            or result.stdout
        ).strip()

        raise RuntimeError(
            f"Falha ao executar: {' '.join(cmd)}\n"
            f"Código: {result.returncode}\n"
            f"{details}"
        )

    return result


def api_get(
    path: str,
    timeout: float = 10.0,
) -> dict:

    request = Request(
        f"{API_BASE}{path}",
        method="GET",
    )

    with urlopen(
        request,
        timeout=timeout,
    ) as response:

        return json.loads(
            response.read().decode("utf-8")
        )


def api_post(
    path: str,
    payload: dict,
    timeout: float = 900.0,
) -> dict:

    body = json.dumps(
        payload
    ).encode("utf-8")

    request = Request(
        f"{API_BASE}{path}",
        data=body,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    with urlopen(
        request,
        timeout=timeout,
    ) as response:

        return json.loads(
            response.read().decode("utf-8")
        )


def extract_message(
    response: dict,
) -> str:

    chunks = []

    for item in response.get(
        "output",
        [],
    ):

        if (
            item.get("type") == "message"
            and item.get("content")
        ):

            chunks.append(
                item["content"]
            )

    return "\n".join(chunks)


def save_result(
    record: dict,
) -> Path:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    path = RESULTS_DIR / (
        f"smoke_gpu"
        f"{GPU_LAYERS_EXPECTED}_"
        f"{stamp}.json"
    )

    path.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def main() -> int:

    banner()

    print(
        "\n[1/5] "
        "Verificando o servidor "
        "do LM Studio..."
    )

    try:

        api_get(
            "/api/v1/models"
        )

    except (
        URLError,
        HTTPError,
        TimeoutError,
    ) as exc:

        print(
            f"ERRO: não consegui acessar "
            f"{API_BASE}: {exc}"
        )

        print(
            "Confirme com: "
            "lms server status"
        )

        return 1

    print("      OK")

    print(
        "\n[2/5] "
        "Limpando modelos carregados..."
    )

    run_lms(
        "unload",
        "--all",
        allow_failure=True,
    )

    print("      OK")

    print(
        "\n[3/5] "
        "Carregando Qwen com "
        "~4/32 camadas na GPU..."
    )

    load = run_lms(
        "load",
        MODEL_KEY,
        "--gpu",
        str(GPU_RATIO),
        "--context-length",
        str(CONTEXT_LENGTH),
        "--identifier",
        INSTANCE_ID,
    )

    print("      OK")

    ps = run_lms(
        "ps"
    ).stdout.strip()

    print(
        "\nEstado após "
        "o carregamento:"
    )

    print(ps)

    print(
        "\nConfiguração pronta."
    )

    input(
        "Pressione ENTER para "
        "iniciar a inferência "
        "de teste... "
    )

    payload = {

        "model":
            INSTANCE_ID,

        "input":
            PROMPT,

        "reasoning":
            "off",

        "temperature":
            TEMPERATURE,

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,

        "store":
            False,
    }

    print(
        "\n[4/5] Inferência "
        "em andamento... "
        "(sem streaming)"
    )

    start = time.perf_counter()

    try:

        response = api_post(
            "/api/v1/chat",
            payload,
        )

    except Exception:

        run_lms(
            "unload",
            "--all",
            allow_failure=True,
        )

        raise

    wall_time = (
        time.perf_counter()
        - start
    )

    stats = response.get(
        "stats",
        {},
    )

    message = extract_message(
        response
    )

    record = {

        "timestamp_local":
            datetime.now()
            .astimezone()
            .isoformat(),

        "benchmark":
            "gpu_smoke",

        "prompt_id":
            PROMPT_ID,

        "prompt":
            PROMPT,

        "model_key":
            MODEL_KEY,

        "instance_id":
            INSTANCE_ID,

        "configuration": {

            "context_length":
                CONTEXT_LENGTH,

            "gpu_ratio":
                GPU_RATIO,

            "gpu_layers_expected":
                GPU_LAYERS_EXPECTED,

            "cpu_threads_target":
                CPU_THREADS_TARGET,

            "reasoning":
                "off",

            "temperature":
                TEMPERATURE,

            "max_output_tokens":
                MAX_OUTPUT_TOKENS,
        },

        "stats":
            stats,

        "wall_time_seconds":
            wall_time,

        "response_text":
            message,

        "lms_ps_after_load":
            ps,

        "lms_load_stdout":
            load.stdout.strip(),
    }

    result_path = save_result(
        record
    )

    print("\nResultado:")

    print(
        "  tokens/s          : "
        f"{stats.get('tokens_per_second', 'n/d')}"
    )

    print(
        "  TTFT              : "
        f"{stats.get('time_to_first_token_seconds', 'n/d')} s"
    )

    print(
        "  input tokens      : "
        f"{stats.get('input_tokens', 'n/d')}"
    )

    print(
        "  output tokens     : "
        f"{stats.get('total_output_tokens', 'n/d')}"
    )

    print(
        "  reasoning tokens  : "
        f"{stats.get('reasoning_output_tokens', 'n/d')}"
    )

    print(
        "  wall time         : "
        f"{wall_time:.3f} s"
    )

    print(
        "\nResposta do modelo:"
    )

    print("-" * 62)

    print(message)

    print("-" * 62)

    print(
        "\nResultado salvo em:\n"
        f"  {result_path}"
    )

    print(
        "\n[5/5] "
        "Descarregando modelo..."
    )

    run_lms(
        "unload",
        "--all",
        allow_failure=True,
    )

    print("      OK")

    return 0


if __name__ == "__main__":

    try:

        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\n\nInterrompido "
            "pelo usuário."
        )

        run_lms(
            "unload",
            "--all",
            allow_failure=True,
        )

        raise SystemExit(130)

    except Exception as exc:

        print(
            f"\nERRO: {exc}",
            file=sys.stderr,
        )

        run_lms(
            "unload",
            "--all",
            allow_failure=True,
        )

        raise SystemExit(1)