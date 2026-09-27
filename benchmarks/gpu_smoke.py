from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE = "http://127.0.0.1:1234"

MODEL_KEY = "qwen/qwen3.5-9b"
EXPECTED_VARIANT = "qwen/qwen3.5-9b@q4_k_m"
INSTANCE_ID = "localia-bench"

CONTEXT_LENGTH = 4096
GPU_RATIO = 0.125
GPU_LAYERS_EXPECTED = 4
TOTAL_MODEL_LAYERS = 32

# Metadata only: current REST/CLI path used here does not expose/verify CPU threads.
CPU_THREADS_TARGET = 8

EXPECTED_EVAL_BATCH_SIZE = 2048
EXPECTED_PARALLEL = 1
EXPECTED_FLASH_ATTENTION = False
EXPECTED_KV_CACHE_GPU_OFFLOAD = False

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 320
REASONING = "off"

PROMPT_ID = "A"
PROMPT = (
    "Explique em aproximadamente 150 palavras o que é uma API REST "
    "e cite suas principais características."
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

LMS_TIMEOUT_SECONDS = 180.0
HTTP_TIMEOUT_SECONDS = 900.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def print_banner() -> None:
    print("=" * 68)
    print(" LocalIA — GPU Smoke Benchmark")
    print("=" * 68)
    print(f" Modelo             : {MODEL_KEY}")
    print(f" Variante esperada  : {EXPECTED_VARIANT}")
    print(f" Instância          : {INSTANCE_ID}")
    print(f" Contexto           : {CONTEXT_LENGTH}")
    print(
        f" GPU offload        : {GPU_RATIO:.3f} "
        f"(~{GPU_LAYERS_EXPECTED}/{TOTAL_MODEL_LAYERS} camadas)"
    )
    print(f" CPU threads alvo   : {CPU_THREADS_TARGET} (metadata; não verificado)")
    print(f" Eval batch esperado: {EXPECTED_EVAL_BATCH_SIZE}")
    print(f" Parallel esperado  : {EXPECTED_PARALLEL}")
    print(f" Flash Attention    : {EXPECTED_FLASH_ATTENTION}")
    print(f" KV cache -> GPU    : {EXPECTED_KV_CACHE_GPU_OFFLOAD}")
    print(f" Think/Reasoning    : {REASONING.upper()}")
    print(f" Temperatura        : {TEMPERATURE}")
    print("=" * 68)


def run_lms(
    *args: str,
    allow_failure: bool = False,
    timeout: float = LMS_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run `lms` without allowing hidden interactive prompts."""
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
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_server() -> None:
    try:
        api_get("/api/v1/models", timeout=5.0)
        return
    except (URLError, HTTPError, TimeoutError):
        pass

    print("      Servidor não respondeu. Tentando `lms server start`...")
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
            "Abra o LM Studio e tente novamente.\n"
            f"{details}"
        )

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            api_get("/api/v1/models", timeout=2.0)
            return
        except (URLError, HTTPError, TimeoutError):
            time.sleep(0.5)

    raise RuntimeError(
        "O servidor foi iniciado, mas /api/v1/models não respondeu em 15s."
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
        if model.get("modelKey") == MODEL_KEY:
            selected = model.get("selectedVariant")
            if selected != EXPECTED_VARIANT:
                raise RuntimeError(
                    "Variante incorreta selecionada.\n"
                    f"  Esperado: {EXPECTED_VARIANT}\n"
                    f"  Atual:    {selected}"
                )

            quant = model.get("quantization") or {}
            print(
                f"      OK — {selected} "
                f"({quant.get('name', 'quantização n/d')})"
            )
            return model

    raise RuntimeError(
        f"Modelo não encontrado pelo LM Studio: {MODEL_KEY}"
    )


def unload_all() -> None:
    run_lms(
        "unload",
        "--all",
        allow_failure=True,
        timeout=60.0,
    )


def load_model() -> tuple[subprocess.CompletedProcess[str], float]:
    start = time.perf_counter()

    result = run_lms(
        "load",
        MODEL_KEY,
        "--gpu",
        str(GPU_RATIO),
        "--context-length",
        str(CONTEXT_LENGTH),
        "--identifier",
        INSTANCE_ID,
        timeout=LMS_TIMEOUT_SECONDS,
    )

    elapsed = time.perf_counter() - start
    return result, elapsed


def find_loaded_instance(
    models_response: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for model in models_response.get("models", []):
        if model.get("key") != MODEL_KEY:
            continue

        for instance in model.get("loaded_instances", []):
            if instance.get("id") == INSTANCE_ID:
                return model, instance

    return None, None


def audit_loaded_config(config: dict[str, Any]) -> list[str]:
    checks = [
        ("context_length", CONTEXT_LENGTH),
        ("eval_batch_size", EXPECTED_EVAL_BATCH_SIZE),
        ("parallel", EXPECTED_PARALLEL),
        ("flash_attention", EXPECTED_FLASH_ATTENTION),
        ("offload_kv_cache_to_gpu", EXPECTED_KV_CACHE_GPU_OFFLOAD),
    ]

    issues: list[str] = []

    for key, expected in checks:
        if key not in config:
            issues.append(
                f"{key}: não exposto pela API; esperado={expected!r}"
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
        if item.get("type") == "message":
            content = item.get("content")
            if isinstance(content, str):
                chunks.append(content)

    return "\n".join(chunks)


def save_json(record: dict[str, Any], prefix: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{prefix}_{stamp}.json"

    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return path


def main() -> int:
    print_banner()

    loaded = False
    diagnostics: dict[str, Any] = {
        "started_at": now_iso(),
        "benchmark": "gpu_smoke",
        "script_version": "2.0",
        "model_key": MODEL_KEY,
        "expected_variant": EXPECTED_VARIANT,
        "configuration_requested": {
            "context_length": CONTEXT_LENGTH,
            "gpu_ratio": GPU_RATIO,
            "gpu_layers_expected": GPU_LAYERS_EXPECTED,
            "total_model_layers": TOTAL_MODEL_LAYERS,
            "cpu_threads_target_metadata_only": CPU_THREADS_TARGET,
            "expected_eval_batch_size": EXPECTED_EVAL_BATCH_SIZE,
            "expected_parallel": EXPECTED_PARALLEL,
            "expected_flash_attention": EXPECTED_FLASH_ATTENTION,
            "expected_offload_kv_cache_to_gpu": EXPECTED_KV_CACHE_GPU_OFFLOAD,
            "reasoning": REASONING,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    }

    try:
        print("\n[1/7] Verificando/iniciando servidor do LM Studio...")
        ensure_server()
        print("      OK")

        print("\n[2/7] Confirmando a variante Q4_K_M...")
        model_inventory = verify_selected_variant()
        diagnostics["model_inventory"] = model_inventory

        print("\n[3/7] Descarregando modelos previamente carregados...")
        unload_all()
        print("      OK")

        print(
            "\n[4/7] Carregando Qwen com "
            f"GPU ratio={GPU_RATIO} "
            f"(~{GPU_LAYERS_EXPECTED}/{TOTAL_MODEL_LAYERS} camadas)..."
        )

        load_result, load_wall_time = load_model()
        loaded = True

        diagnostics["load_wall_time_seconds"] = load_wall_time
        diagnostics["lms_load_stdout"] = load_result.stdout.strip()
        diagnostics["lms_load_stderr"] = load_result.stderr.strip()

        print(f"      OK — {load_wall_time:.2f}s")

        ps_result = run_lms("ps")
        ps_text = ps_result.stdout.strip()
        diagnostics["lms_ps_after_load"] = ps_text

        print("\nEstado reportado por `lms ps`:")
        print("-" * 68)
        print(ps_text)
        print("-" * 68)

        models_response = api_get("/api/v1/models")
        loaded_model, loaded_instance = find_loaded_instance(models_response)

        if loaded_instance is None:
            diagnostics["models_response_after_load"] = models_response
            raise RuntimeError(
                f"A instância `{INSTANCE_ID}` não apareceu em /api/v1/models."
            )

        loaded_config = loaded_instance.get("config") or {}
        diagnostics["loaded_model_from_api"] = loaded_model
        diagnostics["loaded_instance_from_api"] = loaded_instance
        diagnostics["loaded_config"] = loaded_config

        print("\n[5/7] Auditando configuração efetivamente carregada...")
        print("\nConfiguração reportada pela API:")
        print(json.dumps(loaded_config, indent=2, ensure_ascii=False))

        issues = audit_loaded_config(loaded_config)
        diagnostics["config_audit_issues"] = issues
        diagnostics["config_audit_passed"] = len(issues) == 0

        if issues:
            print("\n      ATENÇÃO — divergências/limitações encontradas:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print("\n      OK — campos expostos pela API batem com o baseline.")

        print(
            "\n      Observação: CPU threads=8 e Physical Batch Size=512 "
            "não são confirmados por este endpoint REST."
        )

        print("\n[6/7] Executando Prompt A (sem streaming)...")
        print("      Durante a inferência o Python ficará praticamente ocioso.")

        payload = {
            "model": INSTANCE_ID,
            "input": PROMPT,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "reasoning": REASONING,
            "context_length": CONTEXT_LENGTH,
            "store": False,
            "stream": False,
        }

        inference_start = time.perf_counter()
        response = api_post("/api/v1/chat", payload)
        inference_wall_time = time.perf_counter() - inference_start

        stats = response.get("stats") or {}
        message = extract_message(response)
        reasoning_tokens = stats.get("reasoning_output_tokens")

        diagnostics["prompt_id"] = PROMPT_ID
        diagnostics["prompt"] = PROMPT
        diagnostics["request_payload"] = payload
        diagnostics["response"] = response
        diagnostics["stats"] = stats
        diagnostics["inference_wall_time_seconds"] = inference_wall_time
        diagnostics["reasoning_off_verified"] = reasoning_tokens == 0
        diagnostics["ended_at"] = now_iso()

        result_path = save_json(
            diagnostics,
            prefix=f"smoke_gpu{GPU_LAYERS_EXPECTED}",
        )

        print("\nResultado:")
        print(
            f"  tokens/s         : "
            f"{stats.get('tokens_per_second', 'n/d')}"
        )
        print(
            f"  TTFT             : "
            f"{stats.get('time_to_first_token_seconds', 'n/d')} s"
        )
        print(
            f"  input tokens     : "
            f"{stats.get('input_tokens', 'n/d')}"
        )
        print(
            f"  output tokens    : "
            f"{stats.get('total_output_tokens', 'n/d')}"
        )
        print(
            f"  reasoning tokens : "
            f"{stats.get('reasoning_output_tokens', 'n/d')}"
        )
        print(
            f"  wall time        : "
            f"{inference_wall_time:.3f} s"
        )

        if reasoning_tokens == 0:
            print("  Think/Reasoning  : OFF confirmado ✓")
        else:
            print(
                "  Think/Reasoning  : ATENÇÃO — "
                f"{reasoning_tokens!r} tokens de reasoning"
            )

        print("\nResposta do modelo:")
        print("-" * 68)
        print(message or "(sem texto de resposta)")
        print("-" * 68)

        print(f"\nJSON salvo em:\n  {result_path}")
        print("\n[7/7] Smoke test concluído.")
        return 0

    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário.")
        diagnostics["interrupted"] = True
        diagnostics["ended_at"] = now_iso()

        try:
            path = save_json(diagnostics, prefix="smoke_interrupted")
            print(f"Diagnóstico parcial salvo em:\n  {path}")
        except Exception:
            pass

        return 130

    except Exception as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)

        diagnostics["error"] = str(exc)
        diagnostics["ended_at"] = now_iso()

        try:
            path = save_json(diagnostics, prefix="smoke_error")
            print(f"\nDiagnóstico salvo em:\n  {path}")
        except Exception:
            pass

        return 1

    finally:
        if loaded:
            print("\n[LIMPEZA] Descarregando modelo...")
            try:
                unload_all()
                print("          OK")
            except Exception as cleanup_exc:
                print(
                    f"          AVISO: falha no unload final: {cleanup_exc}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
