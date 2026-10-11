from __future__ import annotations

"""
LocalIA — CPU Thread Pool Load-Time Probe
==========================================

Objetivo:
    Confirmar que o script consegue alterar de verdade o
    "CPU Thread Pool Size" do LM Studio em tempo de carregamento.

Por que este probe existe:
    - cpuThreads é parâmetro de PREDIÇÃO.
    - CPU Thread Pool Size é parâmetro de LOAD.
    - `lms load`, REST /api/v1/models/load e o Python SDK público atual
      não expõem diretamente cpuThreadPoolSize.
    - O LM Studio, porém, usa os Per-model Defaults também em `lms load`.

Estratégia:
    1. Localizar o arquivo JSON de Per-model Defaults do Qwen3.5-9B.
    2. Fazer backup byte-a-byte do arquivo original.
    3. Alterar somente o campo existente de CPU Thread Pool Size.
    4. Carregar o modelo.
    5. Ler o server log e confirmar:
           llama threadpool init, n_threads = X
    6. Testar 4, 6 e 8 em CPU AVX2 e Vulkan.
    7. Restaurar EXATAMENTE o arquivo original no final.

IMPORTANTE:
    O script NÃO inventa uma estrutura de config nova.
    Ele só altera um campo cpuThreadPoolSize que já exista no arquivo.
    Como você já alterou esse valor pela interface do LM Studio, o campo
    deve existir. Se não existir ou houver ambiguidade, o script para.

Requisitos:
    - LM Studio aberto.
    - llama.cpp log level = Debug.
    - Qwen3.5-9B Q4_K_M instalado.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


MODEL_KEY = "qwen/qwen3.5-9b"
INSTANCE_ID = "localia-threadpool-probe"
CONTEXT_LENGTH = 4096

RUNTIMES = {
    "cpu": "llama.cpp-win-x86_64-avx2@2.46.0",
    "vulkan": "llama.cpp-win-x86_64-vulkan-avx2@2.46.0",
}

THREAD_POOLS = [4, 6, 8]

CONFIG_ROOT = (
    Path.home()
    / ".lmstudio"
    / ".internal"
    / "user-concrete-model-default-config"
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"

CONFIG_WATCH_SETTLE_SECONDS = 2.0
AFTER_UNLOAD_SECONDS = 1.0
AFTER_LOAD_LOG_SECONDS = 1.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_lms(
    *args: str,
    allow_failure: bool = False,
    timeout: float = 180.0,
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
        raise RuntimeError("`lms` não foi encontrado no PATH.") from exc

    if result.returncode != 0 and not allow_failure:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Falha ao executar:\n  {' '.join(cmd)}\n"
            f"Código: {result.returncode}\n"
            f"{details or '(sem detalhes)'}"
        )

    return result


def unload_all() -> None:
    run_lms(
        "unload",
        "--all",
        allow_failure=True,
        timeout=60.0,
    )


def runtime_ls_text() -> str:
    return run_lms("runtime", "ls").stdout


def select_runtime(alias: str) -> str:
    unload_all()

    run_lms(
        "runtime",
        "select",
        alias,
        timeout=60.0,
    )

    text = runtime_ls_text()

    for line in text.splitlines():
        if "✓" in line and "GGUF" in line:
            if alias not in line:
                raise RuntimeError(
                    "Runtime selecionado não corresponde ao esperado.\n"
                    f"Esperado: {alias}\n"
                    f"Atual:    {line.strip()}"
                )
            return line.strip()

    raise RuntimeError(
        "Não consegui confirmar o runtime selecionado."
    )


def start_server_log_capture(
    log_path: Path,
) -> tuple[subprocess.Popen[Any] | None, Any | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handle = log_path.open(
        "w",
        encoding="utf-8",
        errors="replace",
    )

    try:
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
    except Exception:
        handle.close()
        raise

    time.sleep(1.0)

    return proc, handle


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

            data = obj.get("data")

            if not isinstance(data, dict):
                continue

            content = data.get("content")

            if isinstance(content, str):
                chunks.append(content)

    return "\n".join(chunks)


def parse_threadpool_from_log(log_path: Path) -> int | None:
    text = extract_server_log_content(log_path)

    matches = list(
        re.finditer(
            r"llama\s+threadpool\s+init,\s*n_threads\s*=\s*(\d+)",
            text,
            flags=re.IGNORECASE,
        )
    )

    if not matches:
        return None

    return int(matches[-1].group(1))


def count_threadpool_occurrences(obj: Any) -> int:
    count = 0

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {
                "cpuThreadPoolSize",
                "llm.load.llama.cpuThreadPoolSize",
            }:
                count += 1

            if (
                key == "key"
                and value == "llm.load.llama.cpuThreadPoolSize"
            ):
                count += 1

            count += count_threadpool_occurrences(value)

    elif isinstance(obj, list):
        for item in obj:
            count += count_threadpool_occurrences(item)

    return count


def find_model_config_file() -> Path:
    if not CONFIG_ROOT.exists():
        raise RuntimeError(
            "Diretório de configs do LM Studio não encontrado:\n"
            f"  {CONFIG_ROOT}"
        )

    candidates: list[
        tuple[int, Path, int]
    ] = []

    for path in CONFIG_ROOT.rglob("*.json"):
        try:
            raw = path.read_text(
                encoding="utf-8"
            )
            data = json.loads(raw)
        except Exception:
            continue

        occurrences = count_threadpool_occurrences(data)

        if occurrences <= 0:
            continue

        normalized_path = str(path).lower().replace("\\", "/")
        raw_lower = raw.lower()

        score = 0

        if "qwen" in normalized_path:
            score += 20

        if "qwen3.5-9b" in normalized_path:
            score += 100

        if "qwen3.5-9b" in raw_lower:
            score += 50

        if "qwen/qwen3.5-9b" in raw_lower:
            score += 100

        candidates.append(
            (
                score,
                path,
                occurrences,
            )
        )

    if not candidates:
        raise RuntimeError(
            "Não encontrei nenhum JSON de Per-model Defaults "
            "contendo cpuThreadPoolSize.\n\n"
            "Como você já alterou o CPU Thread Pool Size pela interface, "
            "isso é inesperado. Não vou criar uma estrutura de config "
            "sem saber o formato usado por sua versão do LM Studio."
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            str(item[1]),
        ),
        reverse=True,
    )

    best_score = candidates[0][0]

    best = [
        item
        for item in candidates
        if item[0] == best_score
    ]

    if len(best) != 1:
        details = "\n".join(
            f"  score={score:3d} occurrences={occ} {path}"
            for score, path, occ in candidates[:10]
        )

        raise RuntimeError(
            "Encontrei mais de um arquivo candidato e não vou escolher "
            "automaticamente.\n\n"
            f"Candidatos:\n{details}"
        )

    return best[0][1]


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
                found.append(
                    {
                        "path": current,
                        "value": value,
                        "style": "direct-key",
                    }
                )

            if (
                key == "key"
                and value == "llm.load.llama.cpuThreadPoolSize"
            ):
                found.append(
                    {
                        "path": path,
                        "value": obj.get("value"),
                        "style": "field-entry",
                    }
                )

            found.extend(
                collect_threadpool_values(
                    value,
                    current,
                )
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
    """
    Substitui TODAS as ocorrências válidas encontradas no mesmo JSON.
    Isso evita deixar um override antigo em outra camada do arquivo.

    Retorna quantas ocorrências foram alteradas.
    """

    changed = 0

    if isinstance(obj, dict):
        # Forma 1:
        # {"cpuThreadPoolSize": 2}
        for key in list(obj.keys()):
            if key in {
                "cpuThreadPoolSize",
                "llm.load.llama.cpuThreadPoolSize",
            }:
                obj[key] = int(new_value)
                changed += 1

        # Forma 2:
        # {"key": "llm.load.llama.cpuThreadPoolSize", "value": 2}
        if (
            obj.get("key")
            == "llm.load.llama.cpuThreadPoolSize"
        ):
            obj["value"] = int(new_value)
            changed += 1

        for key, value in list(obj.items()):
            # Não recursar no scalar que acabamos de alterar.
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


def atomic_write_json(
    path: Path,
    data: Any,
) -> None:
    temp = path.with_suffix(
        path.suffix + ".localia.tmp"
    )

    temp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temp,
        path,
    )


def write_threadpool_to_config(
    config_path: Path,
    value: int,
) -> dict[str, Any]:
    raw = config_path.read_text(
        encoding="utf-8"
    )

    data = json.loads(raw)

    before = collect_threadpool_values(
        data
    )

    changed = set_threadpool_value(
        data,
        value,
    )

    if changed <= 0:
        raise RuntimeError(
            "O arquivo foi encontrado, mas não consegui alterar "
            "cpuThreadPoolSize."
        )

    atomic_write_json(
        config_path,
        data,
    )

    # Reabrir para verificar persistência.
    reloaded = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    after = collect_threadpool_values(
        reloaded
    )

    values_after = {
        item.get("value")
        for item in after
    }

    if values_after != {int(value)}:
        raise RuntimeError(
            "Após salvar, o arquivo não ficou consistente.\n"
            f"Valores encontrados: {sorted(values_after, key=str)}"
        )

    return {
        "changed_occurrences": changed,
        "before": before,
        "after": after,
    }


def load_and_measure(
    *,
    results_dir: Path,
    backend: str,
    runtime_alias: str,
    requested_pool: int,
) -> dict[str, Any]:
    label = (
        f"{backend.upper()}-T{requested_pool}"
    )

    unload_all()

    selected = select_runtime(
        runtime_alias
    )

    log_path = (
        results_dir
        / "logs"
        / f"{label}.jsonl"
    )

    log_proc = None
    log_handle = None

    start = time.perf_counter()

    try:
        log_proc, log_handle = (
            start_server_log_capture(
                log_path
            )
        )

        result = run_lms(
            "load",
            MODEL_KEY,
            "--gpu",
            "off",
            "--context-length",
            str(CONTEXT_LENGTH),
            "--identifier",
            INSTANCE_ID,
            timeout=240.0,
        )

        time.sleep(
            AFTER_LOAD_LOG_SECONDS
        )

    finally:
        stop_log_capture(
            log_proc,
            log_handle,
        )

    load_seconds = (
        time.perf_counter()
        - start
    )

    observed = parse_threadpool_from_log(
        log_path
    )

    unload_all()

    return {
        "timestamp": now_iso(),
        "backend": backend,
        "runtime_alias": runtime_alias,
        "selected_runtime_line": selected,
        "requested_thread_pool": requested_pool,
        "observed_thread_pool": observed,
        "match": observed == requested_pool,
        "load_wall_time_seconds": load_seconds,
        "log_path": str(log_path),
        "lms_stdout": result.stdout.strip(),
        "lms_stderr": result.stderr.strip(),
    }


def main() -> int:
    print("\n[PRECHECK] Localizando Per-model Defaults do Qwen...")

    config_path = find_model_config_file()

    print(
        "  ✓ arquivo encontrado:\n"
        f"    {config_path}"
    )

    original_bytes = config_path.read_bytes()

    try:
        original_json = json.loads(
            original_bytes.decode("utf-8")
        )
    except Exception as exc:
        raise RuntimeError(
            "O arquivo encontrado não é JSON UTF-8 válido."
        ) from exc

    original_values = collect_threadpool_values(
        original_json
    )

    print(
        "  ✓ CPU Thread Pool Size atual:"
    )

    for item in original_values:
        print(
            f"    {item['path']} = "
            f"{item['value']!r}"
        )

    results_dir = (
        RESULTS_ROOT
        / f"threadpool_probe_{timestamp_slug()}"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    backup_path = (
        results_dir
        / "original_model_default_config.json"
    )

    backup_path.write_bytes(
        original_bytes
    )

    report: dict[str, Any] = {
        "created_at": now_iso(),
        "model_key": MODEL_KEY,
        "config_path": str(config_path),
        "original_values": original_values,
        "tests": [],
        "restored_original_config": False,
    }

    report_path = (
        results_dir
        / "report.json"
    )

    try:
        print(
            "\nO probe vai alterar TEMPORARIAMENTE o Per-model Default "
            "e restaurá-lo byte-a-byte no final."
        )
        print(
            "Nenhuma inferência será executada; apenas loads curtos."
        )

        answer = input(
            "\nPressione ENTER para iniciar "
            "ou digite 'cancelar': "
        ).strip().lower()

        if answer in {
            "cancelar",
            "cancel",
            "n",
            "nao",
            "não",
        }:
            print("Probe cancelado.")
            return 0

        # Testar os dois backends para garantir que a configuração
        # load-time é respeitada de forma idêntica.
        for backend, runtime_alias in RUNTIMES.items():
            for requested_pool in THREAD_POOLS:
                print(
                    "\n" + "=" * 78
                )
                print(
                    f" {backend.upper()} | "
                    f"CPU Thread Pool Size = "
                    f"{requested_pool}"
                )
                print(
                    "=" * 78
                )

                edit_info = (
                    write_threadpool_to_config(
                        config_path,
                        requested_pool,
                    )
                )

                print(
                    f"[CONFIG] Salvo no Per-model Default: "
                    f"{requested_pool}"
                )
                print(
                    f"[CONFIG] Ocorrências alteradas: "
                    f"{edit_info['changed_occurrences']}"
                )

                # Dar tempo ao file watcher do LM Studio.
                time.sleep(
                    CONFIG_WATCH_SETTLE_SECONDS
                )

                result = load_and_measure(
                    results_dir=results_dir,
                    backend=backend,
                    runtime_alias=runtime_alias,
                    requested_pool=requested_pool,
                )

                result[
                    "config_edit"
                ] = edit_info

                report[
                    "tests"
                ].append(
                    result
                )

                report_path.write_text(
                    json.dumps(
                        report,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                observed = result.get(
                    "observed_thread_pool"
                )

                if result.get(
                    "match"
                ):
                    print(
                        f"[PASS] solicitado="
                        f"{requested_pool} | "
                        f"threadpool real="
                        f"{observed}"
                    )
                else:
                    print(
                        f"[FAIL] solicitado="
                        f"{requested_pool} | "
                        f"threadpool real="
                        f"{observed}"
                    )

                    raise RuntimeError(
                        "O Per-model Default foi alterado, "
                        "mas o runtime não carregou com o valor esperado.\n"
                        "O probe foi interrompido."
                    )

        print(
            "\n" + "=" * 78
        )
        print(
            " TODOS OS TESTES PASSARAM"
        )
        print(
            "=" * 78
        )

        print(
            "\nA estratégia correta para o benchmark será:"
        )
        print(
            "  editar temporariamente cpuThreadPoolSize"
        )
        print(
            "  → reload do modelo"
        )
        print(
            "  → confirmar `llama threadpool init, n_threads = X`"
        )
        print(
            "  → usar cpuThreads=X na inferência"
        )
        print(
            "  → medir uso real no nível do SO"
        )

        return 0

    finally:
        print(
            "\n[RESTORE] Restaurando o Per-model Default original..."
        )

        try:
            # Restaurar EXATAMENTE os bytes originais para não alterar
            # formatação, metadata ou outros overrides do usuário.
            config_path.write_bytes(
                original_bytes
            )

            time.sleep(
                CONFIG_WATCH_SETTLE_SECONDS
            )

            report[
                "restored_original_config"
            ] = True

            print(
                "[RESTORE] ✓ Configuração original restaurada."
            )

        except Exception as exc:
            report[
                "restore_error"
            ] = str(exc)

            print(
                "[RESTORE] ✗ ERRO ao restaurar automaticamente."
            )
            print(
                "Use o backup salvo em:"
            )
            print(
                f"  {backup_path}"
            )

        try:
            unload_all()
        except Exception:
            pass

        report[
            "finished_at"
        ] = now_iso()

        report_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"\nRelatório:\n  {report_path}"
        )
        print(
            f"Backup original:\n  {backup_path}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
