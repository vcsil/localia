from __future__ import annotations

import re
from typing import Iterable

from .contracts import Capability, JevRequest, normalize_capabilities


# High-precision deterministic evidence. These rules do not replace semantic
# classification; they only add operational facts that should not vary across
# model runs once the user has stated them explicitly.
STRUCTURED_OUTPUT_PATTERNS = (
    re.compile(
        r"\b(?:responda|retorne|devolva|gere|produza|formate)\b.{0,32}"
        r"\b(?:em|como|somente|apenas)\s+(?:json|yaml|xml|csv)\b"
    ),
    re.compile(
        r"\b(?:retorne|devolva)\s+(?:um\s+)?(?:json|yaml|xml|csv)\b"
    ),
)

SUMMARIZATION_PATTERNS = (
    re.compile(r"\bresuma\b"),
    re.compile(r"\bresumir\b"),
    re.compile(r"\bsummar(?:ize|ise|ization|isation)\b"),
)

CLASSIFICATION_PATTERNS = (
    re.compile(r"\bclassifi(?:que|car|cação|cacao|cation|y)\b"),
)

EXTRACTION_PATTERNS = (
    re.compile(r"\bextraia\b"),
    re.compile(r"\bextrair\b"),
    re.compile(r"\bextract(?:ion)?\b"),
)

REPOSITORY_CONTEXT_PATTERN = re.compile(
    r"\b(?:repositório|repositorio|repository|codebase|projeto|project)\b"
)
REPOSITORY_READ_PATTERN = re.compile(
    r"\b(?:abra|analise|inspecione|procure|busque|leia|liste)\b"
)
REPOSITORY_EDIT_PATTERN = re.compile(
    r"\b(?:corrija|corrigir|ajuste|ajustar|edite|editar|modifique|modificar|"
    r"refatore|refatorar|implemente|implementar|renomeie|renomear|apague|"
    r"exclua|remova|adicione|adicionar)\b"
)
NEGATED_REPOSITORY_EDIT_PATTERNS = (
    re.compile(
        r"\bsem\s+(?:alterar|modificar|editar|mexer\s+em)"
        r"(?:\s+(?:nada|arquivos?|o\s+código|o\s+codigo))?\b"
    ),
    re.compile(
        r"\bnão\s+(?:altere|alterar|modifique|modificar|edite|editar|"
        r"mexa|mexer)\b"
    ),
    re.compile(r"\bread[- ]only\b"),
    re.compile(r"\bwithout\s+(?:changing|modifying|editing)\b"),
)

EXECUTION_PATTERNS = (
    re.compile(r"\b(?:rode|rodar|execute|executar)\s+(?:os\s+)?testes\b"),
    re.compile(r"\b(?:rode|rodar|execute|executar)\s+(?:a\s+)?suíte\b"),
    re.compile(r"\b(?:rode|rodar|execute|executar)\s+(?:the\s+)?suite\b"),
    re.compile(r"\b(?:rode|rodar|execute|executar)\s+pytest\b"),
    re.compile(r"\brun\s+(?:the\s+)?tests?\b"),
    re.compile(r"\b(?:usando|via|com)\s+git\b"),
)
NEGATED_EXECUTION_PATTERNS = (
    re.compile(r"\bnão\s+(?:rode|rodar|execute|executar)\b(?:\s+nada)?"),
    re.compile(r"\bsem\s+(?:rodar|executar)\b"),
    re.compile(r"\bdo\s+not\s+run\b"),
)

CODE_REASONING_PATTERNS = (
    re.compile(r"\bpytest\b"),
    re.compile(r"\basyncio\b"),
    re.compile(r"\bthreads?\b"),
    re.compile(r"\blist\s+comprehension\b"),
    re.compile(r"\bfixture\b"),
    re.compile(
        r"\b(?:typeerror|assertionerror|nameerror|valueerror|traceback)\b"
    ),
    re.compile(r"\b(?:código|codigo|code)\b"),
    re.compile(r"\b(?:função|funcao|function)\b"),
    re.compile(r"\bdef\s+[A-Za-z_]\w*\s*\("),
)

FUNCTIONAL_CAPABILITIES = frozenset(
    {
        Capability.GENERAL,
        Capability.CLASSIFICATION,
        Capability.EXTRACTION,
        Capability.SUMMARIZATION,
        Capability.CODE_REASONING,
        Capability.REPOSITORY_READ,
        Capability.REPOSITORY_EDIT,
        Capability.CODE_EXECUTION,
        Capability.REASONING,
    }
)


def _matches_any(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)


def _scrub_patterns(
    text: str,
    patterns: Iterable[re.Pattern[str]],
) -> str:
    scrubbed = text
    for pattern in patterns:
        scrubbed = pattern.sub(" ", scrubbed)
    return scrubbed


def _has_positive_execution_request(text: str) -> bool:
    return _matches_any(
        _scrub_patterns(text, NEGATED_EXECUTION_PATTERNS),
        EXECUTION_PATTERNS,
    )


def _has_positive_repository_edit_request(text: str) -> bool:
    return (
        REPOSITORY_EDIT_PATTERN.search(
            _scrub_patterns(text, NEGATED_REPOSITORY_EDIT_PATTERNS)
        )
        is not None
    )


def refine_required_capabilities(
    request: JevRequest,
    capabilities: Iterable[Capability],
) -> frozenset[Capability]:
    """Add high-confidence operational facts implied by explicit task text."""

    refined = set(normalize_capabilities(capabilities))
    text = request.task.casefold()

    if _matches_any(text, SUMMARIZATION_PATTERNS):
        refined.add(Capability.SUMMARIZATION)
    if _matches_any(text, CLASSIFICATION_PATTERNS):
        refined.add(Capability.CLASSIFICATION)
    if _matches_any(text, EXTRACTION_PATTERNS):
        refined.add(Capability.EXTRACTION)
    if _matches_any(text, STRUCTURED_OUTPUT_PATTERNS):
        refined.add(Capability.STRUCTURED_OUTPUT)

    has_repository_context = REPOSITORY_CONTEXT_PATTERN.search(text) is not None
    if has_repository_context and REPOSITORY_READ_PATTERN.search(text):
        refined.add(Capability.REPOSITORY_READ)

    has_negated_repository_edit = _matches_any(
        text,
        NEGATED_REPOSITORY_EDIT_PATTERNS,
    )
    has_positive_repository_edit = (
        has_repository_context
        and _has_positive_repository_edit_request(text)
    )
    if has_positive_repository_edit:
        refined.add(Capability.REPOSITORY_EDIT)
    elif has_negated_repository_edit:
        refined.discard(Capability.REPOSITORY_EDIT)

    has_positive_execution = _has_positive_execution_request(text)
    if has_positive_execution:
        refined.add(Capability.CODE_EXECUTION)
    elif _matches_any(text, NEGATED_EXECUTION_PATTERNS):
        refined.discard(Capability.CODE_EXECUTION)

    # Chat-only code work remains reasoning, not repository/tool work.
    if (
        not has_repository_context
        and _matches_any(text, CODE_REASONING_PATTERNS)
    ):
        refined.add(Capability.CODE_REASONING)

    # Modifiers such as long_context/structured_output cannot replace the
    # underlying thing the user is asking the route to do.
    if not (refined & FUNCTIONAL_CAPABILITIES):
        if _matches_any(text, CODE_REASONING_PATTERNS):
            refined.add(Capability.CODE_REASONING)
        else:
            refined.add(Capability.GENERAL)

    return normalize_capabilities(refined)


def canonical_task_type(
    model_task_type: str,
    required_capabilities: Iterable[Capability],
) -> str:
    """Stabilize task_type when operational capabilities are decisive."""

    capabilities = normalize_capabilities(required_capabilities)
    precedence = (
        (Capability.REPOSITORY_EDIT, "repository_edit"),
        (Capability.CODE_EXECUTION, "code_execution"),
        (Capability.REPOSITORY_READ, "repository_read"),
        (Capability.CODE_REASONING, "code_reasoning"),
        (Capability.SUMMARIZATION, "summarization"),
        (Capability.CLASSIFICATION, "classification"),
        (Capability.EXTRACTION, "extraction"),
        (Capability.REASONING, "reasoning"),
    )
    for capability, task_type in precedence:
        if capability in capabilities:
            return task_type

    if Capability.GENERAL in capabilities:
        return model_task_type.strip() or "general"
    if Capability.STRUCTURED_OUTPUT in capabilities:
        return "structured_output"
    if Capability.LONG_CONTEXT in capabilities:
        return "long_context"
    return model_task_type.strip() or "general"
