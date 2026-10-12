from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


QWEN_Q4_MODEL_KEY = "qwen/qwen3.5-9b@q4_k_m"
QWEN_BASE_MODEL_KEY = "qwen/qwen3.5-9b"
QWEN_ROUTER_API_IDENTIFIER = QWEN_Q4_MODEL_KEY

EXPECTED_QUANTIZATION = "Q4_K_M"
EXPECTED_Q4_SIZE_BYTES = 6_548_927_711
EXPECTED_Q4_SIZE_TOLERANCE_BYTES = 8 * 1024 * 1024
EXPECTED_CONTEXT_LENGTH = 4096
EXPECTED_CPU_THREADS = 8
DEFAULT_SERVER_BASE_URL = "http://127.0.0.1:1234"


class LMStudioUnavailableError(RuntimeError):
    pass


class LMStudioConfigurationError(RuntimeError):
    pass


def _value(obj: Any, *names: str) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _mapping(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_dict"):
        try:
            value = obj.to_dict()
            if isinstance(value, Mapping):
                return dict(value)
        except Exception:
            pass
    if hasattr(obj, "model_dump"):
        try:
            value = obj.model_dump()
            if isinstance(value, Mapping):
                return dict(value)
        except Exception:
            pass
    data = getattr(obj, "__dict__", None)
    return dict(data) if isinstance(data, dict) else {}


def _normalized_text(value: Any) -> str:
    return str(value or "").strip()


def _looks_like_q4_path(path: str) -> bool:
    normalized = path.upper().replace("-", "_")
    return "Q4_K_M" in normalized


def _size_matches_q4(size_bytes: Any) -> bool:
    try:
        size = int(size_bytes)
    except (TypeError, ValueError):
        return False
    return abs(size - EXPECTED_Q4_SIZE_BYTES) <= EXPECTED_Q4_SIZE_TOLERANCE_BYTES


def _identity_snapshot(info: Any) -> dict[str, Any]:
    return {
        "model_key": _value(info, "model_key", "modelKey"),
        "identifier": _value(info, "identifier"),
        "display_name": _value(info, "display_name", "displayName"),
        "path": _value(info, "path"),
        "size_bytes": _value(info, "size_bytes", "sizeBytes"),
        "context_length": _value(info, "context_length", "contextLength"),
        "format": _value(info, "format"),
    }


def _identity_evidence(info: Any) -> tuple[bool, tuple[str, ...]]:
    key = _normalized_text(_value(info, "model_key", "modelKey"))
    identifier = _normalized_text(_value(info, "identifier"))
    path = _normalized_text(_value(info, "path"))
    size_bytes = _value(info, "size_bytes", "sizeBytes")

    evidence: list[str] = []

    if key == QWEN_Q4_MODEL_KEY:
        evidence.append("exact_model_key")
    if _looks_like_q4_path(path):
        evidence.append("gguf_path_q4_k_m")
    if _size_matches_q4(size_bytes):
        evidence.append("benchmarked_q4_file_size")

    belongs_to_family = (
        key in {QWEN_Q4_MODEL_KEY, QWEN_BASE_MODEL_KEY}
        or identifier in {QWEN_Q4_MODEL_KEY, QWEN_BASE_MODEL_KEY}
        or "QWEN3.5-9B" in path.upper()
        or "QWEN3.5_9B" in path.upper()
    )
    return belongs_to_family and bool(evidence), tuple(evidence)


def _decode_complete_json_object(
    text: Any,
    *,
    channel: str,
) -> dict[str, Any] | None:
    """Return a mapping only when the entire channel is exactly one JSON object."""
    if not isinstance(text, str) or not text.strip():
        return None

    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(value, Mapping):
        raise LMStudioConfigurationError(
            f"LM Studio {channel} contained JSON, but not a JSON object."
        )
    return dict(value)


class LMStudioBackend:
    """SDK-based live-model audit for the calibrated Q4 instance."""

    def __init__(
        self,
        *,
        model_key: str = QWEN_Q4_MODEL_KEY,
        expected_context_length: int = EXPECTED_CONTEXT_LENGTH,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model_key = model_key
        self.expected_context_length = expected_context_length
        self.timeout_seconds = timeout_seconds
        self._model: Any | None = None
        self._lms: Any | None = None
        self._identity_evidence: tuple[str, ...] = ()

    def _import_sdk(self) -> Any:
        if self._lms is not None:
            return self._lms
        try:
            import lmstudio as lms
        except ImportError as exc:
            raise LMStudioUnavailableError(
                "lmstudio-python is not installed in this environment. "
                "Run: python -m pip install -e ."
            ) from exc
        self._lms = lms
        return lms

    def _find_loaded_model(self) -> Any:
        lms = self._import_sdk()
        try:
            lms.set_sync_api_timeout(self.timeout_seconds)
            loaded = lms.list_loaded_models("llm")
        except Exception as exc:
            raise LMStudioUnavailableError(
                "Could not talk to LM Studio. Keep LM Studio running."
            ) from exc

        diagnostics: list[dict[str, Any]] = []
        for model in loaded:
            try:
                info = model.get_info()
            except Exception as exc:
                diagnostics.append(
                    {"get_info_error": f"{type(exc).__name__}: {exc}"}
                )
                continue

            diagnostics.append(_identity_snapshot(info))
            matches, evidence = _identity_evidence(info)
            if matches:
                self._identity_evidence = evidence
                return model

        raise LMStudioConfigurationError(
            "No loaded model could be VERIFIED as the calibrated "
            f"{EXPECTED_QUANTIZATION} router. "
            f"Expected family={QWEN_BASE_MODEL_KEY!r}, "
            f"exact_key={self.model_key!r}. "
            f"Loaded diagnostics={diagnostics!r}."
        )

    def _validate_loaded_model(self, model: Any) -> None:
        try:
            info = model.get_info()
        except Exception as exc:
            raise LMStudioConfigurationError(
                "Could not inspect the loaded LM Studio model."
            ) from exc

        matches, evidence = _identity_evidence(info)
        if not matches:
            raise LMStudioConfigurationError(
                "Loaded model cannot be verified as Qwen3.5-9B Q4_K_M. "
                f"metadata={_identity_snapshot(info)!r}"
            )
        self._identity_evidence = evidence

        context = _value(info, "context_length", "contextLength")
        if context is None:
            try:
                context = model.get_context_length()
            except Exception:
                context = None

        if context is not None and int(context) != self.expected_context_length:
            raise LMStudioConfigurationError(
                f"router context is {context}, expected "
                f"{self.expected_context_length}"
            )

        try:
            load_cfg = _mapping(model.get_load_config())
        except Exception:
            load_cfg = {}

        gpu = load_cfg.get("gpu")
        if isinstance(gpu, Mapping) and "ratio" in gpu:
            ratio = gpu["ratio"]
            if ratio not in (0, 0.0, "off", False, None):
                raise LMStudioConfigurationError(
                    f"GPU offload ratio is {ratio!r}; expected 0/off"
                )

        pool = load_cfg.get("cpuThreadPoolSize")
        if pool is not None and int(pool) != EXPECTED_CPU_THREADS:
            raise LMStudioConfigurationError(
                f"CPU thread pool is {pool}; expected {EXPECTED_CPU_THREADS}"
            )

    def _get_model(self) -> Any:
        if self._model is None:
            model = self._find_loaded_model()
            self._validate_loaded_model(model)
            self._model = model
        return self._model

    def audit(self) -> dict[str, Any]:
        model = self._get_model()
        info = model.get_info()

        try:
            load_cfg = _mapping(model.get_load_config())
        except Exception:
            load_cfg = {}

        context = _value(info, "context_length", "contextLength")
        if context is None:
            try:
                context = model.get_context_length()
            except Exception:
                context = None

        return {
            "expected_exact_model_key": self.model_key,
            "expected_base_model_key": QWEN_BASE_MODEL_KEY,
            "router_api_identifier": QWEN_ROUTER_API_IDENTIFIER,
            "expected_quantization": EXPECTED_QUANTIZATION,
            "quantization_verified": True,
            "identity_evidence": list(self._identity_evidence),
            "model_key": _value(info, "model_key", "modelKey"),
            "identifier": _value(info, "identifier"),
            "display_name": _value(info, "display_name", "displayName"),
            "path": _value(info, "path"),
            "size_bytes": _value(info, "size_bytes", "sizeBytes"),
            "context_length": context,
            "load_config": load_cfg,
            "prediction_cpu_threads": EXPECTED_CPU_THREADS,
            "note": (
                "SDK audit verifies the loaded Q4 identity/context/GPU state "
                "where exposed. Semantic routing uses the OpenAI-compatible "
                "/v1/chat/completions endpoint with JSON-schema constrained "
                "output."
            ),
        }


def _is_transient_engine_fetch_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "engine protocol predict request failed: fetch failed" in message


class LMStudioRestBackend:
    """Structured semantic inference through LM Studio's OpenAI-compatible API.

    The native /api/v1/chat endpoint is excellent for explicitly controlling
    reasoning, but it does not expose JSON-schema structured output. Jev needs
    schema-constrained routing more than free-form chat, so semantic decisions
    use /v1/chat/completions with response_format=json_schema.

    LM Studio/Qwen versions have sometimes placed schema-constrained JSON in
    `reasoning_content` rather than `content`. Jev accepts either channel ONLY
    when the complete channel is exactly one JSON object. Arbitrary prose,
    YAML-like output, partial JSON, or conflicting JSON channels are rejected.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_SERVER_BASE_URL,
        model: str = QWEN_ROUTER_API_IDENTIFIER,
        timeout_seconds: float = 120.0,
        auditor: LMStudioBackend | None = None,
        http_post: Callable[[str, dict[str, Any], float], Mapping[str, Any]]
        | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.auditor = auditor or LMStudioBackend()
        self._http_post = http_post or self._urllib_post
        self._audited = False
        self.last_metadata: dict[str, Any] | None = None

    @staticmethod
    def _urllib_post(
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            try:
                details = exc.read().decode("utf-8", errors="replace")
            except Exception:
                details = ""
            raise LMStudioUnavailableError(
                "LM Studio structured-routing request failed with HTTP "
                f"{exc.code}: {details[:800]}"
            ) from exc
        except (URLError, OSError) as exc:
            raise LMStudioUnavailableError(
                "Could not reach the LM Studio server at "
                f"{url!r}. Start it with: lms server start --port 1234"
            ) from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LMStudioConfigurationError(
                "LM Studio returned invalid HTTP JSON."
            ) from exc

        if not isinstance(decoded, Mapping):
            raise LMStudioConfigurationError(
                "LM Studio response is not a JSON object."
            )
        return decoded

    def _ensure_audited(self) -> None:
        if not self._audited:
            self.auditor.audit()
            self._audited = True

    @staticmethod
    def _extract_structured_choice(
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise LMStudioConfigurationError(
                "Expected exactly one chat-completion choice."
            )

        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise LMStudioConfigurationError(
                "LM Studio chat-completion choice is malformed."
            )

        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise LMStudioConfigurationError(
                "LM Studio hit max_tokens before completing the structured "
                "routing decision."
            )

        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise LMStudioConfigurationError(
                "LM Studio choice is missing message."
            )

        content_obj = _decode_complete_json_object(
            message.get("content"),
            channel="message.content",
        )
        reasoning_obj = _decode_complete_json_object(
            message.get("reasoning_content"),
            channel="message.reasoning_content",
        )

        candidates = [
            ("content", content_obj),
            ("reasoning_content", reasoning_obj),
        ]
        valid = [(name, obj) for name, obj in candidates if obj is not None]

        if not valid:
            diagnostics = {
                "finish_reason": finish_reason,
                "content": message.get("content"),
                "reasoning_content": message.get("reasoning_content"),
            }
            raise LMStudioConfigurationError(
                "LM Studio structured output contained no complete JSON "
                f"object. diagnostics={diagnostics!r}"
            )

        if len(valid) == 2 and valid[0][1] != valid[1][1]:
            raise LMStudioConfigurationError(
                "LM Studio returned conflicting structured JSON in content "
                "and reasoning_content."
            )

        return valid[0][1]

    def respond_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._ensure_audited()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "jev_route_decision",
                    "strict": True,
                    "schema": dict(response_format),
                },
            },
            "temperature": float(config.get("temperature", 0.0)),
            "max_tokens": int(config.get("maxTokens", 128)),
            "stream": False,
            "seed": 0,
        }

        endpoint = f"{self.base_url}/v1/chat/completions"
        try:
            response = self._http_post(
                endpoint,
                payload,
                self.timeout_seconds,
            )
        except LMStudioUnavailableError as exc:
            if not _is_transient_engine_fetch_error(exc):
                raise
            response = self._http_post(
                endpoint,
                payload,
                self.timeout_seconds,
            )

        self.last_metadata = {
            "id": response.get("id"),
            "model": response.get("model"),
            "usage": response.get("usage"),
        }

        return self._extract_structured_choice(response)
