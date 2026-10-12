import unittest
from types import SimpleNamespace

from localia.providers.lmstudio import (
    EXPECTED_Q4_SIZE_BYTES,
    LMStudioBackend,
    LMStudioConfigurationError,
    QWEN_BASE_MODEL_KEY,
    QWEN_Q4_MODEL_KEY,
)


class FakeModel:
    def __init__(
        self,
        *,
        key=QWEN_Q4_MODEL_KEY,
        context=4096,
        path="lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
        size_bytes=EXPECTED_Q4_SIZE_BYTES,
    ):
        self.key = key
        self.context = context
        self.path = path
        self.size_bytes = size_bytes

    def get_info(self):
        return SimpleNamespace(
            model_key=self.key,
            identifier=self.key,
            context_length=self.context,
            path=self.path,
            size_bytes=self.size_bytes,
            display_name="Qwen3.5 9B",
            format="gguf",
        )

    def get_load_config(self):
        return {"gpu": {"ratio": 0.0}}


class FakeLMS:
    def __init__(self, models):
        self.models = models
        self.timeout = None

    def set_sync_api_timeout(self, value):
        self.timeout = value

    def list_loaded_models(self, kind):
        assert kind == "llm"
        return list(self.models)


class LMStudioBackendAuditTests(unittest.TestCase):
    def test_exact_q4_loaded_model_is_verified(self):
        backend = LMStudioBackend()
        backend._lms = FakeLMS([FakeModel()])

        audit = backend.audit()

        self.assertTrue(audit["quantization_verified"])
        self.assertIn("exact_model_key", audit["identity_evidence"])

    def test_generic_identifier_is_accepted_when_q4_path_confirms_quant(self):
        backend = LMStudioBackend()
        backend._lms = FakeLMS([FakeModel(key=QWEN_BASE_MODEL_KEY)])

        audit = backend.audit()

        self.assertEqual(audit["model_key"], QWEN_BASE_MODEL_KEY)
        self.assertIn("gguf_path_q4_k_m", audit["identity_evidence"])

    def test_generic_identifier_can_be_confirmed_by_benchmarked_q4_size(self):
        backend = LMStudioBackend()
        backend._lms = FakeLMS([
            FakeModel(
                key=QWEN_BASE_MODEL_KEY,
                path="qwen/qwen3.5-9b",
                size_bytes=EXPECTED_Q4_SIZE_BYTES,
            )
        ])

        audit = backend.audit()

        self.assertIn(
            "benchmarked_q4_file_size",
            audit["identity_evidence"],
        )

    def test_generic_identifier_with_q6_metadata_is_rejected(self):
        backend = LMStudioBackend()
        backend._lms = FakeLMS([
            FakeModel(
                key=QWEN_BASE_MODEL_KEY,
                path="lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q6_K.gguf",
                size_bytes=8_281_142_495,
            )
        ])

        with self.assertRaises(LMStudioConfigurationError):
            backend.audit()

    def test_wrong_quantized_key_is_rejected(self):
        backend = LMStudioBackend()
        backend._lms = FakeLMS([
            FakeModel(
                key="qwen/qwen3.5-9b@q6_k",
                path="lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q6_K.gguf",
                size_bytes=8_281_142_495,
            )
        ])

        with self.assertRaises(LMStudioConfigurationError):
            backend.audit()

    def test_wrong_context_is_rejected(self):
        backend = LMStudioBackend()
        backend._lms = FakeLMS([FakeModel(context=8192)])

        with self.assertRaises(LMStudioConfigurationError):
            backend.audit()


if __name__ == "__main__":
    unittest.main()
