import contextlib
import io
import unittest
from unittest.mock import patch

from localia.cli import main
from localia.jev.semantic import SemanticRouterError
from localia.jev.validator import DecisionValidationError


class FakeJev:
    def __init__(self, *, error):
        self.registry = self
        self._error = error

    def set_enabled(self, route, enabled):
        pass

    def decide(self, request):
        raise self._error


class CliErrorTests(unittest.TestCase):
    def _run_with_error(self, error):
        stderr = io.StringIO()
        with (
            patch("sys.argv", ["localia", "test task"]),
            patch(
                "localia.cli.LMStudioRestBackend",
                return_value=object(),
            ),
            patch(
                "localia.cli.LMStudioSemanticRouter",
                return_value=object(),
            ),
            patch(
                "localia.cli.Jev",
                return_value=FakeJev(error=error),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            code = main()
        return code, stderr.getvalue()

    def test_semantic_router_error_is_user_friendly(self):
        code, stderr = self._run_with_error(
            SemanticRouterError("bad semantic output")
        )

        self.assertEqual(code, 3)
        self.assertIn("Jev routing failed:", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_validation_error_is_user_friendly(self):
        code, stderr = self._run_with_error(
            DecisionValidationError("bad route")
        )

        self.assertEqual(code, 3)
        self.assertIn("Jev routing failed:", stderr)
        self.assertNotIn("Traceback", stderr)


if __name__ == "__main__":
    unittest.main()
