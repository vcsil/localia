import unittest

from localia.jev import Capability, Route
from localia.jev.registry import RouteRegistry


class RouteRegistryCapabilityTests(unittest.TestCase):
    def test_local_fast_can_reason_about_code_but_cannot_edit_repository(self):
        registry = RouteRegistry()
        capabilities = registry.get(Route.LOCAL_FAST).capabilities

        self.assertIn(Capability.CODE_REASONING, capabilities)
        self.assertNotIn(Capability.REPOSITORY_EDIT, capabilities)
        self.assertNotIn(Capability.CODE_EXECUTION, capabilities)

    def test_codex_declares_repository_and_execution_capabilities(self):
        registry = RouteRegistry()
        capabilities = registry.get(Route.CODEX).capabilities

        self.assertIn(Capability.REPOSITORY_READ, capabilities)
        self.assertIn(Capability.REPOSITORY_EDIT, capabilities)
        self.assertIn(Capability.CODE_EXECUTION, capabilities)

    def test_suitable_routes_requires_all_capabilities(self):
        registry = RouteRegistry()
        registry.set_enabled(Route.CODEX, True)

        suitable = registry.suitable_routes(
            frozenset({Route.LOCAL_FAST, Route.CODEX}),
            frozenset(
                {
                    Capability.REPOSITORY_EDIT,
                    Capability.CODE_EXECUTION,
                }
            ),
        )

        self.assertEqual(suitable, (Route.CODEX,))


if __name__ == "__main__":
    unittest.main()
