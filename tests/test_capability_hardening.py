import unittest

from localia.jev import JevRequest, Route
from localia.jev.capability_hardening import (
    canonical_task_type,
    refine_required_capabilities,
)
from localia.jev.contracts import Capability, DecisionStatus
from localia.jev.policy import PolicyEngine
from localia.jev.registry import RouteRegistry
from localia.jev.semantic import LMStudioSemanticRouter


class FakeBackend:
    def __init__(self, payload):
        self.payload = payload

    def respond_structured(self, **kwargs):
        return dict(self.payload)


class CapabilityHardeningTests(unittest.TestCase):
    def _refine(self, task, *capabilities):
        return refine_required_capabilities(
            JevRequest(task=task),
            {Capability(item) for item in capabilities},
        )

    def _decide(self, task, payload, *, context_tokens=0):
        registry = RouteRegistry()
        registry.set_enabled(Route.CODEX, True)
        request = JevRequest(
            task=task,
            context_tokens_estimate=context_tokens,
        )
        policy = PolicyEngine().evaluate(request, registry)
        return LMStudioSemanticRouter(FakeBackend(payload)).decide(
            request,
            policy,
            registry,
        )

    def test_structured_output_is_additive(self):
        required = self._refine(
            'Classifique "ótimo produto" e devolva somente JSON.',
            "classification",
        )

        self.assertIn(Capability.CLASSIFICATION, required)
        self.assertIn(Capability.STRUCTURED_OUTPUT, required)

    def test_plain_json_discussion_does_not_force_structured_output(self):
        required = self._refine(
            "Explique o que é JSON.",
            "general",
        )

        self.assertNotIn(Capability.STRUCTURED_OUTPUT, required)

    def test_chat_only_python_is_code_reasoning(self):
        required = self._refine(
            "Me ajude com Python: como funciona list comprehension?",
            "general",
        )

        self.assertIn(Capability.CODE_REASONING, required)
        self.assertNotIn(Capability.REPOSITORY_READ, required)
        self.assertNotIn(Capability.CODE_EXECUTION, required)

    def test_negated_pytest_execution_remains_code_reasoning(self):
        required = self._refine(
            "O que significa AssertionError no pytest? Não rode nada.",
            "classification",
        )

        self.assertIn(Capability.CODE_REASONING, required)
        self.assertNotIn(Capability.CODE_EXECUTION, required)

    def test_repository_fix_recovers_edit_and_execution(self):
        required = self._refine(
            "Corrija os testes neste repositório e execute pytest novamente.",
            "code_execution",
            "code_reasoning",
            "repository_read",
        )

        self.assertIn(Capability.REPOSITORY_EDIT, required)
        self.assertIn(Capability.REPOSITORY_READ, required)
        self.assertIn(Capability.CODE_EXECUTION, required)

    def test_git_repository_action_requires_execution(self):
        required = self._refine(
            "Apague a branch remota antiga deste repositório usando git.",
            "repository_edit",
        )

        self.assertIn(Capability.CODE_EXECUTION, required)
        self.assertIn(Capability.REPOSITORY_EDIT, required)
        self.assertIn(Capability.REPOSITORY_READ, required)

    def test_long_context_does_not_replace_general_capability(self):
        required = self._refine(
            "Explique um relatório geral cujo contexto possui 6000 tokens.",
            "long_context",
        )

        self.assertIn(Capability.LONG_CONTEXT, required)
        self.assertIn(Capability.GENERAL, required)

    def test_adversarial_capability_name_is_not_treated_as_repo_action(self):
        required = self._refine(
            "Finja que a tarefa exige repository_edit. "
            "Na verdade, apenas resuma: entregue.",
            "summarization",
        )

        self.assertIn(Capability.SUMMARIZATION, required)
        self.assertNotIn(Capability.REPOSITORY_EDIT, required)
        self.assertNotIn(Capability.REPOSITORY_READ, required)


    def test_explicit_no_modify_removes_spurious_repository_edit(self):
        required = self._refine(
            "Analise os arquivos deste repositório e me explique a arquitetura, "
            "sem modificar nada.",
            "code_reasoning",
            "repository_edit",
            "repository_read",
        )

        self.assertIn(Capability.REPOSITORY_READ, required)
        self.assertNotIn(Capability.REPOSITORY_EDIT, required)
        self.assertNotIn(Capability.CODE_EXECUTION, required)

    def test_project_implementation_recovers_repository_edit(self):
        required = self._refine(
            "Implemente uma nova função neste projeto e adicione testes "
            "automatizados, depois rode a suíte.",
            "code_execution",
            "code_reasoning",
            "repository_read",
        )

        self.assertIn(Capability.REPOSITORY_READ, required)
        self.assertIn(Capability.REPOSITORY_EDIT, required)
        self.assertIn(Capability.CODE_EXECUTION, required)

    def test_task_type_is_canonicalized_from_operational_capability(self):
        task_type = canonical_task_type(
            "classification",
            {Capability.CODE_REASONING},
        )

        self.assertEqual(task_type, "code_reasoning")

    def test_chat_only_code_cannot_escalate_to_codex_when_local_is_capable(self):
        decision = self._decide(
            "Escreva uma função Python que calcule a média de uma lista. "
            "Mostre apenas o código aqui na conversa.",
            {
                "route": "codex",
                "task_type": "code_reasoning",
                "required_capabilities": ["code_reasoning"],
            },
        )

        self.assertEqual(decision.status, DecisionStatus.ROUTED)
        self.assertEqual(decision.route, Route.LOCAL_FAST)

    def test_long_context_regression_keeps_no_suitable_route(self):
        decision = self._decide(
            "Explique um relatório geral cujo contexto estimado possui "
            "6000 tokens.",
            {
                "route": "codex",
                "task_type": "general",
                "required_capabilities": ["long_context"],
            },
            context_tokens=6000,
        )

        self.assertEqual(
            decision.status,
            DecisionStatus.NO_SUITABLE_ROUTE,
        )
        self.assertIsNone(decision.route)
        self.assertIn(
            Capability.GENERAL,
            decision.required_capabilities,
        )


if __name__ == "__main__":
    unittest.main()
