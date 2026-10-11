# Jev Core V0 — primeiro marco de implementação

Este pacote implementa o primeiro esqueleto executável do JEV.

## O que já existe

- contrato genérico `JevRequest`;
- `DecisionType`, começando por `route_task`;
- `RouteRegistry`;
- `PolicyEngine` determinístico;
- interface `SemanticRouter`;
- `StubSemanticRouter` temporário;
- `DecisionValidator`;
- `Jev.decide(request)`;
- CLI mínima;
- testes contra bypass de policy.

## O que deliberadamente NÃO existe ainda

- LM Studio/Qwen no semantic router;
- execução real de `local_fast`;
- Codex CLI;
- Claude Code;
- OpenRouter;
- RAG;
- memória;
- MCP/tools;
- subagentes;
- contexto dinâmico.

Essa ausência é intencional. O objetivo deste marco é provar o contrato e a
separação de responsabilidades antes de colocar um LLM dentro do loop.

## Instalação local

Na raiz do repositório:

```powershell
python -m pip install -e .
```

## Testes

```powershell
python -m unittest discover -s tests -v
```

## Exemplos

Rota simples:

```powershell
localia "Explique cache L3"
```

Ativar Codex para testar o pipeline:

```powershell
localia "Corrija os testes deste repositório Python" --enable codex
```

Ação destrutiva:

```powershell
localia "Delete uma branch remota" --enable codex --destructive
```

Privacidade local-only:

```powershell
localia "Revise estes dados privados" --local-only
```

## Próximo marco

Substituir o `StubSemanticRouter` por `LMStudioSemanticRouter`, usando a
baseline já validada:

- Qwen3.5-9B Q4_K_M
- CPU AVX2 2.46.0
- 8 threads
- GPU offload 0
- contexto 4096
- temperatura 0
- Thinking OFF

O Qwen receberá apenas as rotas previamente permitidas pelas policies. A
decisão continuará passando pelo `DecisionValidator`.
