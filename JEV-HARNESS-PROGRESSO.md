# JEV Harness — Progresso de Implementação

**Última atualização:** 26/09/2026  
**Status:** V0 em preparação — backend local validado e benchmark de CPU concluído.

---

## 1. Objetivo do projeto

Construir um **JEV Harness / LLM Router** capaz de escolher automaticamente o melhor modelo para cada tarefa, priorizando:

1. **Modelos locais**, quando forem suficientes;
2. **Assinaturas já pagas** (ChatGPT Plus via Codex CLI e Claude Pro via Claude Code);
3. **APIs pagas**, como OpenRouter, somente como fallback.

A proposta é usar o **Jev como camada de decisão/orquestração**, e não como modelo principal de geração.

Arquitetura conceitual:

```text
                         TAREFA
                           │
                           ▼
                 REGRAS DETERMINÍSTICAS
                           │
                           ▼
                         JEV
                     LLM ROUTER
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
       LOCAL          ASSINATURAS        API PAGA
          │                │                │
   Qwen3.5-9B       Codex / Claude      OpenRouter
   GPT-OSS-20B       Code (futuro)       (fallback)
```

---

## 2. Decisões arquiteturais tomadas

### Linguagem

O V0 será desenvolvido em **Python**.

Motivos:
- ecossistema mais simples para Jev, OpenAI-compatible APIs e automação;
- velocidade de prototipação;
- integração fácil com LM Studio, Codex CLI e Claude Code;
- o gargalo inicial será inferência/latência dos modelos, não o runtime Python.

**Rust** permanece como possibilidade futura para um runtime mais robusto, especialmente para concorrência, processos, background workers, filas, MCP e controle de estado.

### Modelos previstos

| Rota | Modelo/Backend | Papel |
|---|---|---|
| `local_fast` | Qwen3.5-9B | modelo local padrão |
| `local_heavy` | GPT-OSS-20B | experimento local mais pesado |
| `codex` | Codex CLI | usar franquia do ChatGPT Plus |
| `claude` | Claude Code | usar franquia do Claude Pro |
| `api_fallback` | OpenRouter | fallback pago |

### Estratégia de custo

```text
1. Local
   ↓
2. Assinatura já paga
   ↓
3. API pay-as-you-go
```

---

## 3. Hardware utilizado

**Samsung Galaxy Book3 360**

- CPU: Intel Core i7-1360P
- RAM: 16 GB
- Armazenamento: 1 TB
- GPU: Intel Iris Xe integrada
- Sistema: Windows 11

Características relevantes do processador:

```text
4 P-cores × 2 threads = 8 threads
8 E-cores × 1 thread  = 8 threads

Total:
12 núcleos físicos
16 threads lógicas
```

---

## 4. Backend local escolhido

Foi escolhido o **LM Studio** como backend local por:

- suporte a GGUF;
- servidor local compatível com API OpenAI;
- controle de contexto, CPU e GPU offload;
- possibilidade futura de load/unload programático de modelos.

### LM Studio Bionic

Inicialmente foi instalado o **LM Studio Bionic**.

Ao carregar o Qwen3.5-9B, o `llama-server` encerrava antes de ficar saudável:

```text
Engine protocol runtime llama-server exited before becoming healthy.
exitCode=3221226505
```

O código corresponde a:

```text
0xC0000409
```

O erro foi tratado como crash do runtime, com suspeita principal de incompatibilidade/regressão envolvendo backend gráfico/Vulkan/Intel Iris Xe.

### Migração para LM Studio clássico

Decisão: remover o Bionic e instalar o **LM Studio clássico 0.4.25 para Windows x86**.

O modelo GGUF existente foi preservado.

---

## 5. Modelo local atual

Modelo:

```text
Qwen3.5-9B
GGUF
Q4_K_M
~6,55 GB
```

O Q4_K_M foi usado nos testes iniciais por oferecer maior folga de RAM durante o diagnóstico.

Configuração inicial conservadora:

```text
Context Length:               4096
GPU Offload:                  0
CPU Thread Pool Size:         variável nos benchmarks
Max Concurrent Predictions:   1
Unified KV Cache:             OFF
KV Cache GPU Offload:         OFF
Speculative Decoding:         OFF
Flash Attention:              OFF
mmap:                         ON
Keep Model in Memory:         ON
Think:                        OFF (benchmark normal)
```

Com **GPU Offload = 0**, o modelo carregou corretamente no LM Studio clássico.

Conclusão:

```text
GGUF íntegro ✅
RAM suficiente para Qwen Q4_K_M ✅
Runtime CPU funcional ✅
Crash anterior provavelmente relacionado ao caminho GPU/runtime ✅
```

---

## 6. Observação sobre o modo Think

Foi feito um teste com o modo **Think** habilitado.

Em uma pergunta simples de contagem de letras, o modelo entrou em um longo ciclo de raciocínio sobre instruções de turnos anteriores.

Em um dos testes:

```text
~2.830 tokens gerados
~3,73 tok/s
TTFT ~5,45 s
tempo total ~12 min 43 s
```

Conclusão importante:

- o throughput por token permaneceu próximo ao modo normal;
- o tempo total aumentou principalmente porque o modelo produziu milhares de tokens de raciocínio;
- para `local_fast`, o modo **Think não deve ficar ligado por padrão**;
- no futuro, o Harness pode impor um **reasoning budget** e escalar a tarefa se o modelo exceder o limite.

---

## 7. Prompt cache / KV cache

Durante os testes foi observado que repetir exatamente o mesmo prompt reduzia drasticamente o **TTFT**.

Exemplo:

```text
Primeira execução: ~6 s
Execuções seguintes: ~0,6–0,9 s
```

Mesmo criando chats novos, o backend conseguia reaproveitar parte do prompt cache.

Consequência metodológica:

- `tokens/s` é a principal métrica de throughput;
- o TTFT de prompts repetidos não deve ser tratado como cold start;
- para testar TTFT, usar prompts diferentes ou recarregar o modelo;
- o cache será relevante no futuro para decisões do próprio Harness.

---

## 8. Benchmark de CPU — metodologia

Foram usados:

- Qwen3.5-9B Q4_K_M;
- contexto 4096;
- Think OFF;
- GPU Offload = 0;
- um único modelo carregado;
- carregamento reaplicado ao mudar o número de threads.

O LM Studio clássico limitou a interface a no máximo **8 threads**.

Foram testados:

```text
4 threads
6 threads
8 threads
```

---

## 9. Benchmark — mesmo prompt repetido

Prompt:

> Explique em aproximadamente 150 palavras o que é uma API REST.

| Threads | tok/s | TTFT (s) | Tempo total (s) |
|---:|---:|---:|---:|
| 4 | 3.52 | 6.3 | 75.88 |
| 4 | 3.78 | 0.9 | 64.01 |
| 4 | 3.64 | 0.8 | 61.26 |
| 6 | 4.02 | 4.6 | 59.15 |
| 6 | 3.98 | 0.7 | 58.76 |
| 6 | 4.02 | 0.7 | 62.10 |
| 8 | 4.39 | 4.1 | 50.74 |
| 8 | 4.39 | 0.6 | 47.06 |
| 8 | 4.39 | 0.6 | 53.51 |

Médias de throughput:

```text
4 threads → ~3,65 tok/s
6 threads → ~4,01 tok/s
8 threads → ~4,39 tok/s
```

O TTFT das repetições 2 e 3 é influenciado pelo cache.

---

## 10. Benchmark — prompts diferentes

Prompts:

**A — Computação**

> Explique em aproximadamente 150 palavras o que é uma API REST e cite suas principais características.

**B — Biologia**

> Explique em aproximadamente 150 palavras como funciona a fotossíntese e qual é sua importância para os ecossistemas.

**C — História**

> Explique em aproximadamente 150 palavras quais foram as principais causas da Revolução Francesa.

Resultados:

| Threads | Prompt | tok/s | TTFT (s) | Tempo total (s) |
|---:|:---:|---:|---:|---:|
| 4 | A | 3.63 | 6.29 | 76.44 |
| 4 | B | 3.68 | 4.61 | 74.09 |
| 4 | C | 3.67 | 5.29 | 83.04 |
| 6 | A | 4.39 | 4.0 | 57.06 |
| 6 | B | 4.44 | 4.8 | 65.33 |
| 6 | C | 4.05 | 5.2 | 84.68 |
| 8 | A | 4.37 | 4.0 | 72.44 |
| 8 | B | 4.38 | 4.8 | 66.05 |
| 8 | C | 4.42 | 3.8 | 75.60 |

Médias:

| Threads | Média tok/s | Média TTFT |
|---:|---:|---:|
| 4 | **3,66** | **5,40 s** |
| 6 | **4,29** | **4,67 s** |
| 8 | **4,39** | **4,20 s** |

Ganhos aproximados:

```text
4 → 6 threads: +17,3% em throughput
6 → 8 threads: +2,3%
4 → 8 threads: +20%
```

O conjunto com 8 threads também foi o mais consistente:

```text
4,37
4,38
4,42 tok/s
```

---

## 11. Baseline atual

Configuração recomendada até aqui:

```text
Samsung Galaxy Book3 360
Intel Core i7-1360P
16 GB RAM

Qwen3.5-9B Q4_K_M
Context: 4096
Think: OFF
GPU Offload: 0
CPU Threads: 8

Generation throughput: ~4,39 tok/s
Cold TTFT:            ~4,2 s
```

Essa será a **baseline CPU** para comparar com a Intel Iris Xe.

---

## 12. Observações de hardware

Durante inferência:

- o LM Studio não utiliza necessariamente 100% da CPU;
- inferência de LLM tende a ser fortemente limitada por largura de banda de memória;
- aumentar threads não gera ganho linear;
- 6 → 8 threads trouxe somente ~2,3% de ganho médio;
- RAM chegou a ficar próxima de 90% de utilização quando outros programas estavam abertos;
- benchmarks devem ser feitos com notebook conectado ao carregador e modo de alto desempenho.

Configuração de energia aplicada:

```text
Samsung Settings → Alto desempenho
CPU máximo (tomada) → 100%
CPU mínimo → 5%
```

---

## 13. Próximo experimento

### Intel Iris Xe / GPU Offload

Manter:

```text
Qwen3.5-9B Q4_K_M
Context 4096
Think OFF
CPU threads 8
Max Concurrent 1
```

Variar apenas:

```text
GPU Offload
```

Objetivo:

> descobrir se a Iris Xe consegue superar a baseline de ~4,39 tok/s e qual configuração oferece o melhor equilíbrio entre velocidade, RAM, estabilidade e temperatura.

Os mesmos prompts A/B/C devem ser reutilizados.

---

## 14. Próximas etapas do projeto

### V0 — LLM Router

```text
Task
 ↓
Deterministic Filter
 ↓
Jev Choice
 ↓
Qwen / GPT-OSS / Codex / Claude
 ↓
Response
 ↓
Telemetry
```

Componentes previstos:

```text
jev-harness/
│
├── config/
│   └── models.yaml
│
├── src/
│   └── jev_harness/
│       ├── cli.py
│       ├── orchestrator.py
│       ├── router.py
│       ├── registry.py
│       ├── schemas.py
│       ├── telemetry.py
│       │
│       └── providers/
│           ├── base.py
│           ├── lmstudio.py
│           ├── codex.py
│           └── claude.py
│
└── data/
    └── runs.jsonl
```

### V0.5 — Quality Cascade

```text
modelo barato/local
        ↓
    verificação
        ↓
 suficiente?
   │          │
  sim        não
   │          │
  fim      escalonar
```

### V1 — Dynamic Context

Jev passa a avaliar a relevância de:

- histórico;
- arquivos;
- tool outputs;
- memórias;
- resultados anteriores.

O contexto deixa de ser estático e passa a ser montado especificamente para cada tarefa.

### V2 — Agent Runtime

Adicionar:

- tools;
- subagentes;
- background workers;
- permissões;
- memória;
- evals;
- observabilidade;
- roteamento baseado em custo, privacidade e capacidade.

---

## 15. Princípios do projeto

1. **Código determinístico decide fatos objetivos.**
2. **Jev decide questões semânticas/ambíguas.**
3. **Modelos locais são preferidos quando suficientes.**
4. **Assinaturas já pagas são aproveitadas antes de APIs pay-as-you-go.**
5. **Ações críticas não dependem apenas de probabilidades do Jev.**
6. **Toda execução deve gerar telemetria.**
7. **Routing deve ser calibrado usando resultados reais, não apenas intuição.**
8. **O V0 deve permanecer pequeno e observável.**

---

## 16. Métricas planejadas para o Harness

Cada execução deverá registrar:

```json
{
  "task_id": "...",
  "route": "local_fast",
  "model": "qwen3.5-9b",
  "provider": "lmstudio",
  "tokens_per_second": 4.39,
  "time_to_first_token_seconds": 4.2,
  "input_tokens": 0,
  "output_tokens": 0,
  "latency_seconds": 0,
  "estimated_cost_usd": 0,
  "success": true,
  "user_rating": null
}
```

Com isso será possível responder futuramente:

- quantas tarefas o modelo local resolve sozinho;
- quando o GPT-OSS realmente compensa;
- quando Codex/Claude são necessários;
- quanto uso de API foi evitado;
- quais tarefas merecem escalonamento;
- qual modelo entrega melhor relação qualidade × custo × latência.

---

## 17. Status atual

```text
[✓] Definição conceitual do JEV Harness
[✓] Escolha de Python para o V0
[✓] Estratégia local → assinatura → API
[✓] Seleção do Qwen3.5-9B como local_fast
[✓] LM Studio clássico instalado
[✓] Qwen3.5-9B carregando corretamente em CPU
[✓] Think / reasoning behavior observado
[✓] Efeito do prompt cache identificado
[✓] Benchmark de threads concluído
[✓] 8 threads escolhido como baseline atual
[ ] Benchmark da Intel Iris Xe
[ ] Teste do GPT-OSS-20B
[ ] Ativar servidor OpenAI-compatible do LM Studio
[ ] Instalar/testar Codex CLI
[ ] Instalar/testar Claude Code
[ ] Configurar Jev API
[ ] Criar projeto Python
[ ] Implementar primeiro router
[ ] Implementar telemetria
[ ] Executar primeiro `jev ask "..."`
```

---

## Nota sobre os testes

Nos dados registrados nesta versão do documento existem **18 execuções explicitamente documentadas** (9 do prompt repetido + 9 dos prompts A/B/C). Caso existam outras 6 execuções da série mencionada como “24 testes”, elas ainda precisam ser adicionadas ao histórico.
