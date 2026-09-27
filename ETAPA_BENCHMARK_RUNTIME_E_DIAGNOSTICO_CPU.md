# LocalIA — Etapa de Benchmark de Runtime e Diagnóstico de CPU

**Data:** 27/09/2026  
**Projeto:** LocalIA / Jev Harness  
**Hardware principal:** Samsung Galaxy Book3 360 — Intel Core i7-1360P, 16 GB LPDDR4X, Intel Iris Xe  
**Modelo:** Qwen3.5-9B Q4_K_M  
**Runtime avaliado nesta etapa:** llama.cpp 2.46.0 via LM Studio

---

## 1. Objetivo desta etapa

Esta etapa teve três objetivos:

1. definir se o Qwen3.5-9B Q4_K_M deveria usar o runtime CPU AVX2 ou o runtime Vulkan no Galaxy Book3 360;
2. determinar se havia diferença real entre carregar o Vulkan com `--gpu off` e `--gpu 0.0`;
3. melhorar a metodologia do benchmark para controlar duas variáveis ainda não verificadas: temperatura do processador e número real de threads utilizadas pelo llama.cpp.

A prioridade foi transformar decisões de configuração do LocalIA em decisões sustentadas por medições reproduzíveis, e não por estimativas do LM Studio ou por testes isolados.

---

## 2. Benchmark CPU AVX2 vs Vulkan

Foi executada uma bateria de 18 gerações:

- 2 runtimes;
- 3 prompts;
- 3 rodadas;
- GPU offload desligado;
- mesmo modelo, quantização e contexto;
- temperatura de geração em `0`;
- reasoning desligado.

### Resultado agregado

| Métrica | CPU llama.cpp 2.46.0 | Vulkan llama.cpp 2.46.0 |
|---|---:|---:|
| Throughput médio | 6,638 tok/s | 6,017 tok/s |
| Mediana | 6,592 tok/s | 6,018 tok/s |
| TTFT médio | 1,513 s | 2,482 s |
| Tempo médio de resposta | 39,97 s | 47,19 s |
| Tempo médio de carregamento | 9,78 s | 8,18 s |

O runtime CPU AVX2 apresentou aproximadamente **10,3% mais throughput** e TTFT aproximadamente **39% menor**.

### Decisão

Para o `local_fast`, o candidato preferencial passou a ser:

```text
Runtime: llama.cpp CPU AVX2 2.46.0
Modelo: Qwen3.5-9B Q4_K_M
GPU Offload: OFF
Contexto: 4096
Reasoning: OFF
```

O Vulkan continua útil para diagnóstico e para outros modelos/hardwares, mas não é o backend preferencial desse modelo nesse notebook.

---

## 3. Diagnóstico `--gpu off` vs `--gpu 0.0`

Havia uma hipótese de que o grande salto observado entre benchmarks anteriores pudesse ser explicado por uma diferença semântica entre:

```text
--gpu off
```

e:

```text
--gpu 0.0
```

Foi executada outra bateria de 18 gerações, mantendo o runtime Vulkan 2.46.0 fixo.

### Resultado agregado

| Métrica | `--gpu off` | `--gpu 0.0` |
|---|---:|---:|
| Throughput médio | 5,884 tok/s | 5,876 tok/s |
| Mediana | 5,883 tok/s | 5,901 tok/s |
| TTFT médio | 2,508 s | 2,547 s |
| Tempo médio de resposta | 48,26 s | 48,34 s |

Diferença de throughput: aproximadamente **0,15%**.

### Conclusão

Para este workload:

```text
--gpu off ≈ --gpu 0.0
```

A diferença é pequena demais para ser tratada como efeito real. Portanto, a discrepância antiga de aproximadamente 4,3 tok/s para aproximadamente 6 tok/s **não foi causada por `off` vs `0.0`**.

### Decisão

Encerrar a investigação específica de `--gpu off` contra `--gpu 0.0`.

Para novos testes, usar preferencialmente `--gpu off`, por ser semanticamente mais explícito.

---

## 4. Variável ainda não resolvida: CPU Thread Pool

Os scripts anteriores registraram:

```text
CPU_THREADS_TARGET = 8
```

mas esse valor era apenas metadata. O benchmark não demonstrou que o llama.cpp realmente executou com 8 threads.

Isso passou a ser relevante porque houve uma diferença grande entre baterias antigas e recentes:

```text
Vulkan antigo, GPU 0: ~4,29 tok/s
Vulkan recente, GPU OFF: ~5,9–6,0 tok/s
CPU AVX2 recente: ~6,64 tok/s
```

Como `--gpu off` e `--gpu 0.0` foram equivalentes, o número real de threads passou a ser uma das principais variáveis suspeitas.

### Importante

No ecossistema atual do LM Studio existem conceitos distintos relacionados a threads:

- `CPU Thread Pool Size`, associado ao carregamento do modelo;
- `CPU Threads`, associado à configuração de predição em partes da configuração interna.

Eles não devem ser tratados automaticamente como sinônimos.

Para o benchmark seguinte, o parâmetro prioritário a controlar será o **CPU Thread Pool Size**.

---

## 5. Por que a tentativa de capturar `n_threads` falhou

O script anterior iniciou:

```text
lms log stream --source model
```

Esse source é voltado principalmente para entrada e saída do modelo. Ele não é o source adequado para observar o processo de carregamento/configuração do servidor.

Por isso os arquivos existiram, mas retornaram:

```text
n_threads = null
n_threads_batch = null
n_gpu_layers = null
interesting_lines = []
```

### Correção planejada

Nos próximos scripts:

```text
lms log stream --source server --json
```

será usado durante o carregamento.

Além disso, o nível de log do llama.cpp deverá estar em **Debug** durante o preflight de diagnóstico.

O parser deverá buscar, entre outros:

```text
n_threads
threads
n_threads_batch
cpuThreadPoolSize
--threads
n_gpu_layers
offload
batch
ubatch
```

Se o stream do servidor ainda não expuser a informação, será usado um segundo método de verificação: inspeção da linha de comando do processo llama.cpp/llama-server no Windows.

---

## 6. Variável ainda não resolvida: temperatura da CPU

O benchmark tentou obter temperatura em 54 pontos diferentes:

- início de cada bloco;
- antes e depois de cada warm-up;
- antes e depois de cada geração.

Todos retornaram:

```text
sample = null
```

Portanto, **não houve medição térmica válida** e o thermal gate não funcionou.

Isso não invalida a comparação `off` vs `0.0`, mas impede analisar throttling térmico.

---

## 7. Nova estratégia de temperatura

A próxima versão do benchmark não dependerá prioritariamente do WMI do Windows.

Será usado:

```text
LibreHardwareMonitor
        ↓
Remote Web Server
        ↓
data.json
        ↓
Python
```

O benchmark lerá o sensor de `CPU Package` ou equivalente imediatamente antes e depois de cada inferência.

Não será feito polling contínuo durante a geração principal, para reduzir interferência no desempenho medido.

### Dados térmicos a registrar

Para cada execução:

```text
temp_before_c
temp_after_c
temp_delta_c
sensor_name
sensor_source
```

E para cada bloco:

```text
idle_temp_before_block
warmup_temp_before
warmup_temp_after
```

### Thermal gate

Um bloco só deverá começar normalmente quando a temperatura estiver abaixo do limite configurado, por exemplo:

```text
TARGET_START_TEMP_C = 70
```

Se o limite não for alcançado após um timeout, o teste poderá continuar, mas deverá registrar explicitamente essa condição.

---

## 8. Próximo experimento: thread sweep

Depois de validar temperatura e número real de threads, será executada uma bateria específica no runtime CPU AVX2.

Configurações candidatas:

```text
4 threads
6 threads
8 threads
10 threads
12 threads
```

Cada configuração deverá usar os mesmos prompts e múltiplas repetições.

Métricas principais:

- tokens/s;
- TTFT;
- tempo total;
- temperatura antes/depois;
- variação térmica;
- estabilidade entre rodadas;
- número de threads efetivamente aplicado.

O objetivo não é encontrar apenas o maior valor instantâneo de tokens/s, mas a melhor configuração sustentada para o notebook.

---

## 9. Estado experimental atual

### Confirmado

- Qwen3.5-9B Q4_K_M funciona de forma estável localmente.
- Offload para a Iris Xe piora significativamente a geração nesse modelo.
- CPU AVX2 2.46.0 supera Vulkan 2.46.0 com GPU desligada.
- `--gpu off` e `--gpu 0.0` são equivalentes na prática neste cenário.
- Reasoning permaneceu desligado durante os benchmarks.
- Contexto, eval batch, physical batch, parallel, Flash Attention e KV offload foram auditados.

### Ainda não confirmado

- CPU Thread Pool Size realmente utilizado em cada bateria.
- `n_threads` efetivo durante a geração.
- temperatura real do CPU Package.
- existência ou não de thermal throttling nas baterias mais longas.
- causa exata da diferença entre os benchmarks antigos (~4,3 tok/s) e recentes (~6 tok/s).

---

## 10. Decisão provisória para o LocalIA

Até que o thread sweep seja concluído:

```text
Route: local_fast
Model: Qwen3.5-9B Q4_K_M
Runtime: CPU llama.cpp AVX2 2.46.0
GPU Offload: OFF
Context: 4096
Reasoning: OFF
Temperature: 0
```

O valor de threads ainda não deve ser congelado como configuração definitiva.

---

## 11. Critério para encerrar esta fase

Esta fase poderá ser considerada concluída quando:

1. a temperatura `CPU Package` estiver sendo registrada de forma confiável;
2. o número de threads estiver explicitamente controlado e/ou verificado;
3. o thread sweep estiver completo;
4. uma configuração final de CPU for escolhida com base em throughput, TTFT e estabilidade térmica;
5. essa configuração for registrada como baseline oficial do `local_fast`.

Depois disso, o projeto poderá seguir para comparação de quantização Q4_K_M vs Q6_K e posteriormente para modelos maiores.
