# ADR-001: Reestruturação do Fluxo LangGraph para Assistente Médico Multi-Protocolo

**Data:** 2026-04-29  
**Status:** Proposto  
**Autores:** Equipe de desenvolvimento  

---

## Contexto

A implementação atual utiliza uma arquitetura baseada em múltiplos agentes com um grafo LangGraph simplificado:

- 3 agentes (triagem, clínico, segurança) implementados como chains LCEL (prompt + retriever + LLM)
- Grafo com routing binário: triagem → clínico ou segurança → log
- RAG baseado em FAISS com embeddings semânticos sobre CSV
- State mínimo: `{pergunta, destino, resposta, agente_utilizado}`


---

## Decisões

### D1: Consolidar agentes em nós do LangGraph com lógica explícita

**Situação atual:** Cada agente executa exclusivamente um prompt com contexto de retrieval

**Decisão:** Migrar a lógica dos agentes para **nós do StateGraph**, onde cada nó encapsula: prompt especializado, retrieval filtrado e transformação de state.

**Justificativa:**

1. **Cada agente apenas executa um prompt — não há workflow que justifique a separação.** Os agentes atuais não possuem lógica interna, ferramentas ou fluxo próprio; são funções que montam um prompt, chamam o LLM e retornam texto. Essa lógica cabe naturalmente dentro de um nó do LangGraph, eliminando a indireção sem perda funcional.

2. **Os nós do fluxo precisam de state compartilhado.** O nó de tratamento precisa ler `doenca_confirmada` (preenchida na confirmação), o nó de exames precisa ler `doencas_suspeitas` (preenchida na classificação), o nó de relatório precisa de tudo. O state do LangGraph resolve isso nativamente — cada nó lê e escreve no mesmo `TypedDict`. Com módulos separados, seria necessário passar todo o contexto manualmente entre eles, o que é frágil e não permite routing condicional baseado em dados estruturados.

3. **Alinhamento com o conteúdo das aulas.** Nas aulas da FIAP, o padrão ensinado foi um único LLM orquestrado por nós do LangGraph com state compartilhado definindo o fluxo — não múltiplos agentes independentes. Manter a abordagem de múltiplos agentes pode ser interpretado na avaliação como fora do escopo ensinado ou como não aderente ao que foi demonstrado em aula.

**Abordagem de migração:** O código dos agentes (`agentes.py`) será **reaproveitado** — os prompts e a lógica de invocação do LLM migram para os respectivos nós. O arquivo `agentes.py` pode ser mantido como módulo utilitário se necessário.

**Nós propostos:**

| Nó | Responsabilidade |
|----|-----------------|
| 1. Classificação | Identificar doença(s) suspeita(s) e gravidade via RAG unificado |
| 2. Avaliação | Routing por gravidade (grave → alerta, não grave → exames) |
| 3a. Alerta | Urgência imediata + exames urgentes (casos graves) |
| 3b. Exames | Sugestão de exames para todas as suspeitas (diagnóstico diferencial) |
| 4. Confirmação | Human-in-the-loop — médico confirma doença |
| 5. Tratamento | Sugestão baseada em protocolo da doença confirmada |
| 6. Relatório | Consolidação com fontes e disclaimer |

---

### D2: Retrieval Híbrido (FAISS semântico + BM25 keyword)

**Situação atual:** Busca puramente semântica via FAISS.

**Decisão:** Adicionar BM25 (keyword matching) em paralelo ao FAISS, com fusão via Reciprocal Rank Fusion (RRF).

**Justificativa:**

BM25 complementa a busca semântica (FAISS). Exemplo: query "sintomas de dengue grave".
- **FAISS** busca por similaridade semântica → traz chunks sobre sintomas de dengue em geral (todos os grupos)
- **BM25** busca por palavras-chave → traz chunks que contêm literalmente "grave" 
- **Juntos (RRF):** os chunks que aparecem em ambos (sintomas + grave) sobem no ranking → resultado final são especificamente os sintomas de dengue grave

Sem BM25, a busca semântica sozinha não tem como distinguir "grave" de "leve" com precisão — ambos são semanticamente "sintomas de dengue".

---

### D3: State rico com suporte a diagnóstico diferencial

**Decisão:** Adotar `TriageState` com campos para múltiplas doenças suspeitas, gravidades independentes por doença, exames por suspeita e confirmação médica.

**Justificativa:** Diagnóstico diferencial é prática clínica padrão. Sintomas como "febre + cefaleia + mialgia" são compatíveis com dengue E COVID-19 simultaneamente. O sistema deve:
- Sugerir exames para **todas** as suspeitas
- Permitir que o médico confirme qual doença tratar
- Focar tratamento na doença confirmada

---

### D4: Human-in-the-loop via interrupt do LangGraph

**Decisão:** Usar `interrupt` nas transições críticas (pós-classificação, pós-exames, pós-tratamento).

**O que é interrupt:** Mecanismo do LangGraph que **pausa a execução do grafo** num ponto específico e espera input humano antes de continuar. Exemplo:

```
Nó classificação executa → "suspeita: dengue grupo C"
    ⏸️ INTERRUPT — grafo pausa
    → Médico vê: "Classificação: dengue grupo C. Confirma?"
    → Médico responde: "Sim"
    ⏯️ Grafo retoma com a resposta no state
Próximo nó executa...
```

O grafo salva o state, retorna ao chamador (CLI/API), e quando o médico responde, a execução retoma de onde parou.

**Justificativa:** Garante **estruturalmente** que o sistema nunca avança sem aprovação médica. Não depende de prompt ou boa vontade — o código trava até receber input. Requisito ético para qualquer sistema de sugestão médica.

---

### D5: Vector store unificado com filtro por metadados

**Decisão:** Coleção única contendo chunks de todos os protocolos, com metadados `{doenca, secao_tipo, gravidade_grupo, pagina, fonte}`. Após classificação, nós subsequentes filtram por `doenca` e `secao_tipo`.

**Justificativa:**
- **Classificação sem viés:** O nó 1 busca em todos os protocolos sem filtro prévio — deixa o retrieval trazer chunks relevantes de qualquer doença
- **Precisão pós-classificação:** Nós 3b e 5 filtram por doença confirmada, evitando contaminação de tratamentos cruzados (ex: não sugerir AAS para dengue porque veio chunk de outro protocolo)
- **Extensibilidade:** Adicionar novo protocolo = indexar novo PDF com metadados corretos. Zero mudança no código dos nós.

---

### D6: LLM configurável via strategy (Ollama Cloud / OpenAI / Local)

**Decisão:** Substituir GPT-2/HuggingFace por uma abstração configurável de LLM com strategies intercambiáveis (Ollama Cloud, OpenAI API, Local).

**Justificativa:** Flexibilidade para testes com diferentes providers durante desenvolvimento e futura integração de modelo fine-tuned local sem necessidade de refatoração — basta trocar a configuração.

---

### D7: Embeddings multilíngue (paraphrase-multilingual-MiniLM-L12-v2)

**Decisão:** Utilizar `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` como modelo de embeddings.

**Justificativa:** Os protocolos e queries estão em português. Modelos de embedding treinados apenas em inglês (como `all-MiniLM-L6-v2`) geram representações menos precisas para textos em pt-BR — a busca semântica retorna chunks menos relevantes porque o modelo não entende bem a proximidade entre termos em português. Um modelo multilíngue produz embeddings mais fiéis ao significado real do texto, resultando em retrieval mais eficiente.

---

### D8: Estratégia de migração — mínima remoção de código

**Princípio:** Reaproveitar o máximo possível do código existente, adicionando novos módulos em vez de reescrever.

**Abordagem:**

| Código existente | Ação | Justificativa |
|-----------------|------|---------------|
| `agentes.py` | **Manter** — extrair prompts para `prompts/`, lógica de chain reutilizada nos nós | Prompts já validados |
| `fluxo.py` | **Evoluir** — expandir o grafo de 4 nós para 6+ nós no mesmo arquivo ou novo `graph/builder.py` | Manter histórico git |
| `banco_de_conhecimento.py` | **Estender** — adicionar BM25 e filtro por metadados sobre a base FAISS existente | FAISS já funcional |
| `configuracoes.py` | **Manter** — adicionar novas configs sem remover existentes | Retrocompatibilidade |
| `carregador_do_modelo.py` | **Adaptar** — trocar HuggingFace por Ollama, manter interface | Mudança pontual |
| `main.py` | **Adaptar** — atualizar entry point | Mínimo |

**Novos módulos adicionados:**
- `nodes/` — implementação dos 6 nós
- `core/parser.py` — parsing de PDFs de protocolos
- `configs/base.py` + `configs/dengue.py` — configuração por protocolo
- `prompts/` — prompts externalizados (extraídos de `agentes.py`)

---

## Consequências

### Positivas
- Fluxo multi-nós demonstra domínio real do LangGraph (conditional edges, interrupt, state management)
- Cada decisão clínica é rastreável no state
- Extensível para novos protocolos sem mudança estrutural
- BM25 melhora precision para terminologia médica específica
- Testável unitariamente — cada nó recebe state e retorna state

### Riscos e Mitigações
| Risco | Mitigação |
|-------|-----------|
| Gemma3 4B inconsistente em gerar JSON | Few-shot nos prompts + output parser com retry |
| BM25 requer tokenização adequada para português | Usar tokenizer com suporte a pt-BR (NLTK ou spaCy) |
| Interrupt adiciona complexidade de UX | CLI simples com aprovação por input |

---

## Mudanças Pendentes

> Aguardando detalhamento adicional da equipe.
