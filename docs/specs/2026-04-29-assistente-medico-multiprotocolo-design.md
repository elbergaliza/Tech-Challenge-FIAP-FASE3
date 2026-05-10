# Spec: Assistente Médico Multi-Protocolo com RAG

**Data:** 2026-04-29  
**Status:** Draft  
**Repositório destino:** A definir (repositório separado com implementação inicial)

---

## 1. Objetivo

Sistema de triagem médica baseado em RAG que:
- Recebe sintomas de paciente em texto livre
- Classifica doença(s) provável(is) via busca semântica em protocolos oficiais
- Avalia gravidade conforme protocolo identificado
- Emite alerta de urgência para casos graves
- Sugere exames para **todas** as doenças suspeitas (diagnóstico diferencial)
- Após confirmação médica + exames, sugere tratamento focado na doença confirmada
- Gera relatório final com citação de fonte e disclaimer

**Princípio:** O sistema NUNCA prescreve sozinho — sempre requer aprovação médica em cada step.

---

## 2. Decisões Arquiteturais

### 2.1 Vector Store: Coleção Única com Filtro por Metadado

- **Um único store** contendo chunks de todos os protocolos
- Cada chunk tem metadados: `{doenca, secao_tipo, gravidade_grupo, pagina, fonte}`
- `secao_tipo` ∈ `["sintomas", "classificacao", "exames", "tratamento", "sinais_alarme"]`
- Busca hybrid (FAISS semântico + BM25 keyword) — padrão do covid-parse
- **Após classificação da doença**, os nós seguintes filtram retrieval por `doenca` no metadado

### 2.2 Fluxo de Diagnóstico Diferencial

O sistema aceita **múltiplas suspeitas simultâneas**:

```
Sintomas ambíguos (ex: febre + cefaleia + mialgia)
    → Retrieval traz chunks de dengue E covid
    → LLM classifica: suspeitas = ["dengue", "covid"]
    → Exames sugeridos para AMBAS
    → Médico confirma doença (baseado em exames)
    → Tratamento focado na doença confirmada
```

### 2.3 Aprovação Médica: Sempre Requerida

- Toda transição entre nós requer confirmação do médico
- Implementado via `interrupt` do LangGraph (human-in-the-loop)
- O médico pode: aprovar, rejeitar, ou pular para tratamento direto

### 2.4 LLM: Ollama Local

- Modelo: `gemma3:4b` via Ollama (OpenAI-compatible API)
- Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Reranking (futuro): `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## 3. Protocolos Suportados

| Doença | Fonte | Classificação |
|--------|-------|---------------|
| Dengue | MS/SVS - Diagnóstico e Manejo Clínico | Grupo A / B / C / D |
| COVID-19 | Protocolo MS v9 | Leve / Moderado / Grave / Crítico |
| AVC | PCDT Conitec (futuro) | Tempo-dependente |
| Infarto/SCA | Diretriz SBC (futuro) | Risco baixo/intermediário/alto |

---

## 4. Arquitetura LangGraph — Nós

```
                    ┌─────────────────────────┐
                    │  Nó 1: Classificação    │
                    │  Input: sintomas        │
                    │  Retrieval: unificado   │
                    │  Output: doencas_suspeitas + gravidade │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Nó 2: Avaliação        │
                    │  Gravidade por protocolo│
                    │  Se GRAVE → Alerta      │
                    │  Se NÃO GRAVE → Exames  │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────┼──────────────┐
                  │ GRAVE                       │ NÃO GRAVE
                  ▼                             ▼
    ┌─────────────────────┐      ┌─────────────────────────┐
    │  Nó 3a: Alerta      │      │  Nó 3b: Sugestão Exames │
    │  Urgência imediata  │      │  Para TODAS as suspeitas │
    │  + exames urgentes  │      │  Filtro: doenca + secao  │
    │  → vai para Nó 5    │      └────────────┬────────────┘
    │  (tratamento urgente)│                   │ Médico confirma
    └────────┬────────────┘      ┌────────────▼────────────┐
             │                   │  Nó 4: Confirmação      │
             │                   │  Input: exames + decisão│
             │                   │  Se não confirma → END  │
             │                   │  Se confirma → Nó 5     │
             └──────────┐        └────────────┬────────────┘
                        │                     │
                        └─────────┬───────────┘
                                               │
                                  ┌────────────▼────────────┐
                                  │  Nó 5: Tratamento       │
                                  │  Retrieval: filtrado    │
                                  │  por doenca_confirmada  │
                                  │  + grupo gravidade      │
                                  └────────────┬────────────┘
                                               │ Médico aprova
                                  ┌────────────▼────────────┐
                                  │  Nó 6: Relatório Final  │
                                  │  Consolidação completa  │
                                  │  + fonte + disclaimer   │
                                  └─────────────────────────┘
```

---

## 5. State do LangGraph

```python
class TriageState(TypedDict):
    # Input
    sintomas: str                          # Texto livre do paciente
    
    # Classificação (Nó 1)
    docs_retrieval: list[dict]             # Chunks recuperados
    doencas_suspeitas: list[str]           # ["dengue", "covid"]
    gravidade: dict[str, str]             # {"dengue": "grupo_c", "covid": "moderado"}
    justificativa_classificacao: str
    
    # Avaliação (Nó 2) — por doença, pois gravidades podem ser mistas
    max_gravidade: str                     # "grave" ou "nao_grave" (pior caso entre suspeitas)
    doencas_graves: list[str]              # Subset de doencas_suspeitas que são graves
    doencas_nao_graves: list[str]          # Subset não grave
    alerta: str | None                     # Mensagem de urgência se alguma grave
    exames_urgentes: dict[str, list[str]]  # Exames urgentes para doenças graves
    
    # Exames (Nó 3b) — para doenças não graves
    exames_sugeridos: dict[str, list[str]] # {"dengue": ["hemograma", "NS1"], "covid": ["RT-PCR"]}
    
    # Confirmação (Nó 4)
    doenca_confirmada: str | None          # "dengue" — definido pelo médico
    resultado_exames: str | None           # Input do médico
    pular_para_tratamento: bool            # Médico pode pular exames → tratamento direto
    
    # Tratamento (Nó 5)
    tratamento_sugerido: str
    fontes_tratamento: list[str]
    
    # Relatório (Nó 6)
    relatorio_final: str
    
    # Controle
    aprovacao_medica: dict[str, bool]      # {"classificacao": True, "exames": True, ...}
```

---

## 6. Pipeline RAG — Detalhes Técnicos

### 6.1 Parsing de PDF

```python
# Genérico para qualquer protocolo
def parse_protocol_pdf(pdf_path: str, protocol_config: ProtocolConfig) -> list[Document]:
    """
    1. Extrai texto por página (PyMuPDF)
    2. Remove headers/footers via regex (configurável por protocolo)
    3. Chunking: RecursiveCharacterTextSplitter(800, overlap=200)
    4. Filtra ruído (configurável)
    5. Classifica seção_tipo por keywords do protocolo
    6. Adiciona metadados: {doenca, secao_tipo, pagina, fonte}
    """
```

### 6.2 Configuração por Protocolo

```python
@dataclass
class ProtocolConfig:
    nome: str                           # "dengue"
    pdf_path: str
    header_patterns: list[str]          # Regex para remover cabeçalhos repetidos
    noise_patterns: list[str]           # Padrões de ruído (formulários, índices)
    secao_keywords: dict[str, list[str]]  # {"sintomas": ["febre", "cefaleia", ...], ...}
    gravidade_levels: list[str]         # ["grupo_a", "grupo_b", "grupo_c", "grupo_d"]
    fonte_url: str
```

### 6.3 Retrieval Hybrid

Mantém o padrão do covid-parse:
- FAISS (semântico, k=10) + BM25 (keyword, k=10)
- Reciprocal Rank Fusion (bm25_weight=0.6)
- Dedup por conteúdo

Com filtro adicional:
```python
def retrieve_for_disease(query: str, doenca: str, secao_tipo: str | None = None) -> list[Document]:
    """Retrieval filtrado por metadados após classificação."""
    results = hybrid_retrieve(query, k=10)
    filtered = [r for r in results if r.metadata["doenca"] == doenca]
    if secao_tipo:
        filtered = [r for r in filtered if r.metadata["secao_tipo"] == secao_tipo]
    return filtered
```

---

## 7. Milestones Incrementais

### M1: Parser + Classificação (Nó 1)
**Escopo:** Parser genérico de PDF + indexação + nó de classificação  
**Entregável:** Script que recebe sintomas e retorna doença(s) suspeita(s) + gravidade  
**Teste:** 
- Input: "febre alta, dor abdominal intensa, vômitos persistentes, petéquias"
- Output esperado: "Dengue, Grupo C (sinais de alarme)"

**Componentes:**
- `protocol_config.py` — dataclass de configuração
- `parser.py` — parsing genérico de PDF com config
- `indexer.py` — criação do vector store unificado
- `retriever.py` — hybrid retrieval com filtro
- `nodes/classify.py` — nó LangGraph de classificação
- `configs/dengue.py` — config específica dengue

### M2: Avaliação de Gravidade + Alerta (Nó 2 + 3a)
**Escopo:** Nó que avalia gravidade e emite alerta se grave  
**Entregável:** Classificação → se grave, alerta; se não, segue para exames  
**Teste:**
- Input: estado com "dengue, grupo_d" → Output: alerta urgência + internação UTI
- Input: estado com "dengue, grupo_a" → Output: segue para exames

### M3: Sugestão de Exames (Nó 3b)
**Escopo:** Nó que sugere exames para todas as doenças suspeitas  
**Entregável:** Lista de exames por doença com fonte  
**Teste:**
- Input: suspeitas=["dengue"] gravidade="grupo_b" → Output: "hemograma completo (obrigatório)"
- Input: suspeitas=["dengue","covid"] → Output: exames para ambas

### M4: Confirmação + Tratamento (Nós 4 + 5)
**Escopo:** Human-in-the-loop para confirmação + sugestão de tratamento  
**Entregável:** Médico confirma doença → sistema sugere tratamento do protocolo  
**Teste:**
- Input: doenca_confirmada="dengue", grupo="grupo_a" → Output: "hidratação oral 80ml/kg/dia..."

### M5: Relatório + Fluxo Completo (Nó 6 + integração)
**Escopo:** Relatório final + LangGraph end-to-end  
**Entregável:** Fluxo completo funcional com interrupt/approve

### M6: Segundo Protocolo (COVID)
**Escopo:** Adicionar COVID-19 ao mesmo sistema  
**Entregável:** Sistema diferencia dengue vs covid automaticamente

---

## 8. Estrutura de Projeto Proposta

```
medical-triage-rag/
├── configs/
│   ├── __init__.py
│   ├── base.py              # ProtocolConfig dataclass
│   ├── dengue.py            # Config dengue
│   └── covid.py             # Config covid (M6)
├── core/
│   ├── __init__.py
│   ├── parser.py            # PDF → chunks com metadados
│   ├── indexer.py           # Chunks → vector store
│   └── retriever.py         # Hybrid retrieval + filtro
├── nodes/
│   ├── __init__.py
│   ├── classify.py          # Nó 1: classificação
│   ├── evaluate.py          # Nó 2: gravidade + alerta
│   ├── exams.py             # Nó 3b: sugestão exames
│   ├── confirm.py           # Nó 4: confirmação médica
│   ├── treatment.py         # Nó 5: tratamento
│   └── report.py            # Nó 6: relatório
├── graph/
│   ├── __init__.py
│   ├── state.py             # TriageState
│   └── builder.py           # Montagem do StateGraph
├── data/
│   └── protocols/           # PDFs dos protocolos
├── tests/
│   ├── test_classify.py
│   ├── test_evaluate.py
│   ├── test_exams.py
│   └── test_treatment.py
├── prompts/
│   ├── classify.txt         # Prompt de classificação
│   ├── evaluate.txt         # Prompt de gravidade
│   ├── exams.txt            # Prompt de exames
│   └── treatment.txt        # Prompt de tratamento
└── main.py                  # Entry point
```

---

## 9. Prompts (Diretrizes)

Cada nó terá um prompt estruturado. Exemplo para classificação:

```
Você é um médico especialista em triagem hospitalar.
Com base EXCLUSIVAMENTE nos trechos de protocolos oficiais abaixo, classifique:
1. Doença(s) provável(is)
2. Nível de gravidade conforme o protocolo da doença

CONTEXTO DOS PROTOCOLOS:
{chunks_recuperados}

SINTOMAS DO PACIENTE:
{sintomas}

RESPONDA em JSON:
{
  "doencas_suspeitas": [{"doenca": "...", "gravidade": "...", "justificativa": "..."}],
  "fontes": ["protocolo X, p.Y"]
}
```

---

## 10. Constraints e Regras

1. **Fonte obrigatória:** Toda recomendação cita protocolo + página
2. **Disclaimer:** "Sugestão baseada em protocolo oficial. Validação médica obrigatória."
3. **Medicamentos proibidos:** Dengue → NUNCA sugerir AAS ou anti-inflamatórios não esteroidais
4. **Alerta imediato:** Dengue grupo C/D, COVID grave/crítico → alerta sem esperar exames
5. **Idioma:** Português brasileiro
6. **Offline-first:** Ollama local, sem dependência de APIs externas

---

## 11. Edge Cases e Fallbacks

| Cenário | Comportamento |
|---------|---------------|
| Retrieval retorna zero chunks relevantes | LLM responde "Não foi possível identificar doença com base nos protocolos disponíveis. Avaliação clínica presencial necessária." |
| LLM retorna doença fora dos protocolos indexados | Rejeitar e informar: "Doença sugerida não está nos protocolos suportados." |
| Gravidade mista (dengue grave + covid leve) | Segue path GRAVE (pior caso). Alerta para dengue grave. Exames para ambas. |
| Médico confirma doença que NÃO estava em suspeitas | Aceitar. Fazer retrieval de tratamento para doença confirmada pelo médico. |
| Filtro por metadado retorna lista vazia | Fallback: refazer retrieval sem filtro de `secao_tipo`, apenas filtrando por `doenca`. Se ainda vazio, retrieval completo + aviso. |
| Médico quer pular exames e ir direto para tratamento | `pular_para_tratamento=True` no state → pula Nó 3b/4, vai direto ao Nó 5. |

---

## 12. Esclarecimentos do Diagrama

- **Nó 1 classifica doença E gravidade** no mesmo passo (retrieval unificado traz chunks com info de classificação)
- **Nó 2 não reclassifica** — apenas avalia o output de Nó 1 e decide o routing (grave → 3a, não grave → 3b)
- **Nó 3a (grave) → Nó 5 direto** — paciente grave vai para tratamento urgente sem esperar exames confirmatórios (exames urgentes são solicitados em paralelo)
- **"Se não confirma → END"** significa: médico discorda do diagnóstico, fluxo encerra, paciente volta para avaliação clínica presencial
