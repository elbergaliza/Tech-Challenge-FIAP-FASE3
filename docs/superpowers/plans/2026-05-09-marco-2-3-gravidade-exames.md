# Marco 2 e 3 (Gravidade + Exames) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidar o Marco 2 (avaliacao de gravidade + alerta) e implementar o Marco 3 (sugestao de exames por doenca suspeita), com fluxo `classificacao -> gravidade -> (alerta opcional) -> exames`.

**Architecture:** O M2 permanece deterministico: recebe `gravidade` do M1, normaliza labels e decide pior caso. Se `grave`, o fluxo passa por `alerta` e segue para `exames`; se `nao_grave`, vai direto para `exames`. O M3 usa retrieval filtrado por `doenca` + `secao_tipo="exames"` e aplica fallback em 3 niveis (com secao, sem secao, sem filtro de doenca), garantindo cobertura do edge case de gravidade mista (alerta + exames para todas as suspeitas). O grafo encerra sempre apos `exames`.

**Tech Stack:** Python 3.12, LangGraph StateGraph, LangChain, FAISS + BM25, pytest, unittest.mock.

---

## Contexto de Partida (ultimo daily + estado atual)

- Ultimo daily no Obsidian (`daily/2026-04-29.md`) registra M2-M6 como pendentes no planejamento geral.
- O repositorio ja contem base de M2 (`langchain/nos/gravidade.py`, `langchain/nos/alerta.py`, `langchain/fluxo.py`) e testes de M2.
- Este plano cobre: (1) fechar lacunas de contrato do M2 para integracao e (2) entregar M3 completo com TDD.

---

## Mapa de Arquivos (responsabilidade por unidade)

| Arquivo | Acao | Responsabilidade |
|---|---|---|
| `langchain/nos/gravidade.py` | **Modificar** | Ajustar roteamento `nao_grave -> exames` e hardening de normalizacao |
| `langchain/nos/exames.py` | **Criar** | No M3: retrieval filtrado + invocacao do modelo + parse JSON + merge no estado |
| `langchain/nos/__init__.py` | **Modificar** | Re-exportar `criar_no_exames` |
| `langchain/prompts/exames.txt` | **Criar** | Prompt estruturado para extracao de exames em JSON |
| `langchain/fluxo.py` | **Modificar** | Adicionar no `exames`, novas arestas e campos de estado do M3 |
| `langchain/main.py` | **Modificar** | Exibir `exames_sugeridos` e `fontes_exames` no output CLI |
| `langchain/tests/test_gravidade.py` | **Modificar** | Cobrir normalizacao de doenca/label e roteamento atualizado |
| `langchain/tests/test_exames.py` | **Criar** | Testes unitarios do no de exames (happy path e fallback) |
| `langchain/tests/test_fluxo_m2.py` | **Modificar** | Ajustar expectativa do caminho `nao_grave` para incluir saida do no de exames |
| `langchain/tests/test_fluxo_m3.py` | **Criar** | Integracao de fluxo para M3 (nao grave gera exames; grave gera alerta e tambem exames) |

---

### Task 1: Congelar contrato do M2 para suportar M3

**Files:**
- Modify: `langchain/tests/test_gravidade.py`
- Modify: `langchain/nos/gravidade.py`

- [ ] **Step 1.1: Escrever testes de roteamento (`grave -> alerta_e_exames`, `nao_grave -> exames`)**

```python
def test_roteamento_nao_grave_vai_para_exames(self):
    assert rotear_gravidade({"max_gravidade": "nao_grave"}) == "exames"


def test_roteamento_grave_vai_para_alerta_e_exames(self):
    assert rotear_gravidade({"max_gravidade": "grave"}) == "alerta_e_exames"
```

- [ ] **Step 1.2: Escrever teste de hardening para alias de doenca COVID**

```python
@pytest.mark.parametrize("doenca", ["covid", "COVID", "covid-19", "covid 19"])
def test_classificar_covid_alias_grave(doenca):
    assert classificar_gravidade(doenca, "grave") == "grave"
```

- [ ] **Step 1.3: Rodar teste para verificar falha inicial**

Run: `cd langchain && python -m pytest tests/test_gravidade.py -v -k "roteamento_nao_grave_vai_para_exames or classificar_covid_alias_grave"`
Expected: FAIL (roteamento ainda nao retorna `alerta_e_exames` e/ou alias incompleto).

- [ ] **Step 1.4: Implementar mudancas minimas em `gravidade.py`**

```python
def _normalizar_chave(texto: str) -> str:
    base = unicodedata.normalize("NFD", texto.lower().strip())
    base = "".join(c for c in base if unicodedata.category(c) != "Mn")
    return base.replace("-", "_").replace(" ", "_")


def classificar_gravidade(doenca: str, label_normalizado: str) -> str:
    chave = _normalizar_chave(doenca)
    regras_doenca = _REGRAS.get(chave, {})
    return regras_doenca.get(label_normalizado, "nao_grave")


def rotear_gravidade(estado: dict) -> str:
    if estado.get("max_gravidade") == "grave":
        return "alerta_e_exames"
    return "exames"
```

- [ ] **Step 1.5: Rodar suite do M2 e confirmar regressao zero**

Run: `cd langchain && python -m pytest tests/test_gravidade.py tests/test_alerta.py tests/test_fluxo_m2.py -v`
Expected: PASS.

- [ ] **Step 1.6: Commit**

```bash
git add langchain/nos/gravidade.py langchain/tests/test_gravidade.py
git commit -m "refactor(m2): ajustar roteamento e hardening de normalizacao"
```

---

### Task 2: Definir contrato de estado para M3 no fluxo

**Files:**
- Modify: `langchain/fluxo.py`
- Create: `langchain/tests/test_fluxo_m3.py`

- [ ] **Step 2.1: Adicionar campos do M3 em `EstadoTriagem`**

```python
# Exames (No 3b - M3)
exames_sugeridos: dict[str, list[str]]
fontes_exames: dict[str, list[str]]
justificativa_exames: dict[str, str]
```

- [ ] **Step 2.2: Criar teste de contrato do estado no fluxo (falha inicial)**

Criar em `langchain/tests/test_fluxo_m3.py` um teste simples que espera `exames_sugeridos` no fluxo.

Run: `cd langchain && python -m pytest tests/test_fluxo_m3.py -v`
Expected: FAIL (campo nao populado; no exames inexistente).

- [ ] **Step 2.3: Commit da evolucao de tipagem do estado**

```bash
git add langchain/fluxo.py langchain/tests/test_fluxo_m3.py
git commit -m "feat(state): adicionar contrato de campos do M3"
```

---

### Task 3: Criar prompt de exames (M3)

**Files:**
- Create: `langchain/prompts/exames.txt`

- [ ] **Step 3.1: Criar prompt com formato de saida JSON estrito**

```txt
Voce e um assistente medico focado em extracao de exames de protocolos oficiais.
Use APENAS o contexto abaixo.

DOENCA: {doenca}
GRAVIDADE: {gravidade}

CONTEXTO:
{contexto}

Responda SOMENTE em JSON valido com o formato:
{
  "exames": ["..."],
  "justificativa": "...",
  "fontes": ["fonte, p.X"]
}

Se nao houver evidencias suficientes, retorne:
{
  "exames": [],
  "justificativa": "Evidencia insuficiente no contexto recuperado.",
  "fontes": []
}
```

- [ ] **Step 3.2: Commit**

```bash
git add langchain/prompts/exames.txt
git commit -m "feat(m3): adicionar prompt estruturado de sugestao de exames"
```

---

### Task 4: Implementar no `exames` com TDD

**Files:**
- Create: `langchain/nos/exames.py`
- Create: `langchain/tests/test_exames.py`

- [ ] **Step 4.1: Escrever testes unitarios do no de exames**

Cobrir no minimo:
- gera `exames_sugeridos` para uma doenca suspeita
- gera exames para multiplas doencas suspeitas
- fallback em JSON invalido do modelo
- fallback de retrieval em cascata: com secao -> sem secao -> sem filtro de doenca

Exemplo de teste principal:

```python
def test_no_exames_retorna_dict_por_doenca(indice_fake, modelo_fake):
    no = criar_no_exames(indice_fake, modelo_fake)
    estado = {
        "doencas_suspeitas": ["dengue"],
        "gravidade": {"dengue": "grupo_b"},
    }
    out = no(estado)
    assert "dengue" in out["exames_sugeridos"]
    assert isinstance(out["exames_sugeridos"]["dengue"], list)
```

- [ ] **Step 4.2: Rodar teste para confirmar falha (modulo ainda nao existe)**

Run: `cd langchain && python -m pytest tests/test_exames.py -v`
Expected: FAIL com `ImportError`.

- [ ] **Step 4.3: Implementar `langchain/nos/exames.py`**

```python
import json
import re
from pathlib import Path

from banco_de_conhecimento import IndiceProtocolo, busca_hibrida


CAMINHO_PROMPT = Path(__file__).parent.parent / "prompts" / "exames.txt"


def _parse_resposta_exames(conteudo: str) -> dict:
    try:
        return json.loads(conteudo)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", conteudo, re.DOTALL)
        if match:
            return json.loads(match.group())
    return {"exames": [], "justificativa": "Resposta invalida do modelo.", "fontes": []}


def criar_no_exames(indice: IndiceProtocolo, modelo):
    template = CAMINHO_PROMPT.read_text(encoding="utf-8")

    def no_exames(estado: dict) -> dict:
        doencas = estado.get("doencas_suspeitas", [])
        gravidades = estado.get("gravidade", {})

        exames_sugeridos = {}
        fontes_exames = {}
        justificativa_exames = {}

        for doenca in doencas:
            consulta = f"exames recomendados para {doenca} {gravidades.get(doenca, '')}"
            docs = busca_hibrida(
                indice,
                consulta,
                k=8,
                doenca=doenca,
                secao_tipo="exames",
            )

            aviso_fallback = ""
            if not docs:
                docs = busca_hibrida(indice, consulta, k=8, doenca=doenca)
                aviso_fallback = "fallback_sem_secao"
            if not docs:
                docs = busca_hibrida(indice, consulta, k=8)
                aviso_fallback = "fallback_sem_doenca"

            contexto = "\n\n".join(
                f"[{d.metadata.get('fonte', '')}, p.{d.metadata.get('pagina', '')}]\n{d.page_content}"
                for d in docs
            )

            prompt = template.format(
                doenca=doenca,
                gravidade=gravidades.get(doenca, ""),
                contexto=contexto or "Sem contexto de exames recuperado.",
            )
            resposta = modelo.invoke(prompt)
            conteudo = resposta.content if hasattr(resposta, "content") else str(resposta)
            payload = _parse_resposta_exames(conteudo)

            justificativa = payload.get("justificativa", "")
            if aviso_fallback:
                justificativa = f"{justificativa} [{aviso_fallback}]".strip()

            exames_sugeridos[doenca] = payload.get("exames", [])
            fontes_exames[doenca] = payload.get("fontes", [])
            justificativa_exames[doenca] = justificativa

        return {
            **estado,
            "exames_sugeridos": exames_sugeridos,
            "fontes_exames": fontes_exames,
            "justificativa_exames": justificativa_exames,
        }

    return no_exames
```

- [ ] **Step 4.4: Rodar testes unitarios do no de exames**

Run: `cd langchain && python -m pytest tests/test_exames.py -v`
Expected: PASS.

- [ ] **Step 4.5: Commit**

```bash
git add langchain/nos/exames.py langchain/tests/test_exames.py
git commit -m "feat(m3): implementar no de sugestao de exames"
```

---

### Task 5: Integrar M3 ao grafo LangGraph

**Files:**
- Modify: `langchain/fluxo.py`
- Modify: `langchain/nos/__init__.py`
- Modify: `langchain/tests/test_fluxo_m2.py`
- Modify: `langchain/tests/test_fluxo_m3.py`

 - [ ] **Step 5.1: Escrever testes de integracao para `nao_grave` e gravidade mista**

```python
def test_nao_grave_gera_exames(self, indice):
    modelo = _mock_llm(
        '{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "grupo_b", "justificativa": "..."}], "fontes": ["p.28"]}'
    )
    fluxo = montar_fluxo(modelo, indice)
    out = fluxo.invoke({"sintomas": "febre e mialgia"})
    assert "exames_sugeridos" in out
    assert "dengue" in out["exames_sugeridos"]


def test_gravidade_mista_gera_alerta_e_exames(self, indice):
    modelo = _mock_llm(
        '{"doencas_suspeitas": ['
        '{"doenca": "dengue", "gravidade": "grupo_c", "justificativa": "sinais de alarme"}, '
        '{"doenca": "covid", "gravidade": "moderado", "justificativa": "quadro respiratorio"}'
        '], "fontes": ["p.12", "p.40"]}'
    )
    fluxo = montar_fluxo(modelo, indice)
    out = fluxo.invoke({"sintomas": "febre, dispneia, dor abdominal"})
    assert out["max_gravidade"] == "grave"
    assert out["alerta"] is not None
    assert "dengue" in out["exames_sugeridos"]
    assert "covid" in out["exames_sugeridos"]
```

- [ ] **Step 5.2: Rodar testes para confirmar falha antes da integracao**

Run: `cd langchain && python -m pytest tests/test_fluxo_m3.py -v`
Expected: FAIL (`KeyError`/campo ausente).

- [ ] **Step 5.3: Integrar no `exames` e arestas no `fluxo.py`**

```python
from nos.exames import criar_no_exames

no_exames = criar_no_exames(indice, modelo)

grafo.add_node("exames", no_exames)

grafo.add_conditional_edges(
    "gravidade",
    rotear_gravidade,
    {
        "alerta_e_exames": "alerta",
        "exames": "exames",
    },
)

grafo.add_edge("alerta", "exames")
grafo.add_edge("exames", END)
```

- [ ] **Step 5.4: Atualizar `nos/__init__.py` para exportar o novo no**

```python
from .exames import criar_no_exames
```

- [ ] **Step 5.5: Rodar regressao de classificacao + M2 + M3**

Run: `cd langchain && python -m pytest tests/test_classificacao.py tests/test_gravidade.py tests/test_alerta.py tests/test_exames.py tests/test_fluxo_m2.py tests/test_fluxo_m3.py -v`
Expected: PASS.

- [ ] **Step 5.6: Commit**

```bash
git add langchain/fluxo.py langchain/nos/__init__.py langchain/tests/test_fluxo_m2.py langchain/tests/test_fluxo_m3.py
git commit -m "feat(m3): integrar sugestao de exames ao fluxo de triagem"
```

---

### Task 6: Ajustar saida CLI e validacao final

**Files:**
- Modify: `langchain/main.py`

- [ ] **Step 6.1: Exibir exames no output quando presentes**

```python
if resultado.get("exames_sugeridos"):
    print(f"\nExames sugeridos : {resultado.get('exames_sugeridos', {})}")
    print(f"Fontes exames    : {resultado.get('fontes_exames', {})}")
```

- [ ] **Step 6.2: Rodar suite completa (exceto e2e)**

Run: `cd langchain && python -m pytest tests/ -v --ignore=tests/test_e2e.py`
Expected: PASS.

- [ ] **Step 6.3: Smoke test manual**

Run: `cd langchain && python main.py`
Expected:
- Caso grave (`grupo_c`/`grupo_d`): imprime alerta e tambem `exames_sugeridos` por doenca suspeita.
- Caso nao grave (`grupo_a`/`grupo_b`): imprime `exames_sugeridos` por doenca.

- [ ] **Step 6.4: Commit de fechamento dos marcos**

```bash
git add langchain/main.py
git commit -m "feat(m2,m3): concluir fluxo de gravidade e sugestao de exames"
```

---

## Sequencia de Execucao Recomendada

1. Task 1 (hardening M2)  
2. Task 2 (contrato de estado M3)  
3. Task 3 + Task 4 (prompt + no exames)  
4. Task 5 (integracao no grafo)  
5. Task 6 (CLI + validacao final)

---

## Criterios de Aceite (M2 + M3)

- M2: `max_gravidade`, `doencas_graves`, `doencas_nao_graves`, `alerta` corretos para dengue e covid (incluindo aliases).
- Roteamento: `grave -> alerta -> exames`, `nao_grave -> exames`.
- M3: `exames_sugeridos` preenchido por doenca para todos os suspeitos (inclusive gravidade mista).
- Fallback retrieval em cascata implementado e coberto por teste.
- Fluxo completo nao quebra campos do M1.
- Todos os testes unitarios e de integracao passam (`--ignore=tests/test_e2e.py`).
