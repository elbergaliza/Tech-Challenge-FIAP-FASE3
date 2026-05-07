# M2: Avaliação de Gravidade + Alerta — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar Nó 2 ao grafo LangGraph que lê o campo `gravidade` do estado (produzido pelo M1), normaliza para `"grave"/"nao_grave"` conforme as regras de cada protocolo, e roteia o fluxo: se grave → Nó 3a (alerta de urgência); se não grave → END (M3 ainda não existe).

**Architecture:** O M2 é um nó puramente determinístico — sem chamada ao LLM, sem retrieval. Ele recebe `gravidade: dict[str, str]` (ex: `{"dengue": "grupo_c"}`), aplica as regras de mapeamento por protocolo (dengue: A/B = não grave, C/D = grave; COVID: leve/moderado = não grave, grave/crítico = grave), determina o pior caso entre todas as suspeitas, e popula `max_gravidade`, `doencas_graves`, `doencas_nao_graves` e `alerta` no estado. O nó de alerta (Nó 3a) é um nó simples que formata a mensagem de urgência e encerra.

**Tech Stack:** Python 3.12, LangGraph `StateGraph` + `add_conditional_edges`, pytest com mocks.

---

## Regras de Mapeamento de Gravidade

| Protocolo | Não grave       | Grave           |
|-----------|-----------------|-----------------|
| Dengue    | grupo_a, grupo_b | grupo_c, grupo_d |
| COVID-19  | leve, moderado  | grave, critico  |
| Qualquer  | (não reconhecido) | fallback: não grave |

Normalizações de texto aplicadas antes da comparação:
- lowercase + remover acentos (ex: `"Grupo C"` → `"grupo_c"`, `"Crítico"` → `"critico"`)
- substituir espaço por underscore

---

## Arquivos Alterados / Criados

| Arquivo | Ação | Responsabilidade |
|---------|------|-----------------|
| `langchain/nos/gravidade.py` | **Criar** | Lógica de normalização + nó de avaliação |
| `langchain/nos/alerta.py` | **Criar** | Nó de alerta de urgência (formata mensagem) |
| `langchain/nos/__init__.py` | **Modificar** | Re-exportar `criar_no_gravidade`, `criar_no_alerta` |
| `langchain/fluxo.py` | **Modificar** | Adicionar nós + aresta condicional ao grafo |
| `langchain/fluxo.py` | **Modificar** | Estender `EstadoTriagem` com campos do M2 |
| `langchain/tests/test_gravidade.py` | **Criar** | Testes unitários do nó de gravidade |
| `langchain/tests/test_alerta.py` | **Criar** | Testes do nó de alerta |
| `langchain/tests/test_fluxo_m2.py` | **Criar** | Testes de integração do grafo com M2 |

---

## Task 1: Estender EstadoTriagem com campos do M2

**Files:**
- Modify: `langchain/fluxo.py`

- [ ] **Step 1.1: Adicionar campos M2 ao EstadoTriagem**

Abrir `langchain/fluxo.py` e adicionar ao `EstadoTriagem` (após os campos de Classificação):

```python
    # Avaliação de Gravidade (Nó 2)
    max_gravidade: str              # "grave" ou "nao_grave"
    doencas_graves: list[str]       # subconjunto de doencas_suspeitas
    doencas_nao_graves: list[str]   # subconjunto de doencas_suspeitas
    alerta: str | None              # mensagem de urgência, None se não grave
```

- [ ] **Step 1.2: Verificar que nenhum teste quebra após mudança de tipagem**

```bash
cd langchain && python -m pytest tests/test_classificacao.py -v
```

Esperado: todos os testes do M1 continuam passando (campos novos são `total=False`).

- [ ] **Step 1.3: Commit**

```bash
git add langchain/fluxo.py
git commit -m "feat(state): adicionar campos M2 ao EstadoTriagem"
```

---

## Task 2: Criar módulo de gravidade com normalização e regras

**Files:**
- Create: `langchain/nos/gravidade.py`

- [ ] **Step 2.1: Escrever teste unitário para normalização**

Criar `langchain/tests/test_gravidade.py`:

```python
# tests/test_gravidade.py
"""Testes do nó de avaliação de gravidade (M2)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from nos.gravidade import normalizar_gravidade, classificar_gravidade, criar_no_gravidade


class TestNormalizarGravidade:
    """Testa a normalização do label de gravidade vindo do LLM."""

    def test_grupo_c_maiusculo(self):
        assert normalizar_gravidade("Grupo C") == "grupo_c"

    def test_grupo_d_com_espaco(self):
        assert normalizar_gravidade("grupo d") == "grupo_d"

    def test_grave_simples(self):
        assert normalizar_gravidade("Grave") == "grave"

    def test_critico_com_acento(self):
        assert normalizar_gravidade("Crítico") == "critico"

    def test_nao_grave_com_acento(self):
        assert normalizar_gravidade("Não grave") == "nao_grave"

    def test_moderado(self):
        assert normalizar_gravidade("Moderado") == "moderado"

    def test_vazio(self):
        assert normalizar_gravidade("") == ""


class TestClassificarGravidade:
    """Testa o mapeamento de label normalizado para 'grave'/'nao_grave'."""

    # Dengue
    def test_dengue_grupo_a_nao_grave(self):
        assert classificar_gravidade("dengue", "grupo_a") == "nao_grave"

    def test_dengue_grupo_b_nao_grave(self):
        assert classificar_gravidade("dengue", "grupo_b") == "nao_grave"

    def test_dengue_grupo_c_grave(self):
        assert classificar_gravidade("dengue", "grupo_c") == "grave"

    def test_dengue_grupo_d_grave(self):
        assert classificar_gravidade("dengue", "grupo_d") == "grave"

    # COVID
    def test_covid_leve_nao_grave(self):
        assert classificar_gravidade("covid", "leve") == "nao_grave"

    def test_covid_moderado_nao_grave(self):
        assert classificar_gravidade("covid", "moderado") == "nao_grave"

    def test_covid_grave_grave(self):
        assert classificar_gravidade("covid", "grave") == "grave"

    def test_covid_critico_grave(self):
        assert classificar_gravidade("covid", "critico") == "grave"

    # Fallback
    def test_doenca_desconhecida_nao_grave(self):
        """Doença sem regras definidas → fallback nao_grave."""
        assert classificar_gravidade("avc", "desconhecido") == "nao_grave"

    def test_label_desconhecido_nao_grave(self):
        """Label não reconhecido → fallback nao_grave."""
        assert classificar_gravidade("dengue", "algo_estranho") == "nao_grave"


class TestNoGravidade:
    """Testa o nó de avaliação de gravidade integrado ao estado."""

    def _estado(self, gravidade: dict) -> dict:
        return {
            "sintomas": "teste",
            "doencas_suspeitas": list(gravidade.keys()),
            "gravidade": gravidade,
            "documentos_recuperados": [],
            "justificativa_classificacao": "",
            "fontes": [],
        }

    def test_dengue_grupo_c_max_grave(self):
        no = criar_no_gravidade()
        resultado = no(self._estado({"dengue": "grupo_c"}))
        assert resultado["max_gravidade"] == "grave"
        assert "dengue" in resultado["doencas_graves"]
        assert resultado["doencas_nao_graves"] == []
        assert resultado["alerta"] is not None

    def test_dengue_grupo_a_nao_grave(self):
        no = criar_no_gravidade()
        resultado = no(self._estado({"dengue": "grupo_a"}))
        assert resultado["max_gravidade"] == "nao_grave"
        assert resultado["doencas_graves"] == []
        assert "dengue" in resultado["doencas_nao_graves"]
        assert resultado["alerta"] is None

    def test_sem_doencas_suspeitas(self):
        no = criar_no_gravidade()
        resultado = no(self._estado({}))
        assert resultado["max_gravidade"] == "nao_grave"
        assert resultado["doencas_graves"] == []
        assert resultado["doencas_nao_graves"] == []
        assert resultado["alerta"] is None

    def test_pior_caso_prevalece(self):
        """Dengue A (não grave) + COVID grave → max_gravidade = grave."""
        no = criar_no_gravidade()
        resultado = no(self._estado({"dengue": "grupo_a", "covid": "grave"}))
        assert resultado["max_gravidade"] == "grave"
        assert "covid" in resultado["doencas_graves"]
        assert "dengue" in resultado["doencas_nao_graves"]
        assert resultado["alerta"] is not None
```

- [ ] **Step 2.2: Rodar teste para confirmar falha (módulo ainda não existe)**

```bash
cd langchain && python -m pytest tests/test_gravidade.py -v
```

Esperado: `ImportError: cannot import name 'normalizar_gravidade' from 'nos.gravidade'`

- [ ] **Step 2.3: Implementar `langchain/nos/gravidade.py`**

```python
# nos/gravidade.py
# ─────────────────────────────────────────────────────────────
# Nó de avaliação de gravidade (M2).
#
# Lógica puramente determinística — sem LLM, sem retrieval.
# Normaliza o label de gravidade vindo do LLM e classifica
# cada doença suspeita como "grave" ou "nao_grave".
#
# Regras:
#   Dengue:  grupo_a / grupo_b → nao_grave
#             grupo_c / grupo_d → grave
#   COVID:   leve / moderado   → nao_grave
#             grave / critico   → grave
#   Demais:  fallback nao_grave (conservador)
# ─────────────────────────────────────────────────────────────

import unicodedata


# ─── Tabela de regras por protocolo ──────────────────────────

_REGRAS: dict[str, dict[str, str]] = {
    "dengue": {
        "grupo_a": "nao_grave",
        "grupo_b": "nao_grave",
        "grupo_c": "grave",
        "grupo_d": "grave",
    },
    "covid": {
        "leve": "nao_grave",
        "moderado": "nao_grave",
        "grave": "grave",
        "critico": "grave",
    },
}

# Alias: "covid-19" aponta para as mesmas regras de "covid"
_REGRAS["covid-19"] = _REGRAS["covid"]


# ─── Funções auxiliares ───────────────────────────────────────


def normalizar_gravidade(label: str) -> str:
    """
    Normaliza label de gravidade vindo do LLM para formato canônico.

    Exemplos:
      "Grupo C"    → "grupo_c"
      "Crítico"    → "critico"
      "Não grave"  → "nao_grave"
      "Moderado"   → "moderado"
    """
    if not label:
        return ""
    # lowercase
    texto = label.lower().strip()
    # remove acentos
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    # espaço → underscore
    texto = texto.replace(" ", "_")
    return texto


def classificar_gravidade(doenca: str, label_normalizado: str) -> str:
    """
    Mapeia (doenca, label_normalizado) → 'grave' | 'nao_grave'.

    Usa as regras definidas em _REGRAS por protocolo.
    A chave de doença é normalizada (lowercase, sem hífens extras).
    Fallback: 'nao_grave' para doença ou label desconhecido.
    """
    chave = doenca.lower().strip()
    regras_doenca = _REGRAS.get(chave, {})
    return regras_doenca.get(label_normalizado, "nao_grave")


# ─── Nó LangGraph ────────────────────────────────────────────


def criar_no_gravidade():
    """
    Factory que retorna o nó de avaliação de gravidade.

    Lê `estado["gravidade"]` (dict doença→label do LLM),
    normaliza cada label, classifica como grave/nao_grave,
    determina o pior caso (max_gravidade) e compõe o alerta.
    """

    def no_gravidade(estado: dict) -> dict:
        print("[gravidade] Avaliando gravidade das suspeitas...")

        gravidade_raw: dict[str, str] = estado.get("gravidade", {})

        doencas_graves: list[str] = []
        doencas_nao_graves: list[str] = []

        for doenca, label in gravidade_raw.items():
            label_norm = normalizar_gravidade(label)
            classificacao = classificar_gravidade(doenca, label_norm)
            if classificacao == "grave":
                doencas_graves.append(doenca)
            else:
                doencas_nao_graves.append(doenca)

        max_gravidade = "grave" if doencas_graves else "nao_grave"

        alerta: str | None = None
        if doencas_graves:
            lista = ", ".join(d.upper() for d in doencas_graves)
            alerta = (
                f"⚠️ ALERTA DE URGÊNCIA: suspeita(s) grave(s) identificada(s): {lista}. "
                "Acionar equipe médica imediatamente. "
                "Baseado em protocolo oficial — validação clínica obrigatória."
            )

        print(f"[gravidade] max_gravidade={max_gravidade} | graves={doencas_graves}")

        return {
            **estado,
            "max_gravidade": max_gravidade,
            "doencas_graves": doencas_graves,
            "doencas_nao_graves": doencas_nao_graves,
            "alerta": alerta,
        }

    return no_gravidade


# ─── Função de roteamento (usada pelo grafo) ─────────────────


def rotear_gravidade(estado: dict) -> str:
    """
    Função de roteamento para add_conditional_edges.

    Retorna o nome do próximo nó com base em max_gravidade.
    """
    if estado.get("max_gravidade") == "grave":
        return "alerta"
    return "fim_sem_exames"  # placeholder até M3 existir
```

- [ ] **Step 2.4: Rodar testes de gravidade**

```bash
cd langchain && python -m pytest tests/test_gravidade.py -v
```

Esperado: todos os testes PASS.

- [ ] **Step 2.5: Commit**

```bash
git add langchain/nos/gravidade.py langchain/tests/test_gravidade.py
git commit -m "feat(m2): nó de avaliação de gravidade com regras dengue+covid"
```

---

## Task 3: Criar nó de alerta de urgência

**Files:**
- Create: `langchain/nos/alerta.py`
- Create: `langchain/tests/test_alerta.py`

- [ ] **Step 3.1: Escrever teste do nó de alerta**

Criar `langchain/tests/test_alerta.py`:

```python
# tests/test_alerta.py
"""Testes do nó de alerta de urgência (M2 - path grave)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from nos.alerta import criar_no_alerta


def _estado_grave(doencas_graves=None, alerta=None):
    return {
        "sintomas": "febre alta, dor abdominal intensa, hipotensão",
        "doencas_suspeitas": doencas_graves or ["dengue"],
        "doencas_graves": doencas_graves or ["dengue"],
        "doencas_nao_graves": [],
        "max_gravidade": "grave",
        "alerta": alerta or "⚠️ ALERTA: DENGUE grave.",
        "gravidade": {"dengue": "grupo_d"},
        "documentos_recuperados": [],
        "justificativa_classificacao": "sinais de choque presentes",
        "fontes": ["MS Dengue 2024, p.12"],
    }


class TestNoAlerta:
    def test_imprime_alerta_no_estado(self):
        """O nó de alerta deve manter o campo 'alerta' no estado."""
        no = criar_no_alerta()
        resultado = no(_estado_grave())
        assert resultado["alerta"] is not None
        assert len(resultado["alerta"]) > 0

    def test_estado_preservado(self):
        """Todos os campos anteriores devem ser preservados."""
        no = criar_no_alerta()
        estado = _estado_grave()
        resultado = no(estado)
        assert resultado["sintomas"] == estado["sintomas"]
        assert resultado["doencas_graves"] == estado["doencas_graves"]
        assert resultado["max_gravidade"] == "grave"

    def test_alerta_contem_doenca(self):
        """A mensagem de alerta deve mencionar a doença grave."""
        no = criar_no_alerta()
        resultado = no(_estado_grave(doencas_graves=["dengue"]))
        # alerta foi setado pelo nó de gravidade — nó de alerta apenas loga/exibe
        assert resultado["alerta"] is not None
```

- [ ] **Step 3.2: Rodar para confirmar falha**

```bash
cd langchain && python -m pytest tests/test_alerta.py -v
```

Esperado: `ImportError: cannot import name 'criar_no_alerta'`

- [ ] **Step 3.3: Implementar `langchain/nos/alerta.py`**

```python
# nos/alerta.py
# ─────────────────────────────────────────────────────────────
# Nó de alerta de urgência (M2 — path grave).
#
# Este nó é acionado quando max_gravidade == "grave".
# Ele apenas exibe/loga o alerta composto pelo nó de gravidade
# e encerra o fluxo (até M3 existir, o path grave termina aqui).
# ─────────────────────────────────────────────────────────────


def criar_no_alerta():
    """
    Factory que retorna o nó de alerta de urgência.

    Recebe o estado com `alerta` já preenchido pelo nó de gravidade.
    Imprime o alerta e encerra o fluxo (até M3 adicionar tratamento urgente).
    """

    def no_alerta(estado: dict) -> dict:
        alerta = estado.get("alerta", "")
        print(f"\n{'='*60}")
        print(f"[alerta] {alerta}")
        print(f"{'='*60}\n")
        return {**estado}

    return no_alerta
```

- [ ] **Step 3.4: Rodar testes de alerta**

```bash
cd langchain && python -m pytest tests/test_alerta.py -v
```

Esperado: todos PASS.

- [ ] **Step 3.5: Commit**

```bash
git add langchain/nos/alerta.py langchain/tests/test_alerta.py
git commit -m "feat(m2): nó de alerta de urgência para casos graves"
```

---

## Task 4: Atualizar `__init__.py` dos nós

**Files:**
- Modify: `langchain/nos/__init__.py`

- [ ] **Step 4.1: Re-exportar os novos nós**

Abrir `langchain/nos/__init__.py` e adicionar:

```python
from nos.gravidade import criar_no_gravidade, rotear_gravidade
from nos.alerta import criar_no_alerta
```

- [ ] **Step 4.2: Verificar imports**

```bash
cd langchain && python -c "from nos import criar_no_gravidade, rotear_gravidade, criar_no_alerta; print('OK')"
```

Esperado: `OK`

- [ ] **Step 4.3: Commit**

```bash
git add langchain/nos/__init__.py
git commit -m "feat(m2): re-exportar nós de gravidade e alerta"
```

---

## Task 5: Integrar M2 ao grafo em `fluxo.py`

**Files:**
- Modify: `langchain/fluxo.py`

- [ ] **Step 5.1: Escrever testes de integração do grafo M2**

Criar `langchain/tests/test_fluxo_m2.py`:

```python
# tests/test_fluxo_m2.py
"""Testes de integração do grafo com M2 (gravidade + alerta)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock
from configs.dengue import CONFIG_DENGUE
from banco_de_conhecimento import parsear_protocolo, IndiceProtocolo
from fluxo import montar_fluxo


@pytest.fixture(scope="module")
def indice():
    chunks = parsear_protocolo(CONFIG_DENGUE)
    return IndiceProtocolo(chunks)


def _mock_llm(resposta_json: str):
    modelo = MagicMock()
    modelo.invoke.return_value = MagicMock(content=resposta_json)
    return modelo


class TestFluxoM2:
    def test_dengue_grupo_c_max_grave(self, indice):
        """Fluxo completo: M1 retorna grupo_c → M2 seta max_gravidade=grave."""
        modelo = _mock_llm(
            '{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "grupo_c", '
            '"justificativa": "sinais de alarme"}], "fontes": ["p.12"]}'
        )
        fluxo = montar_fluxo(modelo, indice)
        resultado = fluxo.invoke({"sintomas": "febre, dor abdominal, petéquias"})

        assert resultado["max_gravidade"] == "grave"
        assert "dengue" in resultado["doencas_graves"]
        assert resultado["alerta"] is not None

    def test_dengue_grupo_a_nao_grave(self, indice):
        """Fluxo completo: M1 retorna grupo_a → M2 seta max_gravidade=nao_grave."""
        modelo = _mock_llm(
            '{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "grupo_a", '
            '"justificativa": "sem sinais de alarme"}], "fontes": ["p.28"]}'
        )
        fluxo = montar_fluxo(modelo, indice)
        resultado = fluxo.invoke({"sintomas": "febre, dor de cabeça"})

        assert resultado["max_gravidade"] == "nao_grave"
        assert resultado["doencas_graves"] == []
        assert resultado["alerta"] is None

    def test_sem_doenca_identificada(self, indice):
        """M1 retorna lista vazia → M2 nao_grave sem alerta."""
        modelo = _mock_llm('{"doencas_suspeitas": [], "fontes": []}')
        fluxo = montar_fluxo(modelo, indice)
        resultado = fluxo.invoke({"sintomas": "dor no dedo"})

        assert resultado.get("max_gravidade") == "nao_grave"
        assert resultado.get("alerta") is None

    def test_dengue_grupo_d_alerta_presente(self, indice):
        """Grupo D (choque) → alerta deve conter DENGUE em maiúsculas."""
        modelo = _mock_llm(
            '{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "grupo_d", '
            '"justificativa": "choque"}], "fontes": ["p.32"]}'
        )
        fluxo = montar_fluxo(modelo, indice)
        resultado = fluxo.invoke({"sintomas": "hipotensão, extremidades frias"})

        assert "DENGUE" in resultado["alerta"]
```

- [ ] **Step 5.2: Rodar testes para confirmar falha**

```bash
cd langchain && python -m pytest tests/test_fluxo_m2.py -v
```

Esperado: `KeyError: 'max_gravidade'` — o grafo ainda não tem M2.

- [ ] **Step 5.3: Modificar `fluxo.py` para integrar M2**

Substituir o conteúdo de `langchain/fluxo.py`:

```python
# fluxo.py
# ─────────────────────────────────────────────────────────────
# Monta o fluxo LangGraph.
#
# Fluxo M1 + M2:
#
#   [INÍCIO]
#      │
#      ▼
#   [classificacao]  → recebe sintomas, busca nos protocolos,
#      │               retorna doenças suspeitas + gravidade
#      ▼
#   [gravidade]      → normaliza + classifica grave/nao_grave
#      │
#      ├── grave    → [alerta]    → FIM
#      │
#      └── nao_grave → FIM (M3 adicionará exames aqui)
#
# ─────────────────────────────────────────────────────────────

from typing import TypedDict
from langgraph.graph import StateGraph, END

from banco_de_conhecimento import IndiceProtocolo
from nos.classificacao import criar_no_classificacao
from nos.gravidade import criar_no_gravidade, rotear_gravidade
from nos.alerta import criar_no_alerta


# ─── Estado da conversa ───────────────────────────────────────


class EstadoTriagem(TypedDict, total=False):
    """
    Estado que viaja entre os nós do grafo de triagem.

    Campos preenchidos por:
      - Input (main.py): sintomas
      - Nó classificação (M1): documentos_recuperados, doencas_suspeitas,
                                gravidade, justificativa_classificacao, fontes
      - Nó gravidade (M2): max_gravidade, doencas_graves,
                            doencas_nao_graves, alerta
    """

    # Input
    sintomas: str

    # Classificação (Nó 1 — M1)
    documentos_recuperados: list[dict]
    doencas_suspeitas: list[str]
    gravidade: dict[str, str]       # {"dengue": "grupo_c"}
    justificativa_classificacao: str
    fontes: list[str]

    # Avaliação de Gravidade (Nó 2 — M2)
    max_gravidade: str              # "grave" ou "nao_grave"
    doencas_graves: list[str]       # subconjunto de doencas_suspeitas
    doencas_nao_graves: list[str]   # subconjunto de doencas_suspeitas
    alerta: str | None              # mensagem de urgência, None se nao_grave


# ─── Monta o grafo ────────────────────────────────────────────


def montar_fluxo(modelo, indice: IndiceProtocolo):
    """
    Monta e compila o fluxo LangGraph.

    Parâmetros:
      modelo - LLM carregado (ChatOpenAI / Ollama)
      indice - IndiceProtocolo com busca híbrida

    Retorna: grafo compilado pronto para invoke()
    """

    no_classificacao = criar_no_classificacao(indice, modelo)
    no_gravidade = criar_no_gravidade()
    no_alerta = criar_no_alerta()

    grafo = StateGraph(EstadoTriagem)

    # Nós
    grafo.add_node("classificacao", no_classificacao)
    grafo.add_node("gravidade", no_gravidade)
    grafo.add_node("alerta", no_alerta)

    # Arestas
    grafo.set_entry_point("classificacao")
    grafo.add_edge("classificacao", "gravidade")

    # Aresta condicional: grave → alerta | nao_grave → END
    grafo.add_conditional_edges(
        "gravidade",
        rotear_gravidade,
        {
            "alerta": "alerta",
            "fim_sem_exames": END,   # placeholder até M3
        },
    )
    grafo.add_edge("alerta", END)

    fluxo_compilado = grafo.compile()
    print("[fluxo] Grafo de triagem montado (M1: classificação | M2: gravidade + alerta)")
    return fluxo_compilado
```

- [ ] **Step 5.4: Rodar todos os testes**

```bash
cd langchain && python -m pytest tests/test_classificacao.py tests/test_gravidade.py tests/test_alerta.py tests/test_fluxo_m2.py -v
```

Esperado: todos PASS.

- [ ] **Step 5.5: Rodar a suite completa para garantir que nada quebrou**

```bash
cd langchain && python -m pytest tests/ -v --ignore=tests/test_e2e.py
```

Esperado: todos PASS.

- [ ] **Step 5.6: Commit final**

```bash
git add langchain/fluxo.py langchain/tests/test_fluxo_m2.py
git commit -m "feat(m2): integrar avaliação de gravidade + alerta ao grafo LangGraph"
```

---

## Task 6: Verificação final e smoke test

- [ ] **Step 6.1: Rodar suite completa de testes (exceto e2e)**

```bash
cd langchain && python -m pytest tests/ -v --ignore=tests/test_e2e.py
```

Esperado: todos PASS, nenhum warning crítico.

- [ ] **Step 6.2: Smoke test manual via main.py (se Ollama disponível)**

```bash
cd langchain && OLLAMA_BASE_URL=<url> python main.py
```

Informar sintomas: `"febre alta, dor abdominal intensa, vômitos persistentes, petéquias"`
Esperado: output com `max_gravidade: grave` e alerta de urgência impresso.

- [ ] **Step 6.3: Commit de fechamento do marco**

```bash
git add -A
git commit -m "feat(m2): marco M2 completo — avaliação de gravidade + alerta de urgência"
```

---

## Resumo do que M2 entrega

| Antes (M1) | Depois (M2) |
|-----------|-------------|
| Grafo termina em `classificacao → END` | Grafo: `classificacao → gravidade → [alerta\|END]` |
| `gravidade: {"dengue": "grupo_c"}` no estado | + `max_gravidade`, `doencas_graves`, `doencas_nao_graves`, `alerta` |
| Nenhum roteamento | Aresta condicional: grave aciona nó de alerta |
| LLM decide tudo (incluindo gravidade) | Regras determinísticas mapeiam grupo → grave/nao_grave |
