# M1: Parser Dengue + Classificação — Plano de Implementação

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objetivo:** Evoluir o sistema existente para parsear protocolo de dengue (PDF), indexar com busca híbrida (FAISS + BM25) e classificar doenças via nó LangGraph — mantendo o loop interativo do `main.py`.

**Arquitetura:** Evoluir arquivos existentes (`main.py`, `fluxo.py`, `banco_de_conhecimento.py`, `carregador_do_modelo.py`) e adicionar módulos de suporte (`configs/`, `nos/`, `prompts/`). Reutilizar lógica do `covid-parse.py` adaptada como genérica.

**Tech Stack:** Python, LangGraph, LangChain, PyMuPDF (fitz), FAISS, rank_bm25, HuggingFaceEmbeddings (paraphrase-multilingual-MiniLM-L12-v2), Ollama Cloud (gemma3:4b)

---

## Estrutura de Arquivos

```
langchain/
├── main.py                        # MANTER — adaptar state inicial e imports
├── fluxo.py                       # EVOLUIR — novo grafo com nó de classificação
├── banco_de_conhecimento.py       # EVOLUIR — PDF + BM25 + busca híbrida
├── carregador_do_modelo.py        # ADAPTAR — trocar GPT-2 por Ollama (strategy)
├── configuracoes.py               # MANTER — adicionar novas configs
├── configs/
│   ├── __init__.py
│   ├── base.py                    # NOVO — ConfiguracaoProtocolo dataclass
│   └── dengue.py                  # NOVO — Config específica dengue
├── nos/
│   ├── __init__.py
│   └── classificacao.py           # NOVO — Nó de classificação
└── prompts/
    └── classificacao.txt          # NOVO — Prompt de classificação
```

---

## Task 1: ConfiguracaoProtocolo dataclass

**Files:**
- Create: `langchain/configs/__init__.py`
- Create: `langchain/configs/base.py`
- Create: `langchain/configs/dengue.py`

- [ ] **Step 1: Criar dataclass e config dengue**

```python
# langchain/configs/base.py
from dataclasses import dataclass, field


@dataclass
class ConfiguracaoProtocolo:
    """Configuração genérica para parsing de um protocolo médico."""
    nome: str                                    # ex: "dengue"
    caminho_pdf: str                             # caminho do PDF
    descricao_fonte: str                         # ex: "MS/SVS - Protocolo Dengue 2024"

    # Limpeza
    padroes_cabecalho: list[str] = field(default_factory=list)
    padroes_ruido: list[str] = field(default_factory=list)

    # Classificação de seções — por página (um chunk pode ter múltiplas tags)
    paginas_por_secao: dict[str, list[int]] = field(default_factory=dict)

    # Gravidade
    niveis_gravidade: list[str] = field(default_factory=list)

    # Chunking
    tamanho_chunk: int = 800
    sobreposicao_chunk: int = 200
```

```python
# langchain/configs/__init__.py
from .base import ConfiguracaoProtocolo
```

```python
# langchain/configs/dengue.py
from .base import ConfiguracaoProtocolo

CONFIG_DENGUE = ConfiguracaoProtocolo(
    nome="dengue",
    caminho_pdf="./dados/data/20200504-protocolomanejo-ver09.pdf",
    descricao_fonte="Ministério da Saúde - Protocolo de Manejo Clínico da Dengue (2024)",

    padroes_cabecalho=[
        r"Ministério da Saúde.*?Dengue.*?\n",
        r"DENGUE:.*?MANEJO CLÍNICO.*?\n",
    ],

    padroes_ruido=[
        "sumario", "formulario", "data de nascimento",
        "nome:___", "cpf:", "telefone:",
        "endereco:", "assinatura", "referencias bibliograficas",
        "notificacao imediata", "check-list",
        "composicao da equipe",
    ],

    paginas_por_secao={
        "sintomas": [11, 12],
        "classificacao": [12, 13, 25, 28],
        "sinais_alarme": [12, 32],
        "exames": [28, 31, 33, 36, 56],
        "tratamento": [28, 29, 30, 31, 33, 36, 77],
    },

    niveis_gravidade=["grupo_a", "grupo_b", "grupo_c", "grupo_d"],
)
```

- [ ] **Step 2: Commit**

```bash
git add langchain/configs/
git commit -m "feat(configs): adiciona ConfiguracaoProtocolo e config dengue"
```

---

## Task 2: Evoluir banco_de_conhecimento.py (PDF + BM25 + busca híbrida)

**Files:**
- Modify: `langchain/banco_de_conhecimento.py`
- Modify: `langchain/configuracoes.py`

- [ ] **Step 1: Adicionar novas configs em configuracoes.py**

Adicionar ao final de `configuracoes.py`:

```python
# Modelo de embedding multilíngue (melhor para português)
MODELO_DE_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Peso do BM25 na busca híbrida (0.6 = prioriza keywords)
PESO_BM25 = 0.6
```

- [ ] **Step 2: Reescrever banco_de_conhecimento.py para suportar PDF + busca híbrida**

```python
# banco_de_conhecimento.py
# ─────────────────────────────────────────────────────────────
# Banco de conhecimento com busca híbrida (FAISS + BM25).
#
# Evolução:
#   - Antes: CSV com perguntas e respostas, busca apenas semântica
#   - Agora: PDF de protocolos médicos, busca híbrida (semântica + keyword)
#
# Como funciona:
#   1. Parseia o PDF do protocolo em chunks
#   2. Classifica cada chunk por seção (baseado na página)
#   3. Indexa em FAISS (semântico) e BM25 (keyword)
#   4. Busca híbrida combina os dois rankings via RRF
# ─────────────────────────────────────────────────────────────

import os
import re
import unicodedata

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from configuracoes import (
    CAMINHO_DO_BANCO_DE_VETORES,
    QUANTIDADE_DE_RESULTADOS,
    MODELO_DE_EMBEDDING,
    PESO_BM25,
)
from configs.base import ConfiguracaoProtocolo


# ─── Funções auxiliares de parsing ────────────────────────────

def normalizar(texto: str) -> str:
    """Remove acentos e converte para minúsculo."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto


def _limpar_cabecalhos(texto: str, padroes: list[str]) -> str:
    for padrao in padroes:
        texto = re.sub(padrao, "", texto, flags=re.IGNORECASE)
    return texto.strip()


def _eh_ruido(texto: str, padroes_ruido: list[str]) -> bool:
    t = normalizar(texto)
    if any(p in t for p in padroes_ruido):
        return True
    if t.count(".") > 20 and "..." in t:
        return True
    return False


def _classificar_secao_por_pagina(pagina: int, paginas_por_secao: dict[str, list[int]]) -> list[str]:
    """Classifica chunk pelas seções mapeadas para sua página."""
    secoes = [secao for secao, paginas in paginas_por_secao.items() if pagina in paginas]
    return secoes if secoes else ["geral"]


def _detectar_gravidade(texto: str, niveis: list[str]) -> str | None:
    """Detecta grupo de gravidade mencionado no chunk."""
    t = normalizar(texto)
    for nivel in reversed(niveis):  # prioriza mais grave
        if nivel.replace("_", " ") in t:
            return nivel
    return None


# ─── Parsing de protocolo ─────────────────────────────────────

def parsear_protocolo(config: ConfiguracaoProtocolo) -> list[Document]:
    """
    Parseia PDF de protocolo médico em chunks com metadados ricos.

    Cada chunk recebe:
      - doenca: nome do protocolo
      - pagina: número da página no PDF
      - fonte: descrição da fonte
      - secao_tipo: lista de seções (pode ter múltiplas tags)
      - gravidade_grupo: grupo de gravidade detectado ou None
    """
    pdf = fitz.open(config.caminho_pdf)
    documentos = []

    for i, pagina in enumerate(pdf):
        texto = pagina.get_text()
        texto = _limpar_cabecalhos(texto, config.padroes_cabecalho)
        if texto.strip():
            documentos.append(Document(
                page_content=texto,
                metadata={"source": config.caminho_pdf, "pagina": i + 1}
            ))

    separador = RecursiveCharacterTextSplitter(
        chunk_size=config.tamanho_chunk,
        chunk_overlap=config.sobreposicao_chunk,
    )

    chunks = []
    for doc in documentos:
        for texto_chunk in separador.split_text(doc.page_content):
            if _eh_ruido(texto_chunk, config.padroes_ruido):
                continue

            pagina = doc.metadata["pagina"]
            secoes = _classificar_secao_por_pagina(pagina, config.paginas_por_secao)
            gravidade = _detectar_gravidade(texto_chunk, config.niveis_gravidade)

            chunks.append(Document(
                page_content=texto_chunk,
                metadata={
                    "doenca": config.nome,
                    "pagina": pagina,
                    "fonte": config.descricao_fonte,
                    "secao_tipo": secoes,
                    "gravidade_grupo": gravidade,
                }
            ))

    print(f"[banco_de_conhecimento] {len(chunks)} chunks gerados do protocolo '{config.nome}'")
    return chunks


# ─── Indexação (FAISS + BM25) ─────────────────────────────────

class IndiceProtocolo:
    """Índice híbrido (FAISS + BM25) para chunks de protocolos."""

    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self.embeddings = HuggingFaceEmbeddings(model_name=MODELO_DE_EMBEDDING)
        self.vector_store = FAISS.from_documents(chunks, embedding=self.embeddings)
        self.buscador_faiss = self.vector_store.as_retriever(
            search_kwargs={"k": QUANTIDADE_DE_RESULTADOS}
        )

        # BM25
        tokens = [normalizar(c.page_content).split() for c in chunks]
        self.bm25 = BM25Okapi(tokens)

    def salvar(self, caminho: str):
        """Persiste o FAISS index em disco."""
        os.makedirs(caminho, exist_ok=True)
        self.vector_store.save_local(caminho)

    @classmethod
    def carregar(cls, caminho: str, chunks: list[Document]):
        """Carrega FAISS de disco e reconstrói BM25."""
        instancia = cls.__new__(cls)
        instancia.chunks = chunks
        instancia.embeddings = HuggingFaceEmbeddings(model_name=MODELO_DE_EMBEDDING)
        instancia.vector_store = FAISS.load_local(
            caminho, instancia.embeddings, allow_dangerous_deserialization=True
        )
        instancia.buscador_faiss = instancia.vector_store.as_retriever(
            search_kwargs={"k": QUANTIDADE_DE_RESULTADOS}
        )
        tokens = [normalizar(c.page_content).split() for c in chunks]
        instancia.bm25 = BM25Okapi(tokens)
        return instancia


# ─── Busca híbrida ────────────────────────────────────────────

def busca_hibrida(
    indice: IndiceProtocolo,
    consulta: str,
    k: int = QUANTIDADE_DE_RESULTADOS,
    doenca: str | None = None,
    secao_tipo: str | None = None,
) -> list[Document]:
    """
    Busca híbrida (FAISS + BM25) com filtro opcional por metadados.

    FAISS encontra por similaridade semântica (sinônimos, paráfrases).
    BM25 encontra por correspondência exata de palavras-chave.
    RRF combina os dois rankings: chunks relevantes para ambos sobem no ranking.
    """

    # FAISS (semântico)
    docs_faiss = indice.buscador_faiss.invoke(consulta)

    # BM25 (keyword)
    tokens = normalizar(consulta).split()
    scores_bm25 = indice.bm25.get_scores(tokens)
    top_bm25 = sorted(range(len(scores_bm25)), key=lambda i: scores_bm25[i], reverse=True)[:k]
    docs_bm25 = [indice.chunks[i] for i in top_bm25]

    # Fusão RRF (Reciprocal Rank Fusion)
    peso_faiss = 1 - PESO_BM25
    mapa_scores = {}
    todos_docs = {}

    for posicao, doc in enumerate(docs_bm25):
        chave = doc.page_content[:200]
        mapa_scores[chave] = mapa_scores.get(chave, 0) + PESO_BM25 * (1 / (posicao + 1))
        todos_docs[chave] = doc

    for posicao, doc in enumerate(docs_faiss):
        chave = doc.page_content[:200]
        mapa_scores[chave] = mapa_scores.get(chave, 0) + peso_faiss * (1 / (posicao + 1))
        todos_docs[chave] = doc

    # Ordena por score e coleta
    ranking = sorted(mapa_scores.items(), key=lambda x: x[1], reverse=True)
    resultados = [todos_docs[chave] for chave, _ in ranking]

    # Filtro por metadados
    if doenca:
        resultados = [d for d in resultados if d.metadata.get("doenca") == doenca]
    if secao_tipo:
        resultados = [d for d in resultados if secao_tipo in d.metadata.get("secao_tipo", [])]

    return resultados[:k]


# ─── Interface pública (compatível com main.py) ──────────────

def construir_indice(config: ConfiguracaoProtocolo) -> IndiceProtocolo:
    """
    Constrói ou carrega o índice do protocolo.
    Substitui a antiga obter_buscador() com suporte a busca híbrida.
    """
    if os.path.exists(CAMINHO_DO_BANCO_DE_VETORES):
        print("[banco_de_conhecimento] Índice encontrado. Carregando do disco...")
        chunks = parsear_protocolo(config)
        return IndiceProtocolo.carregar(CAMINHO_DO_BANCO_DE_VETORES, chunks)
    else:
        print("[banco_de_conhecimento] Índice não encontrado. Criando do zero...")
        chunks = parsear_protocolo(config)
        indice = IndiceProtocolo(chunks)
        indice.salvar(CAMINHO_DO_BANCO_DE_VETORES)
        print(f"[banco_de_conhecimento] Índice salvo em: {CAMINHO_DO_BANCO_DE_VETORES}")
        return indice
```

- [ ] **Step 3: Commit**

```bash
git add langchain/banco_de_conhecimento.py langchain/configuracoes.py
git commit -m "feat(banco): evolui para PDF + busca híbrida (FAISS + BM25)"
```

---

## Task 3: Adaptar carregador_do_modelo.py (Ollama strategy)

**Files:**
- Modify: `langchain/carregador_do_modelo.py`

- [ ] **Step 1: Adaptar para suportar Ollama Cloud com fallback**

```python
# carregador_do_modelo.py
# ─────────────────────────────────────────────────────────────
# Carregamento do modelo de linguagem.
#
# Suporta múltiplos providers via variáveis de ambiente:
#   - OLLAMA_BASE_URL + OLLAMA_API_KEY → Ollama Cloud (padrão)
#   - OPENAI_API_KEY → OpenAI API
#
# Configurável para futura integração de modelo fine-tuned local.
# ─────────────────────────────────────────────────────────────

import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def carregar_modelo():
    """
    Carrega o modelo LLM conforme variáveis de ambiente.

    Prioridade:
      1. Ollama Cloud (se OLLAMA_BASE_URL definido)
      2. OpenAI (se OPENAI_API_KEY definido)
    """

    ollama_url = os.environ.get("OLLAMA_BASE_URL")
    ollama_key = os.environ.get("OLLAMA_API_KEY", "")

    if ollama_url:
        print(f"[carregador_do_modelo] Usando Ollama Cloud: {ollama_url}")
        modelo = ChatOpenAI(
            model=os.environ.get("OLLAMA_MODEL", "gemma3:4b"),
            base_url=f"{ollama_url}/v1",
            api_key=ollama_key,
            temperature=0.3,
            max_tokens=1000,
            http_client=httpx.Client(verify=False),
        )
    elif os.environ.get("OPENAI_API_KEY"):
        print("[carregador_do_modelo] Usando OpenAI API")
        modelo = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            max_tokens=1000,
        )
    else:
        raise RuntimeError(
            "Nenhum provider de LLM configurado. "
            "Defina OLLAMA_BASE_URL ou OPENAI_API_KEY no .env"
        )

    print("[carregador_do_modelo] Modelo carregado com sucesso!")
    return modelo
```

- [ ] **Step 2: Commit**

```bash
git add langchain/carregador_do_modelo.py
git commit -m "feat(modelo): adapta carregador para Ollama Cloud e OpenAI"
```

---

## Task 4: Nó de classificação

**Files:**
- Create: `langchain/nos/__init__.py`
- Create: `langchain/nos/classificacao.py`
- Create: `langchain/prompts/classificacao.txt`

- [ ] **Step 1: Criar prompt de classificação**

```text
# langchain/prompts/classificacao.txt
Você é um médico especialista em triagem hospitalar.
Com base EXCLUSIVAMENTE nos trechos de protocolos oficiais abaixo, classifique:
1. Doença(s) provável(is)
2. Nível de gravidade conforme o protocolo da doença

CONTEXTO DOS PROTOCOLOS:
{contexto}

SINTOMAS DO PACIENTE:
{sintomas}

RESPONDA em JSON válido (sem markdown, sem ```):
{{
  "doencas_suspeitas": [
    {{"doenca": "nome_da_doenca", "gravidade": "nivel_conforme_protocolo", "justificativa": "breve explicação"}}
  ],
  "fontes": ["protocolo X, p.Y"]
}}

REGRAS:
- Use APENAS doenças que estão nos protocolos fornecidos
- A gravidade deve usar a nomenclatura exata do protocolo (ex: "grupo_a", "grupo_b", "grupo_c", "grupo_d" para dengue)
- Se não encontrar informação suficiente, retorne doencas_suspeitas vazio
- Cite a página fonte de cada classificação
```

- [ ] **Step 2: Implementar nó de classificação**

```python
# langchain/nos/classificacao.py
# ─────────────────────────────────────────────────────────────
# Nó de classificação: recebe sintomas e retorna doenças suspeitas
# com gravidade, justificativa e fontes.
# ─────────────────────────────────────────────────────────────

import json
import re
from pathlib import Path
from banco_de_conhecimento import IndiceProtocolo, busca_hibrida


CAMINHO_PROMPT = Path(__file__).parent.parent / "prompts" / "classificacao.txt"


def criar_no_classificacao(indice: IndiceProtocolo, modelo):
    """
    Factory que retorna o nó de classificação.

    O nó:
      1. Faz busca híbrida sem filtro (traz de todos os protocolos)
      2. Monta prompt com chunks recuperados
      3. Invoca o LLM para classificar
      4. Parseia JSON e atualiza o estado
    """

    template_prompt = CAMINHO_PROMPT.read_text(encoding="utf-8")

    def no_classificacao(estado: dict) -> dict:
        print("[classificacao] Classificando sintomas...")
        sintomas = estado["sintomas"]

        # Busca sem filtro (classificação busca em tudo)
        docs = busca_hibrida(indice, sintomas)
        docs_resumidos = [
            {"texto": d.page_content, "pagina": d.metadata.get("pagina", ""), "fonte": d.metadata.get("fonte", "")}
            for d in docs
        ]

        # Monta contexto
        contexto = "\n\n".join(
            f"[{d['fonte']}, p.{d['pagina']}]\n{d['texto']}" for d in docs_resumidos
        )

        # Monta prompt e invoca LLM
        prompt = template_prompt.format(contexto=contexto, sintomas=sintomas)
        resposta = modelo.invoke(prompt)
        conteudo = resposta.content if hasattr(resposta, "content") else str(resposta)

        # Parse JSON
        try:
            resultado = json.loads(conteudo)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", conteudo, re.DOTALL)
            if match:
                resultado = json.loads(match.group())
            else:
                resultado = {"doencas_suspeitas": [], "fontes": []}

        # Atualiza estado
        doencas = [d["doenca"] for d in resultado.get("doencas_suspeitas", [])]
        gravidade = {d["doenca"]: d["gravidade"] for d in resultado.get("doencas_suspeitas", [])}
        justificativa = "\n".join(
            f"- {d['doenca']}: {d.get('justificativa', '')}"
            for d in resultado.get("doencas_suspeitas", [])
        )

        print(f"[classificacao] Doenças suspeitas: {doencas}")
        print(f"[classificacao] Gravidade: {gravidade}")

        return {
            **estado,
            "documentos_recuperados": docs_resumidos,
            "doencas_suspeitas": doencas,
            "gravidade": gravidade,
            "justificativa_classificacao": justificativa,
            "fontes": resultado.get("fontes", []),
        }

    return no_classificacao
```

```python
# langchain/nos/__init__.py
from .classificacao import criar_no_classificacao
```

- [ ] **Step 3: Commit**

```bash
git add langchain/nos/ langchain/prompts/
git commit -m "feat(nos): adiciona nó de classificação com busca híbrida"
```

---

## Task 5: Evoluir fluxo.py (novo grafo com classificação)

**Files:**
- Modify: `langchain/fluxo.py`

- [ ] **Step 1: Reescrever fluxo.py com o novo grafo**

```python
# fluxo.py
# ─────────────────────────────────────────────────────────────
# Monta o fluxo LangGraph.
#
# Fluxo M1 (classificação):
#
#   [INÍCIO]
#      │
#      ▼
#   [classificacao]  → recebe sintomas, busca nos protocolos,
#      │               retorna doenças suspeitas + gravidade
#      ▼
#   [FIM]
#
# Próximas milestones adicionarão nós de avaliação, exames,
# tratamento e relatório.
# ─────────────────────────────────────────────────────────────

from typing import TypedDict
from langgraph.graph import StateGraph, END

from banco_de_conhecimento import IndiceProtocolo
from nos.classificacao import criar_no_classificacao


# ─── Estado da conversa ───────────────────────────────────────

class EstadoTriagem(TypedDict, total=False):
    """
    Estado que viaja entre os nós do grafo de triagem.

    Campos preenchidos por:
      - Input (main.py): sintomas
      - Nó classificação: documentos_recuperados, doencas_suspeitas, gravidade, etc.
    """
    # Input
    sintomas: str

    # Classificação (Nó 1)
    documentos_recuperados: list[dict]
    doencas_suspeitas: list[str]
    gravidade: dict[str, str]              # {"dengue": "grupo_c"}
    justificativa_classificacao: str
    fontes: list[str]


# ─── Monta o grafo ────────────────────────────────────────────

def montar_fluxo(modelo, indice: IndiceProtocolo):
    """
    Monta e compila o fluxo LangGraph.

    Parâmetros:
      modelo - LLM carregado (ChatOpenAI)
      indice - IndiceProtocolo com busca híbrida

    Retorna: grafo compilado pronto para invoke()
    """

    # Cria o nó de classificação
    no_classificacao = criar_no_classificacao(indice, modelo)

    # Monta o grafo
    grafo = StateGraph(EstadoTriagem)
    grafo.add_node("classificacao", no_classificacao)
    grafo.set_entry_point("classificacao")
    grafo.add_edge("classificacao", END)

    fluxo_compilado = grafo.compile()
    print("[fluxo] Grafo de triagem montado (M1: classificação)")
    return fluxo_compilado
```

- [ ] **Step 2: Commit**

```bash
git add langchain/fluxo.py
git commit -m "feat(fluxo): evolui grafo para classificação com EstadoTriagem"
```

---

## Task 6: Adaptar main.py

**Files:**
- Modify: `langchain/main.py`

- [ ] **Step 1: Adaptar main.py para novo fluxo**

```python
# main.py
# ─────────────────────────────────────────────────────────────
# Ponto de entrada do sistema.
#
# Para rodar:
#   python main.py
#
# Ordem de execução:
#   1. Carrega o modelo (carregador_do_modelo.py)
#   2. Constrói o índice de protocolos (banco_de_conhecimento.py)
#   3. Monta o fluxo de triagem (fluxo.py)
#   4. Fica em loop esperando sintomas
# ─────────────────────────────────────────────────────────────

from carregador_do_modelo import carregar_modelo
from banco_de_conhecimento import construir_indice
from fluxo import montar_fluxo
from configs.dengue import CONFIG_DENGUE


def fazer_triagem(sintomas: str, fluxo) -> dict:
    """
    Envia sintomas para o fluxo de triagem e retorna o resultado.

    Parâmetros:
      sintomas - texto livre descrevendo sintomas do paciente
      fluxo    - grafo compilado retornado por montar_fluxo()

    Retorna:
      dicionário com: doencas_suspeitas, gravidade, justificativa, fontes
    """
    estado_inicial = {"sintomas": sintomas}
    estado_final = fluxo.invoke(estado_inicial)
    return estado_final


def main():
    """Função principal: inicializa o sistema e inicia o loop."""

    print("=" * 55)
    print("  Assistente de Triagem Médica — Inicializando")
    print("=" * 55)

    # ── Passo 1: Carrega o modelo ─────────────────────────────
    print("\n[1/3] Carregando o modelo de linguagem...")
    modelo = carregar_modelo()

    # ── Passo 2: Constrói o índice de protocolos ──────────────
    print("\n[2/3] Construindo índice de protocolos...")
    indice = construir_indice(CONFIG_DENGUE)

    # ── Passo 3: Monta o fluxo de triagem ─────────────────────
    print("\n[3/3] Montando o fluxo de triagem...")
    fluxo = montar_fluxo(modelo, indice)

    print("\n" + "=" * 55)
    print("  Sistema pronto! Digite 'sair' para encerrar.")
    print("  Descreva os sintomas do paciente para classificação.")
    print("=" * 55 + "\n")

    # ── Loop de triagem ───────────────────────────────────────
    while True:
        sintomas = input("Sintomas: ").strip()

        if sintomas.lower() in ("sair", "exit", "quit"):
            print("Encerrando o assistente. Até logo!")
            break

        if not sintomas:
            continue

        resultado = fazer_triagem(sintomas, fluxo)

        print("\n" + "─" * 55)
        print(f"Doenças suspeitas : {resultado.get('doencas_suspeitas', [])}")
        print(f"Gravidade         : {resultado.get('gravidade', {})}")
        print(f"\nJustificativa:\n{resultado.get('justificativa_classificacao', '')}")
        print(f"\nFontes: {resultado.get('fontes', [])}")
        print("─" * 55 + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add langchain/main.py
git commit -m "feat(main): adapta loop para triagem com classificação"
```

---

## Task 7: Teste end-to-end

**Files:**
- Create: `langchain/tests/test_classificacao.py`

- [ ] **Step 1: Escrever teste de integração**

```python
# langchain/tests/test_classificacao.py
"""Teste end-to-end do M1: sintomas → classificação."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.dengue import CONFIG_DENGUE
from banco_de_conhecimento import parsear_protocolo, IndiceProtocolo
from fluxo import montar_fluxo


def test_classificar_dengue_grave():
    """Sintomas de dengue grupo C devem retornar classificação grave."""
    if not os.environ.get("OLLAMA_BASE_URL"):
        import pytest
        pytest.skip("OLLAMA_BASE_URL não configurado")

    from carregador_do_modelo import carregar_modelo

    chunks = parsear_protocolo(CONFIG_DENGUE)
    indice = IndiceProtocolo(chunks)
    modelo = carregar_modelo()

    fluxo = montar_fluxo(modelo, indice)
    resultado = fluxo.invoke({"sintomas": "febre alta, dor abdominal intensa, vômitos persistentes, petéquias"})

    assert "dengue" in [d.lower() for d in resultado["doencas_suspeitas"]]
    assert any("grupo_c" in v or "grupo_d" in v for v in resultado["gravidade"].values())


def test_parser_gera_chunks():
    """Parser deve gerar chunks com metadados corretos."""
    chunks = parsear_protocolo(CONFIG_DENGUE)
    assert len(chunks) > 0
    for chunk in chunks[:5]:
        assert chunk.metadata["doenca"] == "dengue"
        assert isinstance(chunk.metadata["secao_tipo"], list)
        assert "pagina" in chunk.metadata
```

- [ ] **Step 2: Rodar testes**

Run: `cd langchain && python -m pytest tests/test_classificacao.py -v`
Expected: `test_parser_gera_chunks` PASS, `test_classificar_dengue_grave` PASS ou SKIP

- [ ] **Step 3: Testar manualmente**

Run: `cd langchain && python main.py`
Testar com: "febre alta, dor abdominal intensa, vômitos persistentes, petéquias"
Expected: dengue, grupo_c ou grupo_d

- [ ] **Step 4: Commit**

```bash
git add langchain/tests/
git commit -m "test(m1): testes de integração para classificação"
```

---

## Validação do M1

O M1 está completo quando:
1. ✅ `python main.py` inicia e aceita input de sintomas no loop
2. ✅ Parser gera chunks com metadados `{doenca, secao_tipo (lista), gravidade_grupo, pagina, fonte}`
3. ✅ Busca híbrida retorna chunks relevantes
4. ✅ Caso "febre alta + dor abdominal + vômitos + petéquias" → "dengue, grupo_c"
5. ✅ Caso "febre + dor de cabeça + sem alarme" → "dengue, grupo_a ou grupo_b"
