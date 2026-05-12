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
import pickle
import json
import hashlib
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict

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


def _classificar_secao_por_pagina(
    pagina: int, paginas_por_secao: dict[str, list[int]]
) -> list[str]:
    """Classifica chunk pelas seções mapeadas para sua página."""
    secoes = [
        secao for secao, paginas in paginas_por_secao.items() if pagina in paginas
    ]
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
            documentos.append(
                Document(
                    page_content=texto,
                    metadata={"source": config.caminho_pdf, "pagina": i + 1},
                )
            )

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
            if secoes == ["geral"]:
                continue

            gravidade = _detectar_gravidade(texto_chunk, config.niveis_gravidade)

            chunks.append(
                Document(
                    page_content=texto_chunk,
                    metadata={
                        "doenca": config.nome,
                        "pagina": pagina,
                        "fonte": config.descricao_fonte,
                        "url_fonte": config.url_fonte,
                        "secao_tipo": secoes,
                        "gravidade_grupo": gravidade,
                    },
                )
            )

    print(
        f"[banco_de_conhecimento] {len(chunks)} chunks gerados do protocolo '{config.nome}'"
    )
    return chunks


# ─── Indexação (FAISS + BM25) ─────────────────────────────────


_embedding_cache = None


def _obter_embeddings():
    """Retorna instância única do modelo de embedding (singleton)."""
    global _embedding_cache
    if _embedding_cache is None:
        print("[banco_de_conhecimento] Carregando modelo de embedding...")
        _embedding_cache = HuggingFaceEmbeddings(model_name=MODELO_DE_EMBEDDING)
    return _embedding_cache


class IndiceProtocolo:
    """Índice híbrido (FAISS + BM25) para chunks de protocolos."""

    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self.embeddings = _obter_embeddings()
        self.vector_store = FAISS.from_documents(chunks, embedding=self.embeddings)
        self.buscador_faiss = self.vector_store.as_retriever(
            search_kwargs={"k": QUANTIDADE_DE_RESULTADOS}
        )

        # BM25
        tokens = [normalizar(c.page_content).split() for c in chunks]
        self.bm25 = BM25Okapi(tokens)

    def salvar(self, caminho: str):
        """Persiste o FAISS index e chunks em disco."""
        os.makedirs(caminho, exist_ok=True)
        self.vector_store.save_local(caminho)
        # Salva chunks para não precisar re-parsear o PDF
        with open(os.path.join(caminho, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def carregar(cls, caminho: str):
        """Carrega FAISS + chunks de disco (sem re-parsear PDF)."""
        # Carrega chunks
        with open(os.path.join(caminho, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)

        instancia = cls.__new__(cls)
        instancia.chunks = chunks
        instancia.embeddings = _obter_embeddings()
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
    docs_faiss = indice.vector_store.similarity_search(consulta, k=k)

    # BM25 (keyword)
    tokens = normalizar(consulta).split()
    scores_bm25 = indice.bm25.get_scores(tokens)
    top_bm25 = sorted(
        range(len(scores_bm25)), key=lambda i: scores_bm25[i], reverse=True
    )[:k]
    docs_bm25 = [indice.chunks[i] for i in top_bm25]

    # Fusão RRF
    peso_faiss = 1 - PESO_BM25
    mapa_scores: dict = {}
    todos_docs: dict = {}

    for pos, doc in enumerate(docs_bm25):
        chave = doc.page_content[:200]
        mapa_scores[chave] = mapa_scores.get(chave, 0) + PESO_BM25 * (1 / (pos + 1))
        todos_docs[chave] = doc

    for pos, doc in enumerate(docs_faiss):
        chave = doc.page_content[:200]
        mapa_scores[chave] = mapa_scores.get(chave, 0) + peso_faiss * (1 / (pos + 1))
        todos_docs[chave] = doc

    ranking = sorted(mapa_scores.items(), key=lambda x: x[1], reverse=True)
    resultados = [todos_docs[chave] for chave, _ in ranking]

    # Filtro por metadados
    if doenca:
        resultados = [d for d in resultados if d.metadata.get("doenca") == doenca]
    if secao_tipo:
        resultados = [d for d in resultados if secao_tipo in d.metadata.get("secao_tipo", [])]

    return resultados[:k]


# ─── Interface pública (compatível com main.py) ──────────────


def _normalizar_configs(
    configs: Sequence[ConfiguracaoProtocolo] | ConfiguracaoProtocolo,
) -> list[ConfiguracaoProtocolo]:
    if isinstance(configs, ConfiguracaoProtocolo):
        return [configs]
    return list(configs)


def _montar_manifesto(configs: Sequence[ConfiguracaoProtocolo]) -> dict:
    protocolos = [asdict(c) for c in sorted(configs, key=lambda item: item.nome)]
    return {
        "protocolos": protocolos,
        "modelo_embedding": MODELO_DE_EMBEDDING,
    }


def _chave_cache(manifesto: dict) -> str:
    manifesto_json = json.dumps(manifesto, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(manifesto_json.encode("utf-8")).hexdigest()[:12]
    return f"idx-{digest}"


def _salvar_manifesto(caminho_cache: str, manifesto: dict):
    caminho_manifesto = os.path.join(caminho_cache, "manifest.json")
    with open(caminho_manifesto, "w", encoding="utf-8") as f:
        json.dump(manifesto, f, indent=2, ensure_ascii=False)


def construir_indice(
    configs: Sequence[ConfiguracaoProtocolo] | ConfiguracaoProtocolo,
) -> IndiceProtocolo:
    """
    Constrói ou carrega um índice para 1..N protocolos.

    Se o cache existe (FAISS + chunks.pkl), carrega direto sem re-parsear PDFs.
    Caso contrário, parseia os PDFs, cria o índice e salva em disco.
    """
    lista_configs = _normalizar_configs(configs)
    if not lista_configs:
        raise ValueError("A lista de protocolos para indexação não pode ser vazia.")

    manifesto = _montar_manifesto(lista_configs)
    chave_cache = _chave_cache(manifesto)
    nomes_protocolos = ", ".join(c.nome for c in lista_configs)
    caminho_cache = os.path.join(CAMINHO_DO_BANCO_DE_VETORES, chave_cache)
    chunks_cache = os.path.join(caminho_cache, "chunks.pkl")

    if os.path.exists(chunks_cache):
        print(
            "[banco_de_conhecimento] Cache encontrado para "
            f"'{nomes_protocolos}' ({chave_cache}). Carregando..."
        )
        return IndiceProtocolo.carregar(caminho_cache)

    print(
        "[banco_de_conhecimento] Cache não encontrado para "
        f"'{nomes_protocolos}' ({chave_cache}). Criando..."
    )
    chunks = []
    for config in lista_configs:
        chunks.extend(parsear_protocolo(config))

    indice = IndiceProtocolo(chunks)
    indice.salvar(caminho_cache)
    _salvar_manifesto(caminho_cache, manifesto)
    print(f"[banco_de_conhecimento] Cache salvo em: {caminho_cache}")
    return indice


def construir_indice_unico(config: ConfiguracaoProtocolo) -> IndiceProtocolo:
    """Wrapper de retrocompatibilidade para um único protocolo."""
    return construir_indice([config])
