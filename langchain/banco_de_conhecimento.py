# banco_de_conhecimento.py
# ─────────────────────────────────────────────────────────────
# Esse arquivo é responsável por construir o RAG (Retrieval-Augmented Generation).
#
# O que é RAG?
#   É uma técnica onde, antes de responder, o sistema BUSCA no banco de dados
#   os trechos mais parecidos com a pergunta feita. Esses trechos são
#   passados junto com a pergunta para o modelo, que usa as duas informações
#   para dar uma resposta mais precisa e embasada.
#
# Como funciona aqui:
#   1. Lemos o CSV com perguntas e respostas
#   2. Transformamos cada par (pergunta + resposta) em texto
#   3. Geramos um "vetor" (número que representa o significado) para cada texto
#   4. Salvamos esses vetores no banco de vetores (FAISS)
#   5. Quando chega uma pergunta nova, buscamos os vetores mais parecidos
# ─────────────────────────────────────────────────────────────

import os
import pandas

# FAISS é o banco de vetores — armazena e busca textos por similaridade
from langchain_community.vectorstores import FAISS

# HuggingFaceEmbeddings transforma texto em vetores numéricos
from langchain_community.embeddings import HuggingFaceEmbeddings

# Document é o formato que o LangChain usa para representar um texto
from langchain_core.documents import Document

from configuracoes import (
    CAMINHO_DO_DATASET,
    CAMINHO_DO_BANCO_DE_VETORES,
    QUANTIDADE_DE_RESULTADOS,
)


def carregar_perguntas_como_documentos():
    """
    Lê o arquivo CSV e transforma cada linha em um Document do LangChain.

    Cada Document contém:
    - page_content: o texto que vai ser indexado (pergunta + resposta juntas)
    - metadata: informações extras sobre esse documento (de onde veio, qual linha)

    Retorna: lista de Documents prontos para serem indexados
    """

    # Lê o CSV e guarda em uma tabela
    tabela = pandas.read_csv(CAMINHO_DO_DATASET)

    # Verifica se as colunas certas existem no CSV
    colunas_necessarias = {"question", "answer"}
    if not colunas_necessarias.issubset(tabela.columns):
        raise ValueError(
            f"O CSV precisa ter as colunas 'question' e 'answer'. "
            f"Colunas encontradas: {list(tabela.columns)}"
        )

    lista_de_documentos = []

    # Percorre cada linha da tabela
    for numero_da_linha, linha in tabela.iterrows():

        # Junta a pergunta e a resposta em um único texto
        # Isso ajuda o modelo a entender o contexto completo
        texto_completo = f"Pergunta: {linha['question']}\nResposta: {linha['answer']}"

        # Cria o Document com o texto e metadados
        documento = Document(
            page_content=texto_completo,
            metadata={
                "fonte": "dataset_medico",
                "linha": numero_da_linha,
            }
        )

        lista_de_documentos.append(documento)

    print(f"[banco_de_conhecimento] {len(lista_de_documentos)} documentos carregados do CSV.")
    return lista_de_documentos


def construir_ou_carregar_banco():
    """
    Constrói o banco de vetores se ele não existir ainda.
    Se já existir em disco, apenas carrega (evita reprocessar tudo).

    O modelo de embedding usado é o 'all-MiniLM-L6-v2':
    - É leve e rápido
    - Funciona bem para textos em inglês
    - Roda na CPU sem problemas

    Retorna: o banco de vetores pronto para buscas
    """

    # Modelo que transforma texto em vetores numéricos
    # Esse modelo é baixado automaticamente na primeira execução
    modelo_de_embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Verifica se o banco já existe em disco
    if os.path.exists(CAMINHO_DO_BANCO_DE_VETORES):
        print("[banco_de_conhecimento] Banco de vetores encontrado. Carregando do disco...")

        banco = FAISS.load_local(
            CAMINHO_DO_BANCO_DE_VETORES,
            modelo_de_embedding,
            allow_dangerous_deserialization=True  # necessário para carregar do disco
        )

    else:
        print("[banco_de_conhecimento] Banco de vetores não encontrado. Criando do zero...")

        # Carrega os documentos do CSV
        documentos = carregar_perguntas_como_documentos()

        # Cria o banco indexando todos os documentos
        banco = FAISS.from_documents(documentos, modelo_de_embedding)

        # Salva o banco em disco para não precisar recriar na próxima vez
        os.makedirs(CAMINHO_DO_BANCO_DE_VETORES, exist_ok=True)
        banco.save_local(CAMINHO_DO_BANCO_DE_VETORES)
        print(f"[banco_de_conhecimento] Banco salvo em: {CAMINHO_DO_BANCO_DE_VETORES}")

    return banco


def obter_buscador():
    """
    Retorna o buscador pronto para uso.

    O buscador recebe uma pergunta em texto e devolve os documentos
    mais parecidos do banco de vetores.

    Retorna: retriever do LangChain
    """

    banco = construir_ou_carregar_banco()

    # Cria o buscador com a quantidade de resultados definida nas configurações
    buscador = banco.as_retriever(
        search_type="similarity",               # busca por similaridade de significado
        search_kwargs={"k": QUANTIDADE_DE_RESULTADOS}  # quantos resultados retornar
    )

    return buscador
