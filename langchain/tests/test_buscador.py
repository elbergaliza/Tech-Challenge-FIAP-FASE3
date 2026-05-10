# tests/test_buscador.py
"""Testes da busca híbrida (FAISS + BM25)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from configs.dengue import CONFIG_DENGUE
from banco_de_conhecimento import parsear_protocolo, IndiceProtocolo, busca_hibrida


@pytest.fixture(scope="module")
def indice():
    """Cria índice uma vez para todos os testes do módulo."""
    chunks = parsear_protocolo(CONFIG_DENGUE)
    return IndiceProtocolo(chunks)


def test_busca_hibrida_retorna_resultados(indice):
    """Busca deve retornar ao menos 1 resultado."""
    resultados = busca_hibrida(indice, "sinais de alarme dengue grave")
    assert len(resultados) > 0


def test_busca_hibrida_respeita_limite_k(indice):
    """Busca não deve retornar mais que k resultados."""
    resultados = busca_hibrida(indice, "febre", k=5)
    assert len(resultados) <= 5


def test_busca_hibrida_filtro_doenca(indice):
    """Filtro por doença deve retornar apenas chunks da doença filtrada."""
    resultados = busca_hibrida(indice, "tratamento", doenca="dengue")
    for doc in resultados:
        assert doc.metadata["doenca"] == "dengue"


def test_busca_hibrida_filtro_secao(indice):
    """Filtro por seção deve retornar apenas chunks daquela seção."""
    resultados = busca_hibrida(indice, "hidratação venosa", secao_tipo="tratamento")
    for doc in resultados:
        assert "tratamento" in doc.metadata["secao_tipo"]


def test_busca_hibrida_sem_resultados_filtro_invalido(indice):
    """Filtro por doença inexistente deve retornar lista vazia."""
    resultados = busca_hibrida(indice, "febre", doenca="doenca_inexistente")
    assert len(resultados) == 0
