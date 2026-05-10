# tests/test_classificacao.py
"""Testes do nó de classificação e fluxo completo."""

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
    """Cria índice uma vez para todos os testes do módulo."""
    chunks = parsear_protocolo(CONFIG_DENGUE)
    return IndiceProtocolo(chunks)


def _criar_modelo_mock(resposta_json: str):
    """Cria mock do modelo LLM com resposta fixa."""
    modelo = MagicMock()
    modelo.invoke.return_value = MagicMock(content=resposta_json)
    return modelo


class TestNoClassificacao:
    """Testes do nó de classificação com LLM mockado."""

    def test_classificacao_dengue_grave(self, indice):
        """Deve classificar corretamente dengue grupo C."""
        modelo = _criar_modelo_mock(
            '{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "grupo_c", "justificativa": "sinais de alarme presentes"}], "fontes": ["p.12"]}'
        )
        fluxo = montar_fluxo(modelo, indice)

        resultado = fluxo.invoke(
            {"sintomas": "febre alta, dor abdominal intensa, vômitos persistentes"}
        )

        assert "dengue" in resultado["doencas_suspeitas"]
        assert resultado["gravidade"]["dengue"] == "grupo_c"
        assert len(resultado["fontes"]) > 0

    def test_classificacao_dengue_leve(self, indice):
        """Deve classificar corretamente dengue grupo A."""
        modelo = _criar_modelo_mock(
            '{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "grupo_a", "justificativa": "sem sinais de alarme"}], "fontes": ["p.28"]}'
        )
        fluxo = montar_fluxo(modelo, indice)

        resultado = fluxo.invoke(
            {"sintomas": "febre há 2 dias, dor de cabeça, sem sinais de alarme"}
        )

        assert "dengue" in resultado["doencas_suspeitas"]
        assert resultado["gravidade"]["dengue"] == "grupo_a"

    def test_classificacao_sem_doenca_identificada(self, indice):
        """Deve retornar lista vazia quando LLM não identifica doença."""
        modelo = _criar_modelo_mock('{"doencas_suspeitas": [], "fontes": []}')
        fluxo = montar_fluxo(modelo, indice)

        resultado = fluxo.invoke({"sintomas": "dor no dedo mindinho"})

        assert resultado["doencas_suspeitas"] == []
        assert resultado["gravidade"] == {}

    def test_classificacao_json_invalido_fallback(self, indice):
        """Deve fazer fallback quando LLM retorna JSON inválido."""
        modelo = _criar_modelo_mock(
            "Não consigo determinar a doença com base nas informações fornecidas."
        )
        fluxo = montar_fluxo(modelo, indice)

        resultado = fluxo.invoke({"sintomas": "sintomas vagos"})

        # Não deve crashar — retorna vazio
        assert resultado["doencas_suspeitas"] == []

    def test_estado_contem_documentos_recuperados(self, indice):
        """Estado deve conter os documentos usados na classificação."""
        modelo = _criar_modelo_mock(
            '{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "grupo_b", "justificativa": "risco"}], "fontes": ["p.31"]}'
        )
        fluxo = montar_fluxo(modelo, indice)

        resultado = fluxo.invoke({"sintomas": "febre e dor no corpo"})

        assert "documentos_recuperados" in resultado
        assert len(resultado["documentos_recuperados"]) > 0
        # Cada doc deve ter texto, pagina e fonte
        doc = resultado["documentos_recuperados"][0]
        assert "texto" in doc
        assert "pagina" in doc
        assert "fonte" in doc
