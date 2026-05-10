# tests/test_fluxo_m3.py
"""Testes de integracao do fluxo com M3 (no de exames)."""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from banco_de_conhecimento import IndiceProtocolo, parsear_protocolo
from configs.dengue import CONFIG_DENGUE
from fluxo import montar_fluxo


@pytest.fixture(scope="module")
def indice():
    chunks = parsear_protocolo(CONFIG_DENGUE)
    return IndiceProtocolo(chunks)


def _mock_llm_classificacao(resposta_json: str):
    modelo = MagicMock()
    modelo.invoke.return_value = MagicMock(content=resposta_json)
    return modelo


def test_nao_grave_gera_exames(indice):
    modelo = _mock_llm_classificacao(
        '{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "grupo_b", "justificativa": "sem sinais de alarme"}], "fontes": ["p.28"]}'
    )

    fluxo = montar_fluxo(modelo, indice)
    out = fluxo.invoke({"sintomas": "febre e mialgia"})

    assert out["max_gravidade"] == "nao_grave"
    assert "exames_sugeridos" in out
    assert "dengue" in out["exames_sugeridos"]
    assert isinstance(out["exames_sugeridos"]["dengue"], list)


def test_gravidade_mista_gera_alerta_e_exames(indice):
    modelo = _mock_llm_classificacao(
        '{"doencas_suspeitas": ['
        '{"doenca": "dengue", "gravidade": "grupo_c", "justificativa": "sinais de alarme"}, '
        '{"doenca": "covid", "gravidade": "moderado", "justificativa": "quadro respiratorio"}'
        '], "fontes": ["p.12", "p.40"]}'
    )

    fluxo = montar_fluxo(modelo, indice)
    out = fluxo.invoke({"sintomas": "febre, dispneia, dor abdominal"})

    assert out["max_gravidade"] == "grave"
    assert out["alerta"] is not None
    assert "dengue" in out["doencas_graves"]
    assert "dengue" in out["exames_sugeridos"]
    assert "covid" in out["exames_sugeridos"]
