# tests/test_fluxo_m4.py
"""Testes de integracao para M4 (confirmacao + tratamento)."""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from banco_de_conhecimento import IndiceProtocolo, parsear_protocolo
from configs.dengue import CONFIG_DENGUE
from configs.covid import CONFIG_COVID
from fluxo import montar_fluxo


@pytest.fixture(scope="module")
def indice():
    chunks = parsear_protocolo(CONFIG_DENGUE) + parsear_protocolo(CONFIG_COVID)
    return IndiceProtocolo(chunks)


def test_fluxo_m4_rejeitar_encerra(indice):
    modelo = MagicMock()
    modelo.invoke.return_value = MagicMock(
        content='{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "grupo_b", "justificativa": "ok"}], "fontes": ["p.28"]}'
    )

    fluxo = montar_fluxo(modelo, indice)
    out = fluxo.invoke(
        {
            "sintomas": "febre e mialgia",
            "human_in_the_loop": False,
            "decisao_medica": "rejeitar",
        }
    )

    assert out["encerrar_sem_confirmacao"] is True


def test_fluxo_m4_confirmar_tratamento_singular(indice):
    modelo = MagicMock()
    modelo.invoke.side_effect = [
        MagicMock(
            content='{"doencas_suspeitas": [{"doenca": "covid", "gravidade": "moderado", "justificativa": "ok"}], "fontes": ["p.5"]}'
        ),
        MagicMock(
            content='{"exames": ["rt-pcr"], "justificativa": "ok", "fontes": ["MS, p.5"]}'
        ),
        MagicMock(
            content='{"tratamento": "Suporte clinico", "justificativa": "ok", "fontes": ["MS, p.18"]}'
        ),
    ]

    fluxo = montar_fluxo(modelo, indice)
    out = fluxo.invoke(
        {
            "sintomas": "tosse e febre",
            "human_in_the_loop": False,
            "decisao_medica": "confirmar",
            "doenca_confirmada": "covid",
        }
    )

    assert out["encerrar_sem_confirmacao"] is False
    assert out["tratamento_sugerido"] == "Suporte clinico"


def test_fluxo_m4_pular_tratamento_todas_suspeitas(indice):
    modelo = MagicMock()
    modelo.invoke.side_effect = [
        MagicMock(
            content='{"doencas_suspeitas": ['
            '{"doenca": "dengue", "gravidade": "grupo_b", "justificativa": "ok"}, '
            '{"doenca": "covid", "gravidade": "moderado", "justificativa": "ok"}'
            '], "fontes": ["p.28", "p.5"]}'
        ),
        MagicMock(
            content='{"exames": ["hemograma"], "justificativa": "ok", "fontes": ["MS, p.28"]}'
        ),
        MagicMock(
            content='{"exames": ["rt-pcr"], "justificativa": "ok", "fontes": ["MS, p.5"]}'
        ),
        MagicMock(
            content='{"tratamento": "Conduta dengue", "justificativa": "ok", "fontes": ["MS, p.28"]}'
        ),
        MagicMock(
            content='{"tratamento": "Conduta covid", "justificativa": "ok", "fontes": ["MS, p.18"]}'
        ),
    ]

    fluxo = montar_fluxo(modelo, indice)
    out = fluxo.invoke(
        {
            "sintomas": "febre, tosse, mialgia",
            "human_in_the_loop": False,
            "decisao_medica": "pular",
        }
    )

    assert set(out["tratamento_por_suspeita"].keys()) == {"dengue", "covid"}
    assert out["tratamento_sugerido"] == ""
