# tests/test_e2e.py
"""Testes end-to-end (requerem LLM real configurado via OLLAMA_BASE_URL)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from configs.dengue import CONFIG_DENGUE
from banco_de_conhecimento import parsear_protocolo, IndiceProtocolo
from fluxo import montar_fluxo


pytestmark = pytest.mark.skipif(
    not os.environ.get("OLLAMA_BASE_URL"), reason="OLLAMA_BASE_URL não configurado"
)


@pytest.fixture(scope="module")
def fluxo():
    """Monta fluxo completo com LLM real."""
    from carregador_do_modelo import carregar_modelo

    chunks = parsear_protocolo(CONFIG_DENGUE)
    indice = IndiceProtocolo(chunks)
    modelo = carregar_modelo()
    return montar_fluxo(modelo, indice)


def test_dengue_grave(fluxo):
    """Sintomas de alarme devem classificar como grupo C ou D."""
    resultado = fluxo.invoke(
        {
            "sintomas": "febre alta, dor abdominal intensa, vômitos persistentes, petéquias"
        }
    )
    assert len(resultado["doencas_suspeitas"]) > 0
    assert len(resultado["gravidade"]) > 0


def test_dengue_leve(fluxo):
    """Sintomas leves devem classificar como grupo A ou B."""
    resultado = fluxo.invoke(
        {
            "sintomas": "febre há 2 dias, dor de cabeça, dor no corpo, sem sinais de alarme"
        }
    )
    assert len(resultado["doencas_suspeitas"]) > 0


def test_dengue_choque(fluxo):
    """Sinais de choque devem classificar como grupo D."""
    resultado = fluxo.invoke(
        {
            "sintomas": "febre, hipotensão, pulso rápido e fino, extremidades frias, cianose"
        }
    )
    assert len(resultado["doencas_suspeitas"]) > 0
    assert len(resultado["gravidade"]) > 0
