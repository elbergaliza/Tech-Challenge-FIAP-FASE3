# tests/test_e2e.py
"""Testes end-to-end do fluxo completo com LLM real."""

import sys
import os
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from banco_de_conhecimento import construir_indice
from configs import CONFIG_DENGUE, CONFIG_COVID
from fluxo import montar_fluxo


OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL")
HAS_OLLAMA = bool(OLLAMA_URL and os.environ.get("OLLAMA_API_KEY"))
HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY")) and not OLLAMA_URL


pytestmark = pytest.mark.skipif(
    not (HAS_OLLAMA or HAS_OPENAI),
    reason="Configure OLLAMA_BASE_URL+OLLAMA_API_KEY ou OPENAI_API_KEY",
)


def _assert_saida_m2_m3(resultado: dict):
    assert isinstance(resultado.get("doencas_suspeitas"), list)
    assert isinstance(resultado.get("gravidade"), dict)
    assert resultado.get("max_gravidade") in {"grave", "nao_grave"}
    assert isinstance(resultado.get("doencas_graves"), list)
    assert isinstance(resultado.get("doencas_nao_graves"), list)
    assert isinstance(resultado.get("exames_sugeridos"), dict)
    assert isinstance(resultado.get("fontes_exames"), dict)
    assert isinstance(resultado.get("justificativa_exames"), dict)

    if resultado["max_gravidade"] == "grave":
        assert isinstance(resultado.get("alerta"), str)
        assert resultado.get("alerta")
    else:
        assert resultado.get("alerta") is None

    doencas = set(resultado["doencas_suspeitas"])
    exames = set(resultado["exames_sugeridos"].keys())
    assert doencas.issubset(exames)


def _normalizar_doenca(valor: str) -> str:
    texto = unicodedata.normalize("NFD", valor.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto.strip()


@pytest.fixture(scope="module")
def fluxo():
    """Monta fluxo completo com LLM real."""
    from carregador_do_modelo import carregar_modelo

    indice = construir_indice([CONFIG_DENGUE, CONFIG_COVID])
    modelo = carregar_modelo()
    return montar_fluxo(modelo, indice)


def test_dengue_grave(fluxo):
    """Caso grave deve acionar alerta e manter sugestao de exames."""
    resultado = fluxo.invoke(
        {
            "sintomas": "febre alta, dor abdominal intensa, vômitos persistentes, petéquias"
        }
    )
    _assert_saida_m2_m3(resultado)

    assert resultado["max_gravidade"] == "grave"
    assert resultado["alerta"] is not None


def test_dengue_leve(fluxo):
    """Caso nao critico deve retornar estrutura M2/M3 consistente."""
    resultado = fluxo.invoke(
        {
            "sintomas": "febre há 2 dias, dor de cabeça, dor no corpo, sem sinais de alarme"
        }
    )
    _assert_saida_m2_m3(resultado)


def test_dengue_choque(fluxo):
    """Sinais de choque devem cair no path grave."""
    resultado = fluxo.invoke(
        {
            "sintomas": "febre, hipotensão, pulso rápido e fino, extremidades frias, cianose"
        }
    )
    _assert_saida_m2_m3(resultado)
    assert resultado["max_gravidade"] == "grave"


def test_covid_respiratorio(fluxo):
    """Quadro respiratorio deve produzir suspeita COVID e saida M2/M3 valida."""
    resultado = fluxo.invoke(
        {
            "sintomas": "tosse seca, febre, dispneia e queda de saturacao nos ultimos 2 dias"
        }
    )
    _assert_saida_m2_m3(resultado)
    suspeitas_norm = [_normalizar_doenca(d) for d in resultado.get("doencas_suspeitas", [])]
    assert any("covid" in d for d in suspeitas_norm)


def test_diferencial_dengue_covid(fluxo):
    """Sintomas mistos devem permitir diagnostico diferencial no mesmo caso."""
    resultado = fluxo.invoke(
        {
            "sintomas": "febre, mialgia, cefaleia, tosse e dor de garganta com piora recente"
        }
    )
    _assert_saida_m2_m3(resultado)
    suspeitas_norm = [_normalizar_doenca(d) for d in resultado.get("doencas_suspeitas", [])]
    assert any("dengue" in d or "covid" in d for d in suspeitas_norm)


def test_m4_rejeicao_medica_encerra_sem_tratamento(fluxo):
    resultado = fluxo.invoke(
        {
            "sintomas": "febre, dor no corpo e cefaleia",
            "human_in_the_loop": False,
            "decisao_medica": "rejeitar",
        }
    )
    _assert_saida_m2_m3(resultado)
    assert resultado.get("encerrar_sem_confirmacao") is True


def test_m4_pular_retorna_tratamento_por_suspeita(fluxo):
    resultado = fluxo.invoke(
        {
            "sintomas": "febre, mialgia, tosse e dor de garganta",
            "human_in_the_loop": False,
            "decisao_medica": "pular",
        }
    )
    _assert_saida_m2_m3(resultado)
    assert isinstance(resultado.get("tratamento_por_suspeita"), dict)
