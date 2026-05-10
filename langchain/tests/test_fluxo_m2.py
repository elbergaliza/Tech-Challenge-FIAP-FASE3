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

    def test_dengue_grupo_d_max_grave(self, indice):
        """Fluxo completo: M1 retorna grupo_d (choque) → M2 grave."""
        modelo = _mock_llm(
            '{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "grupo_d", '
            '"justificativa": "choque"}], "fontes": ["p.32"]}'
        )
        fluxo = montar_fluxo(modelo, indice)
        resultado = fluxo.invoke({"sintomas": "hipotensão, extremidades frias"})

        assert resultado["max_gravidade"] == "grave"
        assert "DENGUE" in resultado["alerta"]

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
        assert "exames_sugeridos" in resultado

    def test_dengue_grupo_b_nao_grave(self, indice):
        """Grupo B → nao_grave, sem alerta."""
        modelo = _mock_llm(
            '{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "grupo_b", '
            '"justificativa": "sem sinais de alarme, comorbidade"}], "fontes": ["p.28"]}'
        )
        fluxo = montar_fluxo(modelo, indice)
        resultado = fluxo.invoke({"sintomas": "febre, comorbidade"})

        assert resultado["max_gravidade"] == "nao_grave"
        assert resultado["alerta"] is None
        assert "exames_sugeridos" in resultado

    def test_sem_doenca_identificada(self, indice):
        """M1 retorna lista vazia → M2 nao_grave sem alerta."""
        modelo = _mock_llm('{"doencas_suspeitas": [], "fontes": []}')
        fluxo = montar_fluxo(modelo, indice)
        resultado = fluxo.invoke({"sintomas": "dor no dedo"})

        assert resultado.get("max_gravidade") == "nao_grave"
        assert resultado.get("alerta") is None
        assert resultado.get("exames_sugeridos") == {}

    def test_label_com_maiuscula_normalizado(self, indice):
        """LLM retorna 'Grupo C' com maiúscula → deve classificar como grave."""
        modelo = _mock_llm(
            '{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "Grupo C", '
            '"justificativa": "sinais de alarme"}], "fontes": ["p.12"]}'
        )
        fluxo = montar_fluxo(modelo, indice)
        resultado = fluxo.invoke({"sintomas": "febre, vômitos"})

        assert resultado["max_gravidade"] == "grave"

    def test_campos_m1_preservados_apos_m2(self, indice):
        """Os campos do M1 devem continuar presentes após o M2."""
        modelo = _mock_llm(
            '{"doencas_suspeitas": [{"doenca": "dengue", "gravidade": "grupo_b", '
            '"justificativa": "leve"}], "fontes": ["p.11"]}'
        )
        fluxo = montar_fluxo(modelo, indice)
        resultado = fluxo.invoke({"sintomas": "febre leve"})

        assert "doencas_suspeitas" in resultado
        assert "gravidade" in resultado
        assert "documentos_recuperados" in resultado
        assert "fontes" in resultado
        assert "exames_sugeridos" in resultado
