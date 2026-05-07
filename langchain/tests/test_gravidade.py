# tests/test_gravidade.py
"""Testes do nó de avaliação de gravidade (M2)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from nos.gravidade import normalizar_gravidade, classificar_gravidade, criar_no_gravidade


class TestNormalizarGravidade:
    """Testa a normalização do label de gravidade vindo do LLM."""

    def test_grupo_c_maiusculo(self):
        assert normalizar_gravidade("Grupo C") == "grupo_c"

    def test_grupo_d_com_espaco(self):
        assert normalizar_gravidade("grupo d") == "grupo_d"

    def test_grave_simples(self):
        assert normalizar_gravidade("Grave") == "grave"

    def test_critico_com_acento(self):
        assert normalizar_gravidade("Crítico") == "critico"

    def test_nao_grave_com_acento(self):
        assert normalizar_gravidade("Não grave") == "nao_grave"

    def test_moderado(self):
        assert normalizar_gravidade("Moderado") == "moderado"

    def test_vazio(self):
        assert normalizar_gravidade("") == ""


class TestClassificarGravidade:
    """Testa o mapeamento de label normalizado para 'grave'/'nao_grave'."""

    # Dengue
    def test_dengue_grupo_a_nao_grave(self):
        assert classificar_gravidade("dengue", "grupo_a") == "nao_grave"

    def test_dengue_grupo_b_nao_grave(self):
        assert classificar_gravidade("dengue", "grupo_b") == "nao_grave"

    def test_dengue_grupo_c_grave(self):
        assert classificar_gravidade("dengue", "grupo_c") == "grave"

    def test_dengue_grupo_d_grave(self):
        assert classificar_gravidade("dengue", "grupo_d") == "grave"

    # COVID variantes
    def test_covid_leve_nao_grave(self):
        assert classificar_gravidade("covid", "leve") == "nao_grave"

    def test_covid_moderado_nao_grave(self):
        assert classificar_gravidade("covid", "moderado") == "nao_grave"

    def test_covid_grave_grave(self):
        assert classificar_gravidade("covid", "grave") == "grave"

    def test_covid_critico_grave(self):
        assert classificar_gravidade("covid", "critico") == "grave"

    def test_covid19_alias_grave(self):
        """LLM pode retornar 'covid-19' — deve resolver igual a 'covid'."""
        assert classificar_gravidade("covid-19", "grave") == "grave"

    def test_covid19_alias_leve(self):
        assert classificar_gravidade("covid-19", "leve") == "nao_grave"

    # Fallback
    def test_doenca_desconhecida_nao_grave(self):
        """Doença sem regras definidas → fallback nao_grave."""
        assert classificar_gravidade("avc", "desconhecido") == "nao_grave"

    def test_label_desconhecido_nao_grave(self):
        """Label não reconhecido → fallback nao_grave."""
        assert classificar_gravidade("dengue", "algo_estranho") == "nao_grave"


class TestNoGravidade:
    """Testa o nó de avaliação de gravidade integrado ao estado."""

    def _estado(self, gravidade: dict) -> dict:
        return {
            "sintomas": "teste",
            "doencas_suspeitas": list(gravidade.keys()),
            "gravidade": gravidade,
            "documentos_recuperados": [],
            "justificativa_classificacao": "",
            "fontes": [],
        }

    def test_dengue_grupo_c_max_grave(self):
        no = criar_no_gravidade()
        resultado = no(self._estado({"dengue": "grupo_c"}))
        assert resultado["max_gravidade"] == "grave"
        assert "dengue" in resultado["doencas_graves"]
        assert resultado["doencas_nao_graves"] == []
        assert resultado["alerta"] is not None

    def test_dengue_grupo_d_max_grave(self):
        no = criar_no_gravidade()
        resultado = no(self._estado({"dengue": "grupo_d"}))
        assert resultado["max_gravidade"] == "grave"
        assert "dengue" in resultado["doencas_graves"]

    def test_dengue_grupo_a_nao_grave(self):
        no = criar_no_gravidade()
        resultado = no(self._estado({"dengue": "grupo_a"}))
        assert resultado["max_gravidade"] == "nao_grave"
        assert resultado["doencas_graves"] == []
        assert "dengue" in resultado["doencas_nao_graves"]
        assert resultado["alerta"] is None

    def test_dengue_grupo_b_nao_grave(self):
        no = criar_no_gravidade()
        resultado = no(self._estado({"dengue": "grupo_b"}))
        assert resultado["max_gravidade"] == "nao_grave"
        assert resultado["alerta"] is None

    def test_sem_doencas_suspeitas(self):
        no = criar_no_gravidade()
        resultado = no(self._estado({}))
        assert resultado["max_gravidade"] == "nao_grave"
        assert resultado["doencas_graves"] == []
        assert resultado["doencas_nao_graves"] == []
        assert resultado["alerta"] is None

    def test_pior_caso_prevalece(self):
        """Dengue A (não grave) + COVID grave → max_gravidade = grave."""
        no = criar_no_gravidade()
        resultado = no(self._estado({"dengue": "grupo_a", "covid": "grave"}))
        assert resultado["max_gravidade"] == "grave"
        assert "covid" in resultado["doencas_graves"]
        assert "dengue" in resultado["doencas_nao_graves"]
        assert resultado["alerta"] is not None

    def test_alerta_contem_nome_doenca_maiusculo(self):
        """Alerta deve mencionar a doença em maiúsculas."""
        no = criar_no_gravidade()
        resultado = no(self._estado({"dengue": "grupo_d"}))
        assert "DENGUE" in resultado["alerta"]

    def test_label_com_maiusculas_e_acento(self):
        """LLM pode retornar 'Grupo C' ou 'Crítico' — deve normalizar."""
        no = criar_no_gravidade()
        resultado = no(self._estado({"dengue": "Grupo C"}))
        assert resultado["max_gravidade"] == "grave"
