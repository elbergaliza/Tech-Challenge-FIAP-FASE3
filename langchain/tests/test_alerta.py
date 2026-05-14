# tests/test_alerta.py
"""Testes do nó de alerta de urgência (M2 — path grave)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from nos.alerta import criar_no_alerta


def _estado_grave(doencas_graves=None, alerta=None):
    return {
        "sintomas": "febre alta, dor abdominal intensa, hipotensão",
        "doencas_suspeitas": doencas_graves or ["dengue"],
        "doencas_graves": doencas_graves or ["dengue"],
        "doencas_nao_graves": [],
        "max_gravidade": "grave",
        "alerta": alerta or "ALERTA DE URGENCIA: DENGUE.",
        "gravidade": {"dengue": "grupo_d"},
        "documentos_recuperados": [],
        "justificativa_classificacao": "sinais de choque presentes",
        "fontes": ["MS Dengue 2024, p.12"],
    }


class TestNoAlerta:
    def test_alerta_preservado_no_estado(self):
        """O nó de alerta deve manter o campo 'alerta' no estado."""
        no = criar_no_alerta()
        resultado = no(_estado_grave())
        assert resultado["alerta"] is not None
        assert len(resultado["alerta"]) > 0

    def test_estado_preservado(self):
        """Todos os campos anteriores devem ser preservados."""
        no = criar_no_alerta()
        estado = _estado_grave()
        resultado = no(estado)
        assert resultado["sintomas"] == estado["sintomas"]
        assert resultado["doencas_graves"] == estado["doencas_graves"]
        assert resultado["max_gravidade"] == "grave"
        assert resultado["fontes"] == estado["fontes"]

    def test_nao_modifica_alerta(self):
        """O nó de alerta não deve alterar o conteúdo do campo alerta."""
        no = criar_no_alerta()
        alerta_original = "ALERTA DE URGENCIA: DENGUE. Acionar equipe."
        resultado = no(_estado_grave(alerta=alerta_original))
        assert resultado["alerta"] == alerta_original
