# tests/test_tratamento.py
"""Testes do no de tratamento (M4)."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nos.tratamento import criar_no_tratamento


def _doc(texto: str, pagina: int = 1):
    return SimpleNamespace(
        page_content=texto,
        metadata={"fonte": "MS", "pagina": pagina, "doenca": "covid"},
    )


def test_tratamento_confirmar_retorna_contrato_singular(monkeypatch):
    monkeypatch.setattr("nos.tratamento.busca_hibrida", lambda *args, **kwargs: [_doc("conduta", 18)])

    modelo = MagicMock()
    modelo.invoke.return_value = MagicMock(
        content='{"tratamento": "Suporte", "justificativa": "ok", "fontes": ["MS, p.18"]}'
    )

    no = criar_no_tratamento(object(), modelo)
    out = no(
        {
            "decisao_medica": "confirmar",
            "doenca_confirmada": "covid",
            "doencas_para_tratamento": ["covid"],
            "gravidade": {"covid": "moderado"},
        }
    )

    assert out["tratamento_sugerido"] == "Suporte"
    assert out["fontes_tratamento"] == ["MS, p.18"]


def test_tratamento_pular_retorna_por_suspeita(monkeypatch):
    monkeypatch.setattr("nos.tratamento.busca_hibrida", lambda *args, **kwargs: [_doc("conduta", 22)])

    modelo = MagicMock()
    modelo.invoke.return_value = MagicMock(
        content='{"tratamento": "Conduta geral", "justificativa": "ok", "fontes": ["MS, p.22"]}'
    )

    no = criar_no_tratamento(object(), modelo)
    out = no(
        {
            "decisao_medica": "pular",
            "doencas_para_tratamento": ["dengue", "covid"],
            "gravidade": {"dengue": "grupo_b", "covid": "moderado"},
        }
    )

    assert set(out["tratamento_por_suspeita"].keys()) == {"dengue", "covid"}
    assert out["tratamento_sugerido"] == ""


def test_tratamento_fallback_json_invalido(monkeypatch):
    monkeypatch.setattr("nos.tratamento.busca_hibrida", lambda *args, **kwargs: [_doc("texto", 20)])

    modelo = MagicMock()
    modelo.invoke.return_value = MagicMock(content="saida invalida")

    no = criar_no_tratamento(object(), modelo)
    out = no(
        {
            "decisao_medica": "confirmar",
            "doenca_confirmada": "covid",
            "doencas_para_tratamento": ["covid"],
            "gravidade": {"covid": "moderado"},
        }
    )

    assert out["tratamento_sugerido"] == ""
    assert "Resposta invalida do modelo" in out["justificativa_tratamento"]
