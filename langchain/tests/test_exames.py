# tests/test_exames.py
"""Testes do no de sugestao de exames (M3)."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nos.exames import criar_no_exames


def _doc(texto: str, pagina: int = 1):
    return SimpleNamespace(
        page_content=texto,
        metadata={"fonte": "MS Dengue 2024", "pagina": pagina, "doenca": "dengue"},
    )


def test_no_exames_retorna_dict_por_doenca(monkeypatch):
    def busca_fake(indice, consulta, k=8, doenca=None, secao_tipo=None):
        return [_doc("Hemograma completo e NS1 na fase inicial.", 28)]

    monkeypatch.setattr("nos.exames.busca_hibrida", busca_fake)

    modelo = MagicMock()
    modelo.invoke.return_value = MagicMock(
        content='{"exames": ["hemograma completo", "NS1"], "justificativa": "protocolo", "fontes": ["MS Dengue 2024, p.28"]}'
    )

    no = criar_no_exames(object(), modelo)
    estado = {
        "doencas_suspeitas": ["dengue"],
        "gravidade": {"dengue": "grupo_b"},
    }

    out = no(estado)

    assert "dengue" in out["exames_sugeridos"]
    assert out["exames_sugeridos"]["dengue"] == ["hemograma completo", "NS1"]
    assert out["fontes_exames"]["dengue"] == ["MS Dengue 2024, p.28"]


def test_no_exames_fallback_json_invalido(monkeypatch):
    monkeypatch.setattr("nos.exames.busca_hibrida", lambda *args, **kwargs: [_doc("texto")])

    modelo = MagicMock()
    modelo.invoke.return_value = MagicMock(content="saida sem json")

    no = criar_no_exames(object(), modelo)
    out = no({"doencas_suspeitas": ["dengue"], "gravidade": {"dengue": "grupo_a"}})

    assert out["exames_sugeridos"]["dengue"] == []
    assert "Resposta invalida do modelo" in out["justificativa_exames"]["dengue"]


def test_no_exames_fallback_cascata_busca(monkeypatch):
    chamadas = []

    def busca_fake(indice, consulta, k=8, doenca=None, secao_tipo=None):
        chamadas.append({"doenca": doenca, "secao_tipo": secao_tipo})
        if len(chamadas) < 3:
            return []
        return [_doc("RT-PCR e hemograma.", 40)]

    monkeypatch.setattr("nos.exames.busca_hibrida", busca_fake)

    modelo = MagicMock()
    modelo.invoke.return_value = MagicMock(
        content='{"exames": ["RT-PCR"], "justificativa": "fallback", "fontes": ["MS, p.40"]}'
    )

    no = criar_no_exames(object(), modelo)
    out = no({"doencas_suspeitas": ["covid"], "gravidade": {"covid": "moderado"}})

    assert out["exames_sugeridos"]["covid"] == ["RT-PCR"]
    assert chamadas == [
        {"doenca": "covid", "secao_tipo": "exames"},
        {"doenca": "covid", "secao_tipo": None},
        {"doenca": None, "secao_tipo": None},
    ]
