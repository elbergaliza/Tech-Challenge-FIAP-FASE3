# tests/test_confirmacao.py
"""Testes do no de confirmacao medica (M4)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nos.confirmacao import criar_no_confirmacao, rotear_confirmacao, normalizar_decisao


def _estado_base(**kwargs):
    estado = {
        "human_in_the_loop": False,
        "doencas_suspeitas": ["dengue", "covid"],
    }
    estado.update(kwargs)
    return estado


def test_normalizacao_alias_confirmo_rejeito():
    assert normalizar_decisao("confirmo") == "confirmar"
    assert normalizar_decisao("rejeito") == "rejeitar"


def test_confirmar_exige_doenca_confirmada():
    no = criar_no_confirmacao()
    out = no(_estado_base(decisao_medica="confirmar", doenca_confirmada=None))
    assert out["encerrar_sem_confirmacao"] is True
    assert out["decisao_medica"] == "rejeitar"


def test_pular_nao_define_doenca_confirmada_e_usa_todas_suspeitas():
    no = criar_no_confirmacao()
    out = no(_estado_base(decisao_medica="pular"))
    assert out["encerrar_sem_confirmacao"] is False
    assert out["doenca_confirmada"] is None
    assert out["doencas_para_tratamento"] == ["dengue", "covid"]
    assert rotear_confirmacao(out) == "tratamento"


def test_rejeitar_encerra_fluxo():
    no = criar_no_confirmacao()
    out = no(_estado_base(decisao_medica="rejeitar"))
    assert out["encerrar_sem_confirmacao"] is True
    assert rotear_confirmacao(out) == "encerrar"
