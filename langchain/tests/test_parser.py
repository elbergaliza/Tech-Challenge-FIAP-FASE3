# tests/test_parser.py
"""Testes do parser de protocolos."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.dengue import CONFIG_DENGUE
from configs.covid import CONFIG_COVID
from banco_de_conhecimento import parsear_protocolo


def test_parser_gera_chunks():
    """Parser deve gerar chunks a partir do PDF."""
    chunks = parsear_protocolo(CONFIG_DENGUE)
    assert len(chunks) > 0


def test_parser_metadados_obrigatorios():
    """Cada chunk deve ter metadados obrigatórios."""
    chunks = parsear_protocolo(CONFIG_DENGUE)
    for chunk in chunks[:10]:
        assert "doenca" in chunk.metadata
        assert chunk.metadata["doenca"] == "dengue"
        assert "pagina" in chunk.metadata
        assert "fonte" in chunk.metadata
        assert "secao_tipo" in chunk.metadata
        assert "gravidade_grupo" in chunk.metadata


def test_parser_secao_tipo_eh_lista():
    """secao_tipo deve ser lista (múltiplas tags possíveis)."""
    chunks = parsear_protocolo(CONFIG_DENGUE)
    for chunk in chunks[:10]:
        assert isinstance(chunk.metadata["secao_tipo"], list)
        assert len(chunk.metadata["secao_tipo"]) > 0


def test_parser_filtra_ruido():
    """Chunks não devem conter padrões de ruído."""
    chunks = parsear_protocolo(CONFIG_DENGUE)
    for chunk in chunks:
        conteudo = chunk.page_content.lower()
        assert "cpf:" not in conteudo
        assert "data de nascimento" not in conteudo


def test_parser_detecta_gravidade():
    """Ao menos alguns chunks devem ter gravidade detectada."""
    chunks = parsear_protocolo(CONFIG_DENGUE)
    chunks_com_gravidade = [
        c for c in chunks if c.metadata["gravidade_grupo"] is not None
    ]
    # Deve existir ao menos 1 chunk com gravidade identificada
    assert len(chunks_com_gravidade) > 0


def test_parser_metadados_covid():
    """Parser deve gerar metadados mínimos para chunks de COVID."""
    chunks = parsear_protocolo(CONFIG_COVID)
    assert len(chunks) > 0
    for chunk in chunks[:10]:
        assert chunk.metadata["doenca"] == "covid"
        assert isinstance(chunk.metadata["secao_tipo"], list)
        assert len(chunk.metadata["secao_tipo"]) > 0
