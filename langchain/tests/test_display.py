# tests/test_display.py
"""Testes das funções de exibição do CLI."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import _score_bar


class TestScoreBar:
    def test_barra_2_de_8(self):
        # 2/8 = 0.25 → 0.25*10 = 2.5 → round(2.5) = 2 (banker's rounding em Python)
        barra = _score_bar(2, 8)
        assert "2 de 8" in barra
        assert barra.count("█") == 2

    def test_barra_total_zero_sem_barra(self):
        # Fallback: sem barra quando total desconhecido
        barra = _score_bar(0, 0)
        assert "[" not in barra        # não exibe barra
        assert "compatível" in barra   # exibe contagem textual

    def test_barra_8_de_8(self):
        barra = _score_bar(8, 8)
        assert "8 de 8" in barra
        assert barra.count("█") == 10
        assert barra.count("░") == 0

    def test_barra_0_de_8(self):
        barra = _score_bar(0, 8)
        assert "0 de 8" in barra
        assert barra.count("█") == 0
        assert barra.count("░") == 10
