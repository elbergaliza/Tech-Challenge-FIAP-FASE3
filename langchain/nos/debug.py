# nos/debug.py
# ─────────────────────────────────────────────────────────────
# Utilitário de debug: salva prompts gerados em arquivos para análise.
# Não polui o stdout do usuário final.
# ─────────────────────────────────────────────────────────────

import re
from datetime import datetime
from pathlib import Path

from configuracoes import LOGS_DIR

_DIR_PROMPTS = Path(LOGS_DIR) / "prompts"


def salvar_prompt(no: str, prompt: str, extra: str = "") -> None:
    """
    Salva o prompt gerado em logs/prompts/<no>_<timestamp>.txt

    Parâmetros:
      no    - nome do nó (ex: "classificacao", "exames_dengue")
      prompt - texto completo do prompt enviado ao modelo
      extra  - informação adicional opcional (ex: tamanho do contexto)
    """
    _DIR_PROMPTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome = re.sub(r"[^a-z0-9_]", "_", no.lower())
    caminho = _DIR_PROMPTS / f"{nome}_{ts}.txt"

    cabecalho = f"# no={no} | {datetime.now().isoformat()}"
    if extra:
        cabecalho += f" | {extra}"

    caminho.write_text(f"{cabecalho}\n\n{prompt}\n", encoding="utf-8")
