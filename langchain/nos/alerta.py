# nos/alerta.py
# ─────────────────────────────────────────────────────────────
# Nó de alerta de urgência (M2 — path grave).
#
# Este nó é acionado quando max_gravidade == "grave".
# Ele exibe/loga o alerta composto pelo nó de gravidade
# e encerra o fluxo (até M3 adicionar tratamento urgente).
# ─────────────────────────────────────────────────────────────


def criar_no_alerta():
    """
    Factory que retorna o nó de alerta de urgência.

    Recebe o estado com `alerta` já preenchido pelo nó de gravidade.
    Imprime o alerta e encerra o fluxo (até M3 adicionar tratamento urgente).
    """

    def no_alerta(estado: dict) -> dict:
        alerta = estado.get("alerta", "")
        print(f"\n{'='*60}")
        print(f"[alerta] {alerta}")
        print(f"{'='*60}\n")
        return {**estado}

    return no_alerta
