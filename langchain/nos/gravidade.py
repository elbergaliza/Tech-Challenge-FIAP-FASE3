# nos/gravidade.py
# ─────────────────────────────────────────────────────────────
# Nó de avaliação de gravidade (M2).
#
# Lógica puramente determinística — sem LLM, sem retrieval.
# Normaliza o label de gravidade vindo do LLM e classifica
# cada doença suspeita como "grave" ou "nao_grave".
#
# Regras:
#   Dengue:  grupo_a / grupo_b → nao_grave
#             grupo_c / grupo_d → grave
#   COVID:   leve / moderado   → nao_grave
#             grave / critico   → grave
#   Demais:  fallback nao_grave (conservador)
# ─────────────────────────────────────────────────────────────

import unicodedata


# ─── Tabela de regras por protocolo ──────────────────────────

_REGRAS: dict[str, dict[str, str]] = {
    "dengue": {
        "grupo_a": "nao_grave",
        "grupo_b": "nao_grave",
        "grupo_c": "grave",
        "grupo_d": "grave",
    },
    "covid": {
        "leve": "nao_grave",
        "moderado": "nao_grave",
        "grave": "grave",
        "critico": "grave",
    },
}

# Alias: "covid-19" aponta para as mesmas regras de "covid"
_REGRAS["covid-19"] = _REGRAS["covid"]
_REGRAS["covid_19"] = _REGRAS["covid"]


def _normalizar_chave(texto: str) -> str:
    if not texto:
        return ""
    chave = unicodedata.normalize("NFD", texto.lower().strip())
    chave = "".join(c for c in chave if unicodedata.category(c) != "Mn")
    chave = chave.replace("-", "_").replace(" ", "_")
    return chave


# ─── Funções auxiliares ───────────────────────────────────────


def normalizar_gravidade(label: str) -> str:
    """
    Normaliza label de gravidade vindo do LLM para formato canônico.

    Exemplos:
      "Grupo C"    → "grupo_c"
      "Crítico"    → "critico"
      "Não grave"  → "nao_grave"
      "Moderado"   → "moderado"
    """
    if not label:
        return ""
    # lowercase
    texto = label.lower().strip()
    # remove acentos
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    # espaço → underscore
    texto = texto.replace(" ", "_")
    return texto


def classificar_gravidade(doenca: str, label_normalizado: str) -> str:
    """
    Mapeia (doenca, label_normalizado) → 'grave' | 'nao_grave'.

    Usa as regras definidas em _REGRAS por protocolo.
    A chave de doença é normalizada (lowercase).
    Fallback: 'nao_grave' para doença ou label desconhecido.
    """
    chave = _normalizar_chave(doenca)
    regras_doenca = _REGRAS.get(chave, {})
    return regras_doenca.get(label_normalizado, "nao_grave")


# ─── Nó LangGraph ────────────────────────────────────────────


def criar_no_gravidade():
    """
    Factory que retorna o nó de avaliação de gravidade.

    Lê `estado["gravidade"]` (dict doença→label do LLM),
    normaliza cada label, classifica como grave/nao_grave,
    determina o pior caso (max_gravidade) e compõe o alerta.
    """

    def no_gravidade(estado: dict) -> dict:
        print("[gravidade] Avaliando gravidade das suspeitas...")

        gravidade_raw: dict[str, str] = estado.get("gravidade", {})

        doencas_graves: list[str] = []
        doencas_nao_graves: list[str] = []

        for doenca, label in gravidade_raw.items():
            label_norm = normalizar_gravidade(label)
            classificacao = classificar_gravidade(doenca, label_norm)
            if classificacao == "grave":
                doencas_graves.append(doenca)
            else:
                doencas_nao_graves.append(doenca)

        max_gravidade = "grave" if doencas_graves else "nao_grave"

        alerta: str | None = None
        if doencas_graves:
            lista = ", ".join(d.upper() for d in doencas_graves)
            alerta = (
                f"ALERTA DE URGENCIA: suspeita(s) grave(s) identificada(s): {lista}. "
                "Acionar equipe medica imediatamente. "
                "Baseado em protocolo oficial — validacao clinica obrigatoria."
            )

        print(f"[gravidade] max_gravidade={max_gravidade} | graves={doencas_graves}")

        return {
            **estado,
            "max_gravidade": max_gravidade,
            "doencas_graves": doencas_graves,
            "doencas_nao_graves": doencas_nao_graves,
            "alerta": alerta,
        }

    return no_gravidade


# ─── Função de roteamento (usada pelo grafo) ─────────────────


def rotear_gravidade(estado: dict) -> str:
    """
    Função de roteamento para add_conditional_edges.

    Retorna o nome do próximo nó com base em max_gravidade.
    """
    if estado.get("max_gravidade") == "grave":
        return "alerta_e_exames"
    return "exames"
