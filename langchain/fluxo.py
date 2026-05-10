# fluxo.py
# ─────────────────────────────────────────────────────────────
# Monta o fluxo LangGraph.
#
# Fluxo M1 + M2 + M3:
#
#   [INÍCIO]
#      │
#      ▼
#   [classificacao]  → recebe sintomas, busca nos protocolos,
#      │               retorna doenças suspeitas + gravidade
#      ▼
#   [gravidade]      → normaliza + classifica grave/nao_grave
#      │
#      ├── grave     → [alerta] → [exames] → FIM
#      │
#      └── nao_grave → [exames] → FIM
#
# ─────────────────────────────────────────────────────────────

from typing import TypedDict
from langgraph.graph import StateGraph, END

from banco_de_conhecimento import IndiceProtocolo
from nos.classificacao import criar_no_classificacao
from nos.gravidade import criar_no_gravidade, rotear_gravidade
from nos.alerta import criar_no_alerta
from nos.exames import criar_no_exames


# ─── Estado da conversa ───────────────────────────────────────


class EstadoTriagem(TypedDict, total=False):
    """
    Estado que viaja entre os nós do grafo de triagem.

    Campos preenchidos por:
      - Input (main.py): sintomas
      - Nó classificação: documentos_recuperados, doencas_suspeitas, gravidade, etc.
    """

    # Input
    sintomas: str

    # Classificação (Nó 1 — M1)
    documentos_recuperados: list[dict]
    doencas_suspeitas: list[str]
    gravidade: dict[str, str]       # {"dengue": "grupo_c"}
    justificativa_classificacao: str
    fontes: list[str]

    # Avaliação de Gravidade (Nó 2 — M2)
    max_gravidade: str              # "grave" ou "nao_grave"
    doencas_graves: list[str]       # subconjunto de doencas_suspeitas
    doencas_nao_graves: list[str]   # subconjunto de doencas_suspeitas
    alerta: str | None              # mensagem de urgência, None se nao_grave

    # Exames (Nó 3b - M3)
    exames_sugeridos: dict[str, list[str]]
    fontes_exames: dict[str, list[str]]
    justificativa_exames: dict[str, str]


# ─── Monta o grafo ────────────────────────────────────────────


def montar_fluxo(modelo, indice: IndiceProtocolo):
    """
    Monta e compila o fluxo LangGraph.

    Parâmetros:
      modelo - LLM carregado (ChatOpenAI / Ollama)
      indice - IndiceProtocolo com busca híbrida

    Retorna: grafo compilado pronto para invoke()
    """

    no_classificacao = criar_no_classificacao(indice, modelo)
    no_gravidade = criar_no_gravidade()
    no_alerta = criar_no_alerta()
    no_exames = criar_no_exames(indice, modelo)

    grafo = StateGraph(EstadoTriagem)

    # Nós
    grafo.add_node("classificacao", no_classificacao)
    grafo.add_node("gravidade", no_gravidade)
    grafo.add_node("alerta", no_alerta)
    grafo.add_node("exames", no_exames)

    # Arestas
    grafo.set_entry_point("classificacao")
    grafo.add_edge("classificacao", "gravidade")

    # Aresta condicional: grave → alerta_e_exames | nao_grave → exames
    grafo.add_conditional_edges(
        "gravidade",
        rotear_gravidade,
        {
            "alerta_e_exames": "alerta",
            "exames": "exames",
        },
    )
    grafo.add_edge("alerta", "exames")
    grafo.add_edge("exames", END)

    fluxo_compilado = grafo.compile()
    print("[fluxo] Grafo de triagem montado (M1 classificacao | M2 gravidade+alerta | M3 exames)")
    return fluxo_compilado
