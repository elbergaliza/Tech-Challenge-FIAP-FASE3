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
from configs.base import ConfiguracaoProtocolo
from nos.classificacao import criar_no_classificacao
from nos.gravidade import criar_no_gravidade, rotear_gravidade
from nos.alerta import criar_no_alerta
from nos.exames import criar_no_exames
from nos.confirmacao import criar_no_confirmacao, rotear_confirmacao
from nos.tratamento import criar_no_tratamento


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
    scores: dict[str, int]          # {"dengue": 85, "covid": 40}
    sintomas_compativeis: dict[str, list[str]]  # doenca -> lista de sintomas compatíveis
    total_sintomas_protocolo: dict[str, int]    # doenca -> total de sintomas do protocolo
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

    # Confirmação + Tratamento (Nó 4/5 - M4)
    human_in_the_loop: bool
    decisao_medica: str | None
    doenca_confirmada: str | None
    resultado_exames: str | None
    doencas_para_tratamento: list[str]
    encerrar_sem_confirmacao: bool

    tratamento_sugerido: str
    fontes_tratamento: list[str]
    justificativa_tratamento: str
    tratamento_por_suspeita: dict[str, str]
    fontes_tratamento_por_suspeita: dict[str, list[str]]
    justificativa_tratamento_por_suspeita: dict[str, str]


# ─── Monta o grafo ────────────────────────────────────────────


def montar_fluxo(
    modelo,
    indice: IndiceProtocolo,
    incluir_confirmacao_tratamento: bool = True,
    configs: list[ConfiguracaoProtocolo] | None = None,
):
    """
    Monta e compila o fluxo LangGraph.

    Parâmetros:
      modelo - LLM carregado (ChatOpenAI / Ollama)
      indice - IndiceProtocolo com busca híbrida
      incluir_confirmacao_tratamento - se False, encerra apos exames
      configs - lista de ConfiguracaoProtocolo para customizar queries por doença

    Retorna: grafo compilado pronto para invoke()
    """

    no_classificacao = criar_no_classificacao(indice, modelo)
    no_gravidade = criar_no_gravidade()
    no_alerta = criar_no_alerta()
    no_exames = criar_no_exames(indice, modelo, configs=configs)
    if incluir_confirmacao_tratamento:
        no_confirmacao = criar_no_confirmacao()
        no_tratamento = criar_no_tratamento(indice, modelo)

    grafo = StateGraph(EstadoTriagem)

    # Nós
    grafo.add_node("classificacao", no_classificacao)
    grafo.add_node("gravidade", no_gravidade)
    grafo.add_node("alerta", no_alerta)
    grafo.add_node("exames", no_exames)
    if incluir_confirmacao_tratamento:
        grafo.add_node("confirmacao", no_confirmacao)
        grafo.add_node("tratamento", no_tratamento)

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

    if incluir_confirmacao_tratamento:
        grafo.add_edge("exames", "confirmacao")
        grafo.add_conditional_edges(
            "confirmacao",
            rotear_confirmacao,
            {
                "encerrar": END,
                "tratamento": "tratamento",
            },
        )
        grafo.add_edge("tratamento", END)
    else:
        grafo.add_edge("exames", END)

    fluxo_compilado = grafo.compile()
    if incluir_confirmacao_tratamento:
        print(
            "[fluxo] Grafo de triagem montado "
            "(M1 classificacao | M2 gravidade+alerta | M3 exames | M4 confirmacao+tratamento)"
        )
    else:
        print(
            "[fluxo] Grafo de triagem montado "
            "(M1 classificacao | M2 gravidade+alerta | M3 exames)"
        )
    return fluxo_compilado
