# fluxo.py
# ─────────────────────────────────────────────────────────────
# Monta o fluxo LangGraph.
#
# Fluxo novo (sem supervisor):
#
#   [INÍCIO]
#      │
#      ▼
#   [triagem]  → avalia E decide o encaminhamento
#      │
#      ├── "clinico"   → [agente_clinico]    → [log]
#      └── "seguranca" → [agente_seguranca]  → [log]
#
# A triagem substituiu o supervisor porque ela já é,
# por definição, o processo de avaliar e direcionar.
# ─────────────────────────────────────────────────────────────

import json
import os
from datetime import datetime
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agentes import (
    criar_agente_de_triagem,
    criar_agente_clinico,
    criar_agente_de_seguranca,
)
from configuracoes import CAMINHO_DO_LOG


# ─── Estado da conversa ───────────────────────────────────────

class EstadoDaConversa(TypedDict):
    """
    Pacote de informações que viaja entre os nós.

    Campos:
      pergunta        - o que foi perguntado
      destino         - decisão da triagem: "clinico" ou "seguranca"
      resposta        - resposta gerada pelo agente escolhido
      agente_utilizado - qual agente respondeu
    """
    pergunta: str
    destino: str
    resposta: str
    agente_utilizado: str


# ─── Funções auxiliares ───────────────────────────────────────

def salvar_no_log(estado: EstadoDaConversa):
    """Salva a conversa completa em JSONL para auditoria."""
    os.makedirs(os.path.dirname(CAMINHO_DO_LOG), exist_ok=True)

    registro = {
        "data_hora":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pergunta":         estado["pergunta"],
        "resposta":         estado["resposta"],
        "agente_utilizado": estado["agente_utilizado"],
    }

    with open(CAMINHO_DO_LOG, "a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")

    print(f"[log] Conversa registrada em: {CAMINHO_DO_LOG}")


# ─── Nós do grafo ─────────────────────────────────────────────

def criar_no_triagem(agente_triagem):
    """
    Nó de triagem: avalia a pergunta e decide o encaminhamento.
    Lê a resposta do agente e extrai "clinico" ou "seguranca".
    Qualquer resposta inesperada vai para "clinico" por padrão.
    """
    def no_triagem(estado: EstadoDaConversa) -> EstadoDaConversa:
        print("[triagem] Avaliando pergunta e decidindo encaminhamento...")

        resposta_triagem = agente_triagem.invoke(estado["pergunta"])
        resposta_lower = resposta_triagem.strip().lower()

        # Procura as palavras-chave na resposta
        if "seguranca" in resposta_lower or "emergência" in resposta_lower:
            destino = "seguranca"
        else:
            destino = "clinico"  # padrão seguro

        print(f"[triagem] Encaminhando para: {destino}")
        return {**estado, "destino": destino}

    return no_triagem


def criar_no_clinico(agente_clinico):
    """Nó clínico: responde dúvidas médicas gerais."""
    def no_clinico(estado: EstadoDaConversa) -> EstadoDaConversa:
        print("[agente_clinico] Processando pergunta clínica...")
        resposta = agente_clinico.invoke(estado["pergunta"])
        return {**estado, "resposta": resposta, "agente_utilizado": "clinico"}
    return no_clinico


def criar_no_seguranca(agente_seguranca):
    """Nó de segurança: trata emergências e situações de risco."""
    def no_seguranca(estado: EstadoDaConversa) -> EstadoDaConversa:
        print("[agente_seguranca] ⚠ Processando situação de segurança...")
        resposta = agente_seguranca.invoke(estado["pergunta"])
        return {**estado, "resposta": resposta, "agente_utilizado": "seguranca"}
    return no_seguranca


def criar_no_log():
    """Nó final: salva e encerra."""
    def no_log(estado: EstadoDaConversa) -> EstadoDaConversa:
        salvar_no_log(estado)
        return estado
    return no_log


# ─── Monta o grafo ────────────────────────────────────────────

def montar_fluxo(modelo, buscador):
    """
    Monta e compila o fluxo LangGraph completo.
    Ponto de entrada: triagem (não mais o supervisor).
    """

    # Cria os agentes
    agente_triagem   = criar_agente_de_triagem(modelo, buscador)
    agente_clinico   = criar_agente_clinico(modelo, buscador)
    agente_seguranca = criar_agente_de_seguranca(modelo, buscador)

    # Cria os nós
    no_triagem   = criar_no_triagem(agente_triagem)
    no_clinico   = criar_no_clinico(agente_clinico)
    no_seguranca = criar_no_seguranca(agente_seguranca)
    no_log       = criar_no_log()

    # Monta o grafo
    grafo = StateGraph(EstadoDaConversa)

    grafo.add_node("triagem",   no_triagem)
    grafo.add_node("clinico",   no_clinico)
    grafo.add_node("seguranca", no_seguranca)
    grafo.add_node("log",       no_log)

    # Triagem é o ponto de entrada
    grafo.set_entry_point("triagem")

    # Triagem decide o destino
    grafo.add_conditional_edges(
        "triagem",
        lambda estado: estado["destino"],
        {
            "clinico":   "clinico",
            "seguranca": "seguranca",
        }
    )

    # Ambos terminam no log
    grafo.add_edge("clinico",   "log")
    grafo.add_edge("seguranca", "log")
    grafo.add_edge("log",       END)

    fluxo_compilado = grafo.compile()
    print("[fluxo] Fluxo montado sem supervisor — triagem como ponto de entrada!")
    return fluxo_compilado
