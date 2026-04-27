# fluxo.py
# ─────────────────────────────────────────────────────────────
# Esse arquivo monta o fluxo de decisão usando LangGraph.
#
# O que é LangGraph?
#   É uma biblioteca que permite criar fluxos de decisão como um
#   mapa de estados. Cada "nó" do mapa é uma etapa do processo,
#   e as "arestas" definem para onde ir depois de cada etapa.
#
# Nosso fluxo:
#
#   [INÍCIO]
#      │
#      ▼
#   [supervisor]  → lê a pergunta e decide o caminho
#      │
#      ├── "triagem"   → [agente_de_triagem]   → [registrar_e_encerrar]
#      ├── "clinico"   → [agente_clinico]       → [registrar_e_encerrar]
#      └── "seguranca" → [agente_de_seguranca]  → [registrar_e_encerrar]
#
# O que é o "estado"?
#   É um dicionário que viaja por todos os nós.
#   Cada nó pode ler e escrever no estado.
#   No final, o estado contém toda a informação da interação.
# ─────────────────────────────────────────────────────────────

import json
import os
from datetime import datetime

# TypedDict define o formato do estado (como um dicionário com tipos definidos)
from typing import TypedDict

# StateGraph é o grafo de estados do LangGraph
# END marca o ponto final do fluxo
from langgraph.graph import StateGraph, END

from agentes import (
    criar_agente_de_triagem,
    criar_agente_clinico,
    criar_agente_de_seguranca,
    criar_supervisor,
)
from configuracoes import CAMINHO_DO_LOG


# ─── Definição do estado ──────────────────────────────────────────────────────

class EstadoDaConversa(TypedDict):
    """
    O estado é o "pacote de informações" que passa por todos os nós do fluxo.

    Campos:
      pergunta        - o que o profissional de saúde perguntou
      destino         - qual agente o supervisor escolheu (triagem/clinico/seguranca)
      resposta        - o que o agente respondeu
      agente_utilizado - nome do agente que gerou a resposta
    """
    pergunta: str
    destino: str
    resposta: str
    agente_utilizado: str


# ─── Funções auxiliares ───────────────────────────────────────────────────────

def salvar_no_log(estado: EstadoDaConversa):
    """
    Salva a conversa completa em um arquivo JSONL para auditoria.
    Cada linha do arquivo é um JSON de uma interação.

    Parâmetros:
      estado - o estado final da conversa com pergunta, resposta e agente usado
    """

    # Cria a pasta de logs se não existir
    os.makedirs(os.path.dirname(CAMINHO_DO_LOG), exist_ok=True)

    # Monta o registro da interação
    registro = {
        "data_hora":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pergunta":         estado["pergunta"],
        "resposta":         estado["resposta"],
        "agente_utilizado": estado["agente_utilizado"],
    }

    # Adiciona o registro ao final do arquivo (mode "a" = append)
    with open(CAMINHO_DO_LOG, "a", encoding="utf-8") as arquivo_de_log:
        arquivo_de_log.write(json.dumps(registro, ensure_ascii=False) + "\n")

    print(f"[log] Conversa registrada em: {CAMINHO_DO_LOG}")


# ─── Funções que definem cada nó do fluxo ─────────────────────────────────────
# Cada nó é uma função que:
#   - Recebe o estado atual
#   - Faz alguma coisa (chama um agente, salva log, etc.)
#   - Retorna o estado atualizado

def criar_no_supervisor(supervisor):
    """
    Cria o nó supervisor.
    Lê a pergunta e decide qual agente vai responder.
    """
    def no_supervisor(estado: EstadoDaConversa) -> EstadoDaConversa:
        print(f"[supervisor] Analisando pergunta...")

        # Chama o supervisor com a pergunta
        destino = supervisor.invoke({"pergunta": estado["pergunta"]})

        # Remove espaços extras e deixa em minúsculo
        destino = destino.strip().lower()

        # Segurança: se o modelo responder algo inesperado, vai para clínico
        if destino not in ("triagem", "seguranca", "clinico"):
            print(f"[supervisor] Resposta inesperada '{destino}'. Usando 'clinico' como padrão.")
            destino = "clinico"

        print(f"[supervisor] Direcionando para: {destino}")

        # Atualiza o estado com o destino escolhido
        return {**estado, "destino": destino}

    return no_supervisor


def criar_no_triagem(agente_triagem):
    """Cria o nó que processa perguntas de triagem."""
    def no_triagem(estado: EstadoDaConversa) -> EstadoDaConversa:
        print("[agente_triagem] Processando pergunta de triagem...")

        resposta = agente_triagem.invoke(estado["pergunta"])

        return {**estado, "resposta": resposta, "agente_utilizado": "triagem"}

    return no_triagem


def criar_no_clinico(agente_clinico):
    """Cria o nó que processa perguntas clínicas gerais."""
    def no_clinico(estado: EstadoDaConversa) -> EstadoDaConversa:
        print("[agente_clinico] Processando pergunta clínica...")

        resposta = agente_clinico.invoke(estado["pergunta"])

        return {**estado, "resposta": resposta, "agente_utilizado": "clinico"}

    return no_clinico


def criar_no_seguranca(agente_seguranca):
    """Cria o nó que processa situações de risco e emergência."""
    def no_seguranca(estado: EstadoDaConversa) -> EstadoDaConversa:
        print("[agente_seguranca] ⚠ Processando situação de segurança...")

        resposta = agente_seguranca.invoke(estado["pergunta"])

        return {**estado, "resposta": resposta, "agente_utilizado": "seguranca"}

    return no_seguranca


def criar_no_log():
    """Cria o nó final que salva a conversa e encerra o fluxo."""
    def no_log(estado: EstadoDaConversa) -> EstadoDaConversa:
        salvar_no_log(estado)
        return estado

    return no_log


# ─── Função principal: monta o grafo completo ─────────────────────────────────

def montar_fluxo(modelo, buscador):
    """
    Monta e compila o fluxo LangGraph completo.

    Parâmetros:
      modelo   - o LLM carregado pelo carregador_do_modelo.py
      buscador - o retriever criado pelo banco_de_conhecimento.py

    Retorna: o fluxo compilado, pronto para receber perguntas
    """

    # ── 1. Cria os agentes ────────────────────────────────────────────────────
    supervisor      = criar_supervisor(modelo)
    agente_triagem  = criar_agente_de_triagem(modelo, buscador)
    agente_clinico  = criar_agente_clinico(modelo, buscador)
    agente_seguranca = criar_agente_de_seguranca(modelo, buscador)

    # ── 2. Cria os nós ────────────────────────────────────────────────────────
    no_supervisor  = criar_no_supervisor(supervisor)
    no_triagem     = criar_no_triagem(agente_triagem)
    no_clinico     = criar_no_clinico(agente_clinico)
    no_seguranca   = criar_no_seguranca(agente_seguranca)
    no_log         = criar_no_log()

    # ── 3. Monta o grafo ──────────────────────────────────────────────────────
    grafo = StateGraph(EstadoDaConversa)

    # Adiciona cada nó ao grafo com um nome
    grafo.add_node("supervisor",  no_supervisor)
    grafo.add_node("triagem",     no_triagem)
    grafo.add_node("clinico",     no_clinico)
    grafo.add_node("seguranca",   no_seguranca)
    grafo.add_node("log",         no_log)

    # Define que o fluxo começa pelo supervisor
    grafo.set_entry_point("supervisor")

    # Define o roteamento condicional:
    # O supervisor define o campo "destino" no estado,
    # e aqui mapeamos esse valor para o nó correto
    grafo.add_conditional_edges(
        "supervisor",                      # nó de origem
        lambda estado: estado["destino"],  # função que lê o destino do estado
        {
            "triagem":   "triagem",        # se destino == "triagem"  → vai para o nó triagem
            "clinico":   "clinico",        # se destino == "clinico"  → vai para o nó clinico
            "seguranca": "seguranca",      # se destino == "seguranca"→ vai para o nó seguranca
        }
    )

    # Todos os agentes terminam indo para o nó de log
    grafo.add_edge("triagem",   "log")
    grafo.add_edge("clinico",   "log")
    grafo.add_edge("seguranca", "log")

    # O nó de log encerra o fluxo
    grafo.add_edge("log", END)

    # ── 4. Compila e retorna ──────────────────────────────────────────────────
    fluxo_compilado = grafo.compile()
    print("[fluxo] Fluxo montado e compilado com sucesso!")
    return fluxo_compilado
