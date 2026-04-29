# agentes.py
# ─────────────────────────────────────────────────────────────
# Define os agentes do sistema.
#
# AGENTES:
#
#   ┌─────────────────────────────────────────────────────┐
#   │  AGENTE DE TRIAGEM                                  │
#   │  Recebe a pergunta, avalia gravidade                │
#   │  e decide: "clinico" ou "seguranca"                 │
#   ├─────────────────────────────────────────────────────┤
#   │  AGENTE CLÍNICO                                     │
#   │  Responde dúvidas médicas gerais                    │
#   ├─────────────────────────────────────────────────────┤
#   │  AGENTE DE SEGURANÇA                                │
#   │  Trata emergências, riscos e violência              │
#   └─────────────────────────────────────────────────────┘
#
# O agente de triagem substitui o supervisor.
# Ele não só classifica — ele já avalia clinicamente a gravidade
# e decide para onde encaminhar.
# ─────────────────────────────────────────────────────────────

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ─── Prompt da triagem ────────────────────────────────────────
# A triagem agora tem duas responsabilidades:
#   1. Avaliar clinicamente a pergunta
#   2. Decidir o encaminhamento: "clinico" ou "seguranca"

PROMPT_TRIAGEM = PromptTemplate(
    input_variables=["context", "pergunta"],
    template="""Você é um enfermeiro triagista especializado.
Analise a pergunta ou situação relatada e faça duas coisas:

1. Avalie clinicamente: identifique sintomas, urgência e contexto
2. Decida o encaminhamento respondendo APENAS com uma palavra:
   - "clinico"   → dúvida geral, procedimento, medicação, protocolo
   - "seguranca" → risco imediato, emergência, violência, perigo de vida

Contexto encontrado na base de conhecimento:
{context}

Situação ou pergunta recebida:
{pergunta}

Encaminhamento (responda APENAS: clinico ou seguranca):"""
)

# ─── Prompt clínico ───────────────────────────────────────────
PROMPT_CLINICO = PromptTemplate(
    input_variables=["context", "pergunta"],
    template="""Você é um assistente médico de apoio a profissionais de saúde.
Responda com base no contexto abaixo. Nunca prescreva medicamentos diretamente.

Contexto encontrado na base de conhecimento:
{context}

Pergunta do profissional:
{pergunta}

Resposta:"""
)

# ─── Prompt de segurança ──────────────────────────────────────
PROMPT_SEGURANCA = PromptTemplate(
    input_variables=["context", "pergunta"],
    template="""Você é um agente de segurança médica.
Analise a situação e identifique: risco imediato ao paciente,
necessidade de encaminhamento urgente e protocolos a acionar.

Contexto encontrado na base de conhecimento:
{context}

Situação relatada:
{pergunta}

Avaliação de risco e protocolo recomendado:"""
)


# ─── Funções que criam os agentes ─────────────────────────────

def criar_agente_de_triagem(modelo, buscador):
    """
    Cria o agente de triagem.
    Ele avalia a pergunta E decide o encaminhamento.
    Retorna "clinico" ou "seguranca".
    """
    cadeia = (
        {"context": buscador, "pergunta": RunnablePassthrough()}
        | PROMPT_TRIAGEM
        | modelo
        | StrOutputParser()
    )
    return cadeia


def criar_agente_clinico(modelo, buscador):
    """
    Cria o agente clínico.
    Responde dúvidas médicas gerais com base no RAG.
    """
    cadeia = (
        {"context": buscador, "pergunta": RunnablePassthrough()}
        | PROMPT_CLINICO
        | modelo
        | StrOutputParser()
    )
    return cadeia


def criar_agente_de_seguranca(modelo, buscador):
    """
    Cria o agente de segurança.
    Trata emergências, violência e situações de risco imediato.
    """
    cadeia = (
        {"context": buscador, "pergunta": RunnablePassthrough()}
        | PROMPT_SEGURANCA
        | modelo
        | StrOutputParser()
    )
    return cadeia
