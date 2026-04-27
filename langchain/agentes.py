# agentes.py
# ─────────────────────────────────────────────────────────────
# Esse arquivo define os agentes do sistema.
#
# O que é um agente?
#   É uma "cadeia" (chain) no LangChain que combina:
#   - Um prompt (instrução de como o modelo deve se comportar)
#   - O buscador RAG (para buscar contexto relevante)
#   - O modelo (para gerar a resposta)
#
# Temos 3 agentes especializados:
#
#   ┌─────────────────────────────────────────────────────┐
#   │  AGENTE DE TRIAGEM                                  │
#   │  Recebe sintomas e avalia urgência                  │
#   ├─────────────────────────────────────────────────────┤
#   │  AGENTE DE PERGUNTAS CLÍNICAS                       │
#   │  Responde dúvidas médicas gerais                    │
#   ├─────────────────────────────────────────────────────┤
#   │  AGENTE DE SEGURANÇA                                │
#   │  Detecta situações de risco e aciona alertas        │
#   └─────────────────────────────────────────────────────┘
#
# E 1 supervisor que decide qual agente usar:
#
#   SUPERVISOR → lê a pergunta → escolhe o agente certo
# ─────────────────────────────────────────────────────────────

# PromptTemplate é o molde de texto que enviamos ao modelo
from langchain_core.prompts import PromptTemplate

# StrOutputParser pega a resposta do modelo e retorna como texto simples
from langchain_core.output_parsers import StrOutputParser

# RunnablePassthrough passa o valor de entrada sem modificar
from langchain_core.runnables import RunnablePassthrough


# ─── Prompts de cada agente ───────────────────────────────────────────────────
# Os prompts são as "instruções de comportamento" de cada agente.
# O {context} será preenchido com os resultados do RAG.
# O {pergunta} será preenchido com a pergunta do usuário.

PROMPT_TRIAGEM = PromptTemplate(
    input_variables=["context", "pergunta"],
    template="""Você é um assistente médico especializado em triagem clínica.
Use apenas as informações do contexto abaixo para responder.
Nunca faça diagnósticos definitivos. Sempre sugira consulta presencial para casos graves.

Contexto encontrado na base de conhecimento:
{context}

Pergunta do profissional:
{pergunta}

Avaliação de triagem (indique nível de urgência e próximos passos):"""
)

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

PROMPT_SEGURANCA = PromptTemplate(
    input_variables=["context", "pergunta"],
    template="""Você é um agente de segurança médica.
Analise a situação e identifique: risco imediato ao paciente, necessidade de encaminhamento urgente
e quais protocolos devem ser acionados.

Contexto encontrado na base de conhecimento:
{context}

Situação relatada:
{pergunta}

Avaliação de risco e protocolo recomendado:"""
)

PROMPT_SUPERVISOR = PromptTemplate(
    input_variables=["pergunta"],
    template="""Classifique a pergunta abaixo em uma das três categorias:

- "triagem": envolve sintomas, urgência ou avaliação de quadro clínico
- "seguranca": envolve risco imediato, emergência, violência ou alerta crítico
- "clinico": dúvida geral sobre procedimento, medicação ou protocolo

Pergunta: {pergunta}

Responda APENAS com uma palavra: triagem, seguranca ou clinico"""
)


# ─── Funções que criam os agentes ─────────────────────────────────────────────

def criar_agente_de_triagem(modelo, buscador):
    """
    Cria o agente responsável por avaliar sintomas e urgência.

    Como funciona a cadeia (|):
      1. {"context": buscador, "pergunta": RunnablePassthrough()}
         → busca os documentos relevantes E passa a pergunta original
      2. PROMPT_TRIAGEM
         → monta o texto completo com contexto + pergunta
      3. modelo
         → gera a resposta
      4. StrOutputParser()
         → extrai o texto da resposta

    Parâmetros:
      modelo   - o LLM carregado pelo carregador_do_modelo.py
      buscador - o retriever criado pelo banco_de_conhecimento.py

    Retorna: a cadeia pronta para receber perguntas
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
    Cria o agente responsável por responder dúvidas clínicas gerais.

    Parâmetros:
      modelo   - o LLM carregado pelo carregador_do_modelo.py
      buscador - o retriever criado pelo banco_de_conhecimento.py

    Retorna: a cadeia pronta para receber perguntas
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
    Cria o agente responsável por detectar riscos e acionar protocolos.

    Parâmetros:
      modelo   - o LLM carregado pelo carregador_do_modelo.py
      buscador - o retriever criado pelo banco_de_conhecimento.py

    Retorna: a cadeia pronta para receber perguntas
    """
    cadeia = (
        {"context": buscador, "pergunta": RunnablePassthrough()}
        | PROMPT_SEGURANCA
        | modelo
        | StrOutputParser()
    )
    return cadeia


def criar_supervisor(modelo):
    """
    Cria o supervisor que decide qual agente vai responder.
    O supervisor NÃO usa o buscador — ele só lê a pergunta e decide.

    Parâmetros:
      modelo - o LLM carregado pelo carregador_do_modelo.py

    Retorna: a cadeia que recebe uma pergunta e retorna "triagem", "seguranca" ou "clinico"
    """
    cadeia = PROMPT_SUPERVISOR | modelo | StrOutputParser()
    return cadeia
