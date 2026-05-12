# carregador_do_modelo.py
# ─────────────────────────────────────────────────────────────
# Carregamento do modelo de linguagem.
#
# Suporta múltiplos providers via variáveis de ambiente:
#   - OLLAMA_BASE_URL + OLLAMA_API_KEY → Ollama Cloud (padrão)
#   - OPENAI_API_KEY → OpenAI API
#
# Configurável para futura integração de modelo fine-tuned local.
# ─────────────────────────────────────────────────────────────

import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def carregar_modelo():
    """
    Carrega o modelo LLM conforme variáveis de ambiente.

    Prioridade:
      1. Ollama Cloud (se OLLAMA_BASE_URL definido)
      2. OpenAI (se OPENAI_API_KEY definido)
    """

    ollama_url = os.environ.get("OLLAMA_BASE_URL")
    ollama_key = os.environ.get("OLLAMA_API_KEY", "")

    if ollama_url:
        print(f"[carregador_do_modelo] Usando Ollama Cloud: {ollama_url}")
        modelo = ChatOpenAI(
            model=os.environ.get("OLLAMA_MODEL", "gemma3:4b"),
            base_url=f"{ollama_url}/v1",
            api_key=ollama_key,
            temperature=0.3,
            max_tokens=1000,
            http_client=httpx.Client(verify=False),
        )
    elif os.environ.get("OPENAI_API_KEY"):
        print("[carregador_do_modelo] Usando OpenAI API")
        modelo = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            max_tokens=1000,
        )
    else:
        raise RuntimeError(
            "Nenhum provider de LLM configurado. "
            "Defina OLLAMA_BASE_URL ou OPENAI_API_KEY no .env"
        )

    print("[carregador_do_modelo] Modelo carregado com sucesso!")
    return modelo
