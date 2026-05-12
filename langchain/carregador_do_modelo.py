# carregador_do_modelo.py
# ─────────────────────────────────────────────────────────────
# Carregamento do modelo de linguagem.
#
# Suporta 3 providers via variáveis de ambiente:
#
#   1. Pipeline local → HF_PIPELINE_MODEL  (padrão: "gpt2")
#        Baixa o modelo do HuggingFace Hub e roda localmente.
#        Não requer API key.
#
#   2. Ollama Cloud   → OLLAMA_BASE_URL + OLLAMA_API_KEY
#
#   3. Modelo no disco → LOCAL_MODEL_PATH
#        Carrega modelo de um caminho local — para o fine-tuned futuro.
#        Não requer API key.
#
# Hiperparâmetros dos providers 1 e 3 (configuráveis via .env):
#   LOCAL_MAX_NEW_TOKENS      (padrão: 512)
#   LOCAL_TEMPERATURE         (padrão: 0.3)
#   LOCAL_REPETITION_PENALTY  (padrão: 1.1)
#   LOCAL_TOP_P               (padrão: 0.9)
#   LOCAL_TOP_K               (padrão: 50)
# ─────────────────────────────────────────────────────────────

import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def _carregar_pipeline(origem: str):
    """
    Carrega um modelo via pipeline() do transformers.
    `origem` pode ser um nome do HuggingFace Hub ou um caminho local.
    Não requer API key em nenhum dos casos.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    try:
        from langchain_huggingface import HuggingFacePipeline
    except ImportError:
        from langchain_community.llms import HuggingFacePipeline

    max_new_tokens     = int(os.environ.get("LOCAL_MAX_NEW_TOKENS", 512))
    temperature        = float(os.environ.get("LOCAL_TEMPERATURE", 0.3))
    repetition_penalty = float(os.environ.get("LOCAL_REPETITION_PENALTY", 1.1))
    top_p              = float(os.environ.get("LOCAL_TOP_P", 0.9))
    top_k              = int(os.environ.get("LOCAL_TOP_K", 50))

    device = 0 if torch.cuda.is_available() else -1
    dispositivo_str = "GPU" if device == 0 else "CPU"
    print(f"[carregador_do_modelo] Carregando '{origem}' em {dispositivo_str}...")

    tokenizer = AutoTokenizer.from_pretrained(origem, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        origem,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        device_map="auto" if device == 0 else None,
    )

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(pipeline=pipe)


def carregar_modelo():
    """
    Carrega o modelo LLM conforme variáveis de ambiente.

    Prioridade:
      1. Pipeline local  (HF_PIPELINE_MODEL, padrão: "gpt2" — sem API key)
      2. Ollama Cloud    (OLLAMA_BASE_URL + OLLAMA_API_KEY)
      3. Modelo no disco (LOCAL_MODEL_PATH — fine-tuned local, sem API key)
    """

    hf_pipeline_model = os.environ.get("HF_PIPELINE_MODEL", "gpt2")
    ollama_url        = os.environ.get("OLLAMA_BASE_URL")
    ollama_key        = os.environ.get("OLLAMA_API_KEY", "")
    local_path        = os.environ.get("LOCAL_MODEL_PATH")

    if local_path:
        print(f"[carregador_do_modelo] Usando modelo local no disco: {local_path}")
        modelo = _carregar_pipeline(local_path)

    elif ollama_url:
        print(f"[carregador_do_modelo] Usando Ollama Cloud: {ollama_url}")
        modelo = ChatOpenAI(
            model=os.environ.get("OLLAMA_MODEL", "gemma3:4b"),
            base_url=f"{ollama_url}/v1",
            api_key=ollama_key,
            temperature=0.3,
            max_tokens=1000,
            http_client=httpx.Client(verify=False),
        )

    else:
        print(f"[carregador_do_modelo] Usando pipeline() local: {hf_pipeline_model}")
        modelo = _carregar_pipeline(hf_pipeline_model)

    print("[carregador_do_modelo] Modelo carregado com sucesso!")
    return modelo
