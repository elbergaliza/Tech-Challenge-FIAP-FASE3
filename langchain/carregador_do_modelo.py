# carregador_do_modelo.py
# ─────────────────────────────────────────────────────────────
# Carregamento do modelo de linguagem.
#
# Prioridade (variáveis de ambiente têm precedência sobre configuracoes.py):
#
#   1. LOCAL_MODEL_PATH — modelo causal completo no disco (ex.: merge LoRA).
#   2. LORA_CHECKPOINT_DIR ou CAMINHO_CHECKPOINT_LORA — pasta checkpoint-* do Trainer.
#   3. LORA_ADAPTER_PATH — adapter final PEFT/LoRA; base em LORA_BASE_MODEL,
#      training_metadata.json (base_model_name) ou CAMINHO_DO_MODELO.
#      Opcionalmente detecta pre-trained/qwen2.5-3b-medpt-lora na raiz do repositório.
#   4. OLLAMA_BASE_URL (+ OLLAMA_API_KEY) — API compatível OpenAI.
#   5. HF_PIPELINE_MODEL ou CAMINHO_DO_MODELO — Hub ou caminho local via pipeline.
#
#   LLM_SYSTEM_MESSAGE — sobrescreve a mensagem de sistema (senão usa training_metadata.json
#      ou MENSAGEM_SISTEMA_LLM em configuracoes.py).
#
# dotenv: carrega `.env` no CWD e na raiz do repositório (pasta acima de langchain/).
#
# Hiperparâmetros locais (via .env):
#   LOCAL_MAX_NEW_TOKENS, LOCAL_TEMPERATURE, LOCAL_REPETITION_PENALTY,
#   LOCAL_TOP_P, LOCAL_TOP_K
#   LOCAL_CHAT_TEMPLATE — "1" (padrão) aplica apply_chat_template quando existir;
#                         "0" envia o prompt bruto ao pipeline.
#   LORA_AUTO_DISCOVER — "1" (padrão) tenta pre-trained/qwen2.5-3b-medpt-lora na raiz;
#                        "0" só carrega LoRA via caminhos explícitos (checkpoint, adapter ou config).
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from configuracoes import HF_PIPELINE_MODEL

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv_both() -> None:
    load_dotenv()
    load_dotenv(_REPO_ROOT / ".env")


_load_dotenv_both()


def _chat_template_enabled() -> bool:
    return os.environ.get("LOCAL_CHAT_TEMPLATE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _format_chat_prompt(tokenizer, user_text: str, system_message: str | None) -> str:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_text.strip()})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        if system_message:
            return (
                f"[SYSTEM]\n{system_message}\n\n"
                f"[USER]\n{user_text.strip()}\n\n"
                "[ASSISTANT]\n"
            )
        return user_text


def _default_adapter_dir() -> Path | None:
    """Caminho padrão do adapter do script 02, se a pasta existir."""
    candidate = _REPO_ROOT / "pre-trained" / "qwen2.5-3b-medpt-lora"
    if candidate.is_dir() and (candidate / "adapter_config.json").exists():
        return candidate
    return None


def _read_training_metadata(adapter_dir: Path) -> dict:
    """Lê training_metadata.json gravado pelo tunning/02 junto ao export final."""
    meta_path = adapter_dir / "training_metadata.json"
    if not meta_path.is_file():
        return {}
    try:
        with meta_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_peft_adapter_dir(raw: str | None) -> Path | None:
    """Resolve pasta com adapter PEFT (adapter_config.json); None se inválido."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        if p.is_dir() and (p / "adapter_config.json").exists():
            return p
        return None
    for base in (Path.cwd(), _REPO_ROOT):
        cand = (base / p).resolve()
        if cand.is_dir() and (cand / "adapter_config.json").exists():
            return cand
    cand = (Path.cwd() / p).resolve()
    if cand.is_dir() and (cand / "adapter_config.json").exists():
        return cand
    return None


def _strip_generated_prefix(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].strip()
    return full_text.strip()


class _LocalPipelineRunnable:
    """Envolve pipeline HF para invoke(str) e atributo .content (compatível com ChatOpenAI)."""

    def __init__(self, pipe, tokenizer, system_message: str | None):
        self._pipe = pipe
        self._tokenizer = tokenizer
        self._system_message = system_message

    def invoke(self, input, config=None):
        del config
        text_in = input if isinstance(input, str) else str(input)
        if _chat_template_enabled() and getattr(self._tokenizer, "chat_template", None):
            prompt = _format_chat_prompt(
                self._tokenizer, text_in, self._system_message
            )
        else:
            prompt = text_in

        raw = self._pipe(prompt)
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            full = raw[0].get("generated_text", "")
        else:
            full = str(raw)
        content = _strip_generated_prefix(full, prompt)
        return SimpleNamespace(content=content)


def _carregar_pipeline(origem: str, system_message: str | None):
    """
    Carrega um modelo via pipeline() do transformers.
    `origem` pode ser um nome do HuggingFace Hub ou um caminho local.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    max_new_tokens = int(os.environ.get("LOCAL_MAX_NEW_TOKENS", 512))
    temperature = float(os.environ.get("LOCAL_TEMPERATURE", 0.3))
    repetition_penalty = float(os.environ.get("LOCAL_REPETITION_PENALTY", 1.1))
    top_p = float(os.environ.get("LOCAL_TOP_P", 0.9))
    top_k = int(os.environ.get("LOCAL_TOP_K", 50))

    device = 0 if torch.cuda.is_available() else -1
    dispositivo_str = "GPU" if device == 0 else "CPU"
    print(f"[carregador_do_modelo] Carregando '{origem}' em {dispositivo_str}...")

    tokenizer = AutoTokenizer.from_pretrained(origem, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        max_length=None,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )

    return _LocalPipelineRunnable(pipe, tokenizer, system_message)


def _carregar_lora(adapter_dir: str, base_model_name: str, system_message: str | None):
    """Base + adapter PEFT (mesma ideia que tunning/03_evaluate_model.py)."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
    )
    import torch
    from peft import PeftModel

    max_new_tokens = int(os.environ.get("LOCAL_MAX_NEW_TOKENS", 512))
    temperature = float(os.environ.get("LOCAL_TEMPERATURE", 0.3))
    repetition_penalty = float(os.environ.get("LOCAL_REPETITION_PENALTY", 1.1))
    top_p = float(os.environ.get("LOCAL_TOP_P", 0.9))
    top_k = int(os.environ.get("LOCAL_TOP_K", 50))

    adapter_path = Path(adapter_dir)
    tokenizer_src = str(adapter_path) if adapter_path.is_dir() else base_model_name
    print(
        f"[carregador_do_modelo] Carregando LoRA de '{adapter_path}' "
        f"(base={base_model_name})..."
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print(
            "[carregador_do_modelo] CUDA indisponível: carregando base em float32 "
            "(mais RAM; bitsandbytes 4-bit costuma exigir GPU)."
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        max_length=None,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )

    return _LocalPipelineRunnable(pipe, tokenizer, system_message)


def carregar_modelo():
    """
    Carrega o LLM conforme variáveis de ambiente e defaults em configuracoes.py.
    """
    _load_dotenv_both()

    from configuracoes import (
        BASE_DIR,
        CAMINHO_CHECKPOINT_LORA,
        CAMINHO_DO_ADAPTER_LORA,
        CAMINHO_DO_MODELO,
        MENSAGEM_SISTEMA_LLM,
    )

    local_path = os.environ.get("LOCAL_MODEL_PATH")
    lora_env = os.environ.get("LORA_ADAPTER_PATH")
    if lora_env is not None and not str(lora_env).strip():
        lora_env = None
    ollama_url = os.environ.get("OLLAMA_BASE_URL")
    ollama_key = os.environ.get("OLLAMA_API_KEY", "")
    hf_pipeline_model = os.environ.get("HF_PIPELINE_MODEL")

    lora_path = _resolve_peft_adapter_dir(os.environ.get("LORA_CHECKPOINT_DIR"))
    if lora_path is None and CAMINHO_DO_ADAPTER_LORA:
      p = Path(CAMINHO_DO_ADAPTER_LORA)
      if not p.is_absolute():
        # Mudamos de _REPO_ROOT para BASE_DIR que vem de configuracoes.py
        p = (BASE_DIR / p).resolve() 
      if p.is_dir() and (p / "adapter_config.json").exists():
        lora_path = p

    if lora_path is None:
        lora_path = _resolve_peft_adapter_dir(lora_env)

    if lora_path is None and CAMINHO_DO_ADAPTER_LORA:
        p = Path(CAMINHO_DO_ADAPTER_LORA)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        if p.is_dir() and (p / "adapter_config.json").exists():
            lora_path = p

    if lora_path is None and os.environ.get("LORA_AUTO_DISCOVER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    ):
        auto = _default_adapter_dir()
        if auto is not None:
            lora_path = auto

    metadata = _read_training_metadata(lora_path) if lora_path else {}
    env_base = (os.environ.get("LORA_BASE_MODEL") or "").strip()
    base_for_lora = env_base or metadata.get("base_model_name") or CAMINHO_DO_MODELO
    env_system = (os.environ.get("LLM_SYSTEM_MESSAGE") or "").strip()
    system_for_lora = env_system or metadata.get("system_message") or MENSAGEM_SISTEMA_LLM

    if metadata and not env_base and metadata.get("base_model_name"):
        print(
            f"[carregador_do_modelo] base_model_name do training_metadata.json: "
            f"{metadata['base_model_name']}"
        )

    hf_fallback = os.environ.get("HF_PIPELINE_MODEL") or HF_PIPELINE_MODEL or CAMINHO_DO_MODELO

    if local_path:
        print(f"[carregador_do_modelo] Modelo completo no disco: {local_path}")
        modelo = _carregar_pipeline(local_path, MENSAGEM_SISTEMA_LLM)

    elif lora_path is not None:
        print(f"[carregador_do_modelo] Adapter LoRA: {lora_path}")
        modelo = _carregar_lora(str(lora_path), base_for_lora, system_for_lora)

    elif ollama_url:
        print(f"[carregador_do_modelo] Ollama Cloud: {ollama_url}")
        modelo = ChatOpenAI(
            model=os.environ.get("OLLAMA_MODEL", "gemma3:4b"),
            base_url=f"{ollama_url}/v1",
            api_key=ollama_key,
            temperature=0.3,
            max_tokens=1000,
            http_client=httpx.Client(verify=False),
        )

    else:
        print(f"[carregador_do_modelo] Pipeline HuggingFace: {hf_fallback}")
        modelo = _carregar_pipeline(hf_fallback, MENSAGEM_SISTEMA_LLM)

    print("[carregador_do_modelo] Modelo carregado com sucesso!")
    return modelo
