"""
04_merge_and_push.py
─────────────────────────────────────────────────────────────
Faz merge do adapter LoRA com o modelo base e publica no HuggingFace Hub.

Uso:
    python tunning/04_merge_and_push.py \
        --adapter-path pre-trained/qwen2.5-3b-medpt-lora \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --repo-id SEU_USUARIO/qwen2.5-3b-medpt-lora \
        --private

Pré-requisitos:
    pip install peft transformers torch huggingface_hub
    huggingface-cli login   (ou defina HF_TOKEN no .env)

Após publicar, basta usar no .env:
    HF_PIPELINE_MODEL=SEU_USUARIO/qwen2.5-3b-medpt-lora
"""

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter e publica no HF Hub")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="pre-trained/qwen2.5-3b-medpt-lora",
        help="Caminho do adapter LoRA (saída do 02_finetune_lora.py)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Modelo base do HuggingFace Hub",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="ID do repositório no Hub (ex: seu-usuario/qwen2.5-3b-medpt-lora)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Diretório local para salvar o modelo mergeado (opcional)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Criar repositório privado no Hub",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype para salvar o modelo mergeado",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    output_dir = args.output_dir or f"merged-model-{args.repo_id.split('/')[-1]}"

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "Defina HF_TOKEN no .env ou faça `huggingface-cli login`"
        )

    print(f"[1/4] Carregando modelo base: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="cpu",
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)

    print(f"[2/4] Carregando adapter LoRA: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("[3/4] Fazendo merge dos pesos...")
    model = model.merge_and_unload()

    print(f"[3/4] Salvando modelo mergeado em: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"[4/4] Publicando no HuggingFace Hub: {args.repo_id}")
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=args.repo_id,
        commit_message="Upload modelo fine-tuned mergeado (LoRA + base)",
    )

    print(f"\nPronto! Use no .env:")
    print(f"  HF_PIPELINE_MODEL={args.repo_id}")


if __name__ == "__main__":
    main()
