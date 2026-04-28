#!/usr/bin/env python
"""Executa o fine-tuning LoRA do Qwen2.5-3B-Instruct com Unsloth.

O script foi pensado para rodar no Google Colab com Tesla T4 15 GB:
- usa quantizacao 4-bit;
- aplica PEFT/LoRA com os hiperparametros definidos no PASSO 1;
- salva checkpoints intermediarios em `data/`;
- salva o modelo final (adapter) e tokenizer em `pre-trained/`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

from datasets import DatasetDict, load_from_disk
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


DEFAULT_DATASET_DIR = "data/processed/medpt_qa"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_CHECKPOINT_DIR = "data/checkpoints/qwen2.5-3b-medpt-lora"
DEFAULT_FINAL_DIR = "pre-trained/qwen2.5-3b-medpt-lora"

SYSTEM_MESSAGE = (
    "Você é um assistente virtual médico em português do Brasil com foco em apoio "
    "informacional. Nunca substitua o julgamento clínico humano. Sempre explicite "
    "limites quando houver incerteza e, em sinais de gravidade, oriente busca "
    "imediata por atendimento médico."
)

SAFETY_DISCLAIMER = (
    "Aviso: esta resposta tem caráter informativo e deve ser validada por um "
    "profissional de saúde."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning LoRA do Qwen2.5-3B-Instruct com Unsloth.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Diretorio salvo pelo script 01 com DatasetDict processado.",
    )
    parser.add_argument(
        "--base-model-name",
        default=DEFAULT_BASE_MODEL,
        help="Nome do modelo base no Hugging Face.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Diretorio para checkpoints intermediarios.",
    )
    parser.add_argument(
        "--final-dir",
        default=DEFAULT_FINAL_DIR,
        help="Diretorio para salvar o adapter final e tokenizer.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Comprimento maximo de sequencia. 1024 e conservador para T4.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
        help="Batch size por dispositivo.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Acumulacao de gradientes.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate do fine-tuning LoRA.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Numero de epocas.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Intervalo de log.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Intervalo de checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente global.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limita amostras de treino para dry run.",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="Limita amostras de validacao para dry run.",
    )
    parser.add_argument(
        "--append-safety-disclaimer",
        action="store_true",
        help="Anexa um aviso de seguranca ao final de cada resposta de treino.",
    )
    return parser.parse_args()


def ensure_required_splits(dataset: DatasetDict) -> None:
    required = {"train", "validation"}
    missing = required.difference(dataset.keys())
    if missing:
        raise ValueError(f"Dataset processado sem splits obrigatorios: {sorted(missing)}")


def apply_limits(dataset: DatasetDict, args: argparse.Namespace) -> DatasetDict:
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    if args.max_validation_samples:
        dataset["validation"] = dataset["validation"].select(
            range(min(args.max_validation_samples, len(dataset["validation"])))
        )
    return dataset


def format_example(question: str, answer: str, tokenizer, append_safety_disclaimer: bool) -> Dict[str, str]:
    assistant_answer = answer.strip()
    if append_safety_disclaimer and SAFETY_DISCLAIMER.lower() not in assistant_answer.lower():
        assistant_answer = f"{assistant_answer}\n\n{SAFETY_DISCLAIMER}"

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": assistant_answer},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        text = (
            f"[SYSTEM]\n{SYSTEM_MESSAGE}\n\n"
            f"[USER]\n{question.strip()}\n\n"
            f"[ASSISTANT]\n{assistant_answer}"
        )
    return {"text": text}


def prepare_text_datasets(dataset: DatasetDict, tokenizer, args: argparse.Namespace) -> DatasetDict:
    def formatter(batch):
        texts = []
        for question, answer in zip(batch["question"], batch["answer"]):
            texts.append(
                format_example(
                    question=question,
                    answer=answer,
                    tokenizer=tokenizer,
                    append_safety_disclaimer=args.append_safety_disclaimer,
                )["text"]
            )
        return {"text": texts}

    columns_to_remove = dataset["train"].column_names
    train_dataset = dataset["train"].map(
        formatter,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Formatando dataset de treino",
    )
    validation_dataset = dataset["validation"].map(
        formatter,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Formatando dataset de validacao",
    )
    return DatasetDict({"train": train_dataset, "validation": validation_dataset})


def save_training_metadata(args: argparse.Namespace, final_dir: Path) -> None:
    metadata = {
        "base_model_name": args.base_model_name,
        "max_seq_length": args.max_seq_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "lora": {
            "r": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "system_message": SYSTEM_MESSAGE,
        "append_safety_disclaimer": args.append_safety_disclaimer,
        "safety_disclaimer": SAFETY_DISCLAIMER if args.append_safety_disclaimer else "",
    }
    with (final_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset = load_from_disk(args.dataset_dir)
    ensure_required_splits(dataset)
    dataset = apply_limits(dataset, args)

    checkpoint_dir = Path(args.checkpoint_dir)
    final_dir = Path(args.final_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # 1) Carrega o modelo base em 4-bit, adequado para o ambiente do Colab.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # 2) Aplica LoRA com os hiperparametros definidos no planejamento.
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # 3) Converte o dataset limpo para o formato textual esperado pelo trainer.
    text_datasets = prepare_text_datasets(dataset, tokenizer, args)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=text_datasets["train"],
        eval_dataset=text_datasets["validation"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=SFTConfig(
            output_dir=str(checkpoint_dir),
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_strategy="steps",
            eval_strategy="steps",
            eval_steps=args.save_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            optim="adamw_8bit",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            report_to="none",
            seed=args.seed,
        ),
    )

    trainer.train()

    # 4) Salva o adapter final e o tokenizer para inferencia e avaliacao posterior.
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    save_training_metadata(args, final_dir)

    print(f"Checkpoints salvos em: {checkpoint_dir}")
    print(f"Modelo final (adapter) salvo em: {final_dir}")


if __name__ == "__main__":
    main()
