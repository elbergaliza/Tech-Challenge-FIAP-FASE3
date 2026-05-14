#!/usr/bin/env python
"""Executa o fine-tuning LoRA do Qwen2.5-3B-Instruct com Unsloth.

O script foi pensado para rodar no Google Colab com Tesla T4 15 GB:
- usa quantizacao 4-bit;
- aplica PEFT/LoRA com hiperparametros fixos no codigo (alinhados ao planejamento);
- salva checkpoints intermediarios (padrao sob `data/checkpoints/...`, relativos ao CWD);
- grava relatorio JSON por execucao em `execution_logs/` dentro do diretorio de checkpoints;
- salva o adapter final, tokenizer e `training_metadata.json` (padrao sob `pre-trained/...`).
"""

# PEP 563: avalia anotacoes de tipo de forma tardia (strings).
# Neste modulo: suporta anotacoes nas assinaturas sem referencias circulares em import time.
from __future__ import annotations

# Biblioteca padrao para interface de linha de comando (`ArgumentParser`).
# Neste modulo: `parse_args()` expoe paths, modelo base, hiperparametros de treino (LR, batch,
# epocas, etc.), intervalos de log/salvamento, seed, limites de amostras e flag de disclaimer.
# Os hiperparametros LoRA (r, alpha, dropout, target_modules) ficam fixos em codigo, nao na CLI.
import argparse

# Biblioteca padrao de serializacao JSON.
# Neste modulo: gravar `training_metadata.json` apos o treino com a configuracao usada.
import json

# Biblioteca padrao de interface com o processo (variaveis de ambiente).
# Neste modulo: `os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")` evita avisos do HF tokenizers ao usar o trainer.
# O que é os aqui: uso de os.environ (não só “interface genérica”).
# O que faz o setdefault: roda em main() antes do treino.
# Por que TOKENIZERS_PARALLELISM=false:
#   .tokenizers em Rust pode usar paralelismo;
#   .com DataLoader, multiprocessing ou fork isso pode gerar aviso de não ser
#       fork-safe ou instabilidade;
#   .desligar o paralelismo do tokenizer é o padrão recomendado com HF Trainer.
# Por que setdefault: só preenche se a variável ainda não existir,
#   para o ambiente poder definir antes da execução.
import os

# Modulo padrao para caminhos de arquivo orientados a objeto.
# Neste modulo: diretorios de checkpoint, saida final e arquivo de metadados.
from pathlib import Path

# Data e hora para nome unico de arquivo de relatorio de execucao.
from datetime import datetime, timezone

# Modulo padrao de anotacoes de tipo.
# Neste modulo: `Dict` em `format_example`; `Any` no dict `sft_training_kwargs` antes de
# `SFTConfig(**...)` para alinhar tipagem e versoes do TRL sem ruído estatico.
from typing import Any, Dict, List, Optional

# Callback e tipos do Trainer para registrar salvamentos e avaliacoes durante o treino.
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

# Hugging Face `datasets`: estrutura de splits e leitura de dataset em disco.
# Neste modulo: `load_from_disk` le o diretorio gerado pelo script 01 (tipicamente `DatasetDict`);
# `main()` rejeita se vier um `Dataset` simples. `DatasetDict` tipa splits apos essa checagem.
from datasets import DatasetDict, load_from_disk

# Unsloth: carregamento otimizado de LLMs e integracao com LoRA/PEFT.
# Neste modulo: `FastLanguageModel` carrega o modelo em 4-bit, aplica adapters com
# `use_gradient_checkpointing="unsloth"` em `get_peft_model`; `is_bfloat16_supported`
# define `bf16` vs `fp16` no `SFTConfig`.
from unsloth import FastLanguageModel, is_bfloat16_supported

# Hugging Face TRL (Transformer Reinforcement Learning): treino alinhado a transformers.
# Neste modulo: SFT supervisionado — `SFTConfig` (herda `TrainingArguments`) e `SFTTrainer`
# treinam no campo `text` com `processing_class` = tokenizer.
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer


DEFAULT_DATASET_DIR = "data/processed/medpt_qa"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_CHECKPOINT_DIR = "data/checkpoints/qwen2.5-3b-medpt-lora"
DEFAULT_FINAL_DIR = "pre-trained/qwen2.5-3b-medpt-lora"

EXECUTION_LOGS_SUBDIR = "execution_logs"

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


def _resolved_per_device_eval_batch_size(args: argparse.Namespace) -> int:
    """Batch de avaliacao efetivo: `--per-device-eval-batch-size` ou o do treino."""
    if args.per_device_eval_batch_size is not None:
        return args.per_device_eval_batch_size
    return args.per_device_train_batch_size


def _resolved_eval_steps(args: argparse.Namespace) -> int:
    """Intervalo de avaliacao em steps (`--eval-steps` ou o mesmo que `--save-steps`)."""
    ev = getattr(args, "eval_steps", None)
    if ev is not None:
        return ev
    return args.save_steps


def _should_collect_execution_log(training_args: Optional[TrainingArguments] = None) -> bool:
    """Retorna True no processo que deve gravar o relatorio (rank 0 / processo unico)."""
    if training_args is not None:
        ws = getattr(training_args, "world_size", 1) or 1
        if ws > 1:
            return int(getattr(training_args, "process_index", 0)) == 0
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank not in (None, ""):
        return int(local_rank) == 0
    rank = os.environ.get("RANK")
    if rank not in (None, ""):
        return int(rank) == 0
    return True


def reserve_execution_log_path(log_dir: Path) -> Path:
    """Define um caminho JSON unico: data/hora + microsegundos e sufixo incremental se preciso."""
    log_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    stem = f"run_execution_{now.strftime('%Y%m%d_%H%M%S')}_{now.microsecond:06d}"
    candidate = log_dir / f"{stem}.json"
    if not candidate.exists():
        return candidate
    for n in range(1, 10_000):
        alt = log_dir / f"{stem}_{n}.json"
        if not alt.exists():
            return alt
    raise RuntimeError("Nao foi possivel reservar nome de arquivo de log sem colisao.")


def list_checkpoint_dirs_on_disk(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    """Lista pastas `checkpoint-{step}` sob o diretorio de saida do Trainer."""
    found: List[Dict[str, Any]] = []
    if not checkpoint_dir.is_dir():
        return found
    for path in checkpoint_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        suffix = path.name[len("checkpoint-") :]
        try:
            step = int(suffix)
        except ValueError:
            continue
        found.append({"step": step, "path": str(path.resolve())})
    found.sort(key=lambda item: item["step"])
    return found


def describe_final_export(final_dir: Path) -> Dict[str, Any]:
    """Resume artefatos esperados apos save_model / tokenizer / metadata."""
    adapter_cfg = final_dir / "adapter_config.json"
    weights_sf = final_dir / "adapter_model.safetensors"
    weights_pt = final_dir / "adapter_model.bin"
    tokenizer_ok = (
        (final_dir / "tokenizer.json").exists()
        or (final_dir / "tokenizer_config.json").exists()
    )
    return {
        "final_dir": str(final_dir.resolve()),
        "adapter_config_present": adapter_cfg.exists(),
        "adapter_weights_present": weights_sf.exists() or weights_pt.exists(),
        "tokenizer_present": tokenizer_ok,
        "training_metadata_present": (final_dir / "training_metadata.json").exists(),
    }


class FineTuneRunLogger(TrainerCallback):
    """Acumula eventos de treino (inicio, save, evaluate) para o relatorio JSON final."""

    def __init__(
        self,
        checkpoint_dir: Path,
        final_dir: Path,
        log_json_path: Path,
        dataset_dir: str,
        n_train: int,
        n_val: int,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.final_dir = final_dir
        self.log_json_path = log_json_path
        self.dataset_dir = dataset_dir
        self.n_train = n_train
        self.n_val = n_val
        self.train_begin: Optional[Dict[str, Any]] = None
        self.save_events: List[Dict[str, Any]] = []
        self.eval_events: List[Dict[str, Any]] = []

    def _skip(self, args: TrainingArguments) -> bool:
        return not _should_collect_execution_log(args)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self._skip(args):
            return control
        self.train_begin = {
            "max_steps": state.max_steps,
            "num_train_epochs": args.num_train_epochs,
            "save_steps": args.save_steps,
            "logging_steps": args.logging_steps,
            "eval_steps": getattr(args, "eval_steps", None),
            "output_dir": args.output_dir,
            "final_dir": str(self.final_dir.resolve()),
            "log_json_path": str(self.log_json_path.resolve()),
            "dataset_dir": self.dataset_dir,
            "train_samples": self.n_train,
            "validation_samples": self.n_val,
        }
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self._skip(args):
            return control
        step = state.global_step
        ckpt_path = self.checkpoint_dir / f"checkpoint-{step}"
        self.save_events.append(
            {
                "global_step": step,
                "path": str(ckpt_path.resolve()),
            }
        )
        return control

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if self._skip(args):
            return control
        payload: Dict[str, Any] = {"global_step": state.global_step}
        if metrics:
            payload["metrics"] = dict(metrics)
        self.eval_events.append(payload)
        return control


def write_execution_summary(
    *,
    log_json_path: Path,
    started_at_iso: str,
    finished_at_iso: str,
    cli_args: argparse.Namespace,
    checkpoint_dir: Path,
    final_dir: Path,
    run_logger: Optional[FineTuneRunLogger],
    train_output: Any,
) -> None:
    """Grava o JSON de auditoria da execucao (substituicao atomica via arquivo temporario)."""
    disk_checkpoints = list_checkpoint_dirs_on_disk(checkpoint_dir)
    notes: List[str] = []
    if not disk_checkpoints:
        notes.append(
            "Nenhuma pasta checkpoint-* encontrada sob o diretorio de checkpoints do Trainer; "
            "possivel causa: treino terminou antes do primeiro save_steps ou apenas export final."
        )

    train_metrics: Dict[str, Any] = {}
    if train_output is not None:
        train_metrics = {
            "global_step": getattr(train_output, "global_step", None),
            "training_loss": getattr(train_output, "training_loss", None),
            "metrics": getattr(train_output, "metrics", None),
        }

    payload: Dict[str, Any] = {
        "started_at": started_at_iso,
        "finished_at": finished_at_iso,
        "log_file": str(log_json_path.resolve()),
        "cli_args": vars(cli_args),
        "dataset": {
            "dataset_dir": cli_args.dataset_dir,
            "train_samples": run_logger.n_train if run_logger else None,
            "validation_samples": run_logger.n_val if run_logger else None,
        },
        "training": {
            **((run_logger.train_begin if run_logger else None) or {}),
            "train_output": train_metrics,
        },
        "checkpoint_saves_observed": list(run_logger.save_events) if run_logger else [],
        "eval_events": list(run_logger.eval_events) if run_logger else [],
        "checkpoints_on_disk": disk_checkpoints,
        "final_export": describe_final_export(final_dir),
        "notes": notes,
    }

    log_json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = log_json_path.with_suffix(log_json_path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(log_json_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando do script.

    Argumentos:
        Nenhum diretamente. Via `argparse`, a CLI define: diretório do dataset
        processado (`--dataset-dir`), id do modelo base (`--base-model-name`),
        pastas de checkpoint e de export final, `--max-seq-length`, batch por
        dispositivo, passos de acumulação de gradiente, learning rate, épocas,
        warmup, intervalos de log e de salvamento/avaliação, seed, limites
        opcionais de amostras para dry run e `--append-safety-disclaimer`.
        Hiperparâmetros LoRA (rank, alpha, módulos-alvo, etc.) não são expostos
        aqui; estão fixos em `get_peft_model` e espelhados em
        `save_training_metadata`.

    Funcionalidade:
        Monta o `ArgumentParser` e retorna o namespace parseado.

    Saída:
        `argparse.Namespace` com um atributo por flag definida acima.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tuning LoRA do Qwen2.5-3B-Instruct com Unsloth.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=(
            "Diretorio salvo pelo script 01 com DatasetDict processado. "
            "Padrao: %(default)s."
        ),
    )
    parser.add_argument(
        "--base-model-name",
        default=DEFAULT_BASE_MODEL,
        help="Nome do modelo base no Hugging Face. Padrao: %(default)s.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Diretorio para checkpoints intermediarios. Padrao: %(default)s.",
    )
    parser.add_argument(
        "--final-dir",
        default=DEFAULT_FINAL_DIR,
        help=(
            "Diretorio para salvar o adapter final e tokenizer. "
            "Padrao: %(default)s."
        ),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help=(
            "Comprimento maximo de sequencia (conservador para T4). "
            "Padrao: %(default)s."
        ),
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
        help="Batch size por dispositivo. Padrao: %(default)s.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=None,
        help=(
            "Batch por dispositivo so na avaliacao (validacao). "
            "Padrao: igual a --per-device-train-batch-size. "
            "Use 1 para reduzir VRAM durante o evaluate."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Acumulacao de gradientes. Padrao: %(default)s.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate do fine-tuning LoRA. Padrao: %(default)s.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Numero de epocas. Padrao: %(default)s.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio. Padrao: %(default)s.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Intervalo de log. Padrao: %(default)s.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help=(
            "Intervalo em steps para salvar checkpoint (`checkpoint-*`). "
            "Padrao: %(default)s. Avaliacao usa `--eval-steps` se definido; senao o mesmo valor."
        ),
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help=(
            "Intervalo em steps para rodar validacao (metricas eval_*). "
            "Padrao: igual a --save-steps. "
            "Use junto com --save-steps menor para mais checkpoints sem validar com a mesma frequencia."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Caminho da pasta checkpoint-* para continuar o treino (ex. apos queda no Colab). "
            "Padrao: %(default)s (treino do zero)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente global. Padrao: %(default)s.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help=(
            "Limita amostras de treino para dry run. Padrao: %(default)s "
            "(sem limite)."
        ),
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help=(
            "Limita amostras de validacao para dry run. Padrao: %(default)s "
            "(sem limite)."
        ),
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help=(
            "Dropout aplicado ao caminho LoRA para reduzir sobreajuste. "
            "Padrao: %(default)s (recomendado: 0.0 para habilitar o kernel "
            "otimizado do Unsloth nos layers QKV/O/MLP; usar 0.05 desativa "
            "esse patch e causa queda de desempenho no treino)."
        ),
    )
    parser.add_argument(
        "--append-safety-disclaimer",
        action="store_true",
        help=(
            "Anexa um aviso de seguranca ao final de cada resposta de treino. "
            "Padrao: %(default)s."
        ),
    )
    return parser.parse_args()


def ensure_required_splits(dataset: DatasetDict) -> None:
    """Garante que o dataset contém os splits mínimos para o `SFTTrainer`.

    Argumentos:
        dataset: `DatasetDict` com chaves por split. O script 01 costuma salvar
            também `test`; essa chave extra não interfere nesta validação.

    Funcionalidade:
        Exige a presença de `train` e `validation`. Ausência de algum dos dois
        gera `ValueError` com a lista ordenada de splits faltando.

    Saída:
        `None` em caso de sucesso; caso contrário, lança exceção.
    """
    required = {"train", "validation"}
    missing = required.difference(dataset.keys())
    if missing:
        raise ValueError(
            f"Dataset processado sem splits obrigatorios: {sorted(missing)}")


def apply_limits(dataset: DatasetDict, args: argparse.Namespace) -> DatasetDict:
    """Reduz opcionalmente o tamanho dos splits para testes rápidos (dry run).

    Argumentos:
        dataset: conjunto com pelo menos `train` e `validation`.
        args: namespace com `max_train_samples` e `max_validation_samples`
            opcionais. Apenas valores truthy disparam recorte (`0` ou `None`
            não alteram o split correspondente).

    Funcionalidade:
        Quando o limite está definido e é verdadeiro em Python, recorta com
        `.select(range(...))` até o mínimo entre o limite e o comprimento do
        split.

    Saída:
        O mesmo `DatasetDict`, mutado em `train`/`validation` quando aplicável;
        retornado para encadeamento no fluxo principal.
    """
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(
            range(min(args.max_train_samples, len(dataset["train"]))))
    if args.max_validation_samples:
        dataset["validation"] = dataset["validation"].select(
            range(min(args.max_validation_samples, len(dataset["validation"])))
        )
    return dataset


def format_example(question: str, answer: str, tokenizer, append_safety_disclaimer: bool) -> Dict[str, str]:
    """Monta um único exemplo de SFT no formato chat do modelo.

    Argumentos:
        question: texto da pergunta do usuário.
        answer: texto da resposta do assistente (dataset bruto).
        tokenizer: tokenizer HF (ou compatível) com `apply_chat_template`;
            instância vinda de `FastLanguageModel.from_pretrained`.
        append_safety_disclaimer: se verdadeiro, anexa `SAFETY_DISCLAIMER` ao
            final da resposta quando ainda não estiver presente (comparação
            case-insensitive).

    Funcionalidade:
        Constrói mensagens `system` / `user` / `assistant` usando
        `SYSTEM_MESSAGE`; serializa com `apply_chat_template(..., tokenize=False,
        add_generation_prompt=False)`. Em qualquer exceção nesse caminho, usa
        fallback textual com marcadores `[SYSTEM]`, `[USER]`, `[ASSISTANT]`.

    Saída:
        Dicionário com a chave `"text"` contendo a string do exemplo alinhada
        ao `dataset_text_field` / treino SFT.
    """
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
    """Converte colunas `question`/`answer` em datasets só com coluna `text`.

    Argumentos:
        dataset: `DatasetDict` já validado quanto a `train` e `validation`,
            com colunas `question` e `answer`. Outros splits (ex.: `test`
            salvo pelo script 01) não são mapeados aqui e permanecem intactos
            no objeto original, mas não entram no retorno.
        tokenizer: repassado a `format_example`.
        args: fonte de `append_safety_disclaimer`.

    Funcionalidade:
        Aplica `.map` em batch em `train` e `validation`, gerando `text` via
        `format_example` e removendo todas as colunas originais desses splits.

    Saída:
        Novo `DatasetDict` contendo apenas as chaves `train` e `validation`,
        cada uma com a coluna `text` esperada pelo `SFTTrainer`.
    """
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
    """Persiste hiperparâmetros de treino e bloco fixo LoRA junto ao export final.

    Argumentos:
        args: valores vindos da CLI necessários para auditoria (modelo base,
            seq length, batch, otimização, seed, disclaimer).
        final_dir: diretório de saída do adapter; o arquivo criado é
            `training_metadata.json` (sobrescrito se já existir).

    Funcionalidade:
        Monta um dicionário com campos alinhados ao uso em `main` e ao LoRA
        aplicado em `get_peft_model` (inclui `r`, `alpha`, `dropout`,
        `target_modules`), mais `system_message` e campos de disclaimer;
        grava JSON UTF-8 indentado.

    Saída:
        `None`. Efeito colateral: escrita em `final_dir / "training_metadata.json"`.
    """
    metadata = {
        "base_model_name": args.base_model_name,
        "max_seq_length": args.max_seq_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": _resolved_per_device_eval_batch_size(args),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "save_steps": args.save_steps,
        "eval_steps": _resolved_eval_steps(args),
        "lora": {
            "r": 32,
            "alpha": 64,
            "dropout": args.lora_dropout,
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
    """Orquestra carga do dataset, modelo 4-bit, LoRA, treino SFT e exportação.

    Argumentos:
        Nenhum; usa `parse_args()` para obter a configuração.

    Funcionalidade:
        Carrega o diretório `--dataset-dir` com `load_from_disk` e exige um
        `DatasetDict` (caso contrário `ValueError` orientando o uso do script 01).
        Valida presença de `train` e `validation`, aplica limites opcionais de
        amostras, cria pastas de checkpoint e de saída final, carrega o modelo
        base em 4-bit e o tokenizer via Unsloth, aplica `get_peft_model`,
        converte apenas treino/validação para coluna `text`, instancia
        `SFTTrainer` com `processing_class=tokenizer` e `SFTConfig` (inclui
        `dataset_text_field`, `max_length`, checkpoints a cada `save_steps` e
        avaliacao a cada `eval_steps` resolvido),
        executa `train()`, persiste o modelo com `trainer.save_model(final_dir)`,
        salva o tokenizer em `final_dir`, grava `training_metadata.json` e
        imprime os caminhos de checkpoints, do export e do JSON de execucao
        (sob `execution_logs/`). O split `test`, se existir no disco, não é
        usado neste treino.

    Saída:
        `None`. Efeitos: treino, checkpoints, adapter/tokenizer/metadados em
        disco, opcionalmente relatorio JSON por execucao e mensagens no stdout.
    """
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Carrega o diretório `--dataset-dir` com `load_from_disk` e exige um
    # `DatasetDict` (caso contrário `ValueError` orientando o uso do script 01).
    loaded = load_from_disk(args.dataset_dir)
    if not isinstance(loaded, DatasetDict):
        raise ValueError(
            "Esperado DatasetDict salvo pelo script 01 (diretorio com splits como "
            f"train/). Recebido: {type(loaded).__name__}."
        )
    dataset = loaded
    ensure_required_splits(dataset)
    dataset = apply_limits(dataset, args)

    checkpoint_dir = Path(args.checkpoint_dir)
    final_dir = Path(args.final_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    execution_logs_dir = checkpoint_dir / EXECUTION_LOGS_SUBDIR
    log_json_path: Optional[Path] = None
    run_started_at_iso = ""
    if _should_collect_execution_log(None):
        run_started_at_iso = datetime.now(timezone.utc).isoformat()
        log_json_path = reserve_execution_log_path(execution_logs_dir)

    # 1) Carrega o modelo base em 4-bit, adequado para o ambiente do Colab.
    # Unsloth: `FastLanguageModel.from_pretrained` baixa (se necessario) o causal LM do
    #   Hugging Face Hub e o tokenizer associado, ja preparados para treino eficiente.
    #   - `model_name`: id do repo (ex. Qwen2.5-Instruct) vindo de `--base-model-name`;
    #
    #   - `max_seq_length`: teto de tokens por sequencia (alinhado ao RoPE/comprimento do trainer);
    # “Teto de tokens por sequência” — é o comprimento máximo que cada exemplo de texto
    #   pode ter em tokens: entradas mais longas tendem a ser truncadas e as mais curtas
    #   preenchidas (padding), até esse limite. No seu script, o mesmo valor vem de
    #   --max-seq-length (padrão 1024).
    # “Alinhado ao RoPE” — modelos como o Qwen usam RoPE (Rotary Positional Embedding)
    #   para codificar posição das palavras. O modelo foi treinado com um certo contexto máximo;
    #   definir um max_seq_length coerente com o que o modelo e a biblioteca esperam evita
    #   usar contextos absurdamente maiores que o suportado (e mantém o uso de posições
    #   dentro do que o modelo “conhece”). Em pipelines tipo Unsloth/TRL isso costuma ser
    #   amarrado ao que a lib considera comprimento máximo de atenção.
    # “… / comprimento do trainer” — no mesmo arquivo, esse valor é reutilizado no SFTConfig
    #   como max_length, ou seja, o treinador SFT usa o mesmo teto para tokenizar/
    #   truncar os exemplos no dataset de treino. Por isso o comentário diz que está
    #   alinhado ao trainer: uma única configuração para carga do modelo e para o loop de treino.
    #
    # Em resumo: é o limite único de tamanho de sequência (em tokens) compartilhado
    #   entre o carregamento do modelo e o treino supervisionado.
    #
    #   - `load_in_4bit=True`: quantizacao 4-bit (bitsandbytes) para caber em GPU limitada (ex. T4 16 GB);
    #   - `full_finetuning=False`: nao ajustar todos os pesos do backbone agora; o proximo passo e LoRA/PEFT.
    # full_finetuning=True (hipotético) sinalizaria intenção de atualizar todos os parâmetros do
    #   modelo base no treino — o “backbone” inteiro (camadas do transformer pré-treinado).
    # full_finetuning=False significa que neste carregamento não se está preparando o modelo
    #   para esse modo. Em vez disso, a ideia é manter a maior parte dos pesos fixos
    #   (ou só treinar uma fração pequena depois).
    # No fluxo do seu script, o comentário liga isso ao passo seguinte:
    #   get_peft_model, que coloca adaptadores LoRA em certas camadas.
    #   Só esses adaptadores (e não todas as matrizes gigantes do modelo) são treinados
    #   de forma principal — isto é PEFT (Parameter-Efficient Fine-Tuning).
    # Resumo: "Não vamos fazer fine-tuning completo do modelo agora; vamos treinar em
    #   cima dele com LoRA/PEFT no próximo passo."
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # 2) `FastLanguageModel.get_peft_model` injeta adaptadores LoRA via PEFT: o backbone
    # permanece congelado e so matrizes de baixo rank em projecoes escolhidas recebem
    # gradiente durante o treino.
    # - `model`: instancia retornada por `from_pretrained` (4-bit, preparada para PEFT);
    # - `r`: rank LoRA — dimensao das matrizes A/B; mais rank = mais capacidade e mais VRAM;
    # - `target_modules`: submodulos lineares onde o LoRA e aplicado (atencao multi-cabeca
    #   q/k/v/o_proj + MLP gate/up/down no estilo Qwen/Llama);
    # - `lora_alpha`: fator de escala das atualizacoes em relacao a `r` (escala efetiva ~ alpha/r;
    #   aqui 64 com r=32, padrao comum alpha = 2*r);
    # - `lora_dropout`: dropout aplicado ao caminho LoRA; valor 0.0 (padrao) habilita o
    #   kernel otimizado do Unsloth nos layers QKV/O/MLP — valores > 0 desativam esse patch
    #   e causam queda de desempenho no treino (configuravel via `--lora-dropout`);
    # - `bias="none"`: nao treinar bias adicional nos blocos adaptados;
    # - `use_gradient_checkpointing="unsloth"`: economiza VRAM trocando por mais computacao
    #   no backward (integracao Unsloth);
    # - `random_state`: semente para inicializar os adapters de forma reprodutivel (`--seed`).
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
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # 3) Converte o dataset limpo para o formato textual esperado pelo trainer.
    # Converte apenas treino/validação para coluna `text`. O split `test`, se
    # existir no disco, não é usado neste treino.
    text_datasets = prepare_text_datasets(dataset, tokenizer, args)

    run_logger: Optional[FineTuneRunLogger] = None
    callbacks: Optional[List[TrainerCallback]] = None
    if log_json_path is not None:
        run_logger = FineTuneRunLogger(
            checkpoint_dir=checkpoint_dir,
            final_dir=final_dir,
            log_json_path=log_json_path,
            dataset_dir=args.dataset_dir,
            n_train=len(text_datasets["train"]),
            n_val=len(text_datasets["validation"]),
        )
        callbacks = [run_logger]

    # TRL 0.18+ (ex.: 0.24 com Unsloth): tokenizer -> processing_class;
    # Instancia `SFTTrainer` com `processing_class=tokenizer` e `SFTConfig` (inclui
    # `dataset_text_field`, `max_length`, checkpoint `save_steps`, avaliacao `eval_steps`).
    #  packing / comprimento ficam em SFTConfig (`max_length`).
    # SFTConfig herda TrainingArguments com muitos campos;
    # TRL/dict: usar dict + unpack reduz falsos positivos no basedpyright  ao desempacotar no SFTConfig.

    # TRL 0.18+ (ex.: 0.24 com Unsloth): tokenizer -> processing_class;
    # dataset_text_field / packing / comprimento ficam em SFTConfig (`max_length`).
    # SFTConfig herda TrainingArguments com muitos campos; usar dict + unpack reduz falsos positivos no basedpyright.
    sft_training_kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "dataset_text_field": "text",
        "max_length": args.max_seq_length,
        "packing": False,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "eval_strategy": "steps",
        "eval_steps": _resolved_eval_steps(args),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": _resolved_per_device_eval_batch_size(args),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "optim": "adamw_8bit",
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "report_to": "none",
        "seed": args.seed,
    }
    trainer_kw: dict[str, Any] = {
        "model": model,
        "processing_class": tokenizer,
        "train_dataset": text_datasets["train"],
        "eval_dataset": text_datasets["validation"],
        "args": SFTConfig(**sft_training_kwargs),
    }
    if callbacks:
        trainer_kw["callbacks"] = callbacks
    trainer = SFTTrainer(**trainer_kw)

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt is not None and not Path(resume_ckpt).is_dir():
        raise ValueError(
            "--resume-from-checkpoint deve apontar para um diretorio existente "
            f"(checkpoint-*). Recebido: {resume_ckpt!r}"
        )

    train_output = trainer.train(
        resume_from_checkpoint=resume_ckpt if resume_ckpt else None,
    )

    # 4) Salva o adapter final e o tokenizer para inferencia e avaliacao posterior.
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    save_training_metadata(args, final_dir)

    finished_at_iso = datetime.now(timezone.utc).isoformat()
    if (
        log_json_path is not None
        and run_started_at_iso
        and trainer.is_world_process_zero()
    ):
        write_execution_summary(
            log_json_path=log_json_path,
            started_at_iso=run_started_at_iso,
            finished_at_iso=finished_at_iso,
            cli_args=args,
            checkpoint_dir=checkpoint_dir,
            final_dir=final_dir,
            run_logger=run_logger,
            train_output=train_output,
        )

    checkpoint_dirs_count = len(list_checkpoint_dirs_on_disk(checkpoint_dir))
    print(f"Checkpoints salvos em: {checkpoint_dir}")
    print(f"Pastas checkpoint-* no disco: {checkpoint_dirs_count}")
    if log_json_path is not None and trainer.is_world_process_zero():
        print(f"Relatorio de execucao (JSON): {log_json_path.resolve()}")
    print(f"Modelo final (adapter) salvo em: {final_dir}")


if __name__ == "__main__":
    main()
