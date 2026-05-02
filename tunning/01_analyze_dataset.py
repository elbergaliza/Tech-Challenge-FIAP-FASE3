#!/usr/bin/env python
"""Analisa, limpa e particiona o dataset MedPT para fine-tuning.

Este script implementa as decisoes consolidadas do PASSO 1:
- valida o schema esperado (`question` + `answer`);
- remove ruido promocional/comercial e exemplos inviaveis;
- faz split deterministico por pergunta normalizada, evitando vazamento;
- salva um `DatasetDict` pronto para treino e um manifesto auxiliar
  para auditoria e analise de fatias na etapa de avaliacao.
"""

# PEP 563: avalia anotacoes de tipo de forma tardia (strings).
# Neste modulo: permite usar tipos em assinaturas sem custo circular em import time.
from __future__ import annotations

# Biblioteca padrao para interface de linha de comando (`ArgumentParser`).
# Neste modulo: `parse_args()` expoe paths, proporcoes de split, seed e filtros de tokens.
import argparse

# Biblioteca padrao de funcoes hash criptograficas.
# Neste modulo: MD5 de `f"{seed}:{group_key}"` para bucketing reproduzivel em `assign_split`.
import hashlib

# Biblioteca padrao de serializacao JSON.
# Neste modulo: escrever o manifesto e o relatorio; imprimir o resumo no terminal.
import json

# Biblioteca padrao de expressoes regulares.
# Neste modulo: padroes de HTML/URL/whitespace e deteccao de conteudo promocional.
import re

# Biblioteca padrao do Unicode (normalizacao e propriedades de caracteres).
# Neste modulo: remover acentos na normalizacao da pergunta para agrupamento e split.
import unicodedata

# Estrutura da biblioteca padrao `collections` que conta ocorrencias.
# Neste modulo: agregar motivos de remocao, contagens por split e distribuicoes de metadados.
from collections import Counter

# ABC tipica para objetos mapeaveis (duck-typing de `dict`-like).
# Neste modulo: validar cada linha iterada em `load_medpt_rows` antes de copiar para `dict`.
from collections.abc import Mapping

# Modulo padrao para caminhos de arquivo orientados a objeto.
# Neste modulo: criar diretorios de saida e referenciar manifesto/relatorio em disco.
from pathlib import Path

# Modulo padrao de anotacoes de tipo e genericos.
# Neste modulo: documentar listas, dicionarios, tuplas e iteraveis nas funcoes.
from typing import Any, Dict, Iterable, List, Tuple

# Hugging Face `datasets`: abstracao de datasets tabulares e integracao com Arrow/Parquet.
# Neste modulo: carregar o parquet bruto, montar splits com `Dataset`/`DatasetDict` e
# `save_to_disk` na pasta processada.
from datasets import Dataset, DatasetDict, load_dataset


DEFAULT_DATASET_PATH = "data/train-00000-of-00001.parquet"
DEFAULT_OUTPUT_DIR = "data/processed/medpt_qa"
DEFAULT_MANIFEST_PATH = "data/processed/medpt_qa_manifest.jsonl"
DEFAULT_REPORT_PATH = "data/processed/medpt_qa_report.json"

PROMO_PATTERNS = [
    r"doctoralia",
    r"instagram",
    r"meu instagram",
    r"@[\w\._]+",
    r"whatsapp",
    r"agendar consulta",
    r"agende sua consulta",
    r"reserve uma consulta",
    r"cliqu[eé] no bot[aã]o",
    r"voc[eê] pode reservar",
    r"te convidamos para uma consulta",
    r"site doctoralia",
    r"https?://",
    r"www\.",
    r"consulta\s*-\s*r\$\s*\d+",
    r"\br\$\s*\d+",
]

HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando do script.

    Argumentos:
        Nenhum diretamente. Os valores são lidos da linha de comando via
        `argparse` e incluem caminho do dataset, diretórios de saída, seed,
        proporções de split e limites de tokens da resposta.

    Funcionalidade:
        Define a interface CLI do script e retorna a configuração já parseada
        para ser usada no fluxo principal.

    Saída:
        `argparse.Namespace` com todos os parâmetros necessários para análise,
        limpeza, particionamento e persistência do dataset processado.
    """
    parser = argparse.ArgumentParser(
        description="Analisa, limpa e particiona o dataset MedPT.",
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help=(
            "Caminho local ou URL do parquet do MedPT. "
            "Padrao: %(default)s."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Diretorio de saida para o DatasetDict processado. "
            "Padrao: %(default)s."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        default=DEFAULT_MANIFEST_PATH,
        help=(
            "Arquivo JSONL auxiliar com metadados de auditoria. "
            "Padrao: %(default)s."
        ),
    )
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help=(
            "Arquivo JSON com estatisticas e resumo da limpeza. "
            "Padrao: %(default)s."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Semente usada no split deterministico por pergunta. "
            "Padrao: %(default)s."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proporcao de treino. Padrao: %(default)s.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Proporcao de validacao. Padrao: %(default)s.",
    )
    parser.add_argument(
        "--min-answer-tokens",
        type=int,
        default=10,
        help=(
            "Numero minimo de tokens aceitos na resposta. Padrao: %(default)s."
        ),
    )
    parser.add_argument(
        "--max-answer-tokens",
        type=int,
        default=1000,
        help=(
            "Numero maximo de tokens aceitos na resposta. Padrao: %(default)s."
        ),
    )
    return parser.parse_args()


def remove_accents(text: str) -> str:
    """Remove acentos e marcas diacríticas de um texto.

    Argumentos:
        text: Texto de entrada que pode conter caracteres acentuados.

    Funcionalidade:
        Normaliza o texto para o formato Unicode compatível e remove marcas de
        acentuação para facilitar comparações textuais estáveis.

    Saída:
        `str` com o mesmo conteúdo textual, mas sem acentos.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def clean_text(text: str) -> str:
    """Limpa ruído estrutural básico de um texto.

    Argumentos:
        text: Texto bruto, possivelmente com HTML, URLs, espaços extras ou
            caracteres de espaço não separável.

    Funcionalidade:
        Remove tags HTML, URLs, espaços redundantes e normaliza a formatação
        mínima antes das demais etapas de filtragem e análise.

    Saída:
        `str` limpo e com espaços normalizados.
    """
    text = HTML_RE.sub(" ", text or "")
    text = URL_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_question(text: str) -> str:
    """Normaliza uma pergunta para uso como chave de agrupamento.

    Argumentos:
        text: Pergunta original do dataset.

    Funcionalidade:
        Aplica limpeza textual, converte para minúsculas, remove acentos e
        normaliza espaços, produzindo uma representação estável para evitar
        vazamento no split por perguntas semanticamente idênticas.

    Saída:
        `str` normalizada, adequada para agrupamento e particionamento.
    """
    text = clean_text(text).lower()
    text = remove_accents(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def token_count(text: str) -> int:
    """Conta tokens aproximados com base em separação por espaços.

    Argumentos:
        text: Texto a ser medido.

    Funcionalidade:
        Faz uma contagem simples de tokens para aplicar filtros mínimos e
        máximos de tamanho da resposta.

    Saída:
        `int` com a quantidade de tokens identificados.
    """
    return len((text or "").split())


def contains_promo_content(text: str) -> bool:
    """Verifica se o texto contém indícios de conteúdo promocional.

    Argumentos:
        text: Resposta textual que será inspecionada.

    Funcionalidade:
        Procura padrões associados a propaganda, links, preços, redes sociais
        e convites comerciais que não devem entrar no fine-tuning.

    Saída:
        `bool`: `True` se algum padrão promocional for encontrado; `False`
        caso contrário.
    """
    lowered = (text or "").lower()
    return any(re.search(pattern, lowered) for pattern in PROMO_PATTERNS)


def assign_split(group_key: str, seed: int, train_ratio: float, validation_ratio: float) -> str:
    """Atribui deterministicamente um grupo a treino, validação ou teste.

    Argumentos:
        group_key: Chave estável do grupo, normalmente a pergunta normalizada.
        seed: Semente usada para tornar a partição reproduzível.
        train_ratio: Proporção reservada para treino.
        validation_ratio: Proporção reservada para validação.

    Funcionalidade:
        Usa hash MD5 da combinação de `seed` e `group_key` para mapear cada
        grupo sempre ao mesmo split, impedindo que variações da mesma pergunta
        caiam em conjuntos diferentes.

    Saída:
        `str` com um dos valores: `"train"`, `"validation"` ou `"test"`.
    """
    digest = hashlib.md5(f"{seed}:{group_key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + validation_ratio:
        return "validation"
    return "test"


def validate_ratios(train_ratio: float, validation_ratio: float) -> None:
    """Valida as proporções usadas no particionamento do dataset.

    Argumentos:
        train_ratio: Fração destinada ao split de treino.
        validation_ratio: Fração destinada ao split de validação.

    Funcionalidade:
        Garante que as proporções sejam positivas e que a soma de treino e
        validação deixe espaço para o split de teste.

    Saída:
        Nenhuma. A função retorna `None` e lança `ValueError` em caso de
        configuração inválida.
    """
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError(
            "As proporcoes de treino e validacao devem ser positivas.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("A soma de treino e validacao deve ser menor que 1.")


def load_medpt_rows(dataset_path: str) -> Tuple[List[Dict[str, Any]], str]:
    """Carrega o parquet do MedPT e identifica a coluna de resposta.

    Argumentos:
        dataset_path: Caminho local ou URL do arquivo parquet do dataset.

    Funcionalidade:
        Lê o dataset com `datasets.load_dataset`, valida a presença da coluna
        `question` e aceita `answer` como schema principal, com suporte de
        compatibilidade para `answers`.

    Saída:
        `Tuple[List[Dict[str, Any]], str]` contendo:
        - a lista de linhas do dataset;
        - o nome da coluna de resposta efetivamente utilizada.
    """
    dataset = load_dataset("parquet", data_files={
                           "train": dataset_path}, split="train")
    raw_columns = dataset.column_names
    if raw_columns is None:
        raise ValueError(
            "O dataset retornado nao expoe nomes de colunas (`column_names` ausente)."
        )
    if isinstance(raw_columns, dict):
        raise ValueError(
            "Esperado um Dataset com split unico ao carregar o parquet; "
            "`column_names` nao deve ser um mapa por split."
        )
    columns = set(raw_columns)

    if "answer" in columns:
        answer_column = "answer"
    elif "answers" in columns:
        answer_column = "answers"
    else:
        raise ValueError(
            "Schema invalido: o dataset precisa conter a coluna `answer` "
            "ou, em modo de compatibilidade, `answers`."
        )

    if "question" not in columns:
        raise ValueError(
            "Schema invalido: a coluna `question` nao foi encontrada.")

    rows: List[Dict[str, Any]] = []
    for raw in dataset:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Tipo de linha inesperado ao iterar o dataset: {type(raw)!r}."
            )
        rows.append({str(k): v for k, v in raw.items()})
    return rows, answer_column


def build_processed_outputs(
    rows: Iterable[Dict[str, Any]],
    answer_column: str,
    args: argparse.Namespace,
) -> Tuple[DatasetDict, List[dict], Dict[str, object]]:
    """Constrói os artefatos processados usados nas próximas etapas.

    Argumentos:
        rows: Linhas brutas carregadas do dataset.
        answer_column: Nome da coluna de resposta validada no schema.
        args: Namespace com regras de limpeza, seed e proporções de split.

    Funcionalidade:
        Limpa textos, remove exemplos inválidos, aplica filtros de tamanho e
        conteúdo promocional, executa o split por pergunta normalizada e gera:
        o `DatasetDict` final, o manifesto de auditoria e o relatório
        estatístico da preparação.

    Saída:
        `Tuple[DatasetDict, List[dict], Dict[str, object]]` contendo:
        - dataset processado com splits `train`, `validation` e `test`;
        - linhas do manifesto em formato serializável;
        - relatório consolidado com contagens e métricas da limpeza.
    """
    split_data = {
        "train": {"question": [], "answer": []},
        "validation": {"question": [], "answer": []},
        "test": {"question": [], "answer": []},
    }
    manifest_rows: List[dict] = []

    removed_reasons = Counter()
    split_counts = Counter()
    split_unique_questions = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    question_type_counter = Counter()
    condition_counter = Counter()

    total_rows = 0
    kept_rows = 0

    for row in rows:
        total_rows += 1

        question = clean_text(str(row.get("question", "") or ""))
        answer = clean_text(str(row.get(answer_column, "") or ""))
        question_type = clean_text(str(row.get("question_type", "") or ""))
        condition = clean_text(str(row.get("condition", "") or ""))

        if not question:
            removed_reasons["empty_question"] += 1
            continue
        if not answer:
            removed_reasons["empty_answer"] += 1
            continue

        answer_tokens = token_count(answer)
        if answer_tokens < args.min_answer_tokens:
            removed_reasons["answer_too_short"] += 1
            continue
        if answer_tokens > args.max_answer_tokens:
            removed_reasons["answer_too_long"] += 1
            continue
        if contains_promo_content(answer):
            removed_reasons["promotional_answer"] += 1
            continue

        question_key = normalize_question(question)
        split = assign_split(
            group_key=question_key,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
        )

        split_data[split]["question"].append(question)
        split_data[split]["answer"].append(answer)
        split_counts[split] += 1
        split_unique_questions[split].add(question_key)
        kept_rows += 1

        if question_type:
            question_type_counter[question_type] += 1
        if condition:
            condition_counter[condition] += 1

        manifest_rows.append(
            {
                "question": question,
                "answer": answer,
                "question_normalized": question_key,
                "split": split,
                "question_type": question_type,
                "condition": condition,
            }
        )

    dataset_dict = DatasetDict(
        {
            split_name: Dataset.from_dict(split_rows)
            for split_name, split_rows in split_data.items()
        }
    )

    report = {
        "input_dataset_path": args.dataset_path,
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "removed_rows": total_rows - kept_rows,
        "removed_reasons": dict(removed_reasons),
        "split_counts": dict(split_counts),
        "unique_questions_per_split": {
            key: len(value) for key, value in split_unique_questions.items()
        },
        "config": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": 1 - args.train_ratio - args.validation_ratio,
            "min_answer_tokens": args.min_answer_tokens,
            "max_answer_tokens": args.max_answer_tokens,
        },
        "top_question_types": question_type_counter.most_common(10),
        "top_conditions": condition_counter.most_common(10),
    }
    return dataset_dict, manifest_rows, report


def save_manifest(manifest_rows: Iterable[dict], manifest_path: Path) -> None:
    """Salva o manifesto auxiliar em formato JSONL.

    Argumentos:
        manifest_rows: Coleção de linhas já processadas para auditoria.
        manifest_path: Caminho do arquivo `.jsonl` de saída.

    Funcionalidade:
        Garante a existência do diretório pai e persiste uma linha JSON por
        amostra processada, preservando metadados úteis para avaliação.

    Saída:
        Nenhuma. A função grava o arquivo em disco e retorna `None`.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_report(report: dict, report_path: Path) -> None:
    """Salva o relatório agregado da preparação do dataset.

    Argumentos:
        report: Dicionário com estatísticas, configuração e contagens da
            limpeza e do particionamento.
        report_path: Caminho do arquivo `.json` de saída.

    Funcionalidade:
        Cria o diretório de destino, quando necessário, e grava o relatório
        estruturado em JSON legível.

    Saída:
        Nenhuma. A função persiste o relatório em disco e retorna `None`.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def main() -> None:
    """Orquestra o fluxo completo de preparação do dataset.

    Argumentos:
        Nenhum diretamente. A função usa os parâmetros retornados por
        `parse_args()`.

    Funcionalidade:
        Valida a configuração, carrega o dataset bruto, executa a limpeza e o
        split, salva os artefatos processados em disco e imprime um resumo da
        execução.

    Saída:
        Nenhuma. A função retorna `None` após gerar os arquivos de saída e
        exibir o relatório no terminal.
    """
    args = parse_args()
    validate_ratios(args.train_ratio, args.validation_ratio)

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)
    report_path = Path(args.report_path)

    # 1) Carrega o parquet e valida o schema minimo exigido.
    rows, answer_column = load_medpt_rows(args.dataset_path)

    # 2) Limpa e filtra os exemplos, depois aplica split por pergunta normalizada.
    dataset_dict, manifest_rows, report = build_processed_outputs(
        rows, answer_column, args)

    # 3) Salva o DatasetDict pronto para treino e artefatos auxiliares de auditoria.
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    save_manifest(manifest_rows, manifest_path)
    save_report(report, report_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nDataset processado salvo em: {output_dir}")
    print(f"Manifesto salvo em: {manifest_path}")
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()
