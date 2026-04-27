# carregador_do_modelo.py

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


def carregar_modelo():
    print("[carregador_do_modelo] Carregando modelo: google/flan-t5-small")

    pipeline_de_texto = pipeline(
        task="text-generation",        # task correta para a versão atual do transformers
        model="gpt2",                  # modelo leve que suporta text-generation
        max_new_tokens=256,
        truncation=True,
        pad_token_id=50256,            # evita warning de padding no gpt2
    )

    modelo_para_langchain = HuggingFacePipeline(pipeline=pipeline_de_texto)

    print("[carregador_do_modelo] Modelo carregado com sucesso!")
    return modelo_para_langchain
