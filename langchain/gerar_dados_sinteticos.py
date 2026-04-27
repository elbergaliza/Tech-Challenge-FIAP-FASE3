# gerar_dados_sinteticos.py
# ─────────────────────────────────────────────────────────────
# Esse arquivo cria um CSV de exemplo com perguntas e respostas médicas.
# Serve para testar o sistema enquanto o dataset real não está disponível.
#
# Para rodar esse arquivo:
#   python gerar_dados_sinteticos.py
#
# Resultado: cria o arquivo dados/perguntas_e_respostas.csv
# ─────────────────────────────────────────────────────────────

import os       # biblioteca para criar pastas
import pandas   # biblioteca para criar e salvar tabelas (CSV)

from configuracoes import CAMINHO_DO_DATASET


# Lista de perguntas e respostas médicas de exemplo
# Cada item é um dicionário com duas chaves: "question" e "answer"
# (mantemos em inglês porque é como o dataset real vai vir do time de fine-tuning)
perguntas_e_respostas = [
    {
        "question": "What are the warning signs in a third trimester pregnant woman?",
        "answer": (
            "Warning signs include: blood pressure above 140/90 mmHg, sudden swelling "
            "of face and hands, severe headache, visual disturbances, epigastric pain, "
            "and decreased fetal movements. Any of these signs require immediate evaluation."
        )
    },
    {
        "question": "How to identify signs of domestic violence during a consultation?",
        "answer": (
            "Signs include: injuries at different stages of healing, inconsistent explanations "
            "for injuries, a companion who does not let the patient speak, anxious or apathetic "
            "behavior, history of frequent 'accidents'. Address in a private and safe environment."
        )
    },
    {
        "question": "What tests are requested in low-risk prenatal care?",
        "answer": (
            "Low-risk prenatal tests include: complete blood count, blood type and Rh factor, "
            "fasting blood glucose, urinalysis, urine culture, serology for syphilis, HIV, "
            "hepatitis B and C, toxoplasmosis and rubella, morphological ultrasound between 20-24 weeks."
        )
    },
    {
        "question": "When is colposcopy indicated?",
        "answer": (
            "Colposcopy is indicated after: Pap smear with atypical squamous cells (ASC-US with "
            "positive HPV), low or high-grade intraepithelial lesion (LSIL/HSIL), atypical glandular "
            "cells, or any result suggesting neoplasia."
        )
    },
    {
        "question": "What are the diagnostic criteria for postpartum depression?",
        "answer": (
            "Criteria include: depressed or irritable mood, anhedonia, sleep changes unrelated to "
            "the baby, intense fatigue, feelings of guilt or inadequacy as a mother, difficulty "
            "concentrating, and thoughts of harming oneself or the baby. Symptoms must persist for "
            "more than 2 weeks."
        )
    },
    {
        "question": "What are the most common side effects of combined hormonal contraceptives?",
        "answer": (
            "Common effects: nausea (especially at the start), headache, changes in menstrual flow, "
            "breast tenderness, mood changes, decreased libido. Rare but serious effects: venous "
            "thromboembolism, especially in smokers over 35 years old."
        )
    },
    {
        "question": "What are the protocols for breast cancer screening?",
        "answer": (
            "INCA recommends: biennial mammography for women aged 50-69. For women at high risk "
            "(first-degree family history, BRCA mutation): starting at age 35, annually. "
            "Complementary ultrasound for dense breasts."
        )
    },
    {
        "question": "How to differentiate Braxton Hicks contractions from labor?",
        "answer": (
            "Braxton Hicks: irregular, do not increase in intensity, stop with position change or "
            "hydration, do not cause cervical dilation. Labor: regular contractions with intervals "
            "less than 5 minutes, progressively increase, do not stop with rest, accompanied by "
            "cervical dilation."
        )
    },
]


def criar_dataset():
    """
    Cria a pasta 'dados' se ela não existir,
    transforma a lista acima em uma tabela
    e salva como arquivo CSV.
    """

    # Cria a pasta onde o arquivo vai ser salvo (se já existir, não faz nada)
    pasta = os.path.dirname(CAMINHO_DO_DATASET)
    os.makedirs(pasta, exist_ok=True)

    # Transforma a lista de dicionários em uma tabela (DataFrame)
    tabela = pandas.DataFrame(perguntas_e_respostas)

    # Salva a tabela como arquivo CSV
    tabela.to_csv(CAMINHO_DO_DATASET, index=False, encoding="utf-8")

    print(f"Dataset criado com sucesso!")
    print(f"Arquivo salvo em: {CAMINHO_DO_DATASET}")
    print(f"Total de perguntas: {len(tabela)}")


# Esse bloco só executa quando você roda esse arquivo diretamente
# Exemplo: python gerar_dados_sinteticos.py
if __name__ == "__main__":
    criar_dataset()
