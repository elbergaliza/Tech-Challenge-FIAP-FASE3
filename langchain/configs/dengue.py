from pathlib import Path
from .base import ConfiguracaoProtocolo

_DADOS_DIR = Path(__file__).resolve().parent.parent / "dados" / "data"

CONFIG_DENGUE = ConfiguracaoProtocolo(
    nome="dengue",
    caminho_pdf=str(_DADOS_DIR / "dengue-diagnostico-e-manejo-clinico-adulto-e-crianca.pdf"),
    descricao_fonte="Ministério da Saúde - Protocolo de Manejo Clínico da Dengue (2024)",
    url_fonte="https://www.gov.br/saude/pt-br/centrais-de-conteudo/publicacoes/svsa/dengue/dengue-diagnostico-e-manejo-clinico-adulto-e-crianca",
    padroes_cabecalho=[
        r"Ministério da Saúde.*?Dengue.*?\n",
        r"DENGUE:.*?MANEJO CLÍNICO.*?\n",
    ],
    padroes_ruido=[
        "sumario",
        "formulario",
        "data de nascimento",
        "nome:___",
        "cpf:",
        "telefone:",
        "endereco:",
        "assinatura",
        "referencias bibliograficas",
        "notificacao imediata",
        "check-list",
        "composicao da equipe",
    ],
    paginas_por_secao={
        "sintomas": [13, 14, 22, 24, 28, 56],
        "classificacao": [12, 13, 25, 28],
        "sinais_alarme": [14, 28, 32],
        "exames": [28, 31, 33, 36, 56],
        "tratamento": [28, 29, 30, 31, 33, 36, 77],
    },
    niveis_gravidade=["grupo_a", "grupo_b", "grupo_c", "grupo_d"],
    # Protocolo Dengue usa "exames complementares" e organiza por grupo de gravidade
    query_exames="exames complementares {gravidade}",
)
