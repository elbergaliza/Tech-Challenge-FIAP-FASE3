from pathlib import Path
from .base import ConfiguracaoProtocolo

_DADOS_DIR = Path(__file__).resolve().parent.parent / "dados" / "data"

CONFIG_COVID = ConfiguracaoProtocolo(
    nome="covid",
    caminho_pdf=str(_DADOS_DIR / "20200504-protocolomanejo-ver09.pdf"),
    descricao_fonte="Ministerio da Saude - Protocolo de Manejo Clinico do Coronavirus (COVID-19) na APS (v9)",
    url_fonte="https://www.gov.br/saude/pt-br/coronavirus/publicacoes-tecnicas/guias-e-planos/protocolo-de-manejo-clinico-do-coronavirus-covid-19-na-atencao-primaria-a-saude",
    padroes_cabecalho=[
        r"\d+\s*Ministerio da Saude\s*/?\s*SAPS\s*[–-]\s*PROTOCOLO DE MANEJO CLINICO DO CORONAVIRUS \(COVID-19\) NA ATENCAO PRIMARIA A SAUDE\s*",
    ],
    padroes_ruido=[
        "sumario",
        "check-list",
        "composicao da equipe",
        "notificacao imediata",
        "fluxograma",
        "fast-track",
        "teleatendimento",
        "aplicativo coronavirus",
    ],
    paginas_por_secao={
        "sintomas": [4, 9, 14],
        "classificacao": [10, 12, 13],
        "sinais_alarme": [12, 13, 14],
        # p.5: RT-PCR (padrão ouro), teste rápido sorológico, IgG/IgM, ELISA
        # p.6: limitações dos testes rápidos, manejo por gravidade
        # p.4, 12, 14 removidos — conteúdo é introdução/sinais de gravidade, não indicação de exames
        "exames": [5, 6],
        "tratamento": [18, 19, 20, 21, 22],
    },
    niveis_gravidade=["leve", "moderado", "grave", "critico"],
    # Protocolo COVID usa "diagnóstico laboratorial" — não organiza exames por gravidade
    query_exames="diagnóstico laboratorial {gravidade}",
)
