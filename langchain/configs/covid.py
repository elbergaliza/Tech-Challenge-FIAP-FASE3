from .base import ConfiguracaoProtocolo


CONFIG_COVID = ConfiguracaoProtocolo(
    nome="covid",
    caminho_pdf="./dados/data/20200504-protocolomanejo-ver09.pdf",
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
        "sintomas": [5, 6, 7, 8, 9, 10],
        "classificacao": [10, 11, 12, 13],
        "sinais_alarme": [11, 12, 13],
        "exames": [4, 5],
        "tratamento": [18, 19, 20, 21, 22],
    },
    niveis_gravidade=["leve", "moderado", "grave", "critico"],
)
