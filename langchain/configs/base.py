from dataclasses import dataclass, field


@dataclass
class ConfiguracaoProtocolo:
    """Configuração genérica para parsing de um protocolo médico."""

    nome: str  # ex: "dengue"
    caminho_pdf: str  # caminho do PDF
    descricao_fonte: str  # ex: "MS/SVS - Protocolo Dengue 2024"
    url_fonte: str = ""  # URL pública da publicação original

    # Limpeza
    padroes_cabecalho: list[str] = field(default_factory=list)
    padroes_ruido: list[str] = field(default_factory=list)

    # Classificação de seções — por página (um chunk pode ter múltiplas tags)
    paginas_por_secao: dict[str, list[int]] = field(default_factory=dict)

    # Gravidade
    niveis_gravidade: list[str] = field(default_factory=list)

    # Chunking
    tamanho_chunk: int = 800
    sobreposicao_chunk: int = 200

    # Template de query para busca de exames — use {gravidade} como placeholder
    # Deve usar o vocabulário real do PDF para maximizar o score BM25
    query_exames: str = "exames complementares {gravidade}"
