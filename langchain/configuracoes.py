# configuracoes.py
# ─────────────────────────────────────────────────────────────
# Arquivo central de configurações do projeto.
# Todas as variáveis importantes ficam aqui.
# Se precisar mudar algo, muda só nesse arquivo.
# ─────────────────────────────────────────────────────────────

from pathlib import Path

# Diretório base: sempre aponta para a pasta langchain/, independente de onde o
# processo é iniciado (python main.py, pytest, etc.)
BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / "dados"
LOGS_DIR = BASE_DIR / "logs"

# Caminho do modelo fine-tunado que o time vai entregar
# Enquanto o modelo não chega, usamos esse modelo público como substituto
CAMINHO_DO_MODELO = "google/flan-t5-small"

# Caminho do arquivo CSV com as perguntas e respostas médicas
# O CSV precisa ter duas colunas: "question" e "answer"
CAMINHO_DO_DATASET = str(DADOS_DIR / "perguntas_e_respostas.csv")

# Caminho onde o banco de vetores vai ser salvo
# (o banco de vetores é o que permite o sistema buscar respostas parecidas)
CAMINHO_DO_BANCO_DE_VETORES = str(DADOS_DIR / "banco_de_vetores")

# Quantidade de respostas parecidas que o sistema vai buscar
# quando receber uma pergunta
QUANTIDADE_DE_RESULTADOS = 3

# Arquivo onde todas as conversas vão ser salvas para auditoria
CAMINHO_DO_LOG = str(LOGS_DIR / "conversas.jsonl")

# Modelo de embedding multilíngue (melhor para português)
MODELO_DE_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Peso do BM25 na busca híbrida (0.3 = prioriza semântica via FAISS)
# Experimentos mostraram que peso menor evita que páginas de formulário
# com repetição de keywords dominem sobre chunks de critérios diagnósticos
PESO_BM25 = 0.3
