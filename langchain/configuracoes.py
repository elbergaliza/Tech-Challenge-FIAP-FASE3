# configuracoes.py
# ─────────────────────────────────────────────────────────────
# Arquivo central de configurações do projeto.
# Todas as variáveis importantes ficam aqui.
# Se precisar mudar algo, muda só nesse arquivo.
# ─────────────────────────────────────────────────────────────

# Caminho do modelo fine-tunado que o time vai entregar
# Enquanto o modelo não chega, usamos esse modelo público como substituto
CAMINHO_DO_MODELO = "google/flan-t5-small"

# Caminho do arquivo CSV com as perguntas e respostas médicas
# O CSV precisa ter duas colunas: "question" e "answer"
CAMINHO_DO_DATASET = "./dados/perguntas_e_respostas.csv"

# Caminho onde o banco de vetores vai ser salvo
# (o banco de vetores é o que permite o sistema buscar respostas parecidas)
CAMINHO_DO_BANCO_DE_VETORES = "./dados/banco_de_vetores"

# Quantidade de respostas parecidas que o sistema vai buscar
# quando receber uma pergunta
QUANTIDADE_DE_RESULTADOS = 3

# Arquivo onde todas as conversas vão ser salvas para auditoria
CAMINHO_DO_LOG = "./logs/conversas.jsonl"
