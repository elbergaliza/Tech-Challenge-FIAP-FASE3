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

# Modelo base causal (mesmo ID usado em tunning/02 e 03) — usado com adapter LoRA
# e como fallback do pipeline quando não há LOCAL_MODEL_PATH nem adapter detectado.
# Sem GPU, prefira Ollama ou defina HF_PIPELINE_MODEL=gpt2 (ou outro modelo leve) no .env.
CAMINHO_DO_MODELO = "Qwen/Qwen2.5-3B-Instruct"
HF_PIPELINE_MODEL="Qwen/Qwen2.5-3B-Instruct"
# Adapter LoRA salvo pelo tunning/02 (caminho relativo à raiz do repositório).
# Se a pasta não existir, é ignorado; use LORA_ADAPTER_PATH no .env para forçar.
#CAMINHO_DO_ADAPTER_LORA = "pre-trained/qwen2.5-3b-medpt-lora"
CAMINHO_DO_ADAPTER_LORA = ""
# Checkpoint intermediário do Trainer (pasta checkpoint-* com adapter_config.json).
# Vazio = não usar. Tem precedência sobre CAMINHO_DO_ADAPTER_LORA se existir no disco.
# Ex.: "data/checkpoints/qwen2.5-3b-medpt-lora/checkpoint-3750"
CAMINHO_CHECKPOINT_LORA = ""

# Mensagem de sistema alinhada ao treino (chat template); usada pelo carregador HF local.
MENSAGEM_SISTEMA_LLM = (
    "Você é um assistente virtual médico em português do Brasil com foco em apoio "
    "informacional. Nunca substitua o julgamento clínico humano. Sempre explicite "
    "limites quando houver incerteza e, em sinais de gravidade, oriente busca "
    "imediata por atendimento médico."
)

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

