# Assistente Médico

Sistema de assistente médico com RAG e agentes especializados.

---

## Pipeline de fine-tuning MedPT (`tunning/`)

Os três scripts compartilham o arquivo [requirements-finetune.txt](requirements-finetune.txt) na raiz do repositório (Hugging Face `datasets`, `transformers`, `peft`, `bitsandbytes`, Unsloth para o treino, `bert-score` opcional na avaliação). Detalhes extras (GPU, token HF, persistência no Drive): [docs/colab_02_03_ambiente.md](docs/colab_02_03_ambiente.md).

### Onde executar cada módulo

| Script | Ambiente previsto |
|--------|-------------------|
| [tunning/01_analyze_dataset.py](tunning/01_analyze_dataset.py) | **Cursor** (máquina local): prepara `data/processed/` sem necessidade de GPU. |
| [tunning/02_finetune_lora.py](tunning/02_finetune_lora.py) | **Google Colab** com GPU (ex.: T4): quantização 4-bit e Unsloth. |
| [tunning/03_evaluate_model.py](tunning/03_evaluate_model.py) | **Google Colab** com GPU: inferência com modelo base + adapter LoRA. |

Fluxo de dados: rode o `01` localmente, sincronize a pasta do projeto (em especial `data/processed/` e, após o treino, `pre-trained/`) com o Google Drive ou outra cópia usada no Colab; depois execute os notebooks abaixo na ordem.

### Configuração do ambiente — Cursor (`01_analyze_dataset.py`)

1. Abra o repositório na raiz (`Tech-Challenge-FIAP-FASE3/`).
2. Crie ou ative um ambiente Python (3.10 ou superior recomendado).
3. Instale as dependências do pipeline de tuning:

   ```bash
   pip install -r requirements-finetune.txt
   ```

4. Se o parquet MedPT exigir autenticação no Hugging Face, faça login (`huggingface-cli login` ou variável `HF_TOKEN`).
5. Execute na raiz do projeto (ajuste `--dataset-path` se o parquet não estiver em `data/train-00000-of-00001.parquet`):

   ```bash
   python tunning/01_analyze_dataset.py
   ```

Saídas esperadas: `data/processed/medpt_qa/` (DatasetDict), `data/processed/medpt_qa_manifest.jsonl` e `data/processed/medpt_qa_report.json`.

### Configuração do ambiente — Google Colab (`02` e `03`)

1. **Runtime → Alterar tipo de tempo de execução → acelerador de hardware: GPU** (T4 ou superior é o cenário usual).
2. O projeto deve estar acessível no runtime (ex.: pasta sincronizada no **Google Drive** contendo `tunning/` e, após o passo local, `data/processed/`).
3. Se `Qwen/Qwen2.5-3B-Instruct` (ou outro modelo) exigir aceite de licença, configure token HF (Secrets `HF_TOKEN` ou `huggingface-cli login` na sessão).
4. Todas as células que chamam `!python` devem rodar com **diretório de trabalho na raiz do repositório** (como no `%cd` dos notebooks).

### Execução no Google Colab — notebook do fine-tuning

Use o arquivo [tunning/Execucao do 02_finetune_lora.py.ipynb](tunning/Execucao%20do%2002_finetune_lora.py.ipynb). Ajuste `CW_DIRECTORY_FIAP` e `FIAP_PROJECT` para o caminho real da sua pasta no Drive.

**Montagem do Drive e pasta do projeto** (equivalente à primeira célula do notebook):

```python
from google.colab import drive
drive.mount('/content/drive')

CW_DIRECTORY_FIAP = "/content/drive/MyDrive/elber/#Cursos/##FIAP/"
FIAP_PROJECT = "Tech-Challenge-FIAP-FASE3/"
CW_DIRECTORY = CW_DIRECTORY_FIAP+FIAP_PROJECT
print(CW_DIRECTORY)

#AQUI VOCE LISTA OS ARQUIVOS E FAZ A LEITURA COM ESSE PADRAO
# # artigo_teste = pd.read_csv(CW_DIRECTORY+"teste.csv")
# Isso representa "/content/drive/MyDrive/elber/#Cursos/##FIAP/POSTECH-WIN-2_5-Embeddings/teste.csv"
```

**Raiz do repositório e registro do ambiente** (segunda célula):

```python
#Este é um magic command do IPython.
#Ele altera o diretório de trabalho atual para a sessão do kernel IPython.
#Qualquer código Python subsequente executado neste kernel será executado a partir do novo diretório.
#Esta é a forma que você geralmente quer usar para mudar de diretório para o seu código.
%cd $CW_DIRECTORY
!pip list > pip_collab.txt
```

**Dependências** (terceira célula):

```python
!pip install -q -r requirements-finetune.txt
```

**Verificação do CLI e da GPU** (quarta e quinta células):

```python
#Executando o HELP, eu verifico se vai dar algum erro de PROCESSADOR
!python tunning/02_finetune_lora.py --help
```

```python
# 2. Verificar se o torch tem CUDA:
!python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
#Se o resultado mostrar True, o script vai funcionar.
#!pip list > pip_collab2.txt
```

**Treino** (última célula do fluxo do notebook; amplie amostras ou retire os limites para corrida completa):

```python
!python tunning/02_finetune_lora.py --max-train-samples 512 --max-validation-samples 128
```

Se o Colab pedir reinício após instalar pacotes, use **Runtime → Reiniciar sessão**, reinstale `requirements-finetune.txt`, faça `%cd` de novo e continue.

### Execução no Google Colab — notebook da avaliação

Use o arquivo [tunning/Execucao do 03_evaluate_model.py.ipynb](tunning/Execucao%20do%2003_evaluate_model.py.ipynb). As primeiras células repetem o mesmo padrão de Drive + `%cd`; a lista de pacotes é gravada em `pip_collab_evaluate.txt` para não sobrescrever o arquivo do notebook do treino.

**Montagem do Drive e pasta do projeto** (primeira célula):

```python
from google.colab import drive
drive.mount('/content/drive')

CW_DIRECTORY_FIAP = "/content/drive/MyDrive/elber/#Cursos/##FIAP/"
FIAP_PROJECT = "Tech-Challenge-FIAP-FASE3/"
CW_DIRECTORY = CW_DIRECTORY_FIAP + FIAP_PROJECT
print(CW_DIRECTORY)

# AQUI VOCE LISTA OS ARQUIVOS E FAZ A LEITURA COM ESSE PADRAO
# # artigo_teste = pd.read_csv(CW_DIRECTORY+"teste.csv")
# Isso representa "/content/drive/MyDrive/elber/#Cursos/##FIAP/POSTECH-WIN-2_5-Embeddings/teste.csv"
```

**Raiz do repositório** (segunda célula):

```python
# Este é um magic command do IPython.
# Ele altera o diretório de trabalho atual para a sessão do kernel IPython.
# Qualquer código Python subsequente executado neste kernel será executado a partir do novo diretório.
# Esta é a forma que você geralmente quer usar para mudar de diretório para o seu código.
%cd $CW_DIRECTORY
!pip list > pip_collab_evaluate.txt
```

**Dependências** (terceira célula):

```python
# Cobre torch/transformers/peft/bitsandbytes/datasets e bert-score (opcional em --compute-bertscore).
!pip install -q -r requirements-finetune.txt
```

**Ajuda do script, CUDA e avaliação** (demais células):

```python
# Executando o HELP, eu verifico se vai dar algum erro de PROCESSADOR / imports.
!python tunning/03_evaluate_model.py --help
```

```python
# Verificar se o torch tem CUDA:
!python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Se o resultado mostrar True, o script pode usar GPU na inferência.
```

```python
# Avaliacao no split de teste (dataset + manifest do script 01; adapter do script 02).
# Ajuste --max-eval-samples, --adapter-dir, --dataset-dir e --output-dir conforme o seu Drive.
# Para metricas semanticas adicionais: acrescente --compute-bertscore (bert-score ja esta no requirements-finetune.txt).
!python tunning/03_evaluate_model.py --max-eval-samples 128 --max-new-tokens 256 --temperature 0.0
```

Relatórios padrão em `data/evaluation/qwen2.5-3b-medpt-lora/` (`evaluation_summary.json` e `predictions.jsonl`), salvo alteração de `--output-dir`.

---

## Estrutura de arquivos

```
assistente_medico/
│
├── main.py                      # Ponto de entrada — rode esse para iniciar
├── configuracoes.py             # Todas as configurações em um lugar só
│
├── carregador_do_modelo.py      # Carrega o LLM e prepara para o LangChain
├── banco_de_conhecimento.py     # Constrói o RAG com o dataset question|answer
├── agentes.py                   # Define os 3 agentes + o supervisor
├── fluxo.py                     # Monta o fluxo LangGraph orquestrando tudo
│
├── gerar_dados_sinteticos.py    # Gera CSV de exemplo para testes
│
├── dados/
│   ├── perguntas_e_respostas.csv   # Dataset com colunas: question | answer
│   └── banco_de_vetores/           # Criado automaticamente na primeira execução
│
└── logs/
    └── conversas.jsonl             # Log de todas as interações (criado automaticamente)
```

---

## Como rodar pela primeira vez

```bash
# 1. Instalar as dependências
pip install -r requirements.txt

# 2. Gerar o dataset de exemplo (só precisa fazer uma vez)
python gerar_dados_sinteticos.py

# 3. Iniciar o assistente
python main.py
```

---

## Como trocar o modelo quando o time de fine-tuning entregar

Abra o arquivo `configuracoes.py` e mude apenas essa linha:

```python
# Antes:
CAMINHO_DO_MODELO = "nlpie/tiny-biobert"

# Depois (exemplo):
CAMINHO_DO_MODELO = "./modelo_finetuned"
```

Apague o banco de vetores antigo para recriar com o novo modelo:
```bash
rm -rf ./dados/banco_de_vetores
python main.py
```

---

## Como o fluxo funciona

```
Pergunta do profissional
        │
        ▼
   [supervisor]
   Lê a pergunta e decide qual agente usar
        │
        ├── "triagem"   → [agente_de_triagem]    avalia sintomas e urgência
        ├── "clinico"   → [agente_clinico]        responde dúvidas médicas
        └── "seguranca" → [agente_de_seguranca]   detecta riscos e emergências
                │
                ▼
            [log]  salva a conversa e encerra
```

---

## Formato do dataset

O arquivo `dados/perguntas_e_respostas.csv` precisa ter exatamente essas colunas:

| question | answer |
|---|---|
| What are warning signs in pregnancy? | Warning signs include... |
| When is colposcopy indicated? | Colposcopy is indicated after... |

---

## Formato do log

Cada linha de `logs/conversas.jsonl` é um JSON:

```json
{
  "data_hora": "2024-01-15 10:30:00",
  "pergunta": "What are warning signs in a pregnant woman?",
  "resposta": "Warning signs include blood pressure above 140/90...",
  "agente_utilizado": "triagem"
}
```
