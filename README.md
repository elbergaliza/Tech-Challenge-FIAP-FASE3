# Assistente Médico

Sistema de assistente médico com RAG e agentes especializados.

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
