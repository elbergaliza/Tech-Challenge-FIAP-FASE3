# 🚨 TAREFA: Fine-tuning Assistente Médico com MedPT

## 🎯 OBJETIVO
Fine-tuning de Foundation Model → Assistente Virtual Médico para:
- Apoio em condutas clínicas
- Resposta a dúvidas médicas  
- Sugestões baseadas em protocolos internos

**⚠️ RESTRIÇÕES CRÍTICAS DE SEGURANÇA:**
- NUNCA substituir julgamento clínico
- Sempre incluir avisos de validação humana
- Detectar e sinalizar alucinações/incertezas
- Recusar casos críticos sem evidência

## 📊 DATASET
https://huggingface.co/datasets/AKCIT/MedPT/blob/main/data/train-00000-of-00001.parquet
docs/MedPT.pdf
- somente as colunas "question" e "answer" serão usadas. Descartar todas as outras colunas.

## 🖥️ AMBIENTES
Notebook local: docs/noteinfo.txt
Google Colab: docs/collab.txt


## 🎯 CRITÉRIOS OBRIGATÓRIOS - Foundation Model
Escolha **UM** modelo que atenda **TODOS** estes critérios:
✅ Suporte nativo a quantização 4-bit (bitsandbytes/GPTQ)
✅ Compatibilidade confirmada com Unsloth AI
✅ Licença open-source (Apache 2.0/MIT)
✅ PEFT/LoRA via biblioteca PEFT oficial
✅ Funciona nos 2 ambientes computacionais acima


## 🏗️ ARQUITETURA TÉCNICA (OBRIGATÓRIA)
TOKENIZAÇÃO → Tokenizer nativo do modelo escolhido

QUANTIZAÇÃO → 4-bit (bitsandbytes)

FINE-TUNING → PEFT + LoRA + Unsloth

BIBLIOTECAS:
├─ transformers
├─ datasets
├─ peft (implementa LoRA)
├─ unsloth
└─ bitsandbytes

SPAcY → APENAS pré-processamento auxiliar (NÃO substitui tokenizer)


## 📁 ESTRUTURA DO PROJETO
Tech-Challenge-FIAP-FASE3/
├── tunning/ # ← Código aqui
├── data/ # Dataset + checkpoints
├── pre-trained/ # ← Modelo final + tokenizer
└── docs/ # Documentação


## 📋 EXECUÇÃO EM 3 PASSOS

### PASSO 1: PLANEJAMENTO
1.1 Analisar docs/noteinfo.txt + docs/collab.txt e o dataset
1.2 Listar 3 opções de Foundation Model (com justificativa)
1.3 Escolher 1 modelo final, com justificativa técnica objetiva. (justificar VRAM/tempo)
1.4 Definir hiperparâmetros LoRA (rank, alpha, dropout). Explique técnicamente os hiperparametros LORA propostos.
1.5 Propor plano de avaliação + segurança

#### Decisões consolidadas do PASSO 1

##### 1.1 Diagnóstico do dataset e dos ambientes
- O dataset `AKCIT/MedPT` possui 384.095 linhas e aproximadamente 95,1 MB.
- O schema público do dataset contém as colunas: `id`, `question`, `answer`, `condition`, `medical_specialty` e `question_type`.
- Para este projeto, serão usadas apenas `question` e `answer`.
- Há relação `1 pergunta -> várias respostas`, portanto o split de treino/validação/teste nao pode ser por linha; deve ser por `question` normalizada, evitando vazamento entre conjuntos.
- O dataset é forte para QA médico em portugues-BR, mas nao deve ser usado de forma cega: ha exemplos com promocao de consulta, preco, rede social, convite comercial e variabilidade de qualidade entre respostas.
- Antes do fine-tuning, sera necessario filtrar ruido textual e conteudo promocional.

**Ambiente local (`docs/noteinfo.txt`)**
- Windows 11, 16 GB de RAM.
- Adequado para analise, pre-processamento, inspecao de dados e validacao leve.
- Nao deve ser considerado o ambiente principal de treino.

**Google Colab (`docs/collab.txt`)**
- Linux com GPU Tesla T4 de 15 GB VRAM e 12,67 GB de RAM.
- Este sera o ambiente principal para fine-tuning com `4-bit + PEFT/LoRA + Unsloth`.

##### 1.2 Tres opcoes de Foundation Model

**Opcao A - `Qwen2.5-1.5B-Instruct`**
- Pros:
  - Muito leve para treino em 4-bit.
  - Menor tempo de treino.
  - Maior margem operacional na T4.
- Contras:
  - Menor capacidade para nuances clinicas.
  - Pode perder qualidade em respostas medicas mais detalhadas.

**Opcao B - `Qwen2.5-3B-Instruct`**
- Pros:
  - Melhor equilibrio entre qualidade, VRAM e tempo.
  - Compativel com Unsloth, PEFT e quantizacao 4-bit.
  - Licenca Apache 2.0.
  - Mais seguro para treinar na T4 com margem operacional.
- Contras:
  - Mais lento que o modelo 1.5B.
  - Ainda abaixo de modelos maiores em refinamento de resposta.

**Opcao C - `Qwen3-4B-Instruct`**
- Pros:
  - Maior capacidade que as opcoes anteriores.
  - Compativel com Unsloth, PEFT e 4-bit.
  - Licenca Apache 2.0.
- Contras:
  - Maior pressao de VRAM e tempo.
  - Menor folga operacional na T4.
  - Menos confortavel para validacoes fora do Colab.

##### 1.3 Modelo final escolhido
- Modelo recomendado: `Qwen2.5-3B-Instruct`

**Justificativa tecnica objetiva**
- E o melhor ponto de equilibrio entre qualidade, estabilidade, memoria e tempo de treino.
- Em GPU `Tesla T4 15 GB`, o treino com `4-bit + LoRA + Unsloth` e viavel com margem melhor do que um modelo de 4B.
- O modelo 1.5B e mais barato, mas pode ser pequeno demais para capturar nuances clinicas e linguisticas do dominio medico.
- O modelo 4B pode oferecer ganhos, mas aumenta custo computacional e risco operacional sem necessidade comprovada nesta fase.
- Portanto, o `Qwen2.5-3B-Instruct` foi escolhido por oferecer o melhor compromisso entre capacidade e viabilidade nos dois ambientes definidos.

##### 1.4 Hiperparametros LoRA propostos
- `rank (r) = 32`
- `lora_alpha = 64`
- `lora_dropout = 0.05`

**Justificativa tecnica**
- `r = 32`: oferece capacidade suficiente de adaptacao ao dominio medico sem aumentar demais memoria e custo de treino.
- `alpha = 64`: mantem a relacao `alpha / r = 2`, um ponto estavel para escalar a contribuicao da LoRA sem tornar a adaptacao agressiva demais.
- `dropout = 0.05`: ajuda a reduzir overfitting diante do ruido real do dataset, especialmente respostas promocionais, divergentes ou excessivamente genericas.

##### 1.5 Plano de avaliacao e seguranca

**Plano de avaliacao**
- Validar o schema logo no inicio da pipeline para garantir o uso correto de `question` e `answer`.
- Fazer split por pergunta normalizada, nunca por linha.
- Avaliar o modelo em conjunto holdout real do proprio dataset.
- Medir desempenho por:
  - similaridade semantica da resposta
  - avaliacao qualitativa manual
  - analise por `question_type`
  - analise por tamanho de pergunta
  - analise por condicoes frequentes vs. cauda longa
- Realizar inspecao manual obrigatoria em amostras clinicas reais, pois metricas textuais sozinhas nao garantem seguranca medica.

**Plano de seguranca**
- O assistente deve atuar como apoio informacional, nunca como substituto do julgamento clinico.
- Respostas devem explicitar limites, incerteza e necessidade de validacao humana quando aplicavel.
- Casos potencialmente graves devem ser sinalizados como necessidade de avaliacao medica imediata.
- Devem existir testes especificos para cenarios de maior risco, incluindo:
  - medicacao
  - gestacao
  - pediatria
  - dor toracica e falta de ar
  - risco psiquiatrico
- O processo de avaliacao deve verificar alucinacao, excesso de confianca, omissao de risco e recomendacoes potencialmente perigosas.



### PASSO 2: IMPLEMENTAÇÃO
2.1 tunning/01_analyze_dataset.py
2.1.1 Os seguintes itens deverão ser realizados no dataset:
.Há relação 1 pergunta -> várias respostas, então o split não pode ser por linha; deve ser por question normalizada, senão haverá vazamento entre treino e teste.
.O dataset é valioso, mas não é “limpo o suficiente” para treino cego: nas amostras públicas aparecem respostas promocionais, chamadas para consulta, preços, redes sociais e alguma variabilidade clínica/qualitativa. Realize uma filtragem desses itens, antes do fine-tuning.
2.2 tunning/02_finetune_lora.py
2.2.1 O projeto deve ser pensado para treinar no Colab e usar o ambiente local para análise, preparação e validação leve.
2.3 tunning/03_evaluate_model.py
2.4 Salvar checkpoints intermediarios em data/
2.5 Salvar modelo final e tokenizer em pre-trained/
2.6 Documentar cada macroetapa do código


### PASSO 3: VALIDAÇÃO
3.1 Testes com protocolos médicos
3.2 Testes com perguntas clínicas reais
3.3 Testes de segurança (casos perigosos)
3.4 Relatório de métricas


## 🎯 ENTREGÁVEIS ESPERADOS
✅ 1. Relatório técnico (análise + modelo escolhido + plano)
✅ 2. 3 scripts Python organizados em tunning/
✅ 3. Modelo + tokenizer salvos em pre-trained/
✅ 4. Checkpoints intermediários em data/
✅ 5. README.md com instruções de uso


## 🚫 RESTRIÇÕES
❌ NÃO usar SpaCy como tokenizer principal do treino
❌ NÃO inventar dados de teste
❌ NÃO modificar arquivos fora do escopo
❌ NÃO usar modelos sem suporte Unsloth
❌ NÃO implementar LoRA manualmente (usar PEFT)


## 📤 FORMATO DA RESPOSTA
Quero sua resposta em 4 partes:
1. Diagnóstico do dataset e dos ambientes.
2. Três opções de Foundation Model compatíveis, com prós e contras.
3. Modelo recomendado, com justificativa técnica objetiva.
4. Plano de implementação detalhado com arquivos a criar/alterar (com código comentado), 
etapas, dependências, estratégia de avaliação e riscos de segurança médica.
5. Comandos para executar.