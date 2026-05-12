# Análise da Saída do Script `02_finetune_lora.py`

Mapeamento detalhado de cada bloco de saída para o ponto correspondente no código.

---

## 1. Linhas 1-2 do output — Mensagens de startup do Unsloth

```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
```

**Origem:** Gerado automaticamente pelo `import` na linha 56 do código:

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
```

O simples ato de importar o módulo `unsloth` já dispara essas mensagens de patching.

---

## 2. Linhas 3-7 do output — Banner do Unsloth + info de hardware

```
==((====))==  Unsloth 2026.4.8: Fast Qwen2 patching. Transformers: 5.5.0.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.563 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 7.5. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
```

**Origem:** Gerado internamente por `FastLanguageModel.from_pretrained()` nas linhas 492-497:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.base_model_name,
    max_seq_length=args.max_seq_length,
    load_in_4bit=True,
    full_finetuning=False,
)
```

---

## 3. Linhas 8-18 do output — Download do modelo e tokenizer

```
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
model.safetensors: 100% 2.36G/2.36G [00:42<00:00, 56.1MB/s]
Loading weights: 100% 434/434 [00:03<00:00, 141.43it/s]
generation_config.json: 100% 271/271 [00:00<00:00, 1.68MB/s]
config.json: 1.42kB [00:00, 4.06MB/s]
tokenizer_config.json: 7.36kB [00:00, 22.5MB/s]
vocab.json: 2.78MB [00:00, 61.9MB/s]
merges.txt: 1.67MB [00:00, 93.8MB/s]
tokenizer.json: 100% 11.4M/11.4M [00:00<00:00, 54.0MB/s]
added_tokens.json: 100% 605/605 [00:00<00:00, 2.59MB/s]
special_tokens_map.json: 100% 614/614 [00:00<00:00, 4.02MB/s]
```

**Origem:** Mesmo trecho — `FastLanguageModel.from_pretrained()` (linhas 492-497). A biblioteca baixa pesos e arquivos do tokenizer do Hugging Face Hub.

---

## 4. Linhas 19-21 do output — Avisos de padding e dropout

```
unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit does not have a padding token!
Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = 0.05.
Unsloth will patch all other layers, except LoRA matrices, causing a performance hit.
```

**Origem:** Linha 19 vem de `from_pretrained()` (linhas 492-497). As linhas 20-21 vêm de `get_peft_model()` nas linhas 513-530, porque o `lora_dropout=0.05` não é zero:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=args.seed,
)
```

---

## 5. Linha 22 do output — Resultado do patching

```
Unsloth 2026.4.8 patched 36 layers with 0 QKV layers, 0 O layers and 0 MLP layers.
```

**Origem:** Também de `get_peft_model()` (linhas 513-530). Após aplicar os adapters LoRA, o Unsloth reporta o resumo do patching.

---

## 6. Linhas 23-24 do output — Formatação do dataset

```
Formatando dataset de treino: 100% 512/512 [00:00<00:00, 10158.58 examples/s]
Formatando dataset de validacao: 100% 128/128 [00:00<00:00, 3583.65 examples/s]
```

**Origem:** Função `prepare_text_datasets()`, chamada na linha 536. As strings de descrição vêm diretamente do parâmetro `desc=` do `.map()` nas linhas 347-358:

```python
train_dataset = dataset["train"].map(
    formatter,
    batched=True,
    remove_columns=columns_to_remove,
    desc="Formatando dataset de treino",
)
validation_dataset = dataset["validation"].map(
    formatter,
    batched=True,
    remove_columns=columns_to_remove,
    desc="Formatando dataset de validacao",
)
```

---

## 7. Linha 25 do output — Aviso de warmup_ratio deprecated

```
warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
```

**Origem:** Gerado internamente pelo `SFTConfig` ao receber `warmup_ratio` na construção do trainer (linhas 572-578):

```python
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=text_datasets["train"],
    eval_dataset=text_datasets["validation"],
    args=SFTConfig(**sft_training_kwargs),
)
```

O valor `warmup_ratio` é passado no dicionário `sft_training_kwargs` (linha 563).

---

## 8. Linhas 26-27 do output — Tokenização pelo Unsloth

```
Unsloth: Tokenizing ["text"] (num_proc=6): 100% 512/512 ...
Unsloth: Tokenizing ["text"] (num_proc=6): 100% 128/128 ...
```

**Origem:** Também disparado pela construção do `SFTTrainer` (linhas 572-578). Internamente o trainer tokeniza os datasets `train` (512 exemplos) e `validation` (128 exemplos) usando o campo `dataset_text_field="text"`.

---

## 9. Linhas 29-35 do output — Banner de treino + início do training loop

```
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 512 | Num Epochs = 1 | Total steps = 32
O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 8
\        /    Data Parallel GPUs = 1 | Total batch size (2 x 8 x 1) = 16
 "-____-"     Trainable parameters = 59,867,136 of 3,145,805,824 (1.90% trained)
```

**Origem:** Chamada `trainer.train()` na linha 580:

```python
trainer.train()
```

O Unsloth/HF Trainer imprime esse banner automaticamente ao iniciar o treinamento.

---

## 10. Linhas 36-38 do output — Métricas de loss durante treino

```
{'loss': '2.149', 'grad_norm': '0.7391', 'learning_rate': '0.0001689', 'epoch': '0.3125'}
{'loss': '1.375', 'grad_norm': '0.4727', 'learning_rate': '7.493e-05', 'epoch': '0.625'}
{'loss': '1.369', 'grad_norm': '0.4467', 'learning_rate': '4.586e-06', 'epoch': '0.9375'}
```

**Origem:** Também de `trainer.train()` (linha 580). Logados pelo callback padrão do Trainer a cada `logging_steps=10` (definido na linha 553 do dict `sft_training_kwargs`).

---

## 11. Linhas 39-128 do output — Avaliação + progresso + warnings

Tudo isso (barras de progresso de treino/avaliação, `FutureWarning` do transformers, `eval_loss`, métricas finais de `train_runtime`) é saída interna de `trainer.train()` (linha 580), incluindo a avaliação que roda a cada `eval_steps=200` (que nesse caso coincide com o checkpoint no final da época).

---

## 12. Linha 126 do output — Checkpoint salvo

```
Unsloth: Restored added_tokens_decoder metadata in data/checkpoints/qwen2.5-3b-medpt-lora/checkpoint-32/tokenizer_config.json.
```

**Origem:** Callback automático de salvamento do Trainer dentro de `trainer.train()` (linha 580), disparado pelo `save_steps`.

---

## 13. Linha 129 do output — Adapter final salvo

```
Unsloth: Restored added_tokens_decoder metadata in pre-trained/qwen2.5-3b-medpt-lora/tokenizer_config.json.
```

**Origem:** `trainer.save_model()` e/ou `tokenizer.save_pretrained()` nas linhas 583-584:

```python
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
```

---

## 14. Linhas 130-131 do output — Mensagens finais explícitas do script

```
Checkpoints salvos em: data/checkpoints/qwen2.5-3b-medpt-lora
Modelo final (adapter) salvo em: pre-trained/qwen2.5-3b-medpt-lora
```

**Origem:** Os únicos `print()` explícitos do código, nas linhas 587-588:

```python
print(f"Checkpoints salvos em: {checkpoint_dir}")
print(f"Modelo final (adapter) salvo em: {final_dir}")
```

---

## Resumo

| Linhas do output | Origem no código | Tipo |
|---|---|---|
| 1-2 | Linha 56 (`import unsloth`) | Implícito (import) |
| 3-18 | Linhas 492-497 (`from_pretrained`) | Implícito (biblioteca) |
| 19 | Linhas 492-497 (`from_pretrained`) | Implícito (biblioteca) |
| 20-22 | Linhas 513-530 (`get_peft_model`) | Implícito (biblioteca) |
| 23-24 | Linhas 347-358 (`prepare_text_datasets` / `.map`) | Explícito (`desc=`) |
| 25-28 | Linhas 572-578 (`SFTTrainer` / `SFTConfig`) | Implícito (biblioteca) |
| 29-128 | Linha 580 (`trainer.train()`) | Implícito (trainer loop) |
| 129 | Linhas 583-584 (`save_model` / `save_pretrained`) | Implícito (biblioteca) |
| 130-131 | Linhas 587-588 (`print()`) | **Explícito (seu código)** |

> **Conclusão:** Apenas as duas últimas linhas da saída (130-131) são `print()` explícitos escritos no script. Todo o restante é gerado automaticamente pelas bibliotecas Unsloth, Transformers e TRL.
