Segue o mapeamento da saída em `resultado_03.txt` em relação ao seu script e às bibliotecas.

---

## O que **não** vem do `03_evaluate_model.py` (bibliotecas)

| Trecho na saída | Origem provável |
|-----------------|----------------|
| Linhas 1–2 (CUDA/torch incompatible, HF unauthenticated) | **PyTorch / bitsandbytes / Hugging Face Hub** durante import ou primeiro uso de modelo/tokenizer quantizado |
| Linhas 3–10 (`config.json`, `model.safetensors`, barras de progresso, `generation_config.json`) | **`AutoTokenizer.from_pretrained`** e **`AutoModelForCausalLM.from_pretrained`** em `load_model_and_tokenizer` (carregamento do Hub e escritas no cache) |
| Linha 11 (flags inválidas `temperature`, `top_p`, `top_k`) | **`transformers`** ao chamar **`model.generate(...)`** no loop principal — aviso quando a configuração de geração do modelo não usa essas flags da forma esperada |

Esses pontos são **efeito colateral** destas chamadas no seu código:

```497:517:c:\WORKSPACES\workspace-PYTHON\Tech-Challenge-FIAP-FASE3\tunning\03_evaluate_model.py
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True)
    ...
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
```

```653:661:c:\WORKSPACES\workspace-PYTHON\Tech-Challenge-FIAP-FASE3\tunning\03_evaluate_model.py
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-6),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
```

---

## O que **vem diretamente** do `03_evaluate_model.py`

### 1. Bloco JSON (linhas 12–54 do `resultado_03.txt`)

É o objeto `summary` montado em **`main()`** (dicionário com `model`, `dataset`, `metrics`, `safety`, `slices`) e impresso aqui:

```755:761:c:\WORKSPACES\workspace-PYTHON\Tech-Challenge-FIAP-FASE3\tunning\03_evaluate_model.py
    output_dir = Path(args.output_dir)
    save_json(output_dir / "evaluation_summary.json", summary)
    save_jsonl(output_dir / "predictions.jsonl", predictions)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nRelatorio salvo em: {output_dir / 'evaluation_summary.json'}")
    print(f"Predicoes salvas em: {output_dir / 'predictions.jsonl'}")
```

Os **valores** desse JSON vêm da lógica acima na mesma função: médias (`statistics.mean`), loop por `test_dataset`, `maybe_compute_bertscore`, agregações em `by_question_type` / `by_length_bucket` / `by_condition_bucket`, etc., até o dict `summary` nas linhas 719–753.

### 2. Mensagens “Relatorio salvo…” e “Predicoes salvas…”

São apenas os dois **`print`** seguintes ao `json.dumps`, na mesma faixa (**linhas 760–761**). Os caminhos reais são `output_dir / "evaluation_summary.json"` e `output_dir / "predictions.jsonl"`; no seu `resultado_03.txt` as barras `/` parecem ter sido coladas sem aparecer (`dataevaluation...`), mas no código continuam sendo caminhos com subpastas.

---

## Resumo

- **Todo o texto “estrutural” até o warning de generation flags** sai das **libs** durante `from_pretrained`, download e **`generate`**; não há `print` seu nessa fase.
- **O único lugar do arquivo que imprime texto produzido pelo script** são as **três linhas 759–761**: resumo JSON e confirmação dos arquivos salvos (**`save_json`** / **`save_jsonl`** escrevem em disco sem imprimir, só os `print` informam ao usuário).

Se quiser, no próximo passo podemos apontar exatamente qual campo do JSON (`token_f1_mean`, fatias etc.) corresponde a qual linha de cálculo no loop.