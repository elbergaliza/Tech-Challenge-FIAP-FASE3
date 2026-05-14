# Comparação de avaliação: LoRA pré-treinado vs checkpoint-3750

Comparação entre os resumos em:

- `data/evaluation/qwen2.5-3b-medpt-lora/evaluation_summary.json` (LoRA pré-treinado)
- `data/evaluation/checkpoint-3750/evaluation_summary.json` (fine-tuning até step 3750)

## Contexto dos runs

| Aspecto | LoRA pré-treinado (`qwen2.5-3b-medpt-lora`) | `checkpoint-3750` |
|---------|---------------------------------------------|-------------------|
| Origem do modelo | `pre-trained/qwen2.5-3b-medpt-lora` | `data/checkpoints/qwen2.5-3b-medpt-lora/checkpoint-3750` |
| Base | `Qwen/Qwen2.5-3B-Instruct` | `Qwen/Qwen2.5-3B-Instruct` |
| Amostras avaliadas | **128** | **200** |

**Nota:** o número de amostras difere entre os dois runs. Métricas agregadas e contagens absolutas (por exemplo, casos de alto risco) não são estritamente comparáveis sem o mesmo subconjunto do manifest; ainda assim, as tendências abaixo são úteis.

## Métricas globais

| Métrica | LoRA pré-treinado | checkpoint-3750 | Evolução |
|---------|-------------------|-----------------|----------|
| Token F1 (média) | 0,199 | 0,225 | Subida clara (~+0,026 absoluto; ordem de +13% sobre o valor inicial) |
| ROUGE-L (média) | 0,136 | 0,140 | Ganho pequeno mas consistente (~+0,004 absoluto) |
| BERTScore | Desligado (`flag_disabled`) | Idem | Sem mudança |

## Segurança

| Campo | LoRA pré-treinado | checkpoint-3750 |
|-------|-------------------|-----------------|
| Caution rate | 0,0 | 0,025 |
| Overconfidence rate | 0,0 | 0,0 |
| High-risk case count | 7 | 10 |
| High-risk caution coverage | 0,0 | 0,0 |

O checkpoint passa a registrar uma pequena taxa de cautela (`caution_rate`). A contagem de casos de alto risco aumenta; isso pode refletir o maior número de amostras avaliadas. Em ambos os runs, a cobertura de cautela sobre alto risco permanece zero.

## Fatias: ROUGE-L por tipo de pergunta

| Tipo | LoRA pré-treinado | checkpoint-3750 |
|------|-------------------|-----------------|
| Tratamento | 0,136 | 0,141 |
| Diagnóstico | 0,138 | 0,136 |
| Epidemiologia | 0,127 | 0,137 |
| Escolha de profissionais de saúde | 0,166 | 0,227 |
| Estilo de vida saudável | — | 0,052 |

A categoria **Estilo de vida saudável** só aparece no segundo arquivo; o ROUGE-L nessa fatia é baixo. O salto em **Escolha de profissionais de saúde** é grande; vale verificar se há mais itens desse tipo nas 200 amostras do segundo run.

## Fatias: ROUGE-L por comprimento da pergunta

| Comprimento | LoRA pré-treinado | checkpoint-3750 |
|-------------|-------------------|-----------------|
| short | 0,135 | 0,152 |
| medium | 0,134 | 0,135 |
| long | 0,137 | 0,136 |

Melhora mais visível em perguntas **curtas**; **medium** e **long** ficam estáveis.

## Fatias: ROUGE-L por bucket de condição

| Bucket | LoRA pré-treinado | checkpoint-3750 |
|--------|-------------------|-----------------|
| long_tail | 0,127 | 0,143 |
| frequent | 0,138 | 0,140 |

Ganho mais claro em **long_tail**; **frequent** melhora levemente.

## Síntese

O **checkpoint-3750** melhora sobretudo **F1 de tokens** e mostra **ROUGE-L** um pouco maior no agregado, com destaque em **perguntas curtas**, **epidemiologia**, **long_tail** e **escolha de profissionais de saúde**. **Diagnóstico** cai ligeiramente em ROUGE-L. Para comparação mais rigorosa entre modelos, reavalie ambos no **mesmo** subconjunto fixo de linhas do manifest (mesmo N e mesmas IDs de exemplo).
