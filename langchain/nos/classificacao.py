# nos/classificacao.py
# ─────────────────────────────────────────────────────────────
# Nó de classificação: recebe sintomas e retorna doenças suspeitas
# com gravidade, justificativa e fontes.
# ─────────────────────────────────────────────────────────────

import json
import re
from pathlib import Path
from banco_de_conhecimento import IndiceProtocolo, busca_hibrida
from nos.debug import salvar_prompt


CAMINHO_PROMPT = Path(__file__).parent.parent / "prompts" / "classificacao.txt"

# Limite de chars do contexto para não estourar o context window do modelo
# Prompt fixo tem ~1400 chars; reservamos ~2000 para resposta → ~4000 de contexto seguro
_MAX_CHARS_CONTEXTO = 4000

# k conservador — após filtro por secao_tipo o número real de chunks é menor
_K_CLASSIFICACAO = 6


def criar_no_classificacao(indice: IndiceProtocolo, modelo):
    """
    Factory que retorna o nó de classificação.

    O nó:
      1. Faz busca híbrida sem filtro (traz de todos os protocolos)
      2. Monta prompt com chunks recuperados
      3. Invoca o LLM para classificar
      4. Parseia JSON e atualiza o estado
    """

    template_prompt = CAMINHO_PROMPT.read_text(encoding="utf-8")

    def no_classificacao(estado: dict) -> dict:
        print("[classificacao] Classificando sintomas...")
        sintomas = estado["sintomas"]

        # Busca filtrada por seção de sintomas para evitar chunks de formulários e anexos
        docs = busca_hibrida(indice, sintomas, k=_K_CLASSIFICACAO, secao_tipo="sintomas")
        docs_resumidos = [
            {
                "texto": d.page_content,
                "pagina": d.metadata.get("pagina", ""),
                "fonte": d.metadata.get("fonte", ""),
                "url_fonte": d.metadata.get("url_fonte", ""),
            }
            for d in docs
        ]

        # Monta contexto
        contexto = "\n\n".join(
            f"[{d['fonte']}, p.{d['pagina']}]\n{d['texto']}" for d in docs_resumidos
        )

        # Trunca contexto para não estourar o context window
        if len(contexto) > _MAX_CHARS_CONTEXTO:
            contexto = contexto[:_MAX_CHARS_CONTEXTO] + "\n[...contexto truncado...]"

        # Extrai fontes únicas para o prompt
        fontes_unicas = {
            (d["fonte"], d["url_fonte"]) for d in docs_resumidos if d["fonte"]
        }
        nome_fonte = ", ".join(f for f, _ in fontes_unicas)
        url_fonte = ", ".join(u for _, u in fontes_unicas if u)

        # Monta prompt e invoca LLM
        prompt = template_prompt.format(
            contexto=contexto,
            sintomas=sintomas,
            nome_fonte=nome_fonte,
            url_fonte=url_fonte,
        )

        salvar_prompt("classificacao", prompt, f"{len(docs)} chunks, {len(contexto)} chars de contexto")

        resposta = modelo.invoke(prompt)
        conteudo = resposta.content if hasattr(resposta, "content") else str(resposta)

        # Parse JSON
        try:
            resultado = json.loads(conteudo)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", conteudo, re.DOTALL)
            if match:
                resultado = json.loads(match.group())
            else:
                resultado = {"doencas_suspeitas": [], "fontes": []}

        # Ordena por score decrescente (campo novo; fallback 50 se ausente)
        suspeitas_raw = resultado.get("doencas_suspeitas", [])
        suspeitas_raw.sort(key=lambda d: d.get("score", 50), reverse=True)

        # Atualiza estado
        doencas = [d["doenca"] for d in suspeitas_raw]
        gravidade = {d["doenca"]: d["gravidade"] for d in suspeitas_raw}
        scores = {d["doenca"]: d.get("score", 50) for d in suspeitas_raw}
        justificativa = "\n".join(
            f"- {d['doenca']} (score {d.get('score', '?')}/100): {d.get('justificativa', '')}"
            for d in suspeitas_raw
        )

        # Filtra fontes: mantém apenas fontes cujo texto contém o nome de uma doença suspeita
        # Evita que fontes de dengue apareçam em resultados de COVID e vice-versa
        fontes_raw = resultado.get("fontes", [])
        if doencas:
            termos = [d.lower().replace("-", "").replace(" ", "") for d in doencas]
            fontes_filtradas = [
                f for f in fontes_raw
                if any(t in f.lower().replace("-", "").replace(" ", "") for t in termos)
            ]
            fontes_finais = fontes_filtradas if fontes_filtradas else fontes_raw
        else:
            fontes_finais = fontes_raw

        return {
            **estado,
            "documentos_recuperados": docs_resumidos,
            "doencas_suspeitas": doencas,
            "gravidade": gravidade,
            "scores": scores,
            "justificativa_classificacao": justificativa,
            "fontes": fontes_finais,
        }

    return no_classificacao
