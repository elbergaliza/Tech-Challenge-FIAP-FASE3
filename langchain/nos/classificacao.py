# nos/classificacao.py
# ─────────────────────────────────────────────────────────────
# Nó de classificação: recebe sintomas e retorna doenças suspeitas
# com gravidade, justificativa e fontes.
# ─────────────────────────────────────────────────────────────

import json
import re
from pathlib import Path
from banco_de_conhecimento import IndiceProtocolo, busca_hibrida


CAMINHO_PROMPT = Path(__file__).parent.parent / "prompts" / "classificacao.txt"


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

        # Busca sem filtro (classificação busca em tudo)
        docs = busca_hibrida(indice, sintomas)
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

        # Atualiza estado
        doencas = [d["doenca"] for d in resultado.get("doencas_suspeitas", [])]
        gravidade = {
            d["doenca"]: d["gravidade"] for d in resultado.get("doencas_suspeitas", [])
        }
        justificativa = "\n".join(
            f"- {d['doenca']}: {d.get('justificativa', '')}"
            for d in resultado.get("doencas_suspeitas", [])
        )

        print(f"[classificacao] Doenças suspeitas: {doencas}")
        print(f"[classificacao] Gravidade: {gravidade}")

        return {
            **estado,
            "documentos_recuperados": docs_resumidos,
            "doencas_suspeitas": doencas,
            "gravidade": gravidade,
            "justificativa_classificacao": justificativa,
            "fontes": resultado.get("fontes", []),
        }

    return no_classificacao
