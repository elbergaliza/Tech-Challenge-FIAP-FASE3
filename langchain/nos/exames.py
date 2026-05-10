import json
import re
from pathlib import Path

from banco_de_conhecimento import IndiceProtocolo, busca_hibrida


CAMINHO_PROMPT = Path(__file__).parent.parent / "prompts" / "exames.txt"


def _parse_resposta_exames(conteudo: str) -> dict:
    try:
        return json.loads(conteudo)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", conteudo, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {
        "exames": [],
        "justificativa": "Resposta invalida do modelo.",
        "fontes": [],
    }


def criar_no_exames(indice: IndiceProtocolo, modelo):
    template = CAMINHO_PROMPT.read_text(encoding="utf-8")

    def no_exames(estado: dict) -> dict:
        doencas = estado.get("doencas_suspeitas", [])
        gravidades = estado.get("gravidade", {})

        exames_sugeridos = {}
        fontes_exames = {}
        justificativa_exames = {}

        for doenca in doencas:
            consulta = f"exames recomendados para {doenca} {gravidades.get(doenca, '')}"
            docs = busca_hibrida(
                indice,
                consulta,
                k=8,
                doenca=doenca,
                secao_tipo="exames",
            )

            aviso_fallback = ""
            if not docs:
                docs = busca_hibrida(indice, consulta, k=8, doenca=doenca)
                aviso_fallback = "fallback_sem_secao"
            if not docs:
                docs = busca_hibrida(indice, consulta, k=8)
                aviso_fallback = "fallback_sem_doenca"

            contexto = "\n\n".join(
                f"[{d.metadata.get('fonte', '')}, p.{d.metadata.get('pagina', '')}]\n{d.page_content}"
                for d in docs
            )

            prompt = template.format(
                doenca=doenca,
                gravidade=gravidades.get(doenca, ""),
                contexto=contexto or "Sem contexto de exames recuperado.",
            )
            resposta = modelo.invoke(prompt)
            conteudo = resposta.content if hasattr(resposta, "content") else str(resposta)
            payload = _parse_resposta_exames(conteudo)

            justificativa = payload.get("justificativa", "")
            if aviso_fallback:
                justificativa = f"{justificativa} [{aviso_fallback}]".strip()

            exames_sugeridos[doenca] = payload.get("exames", [])
            fontes_exames[doenca] = payload.get("fontes", [])
            justificativa_exames[doenca] = justificativa

        return {
            **estado,
            "exames_sugeridos": exames_sugeridos,
            "fontes_exames": fontes_exames,
            "justificativa_exames": justificativa_exames,
        }

    return no_exames
