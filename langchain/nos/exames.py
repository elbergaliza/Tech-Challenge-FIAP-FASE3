import json
import re
from pathlib import Path

from banco_de_conhecimento import IndiceProtocolo, busca_hibrida
from nos.debug import salvar_prompt


CAMINHO_PROMPT = Path(__file__).parent.parent / "prompts" / "exames.txt"

# Limite de caracteres do contexto para evitar context window overflow no modelo
# gemma3:4b tem ~8192 tokens; reservamos ~2000 para prompt+resposta → ~6000 chars de contexto
_MAX_CHARS_CONTEXTO = 4000

# Número de chunks recuperados — valor conservador para não estourar o context window
_K_EXAMES = 5


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
        print("[exames] Buscando exames recomendados...")
        doencas = estado.get("doencas_suspeitas", [])
        gravidades = estado.get("gravidade", {})

        exames_sugeridos = {}
        fontes_exames = {}
        justificativa_exames = {}

        for doenca in doencas:
            # Normaliza para bater com config.nome (ex: "Dengue" → "dengue", "COVID-19" → "covid")
            doenca_filtro = doenca.lower().split("-")[0].split("(")[0].strip()

            consulta = f"exames recomendados para {doenca} {gravidades.get(doenca, '')}"
            docs = busca_hibrida(
                indice,
                consulta,
                k=_K_EXAMES,
                doenca=doenca_filtro,
                secao_tipo="exames",
            )

            contexto = "\n\n".join(
                f"[{d.metadata.get('fonte', '')}, p.{d.metadata.get('pagina', '')}]\n{d.page_content}"
                for d in docs
            )

            # Trunca contexto para não exceder o context window do modelo
            if len(contexto) > _MAX_CHARS_CONTEXTO:
                contexto = contexto[:_MAX_CHARS_CONTEXTO] + "\n[...contexto truncado...]"

            contexto_final = contexto or "Sem contexto de exames recuperado."
            prompt = template.format(
                doenca=doenca,
                gravidade=gravidades.get(doenca, ""),
                contexto=contexto_final,
            )

            salvar_prompt(f"exames_{doenca_filtro}", prompt, f"{len(docs)} chunks, {len(contexto_final)} chars de contexto")

            try:
                resposta = modelo.invoke(prompt)
                conteudo = resposta.content if hasattr(resposta, "content") else str(resposta)
                payload = _parse_resposta_exames(conteudo)
            except Exception as e:
                print(f"[exames] Erro ao invocar modelo para '{doenca}': {e}")
                payload = {
                    "exames": [],
                    "justificativa": f"Erro ao consultar modelo: {e}",
                    "fontes": [],
                }

            exames_sugeridos[doenca] = payload.get("exames", [])
            fontes_exames[doenca] = payload.get("fontes", [])
            justificativa_exames[doenca] = payload.get("justificativa", "")

        return {
            **estado,
            "exames_sugeridos": exames_sugeridos,
            "fontes_exames": fontes_exames,
            "justificativa_exames": justificativa_exames,
        }

    return no_exames
