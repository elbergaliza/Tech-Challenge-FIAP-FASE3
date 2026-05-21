import json
import re
from pathlib import Path

from banco_de_conhecimento import IndiceProtocolo, busca_hibrida


CAMINHO_PROMPT = Path(__file__).parent.parent / "prompts" / "tratamento.txt"


def _parse_resposta_tratamento(conteudo: str) -> dict:
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
        "tratamento": "",
        "justificativa": "Resposta invalida do modelo.",
        "fontes": [],
    }


def _gerar_tratamento_para_doenca(
    indice: IndiceProtocolo,
    modelo,
    template: str,
    doenca: str,
    gravidade: str,
    resultado_exames: str,
) -> tuple[str, list[str], str]:
    consulta = f"tratamento recomendado para {doenca} {gravidade}"
    docs = busca_hibrida(
        indice,
        consulta,
        k=8,
        doenca=doenca,
        secao_tipo="tratamento",
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
        gravidade=gravidade,
        resultado_exames=resultado_exames,
        contexto=contexto or "Sem contexto de tratamento recuperado.",
    )
    resposta = modelo.invoke(prompt)
    conteudo = resposta.content if hasattr(resposta, "content") else str(resposta)
    
    payload = _parse_resposta_tratamento(conteudo)

    # ─── ADICIONE ESTE FALLBACK DE SEGURANÇA ───
    tratamento_final = payload.get("tratamento", "")
    
    # Se o modelo gerou um JSON válido mas se recusou a preencher o tratamento:
    if not tratamento_final.strip() and docs:
        print("⚠️ [Aviso] Modelo omitiu o tratamento. Gerando resumo técnico dos chunks automaticamente...")
        # Junta os trechos principais recuperados do protocolo para não deixar em branco
        tratamento_final = "Diretrizes extraídas do protocolo:\n" + "\n".join(
            f"- {d.page_content[:200]}..." for d in docs[:3]
        )
    # ───────────────────────────────────────────

    justificativa = payload.get("justificativa", "")
    if aviso_fallback:
        justificativa = f"{justificativa} [{aviso_fallback}]".strip()

    return (
        tratamento_final, # Retorna a variável tratada com o fallback
        payload.get("fontes", []),
        justificativa,
    )

    justificativa = payload.get("justificativa", "")
    if aviso_fallback:
        justificativa = f"{justificativa} [{aviso_fallback}]".strip()

    return (
        payload.get("tratamento", ""),
        payload.get("fontes", []),
        justificativa,
    )


def criar_no_tratamento(indice: IndiceProtocolo, modelo):
    template = CAMINHO_PROMPT.read_text(encoding="utf-8")

    def no_tratamento(estado: dict) -> dict:
        gravidades = estado.get("gravidade", {})
        resultado_exames = estado.get("resultado_exames") or ""
        doencas_para_tratamento = estado.get("doencas_para_tratamento", [])

        if not doencas_para_tratamento:
            return {
                **estado,
                "tratamento_sugerido": "",
                "fontes_tratamento": [],
                "justificativa_tratamento": "Sem doenca para tratamento.",
                "tratamento_por_suspeita": {},
                "fontes_tratamento_por_suspeita": {},
                "justificativa_tratamento_por_suspeita": {},
            }

        tratamento_por_suspeita = {}
        fontes_por_suspeita = {}
        justificativa_por_suspeita = {}

        for doenca in doencas_para_tratamento:
            tratamento, fontes, justificativa = _gerar_tratamento_para_doenca(
                indice=indice,
                modelo=modelo,
                template=template,
                doenca=doenca,
                gravidade=gravidades.get(doenca, ""),
                resultado_exames=resultado_exames,
            )
            tratamento_por_suspeita[doenca] = tratamento
            fontes_por_suspeita[doenca] = fontes
            justificativa_por_suspeita[doenca] = justificativa

        decisao_medica = estado.get("decisao_medica")
        tratamento_sugerido = ""
        fontes_tratamento = []
        justificativa_tratamento = ""
        if decisao_medica == "confirmar":
            doenca_confirmada = estado.get("doenca_confirmada")
            tratamento_sugerido = tratamento_por_suspeita.get(doenca_confirmada or "", "")
            fontes_tratamento = fontes_por_suspeita.get(doenca_confirmada or "", [])
            justificativa_tratamento = justificativa_por_suspeita.get(
                doenca_confirmada or "", ""
            )

        return {
            **estado,
            "tratamento_sugerido": tratamento_sugerido,
            "fontes_tratamento": fontes_tratamento,
            "justificativa_tratamento": justificativa_tratamento,
            "tratamento_por_suspeita": tratamento_por_suspeita,
            "fontes_tratamento_por_suspeita": fontes_por_suspeita,
            "justificativa_tratamento_por_suspeita": justificativa_por_suspeita,
        }

    return no_tratamento
