import unicodedata


DECISOES_VALIDAS = {"confirmar", "rejeitar", "pular"}


def _normalizar_texto(valor: str) -> str:
    texto = unicodedata.normalize("NFD", (valor or "").lower().strip())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def normalizar_decisao(valor: str | None) -> str:
    texto = _normalizar_texto(valor or "")
    aliases = {
        "confirmo": "confirmar",
        "rejeito": "rejeitar",
    }
    return aliases.get(texto, texto)


def _solicitar_decisao_cli() -> str:
    print("\n[confirmacao] Decisao medica: confirmar / rejeitar / pular")
    while True:
        entrada = input("Decisao medica: ").strip()
        decisao = normalizar_decisao(entrada)
        if decisao in DECISOES_VALIDAS:
            return decisao
        print("Valor invalido. Use: confirmar, rejeitar ou pular.")


def _solicitar_doenca_cli() -> str | None:
    entrada = input("Doenca confirmada: ").strip()
    return entrada or None


def _solicitar_resultado_exames_cli() -> str | None:
    entrada = input("Resultado dos exames (opcional): ").strip()
    return entrada or None


def criar_no_confirmacao():
    def no_confirmacao(estado: dict) -> dict:
        human_in_the_loop = bool(estado.get("human_in_the_loop", False))
        suspeitas = estado.get("doencas_suspeitas", [])
        resultado_exames = estado.get("resultado_exames")

        decisao = normalizar_decisao(estado.get("decisao_medica"))
        if decisao not in DECISOES_VALIDAS:
            decisao = _solicitar_decisao_cli() if human_in_the_loop else "rejeitar"

        doenca_confirmada = estado.get("doenca_confirmada")
        if decisao == "confirmar" and not doenca_confirmada and human_in_the_loop:
            print(f"[confirmacao] Suspeitas atuais: {suspeitas}")
            doenca_confirmada = _solicitar_doenca_cli()

        if (
            decisao in {"confirmar", "pular"}
            and resultado_exames is None
            and human_in_the_loop
        ):
            resultado_exames = _solicitar_resultado_exames_cli()

        if decisao == "rejeitar":
            return {
                **estado,
                "decisao_medica": "rejeitar",
                "doenca_confirmada": None,
                "resultado_exames": resultado_exames,
                "doencas_para_tratamento": [],
                "encerrar_sem_confirmacao": True,
            }

        if decisao == "confirmar":
            if not doenca_confirmada:
                return {
                    **estado,
                    "decisao_medica": "rejeitar",
                    "doenca_confirmada": None,
                    "resultado_exames": resultado_exames,
                    "doencas_para_tratamento": [],
                    "encerrar_sem_confirmacao": True,
                }
            return {
                **estado,
                "decisao_medica": "confirmar",
                "doenca_confirmada": doenca_confirmada,
                "resultado_exames": resultado_exames,
                "doencas_para_tratamento": [doenca_confirmada],
                "encerrar_sem_confirmacao": False,
            }

        # decisao == "pular": gerar tratamento para todas suspeitas
        if not suspeitas:
            return {
                **estado,
                "decisao_medica": "rejeitar",
                "doenca_confirmada": None,
                "resultado_exames": resultado_exames,
                "doencas_para_tratamento": [],
                "encerrar_sem_confirmacao": True,
            }

        return {
            **estado,
            "decisao_medica": "pular",
            "doenca_confirmada": None,
            "resultado_exames": resultado_exames,
            "doencas_para_tratamento": suspeitas,
            "encerrar_sem_confirmacao": False,
        }

    return no_confirmacao


def rotear_confirmacao(estado: dict) -> str:
    if estado.get("encerrar_sem_confirmacao"):
        return "encerrar"
    return "tratamento"
