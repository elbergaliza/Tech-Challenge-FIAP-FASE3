# main.py
# ─────────────────────────────────────────────────────────────
# Ponto de entrada do sistema.
#
# Para rodar:
#   python main.py
#
# Ordem de execução:
#   1. Carrega o modelo (carregador_do_modelo.py)
#   2. Constrói o índice de protocolos (banco_de_conhecimento.py)
#   3. Monta o fluxo de triagem (fluxo.py)
#   4. Fica em loop esperando sintomas
# ─────────────────────────────────────────────────────────────

from carregador_do_modelo import carregar_modelo
from banco_de_conhecimento import construir_indice
from fluxo import montar_fluxo
from configs import CONFIG_DENGUE, CONFIG_COVID
from nos.confirmacao import criar_no_confirmacao
from nos.tratamento import criar_no_tratamento


def fazer_triagem_inicial(sintomas: str, fluxo) -> dict:
    """
    Executa classificacao/gravidade/exames e retorna o estado parcial.

    Parâmetros:
      sintomas - texto livre descrevendo sintomas do paciente
      fluxo    - grafo compilado retornado por montar_fluxo()

    Retorna:
      dicionário com: doencas_suspeitas, gravidade, justificativa, fontes
    """
    estado_inicial = {
        "sintomas": sintomas,
    }
    estado_final = fluxo.invoke(estado_inicial)
    return estado_final


def aplicar_confirmacao_e_tratamento(
    estado_triagem: dict,
    dados_confirmacao: dict,
    no_confirmacao,
    no_tratamento,
) -> dict:
    """Aplica confirmacao medica e, se necessario, gera tratamento."""
    estado = {**estado_triagem, **dados_confirmacao}
    estado = no_confirmacao(estado)

    if estado.get("encerrar_sem_confirmacao"):
        return estado

    return no_tratamento(estado)


def main():
    """Função principal: inicializa o sistema e inicia o loop."""

    print("=" * 55)
    print("  Assistente de Triagem Médica — Inicializando")
    print("=" * 55)

    # ── Passo 1: Carrega o modelo ─────────────────────────────
    print("\n[1/3] Carregando o modelo de linguagem...")
    modelo = carregar_modelo()

    # ── Passo 2: Constrói o índice de protocolos ──────────────
    print("\n[2/3] Construindo índice de protocolos...")
    indice = construir_indice([CONFIG_DENGUE, CONFIG_COVID])

    # ── Passo 3: Monta o fluxo de triagem ─────────────────────
    print("\n[3/3] Montando o fluxo de triagem...")
    fluxo = montar_fluxo(modelo, indice, incluir_confirmacao_tratamento=False)
    no_confirmacao = criar_no_confirmacao()
    no_tratamento = criar_no_tratamento(indice, modelo)

    print("\n" + "=" * 55)
    print("  Sistema pronto! Digite 'sair' para encerrar.")
    print("  Descreva os sintomas do paciente para classificação.")
    print("=" * 55 + "\n")

    # ── Loop de triagem ───────────────────────────────────────
    while True:
        sintomas = input("Sintomas: ").strip()

        if sintomas.lower() in ("sair", "exit", "quit"):
            print("Encerrando o assistente. Até logo!")
            break

        if not sintomas:
            continue

        resultado = fazer_triagem_inicial(sintomas, fluxo)

        print("\n" + "─" * 55)
        print(f"Doenças suspeitas : {resultado.get('doencas_suspeitas', [])}")
        print(f"Gravidade         : {resultado.get('gravidade', {})}")
        print(f"\nJustificativa:\n{resultado.get('justificativa_classificacao', '')}")
        print(f"\nFontes: {resultado.get('fontes', [])}")
        if resultado.get("exames_sugeridos"):
            print(f"\nExames sugeridos : {resultado.get('exames_sugeridos', {})}")
            print(f"Fontes exames    : {resultado.get('fontes_exames', {})}")

        resultado_final = aplicar_confirmacao_e_tratamento(
            resultado,
            {"human_in_the_loop": True},
            no_confirmacao,
            no_tratamento,
        )

        if resultado_final.get("encerrar_sem_confirmacao"):
            print("\nConfirmação médica: rejeitada. Fluxo encerrado sem tratamento.")
        elif resultado_final.get("decisao_medica"):
            print(f"\nDecisão médica   : {resultado_final.get('decisao_medica')}")
            if resultado_final.get("doenca_confirmada"):
                print(f"Doença confirmada: {resultado_final.get('doenca_confirmada')}")

        if resultado_final.get("decisao_medica") == "confirmar":
            print(f"\nTratamento sugerido:\n{resultado_final.get('tratamento_sugerido', '')}")
            print(f"Fontes tratamento : {resultado_final.get('fontes_tratamento', [])}")

        if resultado_final.get("decisao_medica") == "pular":
            print("\nTratamentos por suspeita:")
            print(f"{resultado_final.get('tratamento_por_suspeita', {})}")
        print("─" * 55 + "\n")


if __name__ == "__main__":
    main()
