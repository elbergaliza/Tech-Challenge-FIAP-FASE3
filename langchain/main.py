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

W = 55  # largura do separador


def _barra(char="─"):
    return char * W


def _score_bar(n: int, total: int) -> str:
    """Barra visual de 10 posições: n sintomas compatíveis de total.
    Se total <= 0, exibe apenas a contagem sem barra (sem divisão por zero).
    """
    if total <= 0:
        return f"{n} sintoma(s) compatível(is)"
    preenchido = round((n / total) * 10)
    return f"[{'█' * preenchido}{'░' * (10 - preenchido)}] {n} de {total}"


def _exibir_resultado_triagem(resultado: dict):
    doencas = resultado.get("doencas_suspeitas", [])
    gravidade = resultado.get("gravidade", {})
    scores = resultado.get("scores", {})
    justificativa = resultado.get("justificativa_classificacao", "")
    fontes = resultado.get("fontes", [])
    alerta = resultado.get("alerta")
    exames = resultado.get("exames_sugeridos", {})
    fontes_exames = resultado.get("fontes_exames", {})
    sintomas_compativeis = resultado.get("sintomas_compativeis", {})
    total_sintomas_protocolo = resultado.get("total_sintomas_protocolo", {})

    print("\n" + _barra("═"))
    print("  RESULTADO DA TRIAGEM")
    print(_barra("═"))

    # ── Alerta de urgência ────────────────────────────────────
    if alerta:
        print(f"\n  *** {alerta} ***\n")

    # ── Suspeitas com ranking ─────────────────────────────────
    if not doencas:
        print("\nNenhuma doença identificada com base nos sintomas informados.")
    else:
        print("\nSUSPEITAS DIAGNOSTICAS (ordenadas por probabilidade):\n")
        for i, doenca in enumerate(doencas, 1):
            n = len(sintomas_compativeis.get(doenca, []))
            total = total_sintomas_protocolo.get(doenca, 0)
            lista = ", ".join(sintomas_compativeis.get(doenca, []))
            grav = gravidade.get(doenca, "—")
            print(f"  {i}. {doenca.upper()}")
            print(f"     Sintomas      : {_score_bar(n, total)}")
            if lista:
                print(f"                   ({lista})")
            print(f"     Gravidade     : {grav}")

    # ── Justificativa ─────────────────────────────────────────
    if justificativa:
        print(f"\nJUSTIFICATIVA:\n")
        for linha in justificativa.splitlines():
            print(f"  {linha}")

    # ── Fontes da classificação ───────────────────────────────
    if fontes:
        print(f"\nFONTES DA CLASSIFICACAO:\n")
        for fonte in fontes:
            print(f"  • {fonte}")

    # ── Exames sugeridos ──────────────────────────────────────
    if exames:
        print(f"\nEXAMES SUGERIDOS:\n")
        for doenca, lista in exames.items():
            if not lista:
                print(f"  {doenca.upper()}: protocolo de referência não especifica exames laboratoriais — encaminhar para avaliação especializada")
                continue
            print(f"  {doenca.upper()}:")
            for exame in lista:
                print(f"    • {exame}")
            fontes_d = fontes_exames.get(doenca, [])
            if fontes_d:
                fontes_unicas = sorted(set(fontes_d))
                print(f"    Fontes: {', '.join(fontes_unicas)}")

    print("\n" + _barra())


def _exibir_resultado_final(resultado_final: dict):
    decisao = resultado_final.get("decisao_medica")
    encerrar = resultado_final.get("encerrar_sem_confirmacao")

    print()
    if encerrar:
        print("Confirmacao medica: rejeitada. Fluxo encerrado sem tratamento.")
        print(_barra())
        return

    if decisao:
        print(f"Decisao medica   : {decisao.upper()}")
        if resultado_final.get("doenca_confirmada"):
            print(f"Doenca confirmada: {resultado_final['doenca_confirmada'].upper()}")

    if decisao == "confirmar":
        tratamento = resultado_final.get("tratamento_sugerido", "")
        fontes_t = resultado_final.get("fontes_tratamento", [])
        print(f"\nTRATAMENTO SUGERIDO:\n")
        print(f"  {tratamento}")
        if fontes_t:
            print(f"\n  Fontes:")
            for f in fontes_t:
                print(f"    • {f}")

    if decisao == "pular":
        tratamentos = resultado_final.get("tratamento_por_suspeita", {})
        print(f"\nTRATAMENTOS POR SUSPEITA:\n")
        for doenca, trat in tratamentos.items():
            print(f"  {doenca.upper()}:")
            print(f"    {trat}")

    print(_barra())


def fazer_triagem_inicial(sintomas: str, fluxo) -> dict:
    """
    Executa classificacao/gravidade/exames e retorna o estado parcial.

    Parâmetros:
      sintomas - texto livre descrevendo sintomas do paciente
      fluxo    - grafo compilado retornado por montar_fluxo()

    Retorna:
      dicionário com: doencas_suspeitas, gravidade, justificativa, fontes
    """
    estado_inicial = {"sintomas": sintomas}
    return fluxo.invoke(estado_inicial)


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

    print("=" * W)
    print("  Assistente de Triagem Medica — Inicializando")
    print("=" * W)

    print("\n[1/3] Carregando o modelo de linguagem...")
    modelo = carregar_modelo()

    print("\n[2/3] Construindo indice de protocolos...")
    indice = construir_indice([CONFIG_DENGUE, CONFIG_COVID])

    print("\n[3/3] Montando o fluxo de triagem...")
    fluxo = montar_fluxo(modelo, indice, incluir_confirmacao_tratamento=False, configs=[CONFIG_DENGUE, CONFIG_COVID])
    no_confirmacao = criar_no_confirmacao()
    no_tratamento = criar_no_tratamento(indice, modelo)

    print("\n" + "=" * W)
    print("  Sistema pronto! Digite 'sair' para encerrar.")
    print("  Descreva os sintomas do paciente para classificacao.")
    print("=" * W + "\n")

    while True:
        sintomas = input("Sintomas: ").strip()

        if sintomas.lower() in ("sair", "exit", "quit"):
            print("Encerrando o assistente. Ate logo!")
            break

        if not sintomas:
            continue

        resultado = fazer_triagem_inicial(sintomas, fluxo)
        _exibir_resultado_triagem(resultado)

        resultado_final = aplicar_confirmacao_e_tratamento(
            resultado,
            {"human_in_the_loop": True},
            no_confirmacao,
            no_tratamento,
        )
        _exibir_resultado_final(resultado_final)


if __name__ == "__main__":
    main()
