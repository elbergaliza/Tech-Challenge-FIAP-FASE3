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
from configs.dengue import CONFIG_DENGUE


def fazer_triagem(sintomas: str, fluxo) -> dict:
    """
    Envia sintomas para o fluxo de triagem e retorna o resultado.

    Parâmetros:
      sintomas - texto livre descrevendo sintomas do paciente
      fluxo    - grafo compilado retornado por montar_fluxo()

    Retorna:
      dicionário com: doencas_suspeitas, gravidade, justificativa, fontes
    """
    estado_inicial = {"sintomas": sintomas}
    estado_final = fluxo.invoke(estado_inicial)
    return estado_final


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
    indice = construir_indice(CONFIG_DENGUE)

    # ── Passo 3: Monta o fluxo de triagem ─────────────────────
    print("\n[3/3] Montando o fluxo de triagem...")
    fluxo = montar_fluxo(modelo, indice)

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

        resultado = fazer_triagem(sintomas, fluxo)

        print("\n" + "─" * 55)
        print(f"Doenças suspeitas : {resultado.get('doencas_suspeitas', [])}")
        print(f"Gravidade         : {resultado.get('gravidade', {})}")
        print(f"\nJustificativa:\n{resultado.get('justificativa_classificacao', '')}")
        print(f"\nFontes: {resultado.get('fontes', [])}")
        if resultado.get("exames_sugeridos"):
            print(f"\nExames sugeridos : {resultado.get('exames_sugeridos', {})}")
            print(f"Fontes exames    : {resultado.get('fontes_exames', {})}")
        print("─" * 55 + "\n")


if __name__ == "__main__":
    main()
