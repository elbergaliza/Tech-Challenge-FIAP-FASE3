# main.py
# ─────────────────────────────────────────────────────────────
# Ponto de entrada do sistema.
# É aqui que tudo se conecta e o assistente fica disponível.
#
# Para rodar:
#   python main.py
#
# Ordem de execução:
#   1. Carrega o modelo (carregador_do_modelo.py)
#   2. Constrói o banco de conhecimento RAG (banco_de_conhecimento.py)
#   3. Monta o fluxo com os agentes (fluxo.py)
#   4. Fica em loop esperando perguntas
# ─────────────────────────────────────────────────────────────

from carregador_do_modelo import carregar_modelo
from banco_de_conhecimento import obter_buscador
from fluxo import montar_fluxo


def fazer_pergunta(pergunta: str, fluxo) -> dict:
    """
    Envia uma pergunta para o fluxo e retorna o resultado completo.

    Parâmetros:
      pergunta - a pergunta do profissional de saúde (string)
      fluxo    - o fluxo compilado retornado por montar_fluxo()

    Retorna:
      dicionário com: pergunta, destino, resposta, agente_utilizado
    """

    # Estado inicial: só a pergunta está preenchida, o resto está vazio
    estado_inicial = {
        "pergunta":         pergunta,
        "destino":          "",   # será preenchido pelo supervisor
        "resposta":         "",   # será preenchido pelo agente escolhido
        "agente_utilizado": "",   # será preenchido pelo agente escolhido
    }

    # Executa o fluxo completo e retorna o estado final
    estado_final = fluxo.invoke(estado_inicial)
    return estado_final


def main():
    """
    Função principal: inicializa o sistema e inicia o loop de perguntas.
    """

    print("=" * 55)
    print("  Assistente Médico — Inicializando o sistema")
    print("=" * 55)

    # ── Passo 1: Carrega o modelo ─────────────────────────────
    print("\n[1/3] Carregando o modelo de linguagem...")
    modelo = carregar_modelo()

    # ── Passo 2: Carrega o banco de conhecimento ──────────────
    print("\n[2/3] Carregando o banco de conhecimento (RAG)...")
    buscador = obter_buscador()

    # ── Passo 3: Monta o fluxo com os agentes ────────────────
    print("\n[3/3] Montando o fluxo de agentes...")
    fluxo = montar_fluxo(modelo, buscador)

    print("\n" + "=" * 55)
    print("  Sistema pronto! Digite 'sair' para encerrar.")
    print("=" * 55 + "\n")

    # ── Loop de perguntas ─────────────────────────────────────
    while True:

        # Lê a pergunta do usuário
        pergunta = input("Pergunta: ").strip()

        # Encerra se o usuário digitar 'sair'
        if pergunta.lower() in ("sair", "exit", "quit"):
            print("Encerrando o assistente. Até logo!")
            break

        # Ignora linhas em branco
        if not pergunta:
            continue

        # Envia a pergunta para o fluxo e recebe a resposta
        resultado = fazer_pergunta(pergunta, fluxo)

        # Exibe o resultado formatado
        print("\n" + "─" * 55)
        print(f"Agente utilizado : {resultado['agente_utilizado'].upper()}")
        print(f"\nResposta:\n{resultado['resposta']}")
        print("─" * 55 + "\n")


# Só executa o main() se esse arquivo for rodado diretamente
# (não executa se for importado por outro arquivo)
if __name__ == "__main__":
    main()
