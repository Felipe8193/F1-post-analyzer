"""Classifica textos digitados no terminal usando o modelo treinado.

Uso:
  python teste.py                       # modo interativo (Enter vazio para sair)
  python teste.py "Que corrida incrível do Norris!"
"""
import sys

import config
from modelo import carregar_modelo, prever_proba

sys.stdout.reconfigure(encoding="utf-8")


def mostrar(texto, proba):
    idx = proba.argmax()
    rotulo = config.ID2LABEL[idx].capitalize()
    aviso = "  (baixa confiança – provavelmente neutro)" if proba[idx] < 0.6 else ""
    print(f"\n  Texto: {texto}")
    print(f"  Sentimento: {rotulo}  |  confiança {proba[idx]:.1%}{aviso}")
    print(f"  Boa: {proba[0]:.1%}   Ruim: {proba[1]:.1%}\n")


def main():
    print(f"Carregando modelo de '{config.MODEL_DIR.name}'...")
    try:
        tokenizer, model = carregar_modelo()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if len(sys.argv) > 1:
        texto = " ".join(sys.argv[1:])
        mostrar(texto, prever_proba([texto], tokenizer, model)[0])
        return

    while True:
        texto = input("Digite um post (Enter vazio para sair): ").strip()
        if not texto:
            break
        mostrar(texto, prever_proba([texto], tokenizer, model)[0])


if __name__ == "__main__":
    main()
