"""Funções de gráficos com estilo único. Todas salvam PNG em alta resolução."""
import logging
from pathlib import Path

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_curve)

import config

logger = logging.getLogger(__name__)

TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUDO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
FUNDO = "#fcfcfb"
AZUL_SEQ = LinearSegmentedColormap.from_list("azul_seq", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])
DIVERGENTE = LinearSegmentedColormap.from_list("div", [config.COR_BOA, "#f0efec", config.COR_RUIM])


def aplicar_estilo():
    matplotlib.rcParams.update({
        "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "axes.edgecolor": EIXO, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 22, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUDO, "ytick.color": MUDO, "xtick.labelcolor": TINTA_2, "ytick.labelcolor": TINTA_2,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    })


def salvar(fig, pasta: Path, nome: str):
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{nome}.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    logger.info("Gráfico salvo: %s", caminho.relative_to(config.BASE_DIR))
    return caminho


def subtitulo(ax, texto):
    """Linha de contexto em cinza logo abaixo do título."""
    ax.text(0, 1.01, texto, transform=ax.transAxes, color=MUDO, fontsize=9, va="bottom")


# ---------------------------------------------------------------- avaliação do modelo

def matriz_confusao(y_true, y_pred, pasta, nome, titulo):
    cm = confusion_matrix(y_true, y_pred, labels=config.LABELS)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm_pct, cmap=AZUL_SEQ, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            cor = "white" if cm_pct[i, j] > 0.55 else TINTA
            ax.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]:.0%}", ha="center", va="center",
                    color=cor, fontsize=13, fontweight="semibold")
    ax.set_xticks([0, 1], [l.capitalize() for l in config.LABELS])
    ax.set_yticks([0, 1], [l.capitalize() for l in config.LABELS])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Verdadeiro")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(titulo)
    subtitulo(ax, "contagem e % por classe verdadeira (linha)")
    return fig, salvar(fig, pasta, nome)


def tabela_relatorio(report: dict, pasta, nome, titulo):
    """Renderiza o classification_report do sklearn como uma tabela em imagem."""
    linhas = config.LABELS + ["macro avg", "weighted avg"]
    nomes = {"boa": "Boa", "ruim": "Ruim", "macro avg": "Média macro", "weighted avg": "Média ponderada"}
    dados = [[nomes[l], f"{report[l]['precision']:.3f}", f"{report[l]['recall']:.3f}",
              f"{report[l]['f1-score']:.3f}", f"{int(report[l]['support'])}"] for l in linhas]
    dados.append(["Acurácia", "", "", f"{report['accuracy']:.3f}", f"{int(report['macro avg']['support'])}"])

    fig, ax = plt.subplots(figsize=(7, 2.9))
    ax.axis("off")
    tab = ax.table(cellText=dados, colLabels=["Classe", "Precisão", "Recall", "F1-score", "Suporte"],
                   cellLoc="center", loc="center", colWidths=[0.3, 0.17, 0.17, 0.17, 0.17])
    tab.auto_set_font_size(False)
    tab.set_fontsize(10.5)
    tab.scale(1, 1.7)
    for (r, c), cel in tab.get_celld().items():
        cel.set_edgecolor(GRADE)
        cel.set_facecolor(FUNDO)
        if r == 0:
            cel.set_text_props(color=TINTA_2, fontweight="semibold")
            cel.set_facecolor("#f0efec")
        elif c == 0:
            cel.set_text_props(color=TINTA, fontweight="semibold", ha="left")
            cel.PAD = 0.06
    ax.set_title(titulo)
    return fig, salvar(fig, pasta, nome)


def comparar_modelos(metricas: dict, pasta, nome="03_comparacao_modelos"):
    """metricas = {"Modelo A": {"Acurácia": x, ...}, ...} -> barras agrupadas."""
    modelos = list(metricas)
    nomes_metricas = list(next(iter(metricas.values())))
    cores = ["#6250d6", MUDO, "#1baf7a", "#eda100"]
    x = np.arange(len(nomes_metricas))
    larg = 0.8 / len(modelos)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, m in enumerate(modelos):
        vals = [metricas[m][k] for k in nomes_metricas]
        barras = ax.bar(x + (i - (len(modelos) - 1) / 2) * larg, vals, larg * 0.92,
                        color=cores[i % len(cores)], label=m, zorder=2)
        for b, v in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color=TINTA_2)
    ax.set_xticks(x, nomes_metricas)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Valor (conjunto de teste)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=len(modelos))
    ax.set_title("Comparação de modelos")
    subtitulo(ax, "T5 fine-tuned vs. baseline clássico (TF-IDF + Regressão Logística)")
    return fig, salvar(fig, pasta, nome)


def curvas_aprendizado(log_history: list, pasta, nome="04_curvas_aprendizado"):
    hist = pd.DataFrame(log_history)
    treino = hist[hist.get("loss").notna()] if "loss" in hist else pd.DataFrame()
    val = hist[hist["eval_loss"].notna()] if "eval_loss" in hist else pd.DataFrame()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    if not treino.empty:
        treino_ep = treino.groupby(treino["epoch"].apply(np.ceil))["loss"].mean()
        ax1.plot(treino_ep.index, treino_ep.values, color=MUDO, ls="--", label="Treino")
    ax1.plot(val["epoch"], val["eval_loss"], color="#6250d6", marker="o", ms=5, label="Validação")
    melhor = val.loc[val["eval_f1"].idxmax()] if "eval_f1" in val else None
    if melhor is not None:
        for ax in (ax1, ax2):
            ax.axvline(melhor["epoch"], color=EIXO, lw=1, zorder=0)
        ax1.annotate("melhor checkpoint", (melhor["epoch"], melhor["eval_loss"]),
                     xytext=(6, -18), textcoords="offset points", fontsize=8.5, color=TINTA_2)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax1.set_title("Loss por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    for col, rotulo, cor in [("eval_accuracy", "Acurácia", "#2a78d6"), ("eval_f1", "F1 (macro)", "#1baf7a"),
                             ("eval_recall", "Recall (macro)", "#eda100")]:
        if col in val:
            ax2.plot(val["epoch"], val[col], marker="o", ms=4, color=cor, label=rotulo)
    ax2.set_ylim(0, 1)
    ax2.set_title("Métricas de validação por época")
    ax2.set_xlabel("Época")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    return fig, salvar(fig, pasta, nome)


def roc_e_pr(y_true_bin, scores: dict, pasta, nome="05_curvas_roc_pr"):
    """scores = {"Modelo": prob_da_classe_positiva(ruim)}."""
    cores = ["#6250d6", MUDO]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for (modelo, s), cor in zip(scores.items(), cores):
        fpr, tpr, _ = roc_curve(y_true_bin, s)
        ax1.plot(fpr, tpr, color=cor, label=f"{modelo} (AUC = {auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(y_true_bin, s)
        ax2.plot(rec, prec, color=cor, label=f"{modelo} (AUC = {auc(rec, prec):.3f})")
    ax1.plot([0, 1], [0, 1], color=EIXO, ls=":", lw=1.2, label="Aleatório")
    ax1.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos", xlim=(0, 1), ylim=(0, 1.02))
    ax1.set_title("Curva ROC")
    ax1.legend(loc="lower right")
    ax2.axhline(np.mean(y_true_bin), color=EIXO, ls=":", lw=1.2, label="Aleatório")
    ax2.set(xlabel="Recall", ylabel="Precisão", xlim=(0, 1), ylim=(0, 1.02))
    ax2.set_title("Curva Precisão-Recall")
    ax2.legend(loc="lower left")
    for ax in (ax1, ax2):
        subtitulo(ax, "classe positiva = \"ruim\"")
    fig.tight_layout()
    return fig, salvar(fig, pasta, nome)


def histograma_confianca(confianca, acertos, pasta, nome="06_confianca_previsoes"):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.linspace(0.5, 1, 11)   # a classe prevista sempre tem prob. >= 0.5
    confianca, acertos = np.asarray(confianca), np.asarray(acertos)
    ax.hist([confianca[acertos], confianca[~acertos]], bins=bins, stacked=True,
            color=["#1baf7a", config.COR_RUIM], label=["Acertos", "Erros"],
            edgecolor=FUNDO, linewidth=2, zorder=2)
    ax.set_xlabel("Confiança do modelo (probabilidade da classe prevista)")
    ax.set_ylabel("Nº de posts (teste)")
    ax.grid(axis="x", visible=False)
    ax.legend()
    ax.set_title("Confiança das previsões")
    subtitulo(ax, "probabilidades calibradas (temperature scaling); erros concentrados em baixa confiança = bom sinal")
    return fig, salvar(fig, pasta, nome)


# ---------------------------------------------------------------- análise dos posts

def barras_sentimento_por_grupo(df_grupo: pd.DataFrame, pasta, nome, titulo, sub):
    """df_grupo: index = grupo, colunas 'boa','ruim' (contagens). Barras 100% horizontais."""
    df_grupo = df_grupo.copy()
    df_grupo["total"] = df_grupo["boa"] + df_grupo["ruim"]
    df_grupo["pct_ruim"] = df_grupo["ruim"] / df_grupo["total"]
    df_grupo = df_grupo.sort_values("pct_ruim")

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(df_grupo) + 1.6))
    y = np.arange(len(df_grupo))
    pct_boa = 1 - df_grupo["pct_ruim"]
    ax.barh(y, pct_boa, color=config.COR_BOA, height=0.68, label="Boa", zorder=2)
    ax.barh(y, df_grupo["pct_ruim"], left=pct_boa, color=config.COR_RUIM, height=0.68, label="Ruim", zorder=2)
    for yi, (pb, n) in enumerate(zip(pct_boa, df_grupo["total"])):
        ax.text(0.01, yi, f"{pb:.0%}", va="center", color="white", fontsize=8.5, fontweight="semibold", zorder=4)
        ax.text(0.99, yi, f"{1 - pb:.0%}", va="center", ha="right", color="white", fontsize=8.5,
                fontweight="semibold", zorder=4)
        ax.text(1.02, yi, f"n={n}", va="center", color=MUDO, fontsize=8.5)
    ax.set_yticks(y, df_grupo.index)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.06 if len(df_grupo) > 8 else -0.12), ncol=2)
    ax.set_title(titulo)
    subtitulo(ax, sub)
    return fig, salvar(fig, pasta, nome)


def barras_horizontais(serie: pd.Series, pasta, nome, titulo, sub, cor="#2a78d6", xlabel="Nº de posts"):
    serie = serie.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(serie) + 1.4))
    ax.barh(serie.index, serie.values, color=cor, height=0.7, zorder=2)
    for i, v in enumerate(serie.values):
        ax.text(v + serie.max() * 0.01, i, f"{v:,}".replace(",", "."), va="center", fontsize=8.5, color=TINTA_2)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(titulo)
    subtitulo(ax, sub)
    return fig, salvar(fig, pasta, nome)
