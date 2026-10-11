"""Classifica os ~10k posts coletados e analisa outros aspectos deles.

Aspectos analisados (gráficos em resultados/2_analise_posts/):
  01 distribuição geral de sentimento
  02 volume e sentimento ao longo do tempo (com os GPs marcados)
  03 sentimento por piloto mencionado
  04 sentimento por equipe mencionada
  05 sentimento por tópico (punições, acidentes, estratégia, mercado...)
  06 hashtags mais usadas
  07 palavras mais características de cada sentimento
  08 atividade por dia da semana x hora
  09 sentimento dos perfis que mais postam
  10 tamanho do post x sentimento

Uso:
  python analise_posts.py                  # usa a classificação em cache, se existir
  python analise_posts.py --reclassificar  # força rodar o modelo de novo
  python analise_posts.py --nao-mostrar
"""
import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import dados
import graficos
from graficos import MUDO, TINTA_2, subtitulo

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
PASTA = config.RESULTS_POSTS_DIR
CACHE = PASTA / "posts_classificados.csv"
LIMIAR_INCERTO = 0.60   # abaixo disso o modelo está "em dúvida" (posts neutros/informativos)

# --- Dicionários de entidades e tópicos (comparação sem acento, em minúsculas) ---
PILOTOS = {
    "Verstappen": ["verstappen"], "Norris": ["norris", "lando"], "Piastri": ["piastri"],
    "Leclerc": ["leclerc"], "Hamilton": ["hamilton"], "Russell": ["russell"],
    "Antonelli": ["antonelli", "antoneli"], "Alonso": ["alonso"], "Stroll": ["stroll"],
    "Gasly": ["gasly"], "Colapinto": ["colapinto"], "Ocon": ["ocon"], "Bearman": ["bearman"],
    "Albon": ["albon"], "Sainz": ["sainz"], "Tsunoda": ["tsunoda"], "Lawson": ["lawson"],
    "Hadjar": ["hadjar"], "Hülkenberg": ["hulkenberg"], "Bortoleto": ["bortoleto"],
}
EQUIPES = {
    "Red Bull": ["red bull", "redbull"], "McLaren": ["mclaren"], "Ferrari": ["ferrari", "scuderia"],
    "Mercedes": ["mercedes"], "Aston Martin": ["aston martin", "astonmartin"], "Alpine": ["alpine"],
    "Haas": ["haas"], "Williams": ["williams"], "Racing Bulls": ["racing bulls", "visa cash app", "vcarb"],
    "Sauber / Audi": ["sauber", "audi"],
}
TOPICOS = {
    "Punições e FIA": ["punicao", "punido", "penalidade", "penalizado", "comissarios", "fia", "investigacao",
                       "advertencia", "regulamento"],
    "Acidentes e abandonos": ["acidente", "batida", "bateu", "colisao", "crash", "abandono", "abandonou",
                              "dnf", "safety car", "bandeira vermelha", "rodou"],
    "Classificação (quali)": ["classificacao", "qualify", "quali", "pole", "q1", "q2", "q3"],
    "Estratégia e pneus": ["pneu", "pneus", "pit stop", "pitstop", "parada", "estrategia", "undercut", "box"],
    "Vitória e campeonato": ["vitoria", "venceu", "vence", "vencedor", "podio", "campeao", "campeonato",
                             "lideranca", "titulo", "pontos"],
    "Mercado e contratos": ["contrato", "assina", "assinou", "renovacao", "renova", "anuncia", "anunciou",
                            "substituto", "vaga", "2026"],
    "Chuva": ["chuva", "molhado", "molhada", "wet"],
    "História da F1": ["onthisday", "neste dia", "nesse dia", "senna", "schumacher", "prost"],
}
GPS = {"GP do México": "2025-10-26", "GP de São Paulo": "2025-11-09"}

STOPWORDS = set("""
a à ao aos as às até com como da das de dei do dos e é ela elas ele eles em entre era eram essa esse
esta está estão este eu foi for foram há isso isto já lhe mais mas me mesmo meu minha muito na nas não
nem no nos nós num numa o os ou para pela pelas pelo pelos por qual quando que quem se sem ser seu sua
são só também te tem têm ter teu tua tu um uma umas uns vai vão você vocês ser sobre seus suas aqui
agora ainda depois antes porque pra pro vez vai ja nao esta tambem ate apos após the of and to in is
f1 formula1 formula fórmula gp grande prêmio premio corrida 2025 1 2 3 vs x via ver diz após hoje
""".split())


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.lower()


def compilar(dicionario):
    return {nome: re.compile(r"\b(" + "|".join(re.escape(t) for t in termos) + r")\b")
            for nome, termos in dicionario.items()}


def marcar(df, dicionario, coluna):
    """Cria uma lista com as entidades mencionadas em cada post."""
    regex = compilar(dicionario)
    df[coluna] = df["texto_norm"].map(lambda t: [n for n, r in regex.items() if r.search(t)])
    return df


def contar_por_grupo(df, coluna, minimo):
    exp = df.explode(coluna).dropna(subset=[coluna])
    tab = pd.crosstab(exp[coluna], exp["sentimento"]).reindex(columns=config.LABELS, fill_value=0)
    return tab[tab.sum(axis=1) >= minimo]


# ---------------------------------------------------------------- classificação

def classificar(df, forcar):
    """Classifica os posts, reaproveitando o cache (por Link) se o modelo não mudou."""
    if CACHE.exists() and not forcar and CACHE.stat().st_mtime > (config.MODEL_DIR / "config.json").stat().st_mtime:
        cache = pd.read_csv(CACHE, encoding="utf-8-sig").drop_duplicates("Link").set_index("Link")
        if df["Link"].isin(cache.index).all():
            logger.info("Usando classificação em cache: %s", CACHE.name)
            df["sentimento"] = df["Link"].map(cache["sentimento"])
            df["confianca"] = df["Link"].map(cache["confianca"])
            return salvar_classificados(df)

    from modelo import carregar_modelo, prever_proba
    tokenizer, model = carregar_modelo()
    proba = prever_proba(df["texto"], tokenizer, model, progresso=True)
    df["sentimento"] = [config.ID2LABEL[i] for i in proba.argmax(1)]
    df["confianca"] = proba.max(1).round(4)
    return salvar_classificados(df)


def salvar_classificados(df):
    PASTA.mkdir(parents=True, exist_ok=True)
    df[["Timestamp", "Autor", "texto", "Link", "sentimento", "confianca"]].to_csv(
        CACHE, index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------- gráficos específicos

def g_distribuicao(df):
    cont = df["sentimento"].value_counts().reindex(config.LABELS)
    incertos = (df["confianca"] < LIMIAR_INCERTO).sum()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    esq = 0
    for lab in config.LABELS:
        v = cont[lab]
        ax.barh(0, v, left=esq, color=config.CORES_LABELS[lab], height=0.55)
        ax.text(esq + v / 2, 0, f"{lab.capitalize()}\n{v:,} ({v / len(df):.0%})".replace(",", "."),
                ha="center", va="center", color="white", fontweight="semibold")
        esq += v + len(df) * 0.004
    ax.axis("off")
    ax.set_title(f"Sentimento de {len(df):,} posts sobre F1".replace(",", "."))
    subtitulo(ax, f"{incertos:,} posts ({incertos / len(df):.0%}) têm confiança < {LIMIAR_INCERTO:.0%} "
                  "(geralmente notícias neutras)".replace(",", "."))
    return fig, graficos.salvar(fig, PASTA, "01_distribuicao_sentimento")


def g_tempo(df):
    d = df.dropna(subset=["Timestamp"]).copy()
    d["dia"] = d["Timestamp"].dt.tz_convert("America/Sao_Paulo").dt.date
    tab = pd.crosstab(d["dia"], d["sentimento"]).reindex(columns=config.LABELS, fill_value=0)
    tab.index = pd.to_datetime(tab.index)
    pct_ruim = tab["ruim"] / tab.sum(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax1.bar(tab.index, tab["boa"], color=config.COR_BOA, width=0.8, label="Boa", zorder=2)
    ax1.bar(tab.index, tab["ruim"], bottom=tab["boa"], color=config.COR_RUIM, width=0.8, label="Ruim", zorder=2)
    ax1.set_ylabel("Posts por dia")
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper center", ncol=2)
    ax1.set_title("Volume e sentimento ao longo do tempo")
    subtitulo(ax1, "fuso de Brasília; linhas verticais = domingos de GP")

    ax2.plot(pct_ruim.index, pct_ruim.values, color=config.COR_RUIM, marker="o", ms=4)
    ax2.axhline(pct_ruim.mean(), color=MUDO, ls=":", lw=1.2)
    ax2.text(pct_ruim.index[-1], pct_ruim.mean() + 0.10, f"média {pct_ruim.mean():.0%}",
             color=MUDO, fontsize=8.5, ha="right")
    ax2.set_ylabel("% de posts \"ruim\"")
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax2.set_ylim(0, 1)
    for nome, data in GPS.items():
        dt = pd.Timestamp(data)
        if tab.index.min() <= dt <= tab.index.max():
            for ax in (ax1, ax2):
                ax.axvline(dt, color=graficos.EIXO, lw=1, zorder=0)
            ax1.text(dt, ax1.get_ylim()[1] * 0.97, f" {nome}", color=TINTA_2, fontsize=8.5, va="top")
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m"))
    ax2.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=2))
    fig.tight_layout()
    return fig, graficos.salvar(fig, PASTA, "02_sentimento_ao_longo_do_tempo")


def g_palavras(df, n=15):
    """Palavras mais características de cada classe (log-odds com suavização)."""
    tok = re.compile(r"[#@]?[a-zà-ú]{3,}")
    cont = {lab: Counter() for lab in config.LABELS}
    for texto, lab in zip(df["texto"].str.lower(), df["sentimento"]):
        cont[lab].update(w for w in set(tok.findall(texto)) if w not in STOPWORDS and not w.startswith("@"))
    vocab = [w for w in set(cont["boa"]) | set(cont["ruim"]) if cont["boa"][w] + cont["ruim"][w] >= 15]
    tot = {lab: sum(cont[lab].values()) for lab in config.LABELS}
    lo = pd.Series({w: np.log((cont["ruim"][w] + 1) / tot["ruim"]) - np.log((cont["boa"][w] + 1) / tot["boa"])
                    for w in vocab})

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, lab, serie in [(axes[0], "boa", -lo.nsmallest(n)), (axes[1], "ruim", lo.nlargest(n))]:
        serie = serie.sort_values()
        ax.barh(serie.index, serie.values, color=config.CORES_LABELS[lab], height=0.7, zorder=2)
        ax.set_title(f"Mais associadas a \"{lab}\"")
        ax.set_xlabel("Força da associação (log-odds)")
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    subtitulo(axes[0], "palavras com ≥15 ocorrências; stopwords removidas")
    fig.tight_layout()
    return fig, graficos.salvar(fig, PASTA, "07_palavras_por_sentimento")


def g_atividade(df):
    d = df.dropna(subset=["Timestamp"])
    local = d["Timestamp"].dt.tz_convert("America/Sao_Paulo")
    tab = pd.crosstab(local.dt.dayofweek, local.dt.hour).reindex(index=range(7), columns=range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(tab.values, cmap=graficos.AZUL_SEQ, aspect="auto")
    ax.grid(False)
    ax.set_yticks(range(7), ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"])
    ax.set_xticks(range(0, 24, 2), [f"{h}h" for h in range(0, 24, 2)])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.outline.set_visible(False)
    cb.set_label("Nº de posts", color=TINTA_2)
    ax.set_title("Quando os fãs postam sobre F1")
    subtitulo(ax, "dia da semana x hora (Brasília)")
    return fig, graficos.salvar(fig, PASTA, "08_atividade_dia_hora")


def g_tamanho(df):
    d = df.assign(tam=df["texto"].str.len())
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.arange(0, 330, 20)
    for lab in config.LABELS:
        ax.hist(d.loc[d["sentimento"] == lab, "tam"], bins=bins, density=True, histtype="step", lw=2,
                color=config.CORES_LABELS[lab], label=f"{lab.capitalize()} (mediana {int(d.loc[d['sentimento'] == lab, 'tam'].median())} caract.)")
    ax.set_xlabel("Tamanho do post (caracteres, sem links)")
    ax.set_ylabel("Densidade")
    ax.set_yticks([])
    ax.legend()
    ax.set_title("Tamanho do post x sentimento")
    return fig, graficos.salvar(fig, PASTA, "10_tamanho_post")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reclassificar", action="store_true")
    ap.add_argument("--nao-mostrar", action="store_true")
    a = ap.parse_args()
    graficos.aplicar_estilo()

    df = classificar(dados.carregar_posts(), a.reclassificar)
    df["texto_norm"] = df["texto"].map(normalizar)
    df = marcar(df, PILOTOS, "pilotos")
    df = marcar(df, EQUIPES, "equipes")
    df = marcar(df, TOPICOS, "topicos")
    df["hashtags"] = df["Texto_Traduzido_PT"].str.findall(r"#\w+").map(
        lambda hs: sorted({h.lower() for h in hs} - {"#f1", "#formula1", "#formulaone", "#fórmula1"}))

    tab_pilotos = contar_por_grupo(df, "pilotos", minimo=30)
    tab_equipes = contar_por_grupo(df, "equipes", minimo=30)
    tab_topicos = contar_por_grupo(df, "topicos", minimo=20)
    hashtags = df.explode("hashtags")["hashtags"].value_counts().head(20)
    autores = df["Autor"].value_counts().head(12).index
    tab_autores = pd.crosstab(df.loc[df["Autor"].isin(autores), "Autor"],
                              df["sentimento"]).reindex(columns=config.LABELS, fill_value=0)

    g_distribuicao(df)
    g_tempo(df)
    graficos.barras_sentimento_por_grupo(tab_pilotos, PASTA, "03_sentimento_por_piloto",
                                         "Sentimento por piloto mencionado", "pilotos com ≥30 menções; ordenado por % ruim")
    graficos.barras_sentimento_por_grupo(tab_equipes, PASTA, "04_sentimento_por_equipe",
                                         "Sentimento por equipe mencionada", "equipes com ≥30 menções; ordenado por % ruim")
    graficos.barras_sentimento_por_grupo(tab_topicos, PASTA, "05_sentimento_por_topico",
                                         "Sentimento por tópico", "tópicos detectados por palavras-chave; um post pode ter vários")
    graficos.barras_horizontais(hashtags, PASTA, "06_top_hashtags", "Hashtags mais usadas",
                                "excluindo #f1 e #formula1")
    g_palavras(df)
    g_atividade(df)
    graficos.barras_sentimento_por_grupo(tab_autores, PASTA, "09_sentimento_por_autor",
                                         "Perfis que mais postam", "12 autores com mais posts")
    g_tamanho(df)

    # Tabelas de apoio
    for nome, tab in [("pilotos", tab_pilotos), ("equipes", tab_equipes), ("topicos", tab_topicos)]:
        t = tab.assign(total=tab.sum(axis=1), pct_ruim=(tab["ruim"] / tab.sum(axis=1)).round(3))
        t.sort_values("total", ascending=False).to_csv(PASTA / f"tabela_{nome}.csv", encoding="utf-8-sig")

    resumo = {
        "total_posts": len(df),
        "pct_boa": round(float((df["sentimento"] == "boa").mean()), 4),
        "pct_ruim": round(float((df["sentimento"] == "ruim").mean()), 4),
        "pct_incertos": round(float((df["confianca"] < LIMIAR_INCERTO).mean()), 4),
        "periodo": [str(df["Timestamp"].min()), str(df["Timestamp"].max())],
        "piloto_mais_citado": tab_pilotos.sum(axis=1).idxmax(),
        "equipe_mais_citada": tab_equipes.sum(axis=1).idxmax(),
    }
    (PASTA / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n========== RESUMO DA ANÁLISE ==========")
    for k, v in resumo.items():
        print(f"{k:>20}: {v}")
    print(f"\nGráficos e tabelas em {PASTA.relative_to(config.BASE_DIR)}")

    if a.nao_mostrar:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
