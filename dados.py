"""Carregamento, limpeza e divisão dos dados."""
import logging
import re

import pandas as pd
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|gd|ly|br|nl|fr)(?:/\S*)?)", re.I)
_ESPACOS_RE = re.compile(r"\s+")

# Bots de "trending topics" que só listam #F1 entre dezenas de assuntos (ruído na análise).
_BOTS_RE = re.compile(r"^(trend|ttbrasil|speedrunbot)", re.I)
_SPAM_RE = re.compile(r"^\W*(tendências agora|assuntos dos últimos)|#nsfw", re.I)


def limpar_texto(texto) -> str:
    """Remove links e espaços extras. Hashtags e emojis são mantidos (carregam sentimento)."""
    texto = _URL_RE.sub(" ", str(texto))
    return _ESPACOS_RE.sub(" ", texto).strip()


def carregar_rotulados(caminho=config.DATA_FILE) -> pd.DataFrame:
    """Carrega o CSV rotulado, limpa, remove duplicatas e mapeia label -> id."""
    df = pd.read_csv(caminho, encoding="utf-8-sig")
    total = len(df)

    df = df.dropna(subset=["Texto_Traduzido_PT", "label"])
    df = df[df["label"].isin(config.LABELS)].copy()
    df["texto"] = df["Texto_Traduzido_PT"].map(limpar_texto)
    df = df[df["texto"].str.len() > 0]

    # Duplicatas podem cair uma no treino e outra no teste (vazamento de dados).
    df = df.drop_duplicates(subset="texto").reset_index(drop=True)
    df["label_id"] = df["label"].map(config.LABEL2ID)

    logger.info("Dados rotulados: %d linhas -> %d após limpeza/deduplicação", total, len(df))
    logger.info("Distribuição: %s", df["label"].value_counts().to_dict())
    return df


def dividir(df: pd.DataFrame):
    """Divide em treino / validação / teste estratificados."""
    df_treino_val, df_teste = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=df["label_id"])
    val_rel = config.VAL_SIZE / (1 - config.TEST_SIZE)
    df_treino, df_val = train_test_split(
        df_treino_val, test_size=val_rel, random_state=config.SEED, stratify=df_treino_val["label_id"])
    logger.info("Split: %d treino | %d validação | %d teste", len(df_treino), len(df_val), len(df_teste))
    return df_treino.reset_index(drop=True), df_val.reset_index(drop=True), df_teste.reset_index(drop=True)


def carregar_posts(caminho=config.POSTS_FILE) -> pd.DataFrame:
    """Carrega os posts não rotulados para análise em massa."""
    df = pd.read_csv(caminho, encoding="utf-8-sig")
    # Se a tradução falhou, usa o texto original.
    df["Texto_Traduzido_PT"] = df["Texto_Traduzido_PT"].fillna(df["Texto"])
    df = df.dropna(subset=["Texto_Traduzido_PT"]).copy()
    df["texto"] = df["Texto_Traduzido_PT"].map(limpar_texto)
    df = df[df["texto"].str.len() > 0]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.drop_duplicates(subset=["Autor", "texto"])
    spam = df["Autor"].str.match(_BOTS_RE) | df["texto"].str.contains(_SPAM_RE)
    df = df[~spam].reset_index(drop=True)
    logger.info("Posts para análise: %d (%d de bots/spam removidos)", len(df), spam.sum())
    return df
