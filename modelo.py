"""Carregamento do modelo treinado e inferência em lote."""
import json

import numpy as np
import torch
from transformers import T5ForSequenceClassification, T5Tokenizer

import config


def dispositivo() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def carregar_modelo(caminho=config.MODEL_DIR):
    if not (caminho / "config.json").exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em '{caminho}'. Rode 'python treinar.py' primeiro.")
    tokenizer = T5Tokenizer.from_pretrained(caminho)
    model = T5ForSequenceClassification.from_pretrained(caminho).to(dispositivo())
    model.eval()
    # Temperatura de calibração (ver treinar.calibrar_temperatura); 1.0 = sem calibração.
    arq = caminho / "calibracao.json"
    model.temperatura = json.loads(arq.read_text())["temperatura"] if arq.exists() else 1.0
    return tokenizer, model


@torch.inference_mode()
def prever_logits(textos, tokenizer, model, batch_size=64, progresso=False) -> np.ndarray:
    """Retorna array (n, 2) com os logits brutos [boa, ruim] de cada texto."""
    textos = list(textos)
    saidas = []
    iterador = range(0, len(textos), batch_size)
    if progresso:
        from tqdm import tqdm
        iterador = tqdm(iterador, desc="Classificando", unit="lote")
    for i in iterador:
        lote = [config.PREFIX + t for t in textos[i:i + batch_size]]
        enc = tokenizer(lote, max_length=config.MAX_LENGTH, truncation=True,
                        padding=True, return_tensors="pt").to(model.device)
        saidas.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(saidas) if saidas else np.empty((0, 2))


def softmax(logits, temperatura=1.0) -> np.ndarray:
    z = logits / temperatura
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def prever_proba(textos, tokenizer, model, batch_size=64, progresso=False) -> np.ndarray:
    """Retorna array (n, 2) com as probabilidades calibradas [boa, ruim] de cada texto."""
    logits = prever_logits(textos, tokenizer, model, batch_size, progresso)
    return softmax(logits, getattr(model, "temperatura", 1.0))
