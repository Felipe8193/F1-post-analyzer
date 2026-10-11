"""Treina e avalia o classificador de sentimento (T5) para posts de F1.

Gera em resultados/1_modelo/:
  - matriz de confusão, tabela do classification report, comparação com baseline,
    curvas de aprendizado, curvas ROC/PR, confiança das previsões
  - metricas.json, classification_report.csv, erros_teste.csv, historico_treino.csv

Uso:
  python treinar.py                         # treina do zero e avalia
  python treinar.py --so-avaliar            # só avalia o modelo já salvo
  python treinar.py --modelo-base t5_f1_sentiment_model --dados novos.csv   # continua o treino
  python treinar.py --nao-mostrar           # salva os gráficos sem abrir janelas
"""
import argparse
import json
import logging
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
from sklearn.metrics import (accuracy_score, classification_report, f1_score, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from transformers import (DataCollatorWithPadding, EarlyStoppingCallback,
                          T5ForSequenceClassification, T5Tokenizer, Trainer,
                          TrainingArguments, set_seed)

import config
import dados
import graficos
from modelo import carregar_modelo, dispositivo, prever_logits, prever_proba, softmax

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
PASTA = config.RESULTS_MODEL_DIR


# ---------------------------------------------------------------- métricas

def calcular_metricas(y_true, y_pred, proba_ruim=None) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    m = {"Acurácia": accuracy_score(y_true, y_pred), "Precisão": p, "Recall": r, "F1": f}
    if proba_ruim is not None:
        m["ROC AUC"] = roc_auc_score(y_true, proba_ruim)
    return {k: float(v) for k, v in m.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f}


def apenas_logits(logits, labels):
    # T5ForSequenceClassification devolve (logits, hidden_states...); guardar só os logits economiza memória.
    return logits[0] if isinstance(logits, tuple) else logits


# ---------------------------------------------------------------- baseline

def treinar_baseline(df_treino, df_todos):
    """TF-IDF + Regressão Logística: referência rápida para saber se o T5 agrega valor."""
    def novo():
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode"),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))

    pipe = novo().fit(df_treino["texto"], df_treino["label_id"])

    # Validação cruzada (5 folds) em todo o dataset: mostra a variação esperada com poucos dados.
    cv = cross_validate(novo(), df_todos["texto"], df_todos["label_id"],
                        cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED),
                        scoring=["accuracy", "f1_macro"])
    cv_res = {k: {"media": float(cv[f"test_{k}"].mean()), "desvio": float(cv[f"test_{k}"].std())}
              for k in ["accuracy", "f1_macro"]}
    logger.info("Baseline CV 5-fold: acurácia %.3f ± %.3f | F1 %.3f ± %.3f",
                cv_res["accuracy"]["media"], cv_res["accuracy"]["desvio"],
                cv_res["f1_macro"]["media"], cv_res["f1_macro"]["desvio"])
    return pipe, cv_res


# ---------------------------------------------------------------- T5

def treinar_t5(df_treino, df_val, modelo_base, epocas):
    set_seed(config.SEED)
    logger.info("Carregando modelo base: %s (dispositivo: %s)", modelo_base, dispositivo())
    tokenizer = T5Tokenizer.from_pretrained(modelo_base)
    model = T5ForSequenceClassification.from_pretrained(
        modelo_base, num_labels=len(config.LABELS), id2label=config.ID2LABEL,
        label2id=config.LABEL2ID, ignore_mismatched_sizes=True)

    def tokenizar(df):
        ds = Dataset.from_dict({"texto": df["texto"].tolist(), "labels": df["label_id"].tolist()})
        return ds.map(lambda b: tokenizer([config.PREFIX + t for t in b["texto"]],
                                          max_length=config.MAX_LENGTH, truncation=True),
                      batched=True, remove_columns=["texto"])

    ds_treino, ds_val = tokenizar(df_treino), tokenizar(df_val)

    args = TrainingArguments(
        output_dir=str(config.CHECKPOINT_DIR),
        num_train_epochs=epocas,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,                 # evita gigabytes de checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="f1",         # escolhe pelo F1, não só pela loss
        greater_is_better=True,
        logging_strategy="epoch",
        fp16=torch.cuda.is_available(),
        seed=config.SEED,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds_treino, eval_dataset=ds_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),   # padding dinâmico = treino mais rápido
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=apenas_logits,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
    )
    logger.info("--- Iniciando fine-tuning (%d treino / %d validação) ---", len(ds_treino), len(ds_val))
    trainer.train()

    trainer.save_model(str(config.MODEL_DIR))
    tokenizer.save_pretrained(str(config.MODEL_DIR))
    historico = trainer.state.log_history
    (config.MODEL_DIR / "historico_treino.json").write_text(json.dumps(historico, indent=2), encoding="utf-8")
    shutil.rmtree(config.CHECKPOINT_DIR, ignore_errors=True)
    logger.info("Modelo salvo em %s", config.MODEL_DIR)
    return historico


# ---------------------------------------------------------------- avaliação

def calibrar_temperatura(df_val, tokenizer, model):
    """Temperature scaling: ajusta T na validação para que as probabilidades reflitam
    a chance real de acerto. Não altera as previsões (argmax), só a confiança."""
    logits = prever_logits(df_val["texto"], tokenizer, model)
    y = df_val["label_id"].to_numpy()

    def nll(log_t):
        return log_loss(y, softmax(logits, np.exp(log_t)), labels=[0, 1])

    res = minimize_scalar(nll, bounds=(-3, 5), method="bounded")
    t = float(np.exp(res.x))
    (config.MODEL_DIR / "calibracao.json").write_text(json.dumps({"temperatura": t}), encoding="utf-8")
    model.temperatura = t
    logger.info("Calibração: temperatura %.2f | log-loss validação %.3f -> %.3f", t, nll(0.0), res.fun)
    return {"temperatura": t, "log_loss_val_antes": float(nll(0.0)), "log_loss_val_depois": float(res.fun)}


def avaliar(df_treino, df_val, df_teste, baseline, cv_baseline, historico):
    tokenizer, model = carregar_modelo()
    calibracao = calibrar_temperatura(df_val, tokenizer, model)
    proba = prever_proba(df_teste["texto"], tokenizer, model)
    y_true = df_teste["label_id"].to_numpy()
    y_pred = proba.argmax(1)
    proba_base = baseline.predict_proba(df_teste["texto"])
    y_pred_base = proba_base.argmax(1)

    nomes_true = [config.ID2LABEL[i] for i in y_true]
    nomes_pred = [config.ID2LABEL[i] for i in y_pred]
    report = classification_report(nomes_true, nomes_pred, labels=config.LABELS, output_dict=True, zero_division=0)
    print("\n" + classification_report(nomes_true, nomes_pred, labels=config.LABELS, digits=3, zero_division=0))

    met_t5 = calcular_metricas(y_true, y_pred, proba[:, 1])
    met_base = calcular_metricas(y_true, y_pred_base, proba_base[:, 1])

    # Overfitting: compara desempenho no treino x teste
    proba_treino = prever_proba(df_treino["texto"], tokenizer, model)
    f1_treino = f1_score(df_treino["label_id"], proba_treino.argmax(1), average="macro")

    figs = []
    figs.append(graficos.matriz_confusao(nomes_true, nomes_pred, PASTA, "01_matriz_confusao",
                                         "Matriz de confusão – T5 (teste)")[0])
    figs.append(graficos.tabela_relatorio(report, PASTA, "02_relatorio_classificacao",
                                          "Relatório de classificação – T5 (teste)")[0])
    figs.append(graficos.comparar_modelos({"T5 (PTT5-small)": met_t5, "TF-IDF + Reg. Logística": met_base}, PASTA)[0])
    if historico:
        figs.append(graficos.curvas_aprendizado(historico, PASTA)[0])
        pd.DataFrame(historico).to_csv(PASTA / "historico_treino.csv", index=False)
    figs.append(graficos.roc_e_pr(y_true, {"T5": proba[:, 1], "Baseline": proba_base[:, 1]}, PASTA)[0])
    figs.append(graficos.histograma_confianca(proba.max(1), y_pred == y_true, PASTA)[0])
    figs.append(graficos.matriz_confusao([config.ID2LABEL[i] for i in y_true],
                                         [config.ID2LABEL[i] for i in y_pred_base], PASTA,
                                         "07_matriz_confusao_baseline", "Matriz de confusão – Baseline (teste)")[0])

    # Tabelas para inspeção
    pd.DataFrame(report).T.round(4).to_csv(PASTA / "classification_report.csv")
    erros = df_teste.assign(previsto=nomes_pred, confianca=proba.max(1).round(3))
    erros = erros[erros["previsto"] != erros["label"]][["texto", "label", "previsto", "confianca", "Link"]]
    erros.sort_values("confianca", ascending=False).to_csv(PASTA / "erros_teste.csv", index=False, encoding="utf-8-sig")

    resumo = {
        "amostras": {"treino": len(df_treino), "teste": len(df_teste)},
        "t5_teste": met_t5,
        "t5_f1_treino": float(f1_treino),
        "baseline_teste": met_base,
        "baseline_cv_5fold": cv_baseline,
        "melhor_epoca": _melhor_epoca(historico),
        "calibracao": calibracao,
    }
    (PASTA / "metricas.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ RESUMO (conjunto de teste) ================")
    print(pd.DataFrame({"T5": met_t5, "Baseline": met_base}).round(3).to_string())
    print(f"\nF1 no treino: {f1_treino:.3f}  |  F1 no teste: {met_t5['F1']:.3f}  (diferença grande = overfitting)")
    print(f"Erros salvos em {(PASTA / 'erros_teste.csv').relative_to(config.BASE_DIR)}")
    return figs


def _melhor_epoca(historico):
    evals = [h for h in historico or [] if "eval_f1" in h]
    return max(evals, key=lambda h: h["eval_f1"])["epoch"] if evals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", type=Path, default=config.DATA_FILE)
    ap.add_argument("--modelo-base", default=config.MODEL_NAME,
                    help="modelo do HuggingFace ou pasta local (para continuar o treino)")
    ap.add_argument("--epocas", type=int, default=config.EPOCHS)
    ap.add_argument("--so-avaliar", action="store_true", help="não treina; avalia o modelo salvo")
    ap.add_argument("--nao-mostrar", action="store_true", help="não abre as janelas dos gráficos")
    a = ap.parse_args()

    graficos.aplicar_estilo()
    df = dados.carregar_rotulados(a.dados)
    df_treino, df_val, df_teste = dados.dividir(df)

    baseline, cv_baseline = treinar_baseline(df_treino, df)

    if a.so_avaliar:
        arq = config.MODEL_DIR / "historico_treino.json"
        historico = json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else None
    else:
        historico = treinar_t5(df_treino, df_val, a.modelo_base, a.epocas)

    avaliar(df_treino, df_val, df_teste, baseline, cv_baseline, historico)
    logger.info("Resultados salvos em %s", PASTA)

    if a.nao_mostrar:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
