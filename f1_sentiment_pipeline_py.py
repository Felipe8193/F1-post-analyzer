

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from transformers import (
    T5ForSequenceClassification,
    T5Tokenizer,
    TrainingArguments,
    Trainer
)
import torch
import numpy as np
import warnings
import logging


# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suprimir avisos de datasets
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configurações ---
MODEL_NAME = "unicamp-dl/ptt5-small-portuguese-vocab"
DATA_FILE = "f1_posts_rotulados_balanceado.csv"
MODEL_OUTPUT_DIR = "./t5_f1_sentiment_model"
LABELS = ["boa", "ruim"] # 0 = boa, 1 = ruim
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}
# ---------------------

def load_and_preprocess_data(data_file):
    """
    Carrega o CSV rotulado E CONVERTE labels de texto para inteiros.
    """

    logger.info(f"Carregando dados do arquivo: {data_file}")
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        logger.error(f"Arquivo '{data_file}' não encontrado.")
        logger.error("Você executou a Etapa 3 (Rotulagem Manual) e salvou o arquivo?")
        return None

    # Limpeza básica

    df = df.dropna(subset=['Texto_Traduzido_PT', 'label'])
    logger.info(f"Após remoção de NaNs, {len(df)} amostras restantes.")

    # Filtra para garantir que só temos as labels esperadas
    df = df[df['label'].isin(LABELS)]
    logger.info(f"Após filtro de labels válidas, {len(df)} amostras restantes.")

    if len(df) < 20:
        logger.error("Dados insuficientes para treinar. Rotule mais posts.")
        return None


    # --- CORREÇÃO CRÍTICA ---
    # Mapeia "boa" -> 0 e "ruim" -> 1
    df['label'] = df['label'].map(LABEL_MAP)
    # --- FIM DA CORREÇÃO ---

    logger.info(f"Dados carregados. Total de amostras rotuladas: {len(df)}")
    logger.info(f"Contagem de labels (0=boa, 1=ruim):\n{df['label'].value_counts()}")

    # Dividir em treino e teste (80% treino, 20% teste)
    # Stratify usa a coluna 'label' numérica

    df_train, df_test = train_test_split(df, test_size=0.25, random_state=42, stratify=df['label'])
    logger.info(f"Split: {len(df_train)} treino, {len(df_test)} teste.")

    # Converter para o formato Dataset do Hugging Face
    dataset = DatasetDict({
        'train': Dataset.from_pandas(df_train, preserve_index=False),
        'test': Dataset.from_pandas(df_test, preserve_index=False)
    })

    logger.info("Dados convertidos para HuggingFace DatasetDict.")
    return dataset

def tokenize_data(dataset, tokenizer):
    """
    Prepara os dados para o T5 para Classificação.
    Os rótulos (labels) NÃO são tokenizados.
    """
    prefix = "classifique o sentimento: "

    def preprocess_function(examples):
        # Tokeniza a entrada de texto
        inputs = [prefix + doc for doc in examples["Texto_Traduzido_PT"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

        # --- CORREÇÃO CRÍTICA ---
        # Passa o rótulo numérico (0 ou 1) diretamente
        model_inputs["labels"] = examples["label"]
        # --- FIM DA CORREÇÃO ---
        return model_inputs

    logger.info("Tokenizando dados para classificação...")
    # Remove colunas desnecessárias APENAS para o map.
    # 'text' e 'label' são necessários para a preprocess_function.
    # Correção do 'KeyError'
    tokenized_datasets = dataset.remove_columns(["Timestamp", "Autor", "Link"]).map(preprocess_function, batched=True)

    return tokenized_datasets

def plot_confusion_matrix(y_true, y_pred, title, labels_for_plot):
    """
    Plota a matriz de confusão.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels_for_plot)
    logger.info(f"Matriz de confusão para {title}:\n{cm}")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_for_plot, yticklabels=labels_for_plot)
    plt.title(title)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.show()

def main():
    # 1. Carregar e pré-processar dados
    raw_dataset = load_and_preprocess_data(DATA_FILE)
    if raw_dataset is None:
        logger.error("Pipeline encerrado por erro nos dados.")
        return

    # Verificar se a GPU está disponível
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Usando dispositivo: {device}")

    # 2. Carregar Tokenizador e Modelo de Classificação
    logger.info(f"Carregando tokenizador e modelo: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    model = T5ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2, # Informa ao modelo que temos 2 classes (0=boa, 1=ruim)
        ignore_mismatched_sizes=True # Necessário ao adaptar um T5
    ).to(device)

    tokenized_datasets = tokenize_data(raw_dataset, tokenizer)

    # 3. Definir Argumentos de Treinamento
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        max_steps=5000,
        num_train_epochs=50,             # 5 épocas (ajuste conforme necessário)
        per_device_train_batch_size=4,  # Reduza se tiver OOM (Out of Memory)
        per_device_eval_batch_size=4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",    # Avalia no final de cada época
        save_strategy="epoch",          # Salva no final de cada época
        load_best_model_at_end=True,
        report_to="none",                # Desativa relatórios online (Wandb, etc.)
        remove_unused_columns=False, # Keep columns like 'label' for the trainer
    )

    # 4. Inicializar o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # 5. Treinar o Modelo
    logger.info("--- Iniciando o Finetuning do T5 ---")
    trainer.train()
    logger.info("--- Treinamento Concluído ---")

    # 6. Salvar o Modelo Final e Tokenizador
    trainer.save_model()
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    logger.info(f"Modelo salvo com sucesso em: {MODEL_OUTPUT_DIR}")

    # 7. Mostrar Evolução do Aprendizado
    logger.info("--- Evolução do Aprendizado (Histórico de Loss) ---")
    loss_history_df = pd.DataFrame(trainer.state.log_history)

    eval_logs = loss_history_df[loss_history_df['eval_loss'].notna()].set_index('epoch')
    train_logs = loss_history_df[loss_history_df['loss'].notna()].set_index('epoch')

    combined_logs = eval_logs[['eval_loss']].join(train_logs[['loss']], how='left')
    combined_logs.rename(columns={'loss': 'train_loss'}, inplace=True)

    logger.info(f"Histórico de loss:\n{combined_logs}")

    plt.figure(figsize=(10, 4))
    plt.plot(combined_logs.index, combined_logs['eval_loss'], label="Perda (Validação)", marker='o')
    plt.plot(combined_logs.index, combined_logs['train_loss'], label="Perda (Treino)", marker='o', linestyle='--')
    plt.title("Evolução da Perda (Loss) durante o Treinamento")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 8. Avaliação e Matriz de Confusão (Correto para Classificação)
    logger.info("--- Avaliação Detalhada e Matrizes de Confusão ---")

    for split in ['train', 'test']:
        logger.info(f"Avaliando no conjunto de {split}...")
        dataset_to_eval = tokenized_datasets[split]

        predictions_output = trainer.predict(dataset_to_eval)

        # --- CORREÇÃO ---
        # Access the logits from the predictions_output.predictions tuple
        # The first element is usually the logits
        # print(f"Shape of predictions_output.predictions: {predictions_output.predictions.shape}") # Debugging line
        # print(f"Shape of predictions_output.predictions[0]: {predictions_output.predictions[0].shape}") # Debugging line

        # Pega o índice da maior logit (0 ou 1)
        predicted_indices = np.argmax(predictions_output.predictions[0], axis=1)
        # --- FIM DA CORREÇÃO ---

        true_labels = predictions_output.label_ids

        # Converte índices (0, 1) de volta para nomes ("boa", "ruim")
        predicted_text = [LABELS[i] for i in predicted_indices]
        true_text = [LABELS[i] for i in true_labels]

        logger.info(f"Relatório de Classificação ({split}):\n{classification_report(true_text, predicted_text, labels=LABELS, zero_division=0)}")

        # Plota a matriz usando os nomes
        plot_confusion_matrix(
            true_text,
            predicted_text,
            title=f"Matriz de Confusão ({split})",
            labels_for_plot=LABELS
        )

if __name__ == "__main__":
    main()