import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# --- 1. Definir o Modelo Salvo ---
#
# Escolha o diretório do modelo que você já treinou e quer continuar
#
# Caminho do modelo T5 salvo
MODEL_PATH = "D:/100MEDIA/t5_f1_sentiment_model"
# ou "./meu_modelo_f1_reddit", etc.

# --- 2. Carregar o Modelo e Tokenizador JÁ TREINADOS ---
print(f"Carregando modelo e tokenizador de '{MODEL_PATH}'...")

# Detectar GPU e mover modelo para CUDA se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForSequenceClassification.from_pretrained(MODEL_PATH)
    model = model.to(device)
    print(f"Modelo T5 anterior carregado com sucesso! Usando dispositivo: {device}")
except OSError:
    print(f"Erro: Não foi possível encontrar um modelo em '{MODEL_PATH}'.")
    print("Verifique o caminho antes de continuar.")
    exit()

# --- 3. Preparar os NOVOS Dados ---
#
# Para continuar o treinamento, você precisa de NOVOS dados que o modelo
# ainda não viu. Aqui, estamos simulando a chegada de novos posts.
#
# Em um caso real, você pode carregar um novo CSV
# (ex: pd.read_csv("novos_dados_rotulados.csv"))
#
new_data = {
    'text': [
        "A Alpine está completamente perdida, que performance horrível.",
        "Essa briga entre Russell e Leclerc foi o ponto alto da corrida!",
        "Chuva no GP do Brasil é sempre sinônimo de corrida boa.",
        "Não aguento mais essa dominância, sempre o mesmo vencedor.",
        "O safety car demorou muito para sair, prejudicou a corrida.",
        "Que volta de qualificação absurda do Verstappen!",
    ],
    'label': [0, 1, 1, 0, 0, 1] # 0=Ruim, 1=Bom
}
df_new = pd.DataFrame(new_data)

# É uma boa prática ter dados de treino e validação
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_new['text'], df_new['label'], test_size=0.3, random_state=42
)


# Função de tokenização para T5
def tokenize_function(examples):
    prefix = "classifique o sentimento: "
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = examples["label"]
    return model_inputs

# Criar os datasets
train_dataset = Dataset.from_dict({'text': list(train_texts), 'label': list(train_labels)})
val_dataset = Dataset.from_dict({'text': list(val_texts), 'label': list(val_labels)})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)


# --- 4. Configurar o NOVO Treinamento ---
#
# Vamos salvar os resultados em um novo diretório
#
training_args = TrainingArguments(
    output_dir="./results_continuacao",       # Novo diretório de resultados
    max_steps=3000,
    num_train_epochs=50,             # 5 épocas (ajuste conforme necessário)
    per_device_train_batch_size=4,  # Reduza se tiver OOM (Out of Memory)
    per_device_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="steps",    # Avalia no final de cada época
    save_strategy="steps",          # Salva a cada N passos
    save_steps=1000,                 # Salva checkpoints de 1000 em 1000 passos
    load_best_model_at_end=True,
    report_to="none",                # Desativa relatórios online (Wandb, etc.)
    remove_unused_columns=False, # Keep columns like 'label' for the trainer
)




# Criar o Trainer, passando o modelo JÁ CARREGADO
trainer = Trainer(
    model=model, # <-- Aqui está a mágica! Usando o modelo carregado.
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\n--- Iniciando a continuação do treinamento ---")
# O modelo não vai começar do zero, ele vai continuar de onde parou.
trainer.train()


print("--- Treinamento adicional concluído! ---")

# --- Visualizar gráficos de evolução do loss ---
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("\n--- Evolução do Loss durante o Treinamento ---")
loss_history_df = pd.DataFrame(trainer.state.log_history)
eval_logs = loss_history_df[loss_history_df['eval_loss'].notna()].set_index('epoch')
train_logs = loss_history_df[loss_history_df['loss'].notna()].set_index('epoch')
combined_logs = eval_logs[['eval_loss']].join(train_logs[['loss']], how='left')
combined_logs.rename(columns={'loss': 'train_loss'}, inplace=True)
print(combined_logs)

plt.figure(figsize=(10, 4))
plt.plot(combined_logs.index, combined_logs['eval_loss'], label="Perda (Validação)", marker='o')
plt.plot(combined_logs.index, combined_logs['train_loss'], label="Perda (Treino)", marker='o', linestyle='--')
plt.title("Evolução da Perda (Loss) durante o Treinamento")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.legend()
plt.grid(True)
plt.show()

# --- Avaliação automática ---
from sklearn.metrics import classification_report, confusion_matrix

print("\n--- Avaliação automática no conjunto de validação ---")
predictions_output = trainer.predict(val_dataset)
predicted_indices = np.argmax(predictions_output.predictions[0], axis=1)
true_labels = predictions_output.label_ids

labels_map = {0: "Ruim", 1: "Bom"}
predicted_text = [labels_map[i] for i in predicted_indices]
true_text = [labels_map[i] for i in true_labels]

print(classification_report(true_text, predicted_text, labels=["Ruim", "Bom"], zero_division=0))
cm = confusion_matrix(true_text, predicted_text, labels=["Ruim", "Bom"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ruim", "Bom"], yticklabels=["Ruim", "Bom"])
plt.title("Matriz de Confusão (Validação)")
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
plt.show()

# --- Testar modelo com novos textos ---
print("\n--- Teste do modelo com novos textos ---")
novos_textos = [
    "Que corrida espetacular! O Hamilton pilotou muito hoje.",
    "A Ferrari errou a estratégia de novo, que desastre.",
    "Péssima qualificação, o carro não tinha ritmo nenhum.",
    "O Norris conseguiu um pódio incrível, estou muito feliz!"
]
for texto in novos_textos:
    inputs = tokenizer(["classifique o sentimento: " + texto], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0].detach().cpu()
    prediction_index = np.argmax(logits.numpy())
    prediction_label = labels_map[prediction_index]
    print(f"Texto: '{texto}'\nSentimento previsto: {prediction_label} (Logits: {logits.numpy()})\n")


# --- 5. Salvar a Versão Final (v2) do Modelo ---
#
# É recomendado salvar o modelo final em um novo local
#
NEW_MODEL_SAVE_PATH = "D:/100MEDIA/meu_modelo_f1_v2"
model.save_pretrained(NEW_MODEL_SAVE_PATH)
tokenizer.save_pretrained(NEW_MODEL_SAVE_PATH)

print(f"\nModelo atualizado e salvo com sucesso em '{NEW_MODEL_SAVE_PATH}'.")
print("Este novo modelo agora contém o conhecimento do treino original + o novo treino.")