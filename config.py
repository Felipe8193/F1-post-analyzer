"""Configurações centrais do projeto F1 Sentiment.

Todos os caminhos são relativos à pasta do projeto, então os scripts
funcionam em qualquer máquina sem editar caminhos absolutos.
"""
from pathlib import Path

# --- Caminhos ---
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "f1_posts_rotulados_balanceado.csv"   # dados rotulados (treino/teste)
POSTS_FILE = BASE_DIR / "posts_traduzidos.csv"               # ~10k posts para análise
MODEL_DIR = BASE_DIR / "t5_f1_sentiment_model"               # modelo final
CHECKPOINT_DIR = BASE_DIR / "checkpoints"                    # temporário durante o treino

RESULTS_DIR = BASE_DIR / "resultados"
RESULTS_MODEL_DIR = RESULTS_DIR / "1_modelo"                 # métricas e avaliação
RESULTS_POSTS_DIR = RESULTS_DIR / "2_analise_posts"          # análise dos 10k posts

# --- Modelo ---
MODEL_NAME = "unicamp-dl/ptt5-small-portuguese-vocab"
PREFIX = "classifique o sentimento: "
LABELS = ["boa", "ruim"]                     # 0 = boa, 1 = ruim
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Os posts têm no máximo ~350 caracteres; 128 tokens cobrem praticamente
# todos e deixam o treino ~4x mais rápido que o antigo padding de 512.
MAX_LENGTH = 128

# --- Treino ---
SEED = 42
TEST_SIZE = 0.15          # conjunto de teste (nunca visto no treino)
VAL_SIZE = 0.15           # validação (early stopping / escolha do melhor checkpoint)
EPOCHS = 20               # máximo; o early stopping costuma parar antes
EARLY_STOPPING_PATIENCE = 4
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# --- Visual dos gráficos ---
COR_BOA = "#2a78d6"       # azul
COR_RUIM = "#e34948"      # vermelho
COR_NEUTRA = "#898781"
CORES_LABELS = {"boa": COR_BOA, "ruim": COR_RUIM}
CORES_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#6250d6", "#e34948"]
