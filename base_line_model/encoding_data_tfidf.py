import logging
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Configuración
DATA_FILE = "../test/processed_dataset_remove_punctuation_true.parquet"
OUTPUT_DIR = "./tfidf_outputs"
TRACKING_URI = "http://ec2-34-201-213-246.compute-1.amazonaws.com:8080/"
RANDOM_STATE = 42
TEST_SIZE_TOTAL = 0.20
DEV_RATIO_OF_TEMP = 0.50

VEC_KW = dict(
    lowercase=False,
    max_df=0.9,
    min_df=5,
    max_features=310_291,
    dtype=np.float32
)

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("tfidf_vectorizer")


def split_80_10_10(y: np.ndarray):
    idx = np.arange(len(y))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=TEST_SIZE_TOTAL,
        random_state=RANDOM_STATE, stratify=y
    )
    idx_dev, idx_test, y_dev, y_test = train_test_split(
        idx_temp, y_temp, test_size=1 - DEV_RATIO_OF_TEMP,
        random_state=RANDOM_STATE, stratify=y_temp
    )
    return (idx_train, y_train), (idx_dev, y_dev), (idx_test, y_test)


def main():
    if not os.path.exists(DATA_FILE):
        logger.error("No se encontró el dataset: %s", DATA_FILE)
        return

    df = pd.read_parquet(DATA_FILE)
    if "text" not in df.columns:
        logger.error("El dataset no contiene la columna necesaria: 'text'")
        return

    X_all = df["text"].astype(str).to_numpy()
    y_dummy = np.zeros(len(X_all))  # Dummy labels solo para dividir
    (idx_tr, _), (_, _), (_, _) = split_80_10_10(y_dummy)
    X_tr = X_all[idx_tr]

    mlflow.set_tracking_uri(TRACKING_URI)

    # Crear vectorizador ngrama (1,2)
    vec = TfidfVectorizer(ngram_range=(1, 2), **VEC_KW)
    vec.fit(X_tr)

    # (Opcional) Guardar matrices locales
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), vec.transform(X_tr))

    # Guardar directamente en el Model Registry
    with mlflow.start_run(run_name="tfidf_vectorizer_1-2"):
        mlflow.sklearn.log_model(
            sk_model=vec,
            name="model",
            registered_model_name="TFIDF_Vectorizer_1_2"
        )

    logger.info("Vectorizador TF-IDF (1,2) registrado en MLflow Model Registry")


if __name__ == "__main__":
    main()
