import glob
import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

try:
    import mlflow
    import mlflow.sklearn
except ModuleNotFoundError:
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "mlflow>=2.12,<3"]
    )
    import mlflow
    import mlflow.sklearn

# Config
DATA_DIR = "../test"
TRACKING_URI = "http://ec2-34-201-213-246.compute-1.amazonaws.com:8080"
RANDOM_STATE = 42

# Configuración fija
TEST_SIZE_TOTAL = 0.20
DEV_RATIO_OF_TEMP = 0.50
FIXED_ALPHA = 1.0
FIXED_NGRAM = (1, 1)

VEC_KW = dict(
    lowercase=False, max_df=0.9, min_df=5, max_features=500_000, dtype=np.float32
)

# Logging
logging.basicConfig(
    level="INFO", format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("bow_mnb_pipeline")


# Utils
def split_80_10_10(y: np.ndarray):
    idx = np.arange(len(y))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=TEST_SIZE_TOTAL, random_state=RANDOM_STATE, stratify=y
    )
    idx_dev, idx_test, y_dev, y_test = train_test_split(
        idx_temp,
        y_temp,
        test_size=1 - DEV_RATIO_OF_TEMP,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return (idx_train, y_train), (idx_dev, y_dev), (idx_test, y_test)


def metrics_block(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    cm = confusion_matrix(y_true, y_pred)
    return dict(
        acc=float(acc),
        prec=float(prec),
        rec=float(rec),
        f1=float(f1),
        auc=(float(auc) if auc is not None else None),
        cm=cm.tolist(),
    )


def train_and_test(
    X_train_text: np.ndarray,
    y_train: np.ndarray,
    X_test_text: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
):
    vec = CountVectorizer(ngram_range=FIXED_NGRAM, **VEC_KW)
    Xtr = vec.fit_transform(X_train_text)
    Xte = vec.transform(X_test_text)

    clf = MultinomialNB(alpha=FIXED_ALPHA)
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    try:
        y_prob = clf.predict_proba(Xte)[:, 1]
    except Exception:
        y_prob = None

    metr = metrics_block(y_test, y_pred, y_prob)
    report = classification_report(y_test, y_pred, digits=4)

    mlflow.log_metrics(
        {
            "test_acc": metr["acc"],
            "test_prec": metr["prec"],
            "test_rec": metr["rec"],
            "test_f1": metr["f1"],
        }
    )
    if metr["auc"] is not None:
        mlflow.log_metric("test_auc", metr["auc"])
    mlflow.log_dict({"confusion_matrix": metr["cm"]}, "test_confusion_matrix.json")
    mlflow.log_text(report, "classification_report_test.txt")

    mlflow.sklearn.log_model(
        clf, artifact_path="model", registered_model_name=model_name
    )
    mlflow.log_dict(vec.get_params(), "vectorizer_params.json")

    return metr, report


# Main
def main():
    parquets = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    if not parquets:
        logger.error("No se encontraron .parquet en %s", DATA_DIR)
        return

    for pq in parquets:
        dataset_name = Path(pq).stem
        logger.info("========== Dataset: %s ==========", dataset_name)
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(f"BoW-MNB_{dataset_name}")

        df = pd.read_parquet(pq)
        if not {"text", "label"}.issubset(df.columns):
            logger.warning(
                "Saltando %s: faltan columnas 'text' y/o 'label'.", dataset_name
            )
            continue

        X_all = df["text"].astype(str).to_numpy()
        y_all = df["label"].astype(int).to_numpy()

        (idx_tr, y_tr), (_, _), (idx_te, y_te) = split_80_10_10(y_all)
        X_tr, X_te = X_all[idx_tr], X_all[idx_te]

        with mlflow.start_run(run_name=dataset_name):
            mlflow.log_params(
                {
                    "random_state": RANDOM_STATE,
                    "alpha": FIXED_ALPHA,
                    "ngram_range": FIXED_NGRAM,
                    "vec_max_features": VEC_KW["max_features"],
                    "vec_min_df": VEC_KW["min_df"],
                    "vec_max_df": VEC_KW["max_df"],
                }
            )

            t0 = time.time()
            metr, report = train_and_test(
                X_tr, y_tr, X_te, y_te, f"{dataset_name}_bow_mnb"
            )

            mlflow.log_dict(
                {
                    "test_metrics": metr,
                },
                "final_summary.json",
            )

            elapsed_min = (time.time() - t0) / 60.0
            mlflow.log_metric("elapsed_minutes", elapsed_min)
            logger.info("Tiempo total dataset %s: %.2f min", dataset_name, elapsed_min)


if __name__ == "__main__":
    main()
