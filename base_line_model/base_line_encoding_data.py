import os, time, logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import mlflow
import mlflow.sklearn

# Configuración
DATA_FILE = "../selected_data/processed_dataset_remove_punctuation_true.parquet"
OUTPUT_DIR = "./tfidf_outputs"
TRACKING_URI = "http://ec2-34-201-213-246.compute-1.amazonaws.com:8080/"
RANDOM_STATE = 42
ALPHAS = [0.1, 0.5, 1.0, 2.0]
NGRAMS = [(1, 1), (1, 2), (2, 2)]
TEST_SIZE_TOTAL = 0.20
DEV_RATIO_OF_TEMP = 0.50

VEC_KW = dict(
    lowercase=False,
    max_df=0.9,
    min_df=5,
    max_features=500_000,
    dtype=np.float32
)

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("tfidf_mnb_pipeline")


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


def metrics_block(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    return dict(acc=float(acc), prec=float(prec), rec=float(rec), f1=float(f1),
                auc=(float(auc) if auc is not None else None))


def hyperparameter_search_for_ngram(X_train, y_train, X_dev, y_dev, ngram):
    best_alpha = None
    best_f1 = -1
    vec = TfidfVectorizer(ngram_range=ngram, **VEC_KW)
    Xtr = vec.fit_transform(X_train)
    Xdv = vec.transform(X_dev)

    for alpha in ALPHAS:
        clf = MultinomialNB(alpha=alpha)
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xdv)
        try:
            y_prob = clf.predict_proba(Xdv)[:, 1]
        except Exception:
            y_prob = None

        metr = metrics_block(y_dev, y_pred, y_prob)
        if metr["f1"] > best_f1:
            best_f1 = metr["f1"]
            best_alpha = alpha

    return {"ngram_range": ngram, "alpha": best_alpha, "f1_dev": best_f1}, vec


def evaluate_on_test(X_train, y_train, X_test, y_test, vec, alpha):
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = MultinomialNB(alpha=alpha)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    try:
        y_prob = clf.predict_proba(Xte)[:, 1]
    except Exception:
        y_prob = None

    metr = metrics_block(y_test, y_pred, y_prob)
    return metr, clf, Xtr, Xte


def main():
    if not os.path.exists(DATA_FILE):
        logger.error("No se encontró el dataset: %s", DATA_FILE)
        return

    df = pd.read_parquet(DATA_FILE)
    if not {"text", "label"}.issubset(df.columns):
        logger.error("El dataset no contiene las columnas necesarias: 'text', 'label'")
        return

    X_all = df["text"].astype(str).to_numpy()
    y_all = df["label"].astype(int).to_numpy()
    (idx_tr, y_tr), (idx_dv, y_dv), (idx_te, y_te) = split_80_10_10(y_all)
    X_tr, X_dv, X_te = X_all[idx_tr], X_all[idx_dv], X_all[idx_te]

    mlflow.set_tracking_uri(TRACKING_URI)

    best_model = {"f1_test": -1, "cfg": None, "vec": None, "data": None}

    for ngram in NGRAMS:
        mlflow.set_experiment(f"TF-IDF_ngram_{ngram[0]}-{ngram[1]}-MNB_selected_data")
        with mlflow.start_run(run_name=f"ngram_{ngram[0]}-{ngram[1]}"):
            t0 = time.time()

            cfg, vec = hyperparameter_search_for_ngram(X_tr, y_tr, X_dv, y_dv, ngram)
            metr_test, clf, Xtr_tfidf, Xte_tfidf = evaluate_on_test(X_tr, y_tr, X_te, y_te, vec, cfg["alpha"])

            mlflow.log_params({
                "alpha": cfg["alpha"],
                "ngram_range": cfg["ngram_range"],
                "vectorizer": "tfidf"
            })

            mlflow.log_metrics({
                "test_acc": metr_test["acc"],
                "test_prec": metr_test["prec"],
                "test_rec": metr_test["rec"],
                "test_f1": metr_test["f1"],
            })
            if metr_test["auc"] is not None:
                mlflow.log_metric("test_auc", metr_test["auc"])

            elapsed_min = (time.time() - t0) / 60.0
            mlflow.log_metric("elapsed_minutes", elapsed_min)
            logger.info("Tiempo total para ngram=%s: %.2f min", ngram, elapsed_min)

            if metr_test["f1"] > best_model["f1_test"]:
                best_model.update({
                    "f1_test": metr_test["f1"],
                    "cfg": cfg,
                    "vec": vec,
                    "data": (X_tr, y_tr, X_dv, y_dv, X_te, y_te)
                })

    # Guardar solo el mejor n-grama
    if best_model["vec"] is not None:
        logger.info("Guardando solo el mejor ngrama: %s", best_model["cfg"]["ngram_range"])
        Xtr = best_model["vec"].fit_transform(best_model["data"][0])
        Xdv = best_model["vec"].transform(best_model["data"][2])
        Xte = best_model["vec"].transform(best_model["data"][4])

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), Xtr)
        np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), best_model["data"][1])
        np.save(os.path.join(OUTPUT_DIR, "X_dev.npy"), Xdv)
        np.save(os.path.join(OUTPUT_DIR, "y_dev.npy"), best_model["data"][3])
        np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), Xte)
        np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), best_model["data"][5])


if __name__ == "__main__":
    main()
