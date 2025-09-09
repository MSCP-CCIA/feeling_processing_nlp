import time, logging
import scipy.sparse as sps
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

try:
    import mlflow
    import mlflow.sklearn
except ModuleNotFoundError:
    import subprocess, sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "mlflow>=2.12,<3"]
    )
    import mlflow
    import mlflow.sklearn

# Config
TRACKING_URI = (
    "http://ec2-34-201-213-246.compute-1.amazonaws.com:8080"  # "http://ec2-34-201-213-246.compute-1.amazonaws.com:8080"
)
RANDOM_STATE = 42
VEC_KW = dict(
    lowercase=False, max_df=0.9, min_df=5, max_features=500_000, dtype=np.float32
)
NGRAM = (1, 2)

# Logging
logging.basicConfig(
    level="INFO", format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("tfidf_rf_pipeline")


# ==========================
# Utils
# ==========================
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


def train_and_validate(X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    # Si YA tienes TF-IDF cargado desde disco, NO vectorices de nuevo
    Xtr, Xval, Xte = X_train, X_val, X_test

    # PredefinedSplit para separar validación
    test_fold = np.concatenate(
        [np.full(Xtr.shape[0], -1), np.zeros(Xval.shape[0])]
    )
    ps = PredefinedSplit(test_fold)

    # Concatenar train + val
    from scipy.sparse import vstack
    X_trval = vstack([Xtr, Xval])
    y_trval = np.concatenate([y_train, y_val])

    # Random Forest + GridSearch
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 20, 50],
        "max_features": ["sqrt", "log2"],
    }

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=ps,
        scoring="f1",
        verbose=2,
        n_jobs=-1,
    )
    grid.fit(X_trval, y_trval)

    best_model = grid.best_estimator_

    # Evaluación en test
    y_pred = best_model.predict(Xte)
    try:
        y_prob = best_model.predict_proba(Xte)[:, 1]
    except Exception:
        y_prob = None

    metr = metrics_block(y_test, y_pred, y_prob)
    report = classification_report(y_test, y_pred, digits=4)

    # Log en MLflow
    mlflow.log_params(grid.best_params_)
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
        best_model, artifact_path="model", registered_model_name=model_name
    )

    return metr, report, grid.best_params_


# ==========================
# Main
# ==========================
def main():
    # --- CARGA Y COMBINACIÓN DE DATOS ---
    try:
        logger.info("Cargando datos desde los archivos...")

        # Carga los archivos que son matrices dispersas directamente con np.load
        # y los verifica.
        X_train_obj = np.load('../data_process/X_train.npy', allow_pickle=True)
        X_train = X_train_obj.item() if isinstance(X_train_obj, np.ndarray) and X_train_obj.dtype == object else X_train_obj

        X_val_obj = np.load('../data_process/X_dev.npy', allow_pickle=True)
        X_val = X_val_obj.item() if isinstance(X_val_obj, np.ndarray) and X_val_obj.dtype == object else X_val_obj

        X_test_obj = np.load('../data_process/X_test.npy', allow_pickle=True)
        X_test = X_test_obj.item() if isinstance(X_test_obj, np.ndarray) and X_test_obj.dtype == object else X_test_obj

        # Carga los arrays de etiquetas
        y_train = np.load('../data_process/y_train.npy', allow_pickle=True)
        y_val = np.load('../data_process/y_dev.npy', allow_pickle=True)
        y_test = np.load('../data_process/y_test.npy', allow_pickle=True)

        # Combina las matrices dispersas con vstack
        X_combined = sps.vstack((X_train, X_val))

        # Combina los arrays de etiquetas con np.concatenate
        y_combined = np.concatenate((y_train, y_val), axis=0)

        logger.info(f"Forma de X_combined: {X_combined.shape}")
        logger.info(f"Forma de y_combined: {y_combined.shape}")

    except FileNotFoundError as e:
        logger.error("Error: Asegúrate de que los archivos de datos estén en la carpeta correcta.")
        exit(1)
    except Exception as e:
        logger.error(f"Error inesperado al cargar los archivos: {e}")
        exit(1)

    dataset_name = "processed_dataset_remove_punctuation_true"

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(f"TF-IDF_ngram_1-2-RF-{dataset_name}")

    with mlflow.start_run(run_name=dataset_name):
        mlflow.log_param("random_state", RANDOM_STATE)

        t0 = time.time()
        metr, report, best_params = train_and_validate(
            X_train, y_train, X_val, y_val, X_test, y_test, f"{dataset_name}_tfidf_rf"
        )

        mlflow.log_dict(
            {
                "test_metrics": metr,
                "best_params": best_params,
            },
            "final_summary.json",
        )

        elapsed_min = (time.time() - t0) / 60.0
        mlflow.log_metric("elapsed_minutes", elapsed_min)
        logger.info("Tiempo total dataset %s: %.2f min", dataset_name, elapsed_min)


if __name__ == "__main__":
    main()
