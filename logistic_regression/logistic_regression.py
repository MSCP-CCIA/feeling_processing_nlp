import logging
import os
import time

import mlflow
import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, PredefinedSplit

# Configura el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DEL ENDPOINT DE MLFLOW ---
mlflow.set_tracking_uri("http://ec2-34-201-213-246.compute-1.amazonaws.com:8080")

# --- CARGA Y COMBINACIÓN DE DATOS ---
try:
    logger.info("Cargando datos desde los archivos...")

    # Carga los archivos que son matrices dispersas directamente con np.load
    # y los verifica.
    X_train_obj = np.load('X_train.npy', allow_pickle=True)
    X_train = X_train_obj.item() if isinstance(X_train_obj, np.ndarray) and X_train_obj.dtype == object else X_train_obj

    X_val_obj = np.load('X_dev.npy', allow_pickle=True)
    X_val = X_val_obj.item() if isinstance(X_val_obj, np.ndarray) and X_val_obj.dtype == object else X_val_obj

    X_test_obj = np.load('X_test.npy', allow_pickle=True)
    X_test = X_test_obj.item() if isinstance(X_test_obj, np.ndarray) and X_test_obj.dtype == object else X_test_obj

    # Carga los arrays de etiquetas
    y_train = np.load('y_train.npy', allow_pickle=True)
    y_val = np.load('y_dev.npy', allow_pickle=True)
    y_test = np.load('y_test.npy', allow_pickle=True)

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

# Crea el índice para PredefinedSplit
split_index = np.zeros(X_combined.shape[0], dtype=int)
split_index[:X_train.shape[0]] = -1
split_index[X_train.shape[0]:] = 0
pds = PredefinedSplit(test_fold=split_index)

# --- RESTO DEL SCRIPT ---
experiment_name = "TFIDF_ngram_1-2-LGR_processed_dataset_remove_punctuation_true"
mlflow.set_experiment(experiment_name)

start_time = time.time()

with mlflow.start_run():
    logger.info("Iniciando experimento MLflow...")
    logistic_regression = LogisticRegression()
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }

    mlflow.log_param('param_grid', str(param_grid))

    logger.info(f"Realizando Grid Search con los parámetros: {param_grid}")
    grid_search = GridSearchCV(logistic_regression, param_grid, cv=pds)

    grid_search.fit(X_combined, y_combined)

    best_params = grid_search.best_params_
    mlflow.log_params(best_params)
    mlflow.log_metric('best_validation_accuracy', grid_search.best_score_)
    logger.info(f"Mejores hiperparámetros encontrados: {best_params}")

    logger.info("Evaluando el mejor modelo en el conjunto de prueba...")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60

    metrics = {
        'test_prec': precision_score(y_test, y_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_pred, zero_division=0),
        'elapsed_minutes': elapsed_minutes,
        'test_rec': recall_score(y_test, y_pred, zero_division=0),
        'test_acc': accuracy_score(y_test, y_pred),
        'test_auc': roc_auc_score(y_test, y_prob)
    }

    mlflow.log_metrics(metrics)

    logger.info("Métricas del modelo de prueba registradas:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name=experiment_name)
    logger.info("Modelo guardado en MLflow.")
    logger.info("Experimento finalizado.")