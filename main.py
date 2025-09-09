import json
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

# ---------------------------
# MLflow config
# ---------------------------
TRACKING_URI = "http://ec2-34-201-213-246.compute-1.amazonaws.com:8080"
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Model Comparison & MLflow API",
    description="Comparación de modelos (LR vs DistilBERT) y consulta de modelos en MLflow.",
)


# Métricas fijas tomadas de tus capturas
LR_METRICS = {
    "test_prec": 0.8175051553939453,
    "test_f1": 0.8273634943471254,
    "elapsed_minutes": 13.83313014904658,
    "test_rec": 0.8374625,
    "test_acc": 0.82525625,
    "best_validation_accuracy": 0.82261875,
    "test_auc": 0.90422496546875,
}
LR_PARAMS = {
    "C": 1,
    "solver": "liblinear",
    "param_grid": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"],
    },
}
BERT_METRICS = {
    "f1_binary": 0.9137168141592921,
    "f1_macro": 0.9104298356510746,
    "f1_weighted": 0.9104901471833355,
}
BERT_INFO = {"hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english"}


def build_payload():
    f1_lr = float(LR_METRICS["test_f1"])
    f1_bin = float(BERT_METRICS["f1_binary"])
    deltas = {
        "f1_binary_delta": round(f1_bin - f1_lr, 6),
        "f1_macro_delta_vs_lr_f1": round(float(BERT_METRICS["f1_macro"]) - f1_lr, 6),
        "f1_weighted_delta_vs_lr_f1": round(
            float(BERT_METRICS["f1_weighted"]) - f1_lr, 6
        ),
        "relative_improvement_%": round((f1_bin - f1_lr) / f1_lr * 100, 4),
    }
    winner = "distilbert" if f1_bin > f1_lr else "logistic_regression"
    return {
        "logistic_regression": {"metrics": LR_METRICS, "hyperparams": LR_PARAMS},
        "distilbert": {"metrics": BERT_METRICS, **BERT_INFO},
        "diff": deltas,
        "winner_by_f1_binary": winner,
        "notes": "LR (métricas/params de la captura) vs DistilBERT (F1 binary/macro/weighted).",
    }


def render_html(payload: dict) -> str:
    lr = payload["logistic_regression"]["metrics"]
    lrp = payload["logistic_regression"]["hyperparams"]
    bt = payload["distilbert"]["metrics"]
    df = payload["diff"]
    winner = payload["winner_by_f1_binary"]

    return f"""<!doctype html>
<html lang="es"><head>
<meta charset="utf-8"/>
<title>Comparación de Modelos</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
h1,h2,h3 {{ margin: 0 0 10px; }}
.card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding:16px; margin-bottom:16px; }}
.table {{ border-collapse: collapse; width: 100%; }}
.table th, .table td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
.badge {{ display:inline-block; background:#eef2ff; color:#3730a3; padding:4px 8px; border-radius:999px; font-weight:600; }}
.kv {{ background:#0b1020; color:#e5e7eb; padding:12px; border-radius:8px; overflow:auto; }}
.small {{ color:#64748b; font-size: 12px; }}
</style></head>
<body>
<h1>Comparación de Modelos</h1>

<div class="card">
  <h2>Resumen</h2>
  <p>Ganador por F1 (binary): <span class="badge">{winner}</span></p>
  <p>Mejora relativa vs LR (F1 binary): <b>{df["relative_improvement_%"]}%</b></p>
</div>

<div class="card">
  <h3>Métricas</h3>
  <table class="table">
    <tr><th>Métrica</th><th>Logistic Regression</th><th>DistilBERT</th></tr>
    <tr><td>F1</td><td>{lr["test_f1"]:.6f}</td><td>{bt["f1_binary"]:.6f} (binary)</td></tr>
    <tr><td>Precision</td><td>{lr["test_prec"]:.6f}</td><td>—</td></tr>
    <tr><td>Recall</td><td>{lr["test_rec"]:.6f}</td><td>—</td></tr>
    <tr><td>Accuracy</td><td>{lr["test_acc"]:.6f}</td><td>—</td></tr>
    <tr><td>AUC</td><td>{lr["test_auc"]:.6f}</td><td>—</td></tr>
    <tr><td>F1 (macro)</td><td>—</td><td>{bt["f1_macro"]:.6f}</td></tr>
    <tr><td>F1 (weighted)</td><td>—</td><td>{bt["f1_weighted"]:.6f}</td></tr>
  </table>
</div>

<div class="card">
  <h3>Diferencias (DistilBERT − LR)</h3>
  <ul>
    <li>Δ F1 (binary): <b>{df["f1_binary_delta"]}</b></li>
    <li>Δ vs LR F1 (macro): <b>{df["f1_macro_delta_vs_lr_f1"]}</b></li>
    <li>Δ vs LR F1 (weighted): <b>{df["f1_weighted_delta_vs_lr_f1"]}</b></li>
  </ul>
</div>

<div class="card">
  <h3>Hiperparámetros LR</h3>
  <div class="kv"><pre>{json.dumps(lrp, indent=2)}</pre></div>
  <p class="small">Nota: hyperparams fijos para visualización.</p>
</div>

</body></html>"""


# ------------------------------
# Pydantic Schemas
# ------------------------------
class ModelDetails(BaseModel):
    name: str
    version: str
    run_id: str
    current_stage: str
    artifact_uri: str
    description: Optional[str] = None
    tags: Dict[str, str]
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]


class PredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    predictions: List[Any]


# ------------------------------
# Load Models (lazy loading)
# ------------------------------
VEC_MODEL_NAME = "TFIDF_Vectorizer_1_2"
CLS_MODEL_NAME = "TFIDF_ngram_1-2-LGR_processed_dataset_remove_punctuation_true"

_vectorizer = None
_classifier = None


def load_models():
    global _vectorizer, _classifier
    if _vectorizer is None:
        _vectorizer = mlflow.sklearn.load_model(
            f"models:/{VEC_MODEL_NAME}/2"
        )  # usa versión 1
    if _classifier is None:
        _classifier = mlflow.sklearn.load_model(
            f"models:/{CLS_MODEL_NAME}/1"
        )  # usa versión 1
    return _vectorizer, _classifier


# ---------------------------
# 1) /comparison (y /com)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        "<h2>API OK</h2><ul>"
        '<li><a href="/comparison">/comparison</a> (HTML)</li>'
        '<li><a href="/comparison/json">/comparison/json</a> (JSON)</li>'
        '<li><a href="/docs">/docs</a> (Swagger)</li>'
        "</ul>"
    )


@app.get("/comparison", response_class=HTMLResponse)
def comparison_html():
    return HTMLResponse(render_html(build_payload()))


@app.get("/comparison/json")
def comparison_json(pretty: bool = Query(False, description="True para indentado")):
    payload = build_payload()
    if pretty:
        return HTMLResponse(
            "<pre>" + json.dumps(payload, indent=2, ensure_ascii=False) + "</pre>"
        )
    return JSONResponse(payload)


# ---------------------------
# 2) /model_details (tu endpoint MLflow)
# ---------------------------
@app.get("/model_details", response_model=ModelDetails)
async def get_model_details(
    model_name: str, model_version: Optional[Union[str, int]] = 1
):
    """Obtiene metadata, hiperparámetros y métricas de un modelo registrado en MLflow.
    Ejemplo: /model_details?model_name=BoW-MNB_mimodelo&model_version=1"""
    try:
        model_version_info = client.get_model_version(
            name=model_name, version=str(model_version)
        )
        run_id = model_version_info.run_id
        run = client.get_run(run_id)

        response_data = {
            "name": model_version_info.name,
            "version": model_version_info.version,
            "run_id": run_id,
            "current_stage": model_version_info.current_stage,
            "artifact_uri": model_version_info.source,
            "description": model_version_info.description,
            "tags": model_version_info.tags,
            "hyperparameters": dict(run.data.params),
            "metrics": dict(run.data.metrics),
        }
        return response_data

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Error: Could not get model or run data. {e}",
        )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Run batch predictions on a list of input texts."""
    try:
        vec, clf = load_models()
        X = vec.transform(request.texts)  # vectorize input batch
        preds = clf.predict(X)  # run inference
        return {"predictions": preds.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ---------------------------
# Run:
# uvicorn main:app --reload --port 8000
# ---------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
