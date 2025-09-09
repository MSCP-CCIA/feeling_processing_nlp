import json
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
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

# main.py
rows = [
    ("Baseline", 0.77, 0.46, 0.85),
    ("MNB-BOW-TranslateEmojis", 0.77, 0.53, 0.85),
    ("MNB-BOW-Emojis-RemovePunctuationTEmojis", 0.78, 0.43, 0.85),
    ("MNB-BOW-LemmatizationFalse", 0.76, 0.41, 0.84),
    ("MNB-BOW-StopWordsFalse", 0.77, 0.44, 0.85),
    ("MNB-BOW-AllSteps", 0.76, 0.41, 0.83),
    ("MNB-TF-IDF_unigram-Punctutation", 0.77, 0.37, 0.86),
    ("MNB-TF-IDF_bigram-Punctuation", 0.77, 0.98, 0.86),
    ("MNB-TF-IDF_unigram_&_bigram-Punctuation", 0.80, 1.20, 0.88),
    ("LGR-TF-IDF_unigram&bigram-Punctutation", 0.83, 13.80, 0.90),
    ("DT-TF-IDF_unigram&bigram-Punctuation", 0.70, 12.80, 0.73),
    ("DNN-TF-IDF_unigram&bigram-Punctuation", 0.81, 30.60, 0.87),
]

df = pd.DataFrame(rows, columns=["experiment", "f1", "elapsed_minutes", "auc"])

# === Helper: crear imagen (scatter) y devolver base64 PNG ===
def plot_f1_vs_time_base64(df: pd.DataFrame) -> str:
    plt.figure(figsize=(10, 6))

    # Normalizar colores según F1 (mejor interpretación visual)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=df["f1"].min(), vmax=df["f1"].max())
    colors = cmap(norm(df["f1"]))

    # Tamaño de puntos proporcional al AUC
    sizes = ((df["auc"] - 0.7) * 400).clip(lower=20)

    scatter = plt.scatter(df["elapsed_minutes"], df["f1"], s=sizes, c=df["f1"],
                          cmap=cmap, alpha=0.8, edgecolors='k', linewidth=0.5)

    # Líneas de referencia
    plt.axhline(df["f1"].mean(), color="gray", linestyle="--", alpha=0.5, label="Mean F1")
    plt.axvline(df["elapsed_minutes"].mean(), color="gray", linestyle="--", alpha=0.5, label="Mean Time")

    plt.xlabel("Elapsed time (minutes)")
    plt.ylabel("F1 score")
    plt.title("Trade-off: F1 vs Training time (point size ~ AUC)")

    # Anotar puntos
    for _, row in df.iterrows():
        label = row["experiment"]
        short = label if len(label) <= 24 else label[:21] + "..."
        plt.annotate(short,
                     (row["elapsed_minutes"], row["f1"]),
                     textcoords="offset points", xytext=(8, 5),
                     ha="left", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, lw=0))

    plt.grid(True, linestyle=":", linewidth=0.5)

    # Barra de color para F1
    cbar = plt.colorbar(scatter)
    cbar.set_label("F1 Score")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    return img_b64


# === HTML renderer ===
def build_html(df: pd.DataFrame) -> str:
    # tabla (solo valores exactos; separada de la gráfica)
    table_html = df.round({"f1": 3, "elapsed_minutes": 2, "auc": 3}).to_html(
        index=False, classes="abl-table", border=0, justify="left"
    )

    # gráfica como base64
    img_b64 = plot_f1_vs_time_base64(df)

    # conclusiones automáticas (texto breve, útil para informe)
    # Aquí construimos un párrafo de conclusión enfocado en trade-offs F1/time/auc
    best_f1_row = df.loc[df["f1"].idxmax()]
    best_tradeoff_candidates = df[
        (df["f1"] >= 0.80) & (df["elapsed_minutes"] <= 2.0)
    ]
    if not best_tradeoff_candidates.empty:
        tradeoff_example = best_tradeoff_candidates.sort_values(
            ["f1", "elapsed_minutes"], ascending=[False, True]
        ).iloc[0]
        tradeoff_text = (
            f"Una buena compensación rendimiento/tiempo es <b>{tradeoff_example['experiment']}</b> "
            f"con F1={tradeoff_example['f1']:.3f}, tiempo={tradeoff_example['elapsed_minutes']:.2f} min y AUC={tradeoff_example['auc']:.3f}."
        )
    else:
        tradeoff_text = "No hay un candidato claro con F1 >= 0.80 y tiempo de entrenamiento pequeño; los mejores F1 requieren tiempos más largos."

    conclusions = f"""
        <p>
          El modelo con <b>mejor F1 absoluto</b> fue <b>{best_f1_row['experiment']}</b>:
          <b>F1={best_f1_row['f1']:.3f}</b>, tiempo={best_f1_row['elapsed_minutes']:.2f} min y AUC={best_f1_row['auc']:.3f}.
          Este resultado representa el punto más alto de precisión entre todos los experimentos realizados.
        </p>

        <p>
          {tradeoff_text}
        </p>

        <p>
          <b>Hallazgos clave:</b>
          <ul>
            <li>Los modelos <b>LGR-TF-IDF</b> y <b>DNN-TF-IDF</b> presentan las métricas más equilibradas entre F1 (0.81–0.83) y AUC (≥0.87),
                pero su <u>tiempo de entrenamiento</u> es significativamente mayor (13.8 y 30.6 min), lo que los hace menos eficientes en escenarios de recursos limitados.</li>

            <li>La combinación <b>MNB-TF-IDF_unigram_&_bigram-Punctuation</b> destaca como una opción
                de <b>alto rendimiento (F1=0.80, AUC=0.88)</b> con un tiempo de entrenamiento moderado (1.2 min),
                siendo el mejor compromiso entre precisión y eficiencia.</li>

            <li>Las variantes MNB con <b>BoW</b> o <b>preprocesamientos ligeros</b> (p. ej. emojis, stopwords, lematización)
                ofrecen resultados estables (F1≈0.76–0.78) y entrenan en menos de 1 min,
                pero no alcanzan la calidad de los modelos TF-IDF.</li>

            <li>El <b>árbol de decisión (DT)</b> es claramente ineficiente en este contexto:
                bajo F1=0.70, AUC=0.73 y un tiempo elevado (12.8 min), por lo que se descarta para producción.</li>
          </ul>
        </p>

        <p>
          <b>Recomendación estratégica:</b><br>
          - Si el objetivo es <u>máxima precisión sin restricciones de tiempo</u>, utiliza <b>LGR-TF-IDF</b>.<br>
          - Si se requiere <u>rapidez con buen desempeño</u>, adopta <b>MNB-TF-IDF_unigram_&_bigram-Punctuation</b>.<br>
          - Para <u>entornos en tiempo real</u> o con limitaciones de cómputo, modelos MNB-BoW son adecuados pero sacrifican F1.
        </p>

        <p>
          En resumen, la <b>ablación demuestra que la inclusión de TF-IDF (especialmente con bigramas)
          y clasificadores más sofisticados</b> (LGR/DNN) incrementa el rendimiento, pero introduce un
          costo de tiempo considerable. La decisión óptima dependerá del balance entre
          <i>precisión deseada y tiempo de despliegue disponible</i>. Modelos como la red neuronal densa podrían
          presentar un rendimiento mucho mayor, sin embargo las limitaciones de recursos computacionales no 
          permiten explorar todo su potencial y su poca mejora respecto a modelos mucho menos costosos no 
          justifica su uso.
        </p>
        """

    html = f"""
    <!doctype html>
    <html lang="es">
    <head>
      <meta charset="utf-8"/>
      <title>Análisis de Ablación</title>
      <style>
        body {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin: 28px; color:#0b1220; }}
        h1 {{ margin-bottom: 6px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start; }}
        .card {{ border: 1px solid #e6e9ef; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(12,15,20,0.03); }}
        .abl-table {{ width: 100%; border-collapse: collapse; margin-bottom: 8px; }}
        .abl-table th, .abl-table td {{ padding: 8px 10px; border-bottom: 1px solid #f0f2f5; text-align: left; }}
        .note {{ font-size: 0.9rem; color: #444; }}
        img.plot {{ max-width: 100%; border-radius: 6px; border: 1px solid #f0f2f5; }}
        ul {{ margin-top: 6px; }}
      </style>
    </head>
    <body>
      <h1>Análisis de Ablación</h1>

      <div class="grid">
        <div class="card">
          <h2>Tabla de resultados</h2>
          {table_html}
        </div>

        <div class="card">
          <h2>Gráfica: F1 vs Tiempo</h2>
          <img class="plot" src="data:image/png;base64,{img_b64}" alt="F1 vs time">
          <p class="note">Cada punto representa un experimento; el área del punto se escala con el AUC (mayor AUC → punto más grande).</p>
        </div>
      </div>

      <div class="card" style="margin-top:18px;">
        <h2>Conclusiones</h2>
        {conclusions}
      </div>
    </body>
    </html>
    """
    return html

# === Rutas ===
@app.get("/ablation", response_class=HTMLResponse)
def ablation_html():
    return HTMLResponse(build_html(df))

@app.get("/ablation/json")
def ablation_json():
    return JSONResponse(df.to_dict(orient="records"))

# ---------------------------
# Run:
# uvicorn main:app --reload --port 8000
# ---------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000,host="0.0.0.0")
