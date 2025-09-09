import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from typing import Optional, Union, Dict, Any
from pydantic import BaseModel

# 1. Configure the MLflow server connection
TRACKING_URI = "http://ec2-34-201-213-246.compute-1.amazonaws.com:8080"
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# Define a Pydantic model for the response to ensure a clear structure
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

# Initialize the FastAPI application
app = FastAPI(
    title="MLflow Model API",
    description="A simple API to get details of a registered MLflow model."
)

@app.get("/model_details", response_model=ModelDetails)
async def get_model_details(model_name: str, model_version: Optional[Union[str, int]] = 1):
    """
    Retrieves metadata, hyperparameters, and metrics for a registered MLflow model.
    """
    try:
        # Get the model's metadata and run_id
        model_version_info = client.get_model_version(name=model_name, version=str(model_version))
        run_id = model_version_info.run_id

        # Use the run_id to get the run details
        run = client.get_run(run_id)

        # Prepare the response data in a dictionary
        response_data = {
            "name": model_version_info.name,
            "version": model_version_info.version,
            "run_id": run_id,
            "current_stage": model_version_info.current_stage,
            "artifact_uri": model_version_info.source,
            "description": model_version_info.description,
            "tags": model_version_info.tags,
            "hyperparameters": run.data.params,
            "metrics": run.data.metrics,
        }

        # Return the structured data
        return response_data

    except Exception as e:
        # Raise an HTTP exception if the model or run data is not found
        raise HTTPException(
            status_code=404,
            detail=f"Error: Could not get model or run data. {e}"
        )

# Optional: Add a main block to run the application with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
