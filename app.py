from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib, os
import pandas as pd

MODEL_PATH = os.environ.get("MODEL_PATH", "price_model.joblib")

app = FastAPI(title="PricePredictor", version="0.1")

model = None
load_error = None

@app.on_event("startup")
def load_model():
    global model, load_error
    try:
        model = joblib.load(MODEL_PATH)
        load_error = None
    except Exception as e:
        model = None
        load_error = str(e)

class PredictRequest(BaseModel):
    dish_name: str
    quantity: int = Field(1, ge=1)
    is_combo: int = Field(0, ge=0)
    is_weekend: int = Field(0, ge=0)

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    # Build a pandas DataFrame — the scikit-learn pipeline expects a 2D array / DataFrame
    try:
        X_df = pd.DataFrame([{
            "dish_name": req.dish_name,
            "quantity": req.quantity,
            "is_combo": req.is_combo,
            "is_weekend": req.is_weekend
        }])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    try:
        preds = model.predict(X_df)    # now pipeline will apply OneHotEncoder etc.
        return {"success": True, "predicted_price1": float(preds[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
