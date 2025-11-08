from fastapi import FastAPI
import joblib
from .schemas import InputRow

app = FastAPI()
_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load("models/best.joblib")
    return _model

@app.get("/health")
def health(): return {"ok": True}

@app.post("/predict")
def predict(row: InputRow):
    model = get_model()
    X = [[row.f1, row.f2, row.f3_cat]]
    y = model.predict(X)[0]
    proba = getattr(model.named_steps["model"], "predict_proba", None)
    p = proba(model.named_steps["pre"].transform([[row.f1, row.f2, row.f3_cat]])).tolist()[0] if proba else None
    return {"y": int(y), "proba": p}
