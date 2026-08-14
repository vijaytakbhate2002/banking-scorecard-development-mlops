"""FastAPI app to serve the trained credit scorecard model.

Endpoints:
- GET /health -> simple liveness
- GET /metadata -> model metadata and feature list
- POST /predict -> accepts JSON records, returns `probability` and `score`

This app loads the latest artifact directory under `artifacts/` at startup using
`model_development.model_artifact_loader`.
"""

from typing import List, Dict, Any
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_development.model_artifact_loader import load_model_artifacts, list_artifact_dirs


app = FastAPI(title="Credit Scorecard Model API")


class PredictionRecord(BaseModel):
	data: Dict[str, Any]


class PredictionRequest(BaseModel):
	records: List[PredictionRecord]


class PredictionItem(BaseModel):
	probability: float
	score: float


class PredictionResponse(BaseModel):
	predictions: List[PredictionItem]


# Globals populated at startup
ARTIFACT_DIR = None
ART = {}
MODEL = None
TOP_FEATURES = []
BINS = {}
IMPUTE = {}


def preprocess_dates(df: pd.DataFrame) -> pd.DataFrame:
	DATE_COLS = ["issue_d", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d"]
	out = df.copy()
	for col in DATE_COLS:
		if col not in out.columns:
			continue
		try:
			if col == "issue_d":
				out[col] = pd.to_datetime(out[col], errors="coerce")
			else:
				out[col] = pd.to_datetime(out[col], format="%b-%Y", errors="coerce")
			out[col] = out[col].map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
		except Exception:
			out[col] = out[col]
	return out


def _make_woe_df(df: pd.DataFrame, top_features: List[str]):
	out = pd.DataFrame(index=df.index)
	for col in top_features:
		if col in BINS and BINS[col] is not None:
			try:
				out[col] = BINS[col].transform(df[col], metric="woe")
			except Exception:
				out[col] = 0.0
		else:
			out[col] = 0.0
	return out


@app.on_event("startup")
def startup_event():
	global ARTIFACT_DIR, ART, MODEL, TOP_FEATURES, BINS, IMPUTE
	dirs = list_artifact_dirs("model_development\\artifacts")
	if not dirs:
		raise RuntimeError("No artifact directories found under 'artifacts/'. Run training notebook first.")
	ARTIFACT_DIR = dirs[-1]
	ART = load_model_artifacts(ARTIFACT_DIR)
	MODEL = ART.get("model")
	TOP_FEATURES = ART.get("top5_features") or []
	BINS = ART.get("binners") or {}
	IMPUTE = ART.get("impute_values") or {}


@app.get("/health")
def health():
	return {"status": "ok", "artifact_dir": ARTIFACT_DIR}


@app.get("/metadata")
def metadata():
	md = ART.get("metadata") or {}
	return {"metadata": md, "top_features": TOP_FEATURES}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
	if MODEL is None:
		raise HTTPException(status_code=500, detail="Model not loaded")

	records = [r.data for r in req.records]
	if not records:
		raise HTTPException(status_code=400, detail="No records provided")

	df = pd.DataFrame(records)
	# Ensure all required columns exist
	df = preprocess_dates(df)

	# Impute missing values using saved mapping
	for col, val in IMPUTE.items():
		if col in df.columns:
			df[col] = df[col].fillna(val)
		else:
			df[col] = val
		# keep categorical columns as string
		if not pd.api.types.is_numeric_dtype(df[col]):
			df[col] = df[col].astype(str)

	# Build WoE features for top features
	X_woe = _make_woe_df(df, TOP_FEATURES)

	# Align column order
	X_woe = X_woe[TOP_FEATURES]

	probs = MODEL.predict_proba(X_woe)[:, 1]
	preds = []
	for p in probs:
		# example score: map probability to 300-850 range (simple linear scaling)
		score = float(round(p * 550 + 300, 2))
		preds.append(PredictionItem(probability=float(p), score=score))

	return PredictionResponse(predictions=preds)

