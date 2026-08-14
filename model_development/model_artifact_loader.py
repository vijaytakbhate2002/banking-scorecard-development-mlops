import os
import joblib
import json
import pandas as pd


def load_model_artifacts(artifact_dir: str) -> dict:
    """
    Load model artifacts saved by the training notebook.

    Returns a dict with keys:
      - model: sklearn estimator (if found)
      - impute_values: dict
      - binners: dict of feature -> binner object (when joblib-loaded)
      - top5_features: list
      - woe_features: list
      - metrics: dict
      - metadata: dict
      - dataframes: dict of csv-loaded dataframes (iv_df, gain_df, shap_df, importance)
    """
    out = {
        "model": None,
        "impute_values": None,
        "binners": {},
        "top5_features": None,
        "woe_features": None,
        "metrics": None,
        "metadata": None,
        "dataframes": {},
    }

    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

    model_path = os.path.join(artifact_dir, "model_top5.joblib")
    if os.path.exists(model_path):
        out["model"] = joblib.load(model_path)

    impute_path = os.path.join(artifact_dir, "impute_values.joblib")
    if os.path.exists(impute_path):
        out["impute_values"] = joblib.load(impute_path)

    binners_dir = os.path.join(artifact_dir, "binners")
    if os.path.isdir(binners_dir):
        for fname in os.listdir(binners_dir):
            fpath = os.path.join(binners_dir, fname)
            name, ext = os.path.splitext(fname)
            try:
                out["binners"][name] = joblib.load(fpath)
            except Exception:
                out["binners"][name] = None

    for key in ("top5_features.json", "woe_features.json", "metrics.json", "metadata.json"):
        p = os.path.join(artifact_dir, key)
        if os.path.exists(p):
            with open(p, "r", encoding="utf8") as f:
                out_key = key.replace(".json", "")
                out[out_key] = json.load(f)

    # load CSV dataframes if present
    for df_name in ("iv_df.csv", "gain_df.csv", "shap_df.csv", "importance.csv"):
        p = os.path.join(artifact_dir, df_name)
        if os.path.exists(p):
            try:
                out["dataframes"][df_name.replace(".csv", "")] = pd.read_csv(p)
            except Exception:
                out["dataframes"][df_name.replace(".csv", "")] = None

    return out


def list_artifact_dirs(root: str = "artifacts") -> list:
    """Return list of artifact directories under `root` sorted by name (most recent last if timestamped)."""
    if not os.path.isdir(root):
        return []
    items = [os.path.join(root, p) for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    items.sort()
    return items
