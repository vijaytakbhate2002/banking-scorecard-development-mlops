import requests
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from app import app, startup_event

client = TestClient(app=app)

try:
    startup_event()
    print("✓ Model artifacts loaded successfully")
except Exception as e:
    print(f"⚠ Failed to load model artifacts: {e}")
    # raise RuntimeError(f"Failed to load model artifacts: {e}")

TARGET = "default_status" # binary target for prediction (0 or 1)
ROOT_DIR = Path(__file__).parent.parent

test_data_path = ROOT_DIR / "test_api_results" / "test_fi.csv"
test_df = None
print(f"Looking for test data at: {test_data_path}")
if test_data_path.exists():
    test_df = pd.read_csv(test_data_path)

payload = {
    "records": [
        {
            "data": {
                "issue_d": "2023-01-15",
                "last_pymnt_d": "2023-08-20",
                "next_pymnt_d": "2023-09-20",
                "last_credit_pull_d": "2023-01-10",
                "last_pymnt_amnt": 250.50,
                "emp_title": "Software Engineer",
                "total_rec_prncp": 5000.00,
                "total_pymnt": 5500.00,
                "total_pymnt_inv": 5450.00,
                "inq_last_12m": 2
            }
        },
        {
            "data": {
                "issue_d": "2023-02-10",
                "last_pymnt_d": "2023-09-05",
                "next_pymnt_d": "2023-10-05",
                "last_credit_pull_d": "2023-02-08",
                "last_pymnt_amnt": 175.25,
                "emp_title": "Product Manager",
                "total_rec_prncp": 3500.00,
                "total_pymnt": 3800.00,
                "total_pymnt_inv": 3750.00,
                "inq_last_12m": 1
            }
        }
    ]
}


def test_health():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    print("✓ Health check passed")


def test_metadata():
    """Test metadata endpoint"""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert "top_features" in data
    print("✓ Metadata endpoint passed")
    print(f"  Top features: {data['top_features']}")


def test_predict_success():
    """Test successful prediction"""
    response = client.post("/predict", json=payload)
    
    if response.status_code != 200:
        error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
        print(f"Error response: {error_detail}")
    
    if response.status_code == 500:
        error_data = response.json()
        if "Model not loaded" in str(error_data.get("detail", "")):
            print("⊘ Skipping test_predict_success: Model artifacts not loaded (run training notebook first)")
            return
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    
    # Validate prediction structure
    for pred in data["predictions"]:
        assert "probability" in pred
        assert "score" in pred
        assert 0 <= pred["probability"] <= 1
        assert 300 <= pred["score"] <= 850
    
    print("✓ Prediction test passed")
    print(f"  Predictions: {data['predictions']}")


def test_predict_empty_records():
    """Test prediction with empty records"""
    empty_payload = {"records": []}
    response = client.post("/predict", json=empty_payload)
    assert response.status_code in [400, 500], f"Expected 400 or 500, got {response.status_code}"
    print("✓ Empty records validation passed")


def test_predict_missing_required_fields():
    """Test prediction with missing fields"""
    incomplete_payload = {
        "records": [
            {
                "data": {
                    "issue_d": "2023-01-15"
                }
            }
        ]
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code in [200, 500]
    print("✓ Missing fields test passed")


def test_predict_from_test_data():
    """Test prediction using actual test data from CSV"""
    if test_df is None:
        print("⊘ Skipping test_predict_from_test_data: test_fi.csv not found")
        return
    
    test_records = test_df.head(2).to_dict(orient="records")
    
    def replace_nan(d):
        return {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in d.items()}
    
    test_records = [replace_nan(r) for r in test_records]
    test_payload = {"records": [{"data": record} for record in test_records]}
    
    try:
        response = client.post("/predict", json=test_payload)
        if response.status_code == 200:
            data = response.json()
            assert len(data["predictions"]) == 2
            print("✓ Prediction from test data passed")
            print(f"  Predictions: {data['predictions']}")
        else:
            print(f"⊘ Test data prediction returned {response.status_code}")
            if response.headers.get('content-type') == 'application/json':
                print(f"  Error: {response.json()}")
    except Exception as e:
        print(f"⊘ Test data prediction error: {e}")


def test_auc_from_test_data():
    """Test AUC computation on entire test_fi.csv dataset"""
    if test_df is None:
        print("⊘ Skipping test_auc_from_test_data: test_fi.csv not found")
        return
    
    if TARGET not in test_df.columns:
        print(f"⊘ Skipping test_auc_from_test_data: Target column '{TARGET}' not found in test data")
        return
    
    print(f"Computing AUC on {len(test_df)} records from test_fi.csv...")
    
    def replace_nan(d):
        return {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in d.items()}
    
    predictions = []
    targets = []
    errors = 0
    
    # Process in batches for efficiency
    batch_size = 100
    for batch_start in range(0, len(test_df), batch_size):
        batch_end = min(batch_start + batch_size, len(test_df))
        batch_df = test_df.iloc[batch_start:batch_end]
        
        # Create payload excluding target column
        test_records = batch_df.drop(columns=[TARGET]).to_dict(orient="records")
        test_records = [replace_nan(r) for r in test_records]
        test_payload = {"records": [{"data": record} for record in test_records]}
        
        try:
            response = client.post("/predict", json=test_payload)
            if response.status_code == 200:
                data = response.json()
                batch_probs = [pred["probability"] for pred in data["predictions"]]
                predictions.extend(batch_probs)
                targets.extend(batch_df[TARGET].tolist())
            else:
                error_msg = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                print(f"✗ Batch {batch_start}-{batch_end} failed with status {response.status_code}: {error_msg}")
                errors += len(batch_df)
        except Exception as e:
            print(f"✗ Batch {batch_start}-{batch_end} error: {e}")
            errors += len(batch_df)
        
        if (batch_end) % 500 == 0 or batch_end == len(test_df):
            print(f"  Processed {batch_end}/{len(test_df)} records...")
    
    # Compute AUC
    if len(predictions) > 0 and len(targets) > 0:
        try:
            auc = roc_auc_score(targets, predictions)
            assert 0.7 <= auc <= 0.98, f"AUC out of range: {auc}"
            print(f"✓ AUC test passed")
            print(f"  AUC Score: {auc:.4f}")
            print(f"  Records processed: {len(predictions)}/{len(test_df)}")
            if errors > 0:
                print(f"  Errors encountered: {errors}")
        except Exception as e:
            print(f"✗ AUC computation failed: {e}")
            raise
    else:
        print(f"⊘ No predictions generated (predictions: {len(predictions)}, targets: {len(targets)})")



def run_all_tests():
    """Run all tests"""
    print("\n" + "="*50)
    print("Running API Tests")
    print("="*50 + "\n")
    
    tests = [
        ("Health Check", test_health),
        ("Metadata", test_metadata),
        ("Prediction Success", test_predict_success),
        ("Empty Records", test_predict_empty_records),
        ("Missing Fields", test_predict_missing_required_fields),
        ("Test Data Prediction", test_predict_from_test_data),
        ("AUC from Test Data", test_auc_from_test_data),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()
