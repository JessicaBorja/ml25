from ml25.P01_customer_purchases.boilerplate.evil_data_processing import read_test_data
from pathlib import Path
import joblib
import pandas as pd
import matplotlib_inline as plt

CURRENT_FILE = Path(__file__).resolve()
RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR = CURRENT_FILE.parent / "trained_models"


def run_inference(model_name: str, X: pd.DataFrame) -> pd.DataFrame:
    full_path = MODELS_DIR / model_name
    print(f"Loading model from {full_path}")
    model = joblib.load(full_path)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    results = pd.DataFrame(
        {"ID": X.index, "pred": preds, "prob": probs}
    )
    return results

def plot_roc(y_true, y_proba, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X = read_test_data()
    model_name = "logreg_baseline_20251013_164514.pkl"
    model = joblib.load(MODELS_DIR / model_name)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({"ID": X.index, "pred": preds, "prob": probs})
    out.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    print(f"Saved {len(out)} predictions to {RESULTS_DIR / 'predictions.csv'}")