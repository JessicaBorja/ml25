# model.py
from pathlib import Path
import joblib
from datetime import datetime
import os

from sklearn.linear_model import LogisticRegression

CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


class PurchaseModel:
    def __init__(self, C=1.0, class_weight=None, max_iter=1000, solver="liblinear", random_state=42):
        self.params = dict(C=C, class_weight=class_weight, max_iter=max_iter, solver=solver, random_state=random_state)
        self.model = LogisticRegression(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # Devuelve prob de clase positiva en la segunda columna
        return self.model.predict_proba(X)

    def get_config(self):
        return {"estimator": "LogisticRegression", **self.params}

    def save(self, prefix: str):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{now}.pkl"
        filepath = Path(MODELS_DIR) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath = os.path.abspath(filepath)
        joblib.dump(self, filepath)
        print(f"{repr(self)} || Model saved to {filepath}")
        return filepath

    def load(self, filename: str):
        filepath = Path(MODELS_DIR) / filename
        model = joblib.load(filepath)
        print(f"{self.__repr__} || Model loaded from {filepath}")
        return model
