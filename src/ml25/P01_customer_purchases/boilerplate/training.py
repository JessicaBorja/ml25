from sklearn.model_selection import train_test_split
from ml25.P01_customer_purchases.boilerplate.evil_data_processing import read_train_raw, process_df
from ml25.P01_customer_purchases.boilerplate.model import PurchaseModel
from ml25.P01_customer_purchases.boilerplate.utils import setup_logger
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score
import numpy as np
import pandas as pd

def run_training(classifier_name: str = "logreg_baseline"):
    logger = setup_logger(f"training_{classifier_name}")
    logger.info("Cargando y procesando train...")

    dfm_raw, y = read_train_raw()

    # split ANTES del preprocesamiento
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        dfm_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ajusta preprocesador SOLO en train y reutilízalo en val
    X_train = process_df(X_train_raw, training=True)
    X_val   = process_df(X_val_raw,   training=False)

    print("Distribución de etiquetas (train):", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Distribución de etiquetas (val):",   dict(zip(*np.unique(y_val,   return_counts=True))))

    logger.info(f"Shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}")

    model = PurchaseModel(C=1.0, class_weight=None, max_iter=1000, solver="liblinear", random_state=42)
    logger.info(f"Hiperparámetros: {model.get_config()}")

    logger.info("Entrenando...")
    model.fit(X_train, y_train)

    logger.info("Validando...")
    y_val_pred  = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_val_pred)
    f1  = f1_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_proba)

    logger.info(f"Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    logger.info("\n" + classification_report(y_val, y_val_pred, digits=4))

    path = model.save(prefix=classifier_name)
    logger.info(f"Modelo guardado en: {path}")

    # logging a CSV
    exp_row = pd.DataFrame([{
        "model": classifier_name, "params": str(model.get_config()),
        "val_acc": acc, "val_f1": f1, "val_auc": auc
    }])
    exp_file = path.replace("trained_models", "logs").replace(".pkl", "_experiments.csv")
    try:
        prev = pd.read_csv(exp_file)
        pd.concat([prev, exp_row], ignore_index=True).to_csv(exp_file, index=False)
    except Exception:
        exp_row.to_csv(exp_file, index=False)
    logger.info(f"Registro de experimento actualizado: {exp_file}")

if __name__ == "__main__":
    run_training("logreg_baseline")
