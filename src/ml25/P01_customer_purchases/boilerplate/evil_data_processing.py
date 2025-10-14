from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from ml25.P01_customer_purchases.boilerplate.negative_generation import gen_random_negatives
import joblib

DATA_DIR = (Path(__file__).resolve() / "../../../datasets/customer_purchases/").resolve()
DATA_DIR = Path(DATA_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVEPATH_PREPROCESSOR = DATA_DIR / "preprocessor.pkl"
FEATURES_CSV = DATA_DIR / "customer_features.csv"

def _build_train_with_negatives(train_df: pd.DataFrame, n_per_positive: int = 2) -> pd.DataFrame:
    # Columnas base de compra (positivos)
    pos_cols = train_df.columns.tolist()

    # Tablas “maestras” para poder reconstruir negativos con atributos
    item_cols = [
        "item_id","item_title","item_category","item_price","item_img_filename",
        "item_avg_rating","item_num_ratings","item_release_date"
    ]
    cust_cols = ["customer_id","customer_date_of_birth","customer_gender","customer_signup_date"]

    items = train_df[item_cols].drop_duplicates(subset=["item_id"]).copy()
    customers = train_df[cust_cols].drop_duplicates(subset=["customer_id"]).copy()

    # Negativos: (customer_id, item_id, label=0)
    neg_df = gen_random_negatives(train_df, n_per_positive=n_per_positive)  # cols: customer_id, item_id, label

    # Une atributos de cliente e ítem
    neg_full = (
        neg_df.merge(customers, on="customer_id", how="left")
              .merge(items, on="item_id", how="left")
              .copy()
    )
    # Imitar distribuciones de features de compra para que no sean triviales
    pos_views  = train_df["customer_item_views"].dropna()
    if not pos_views.empty:
        neg_full["customer_item_views"] = np.random.choice(pos_views.values, size=len(neg_full))

    pos_devices = train_df["purchase_device"].dropna()
    if not pos_devices.empty:
        neg_full["purchase_device"] = np.random.choice(pos_devices.values, size=len(neg_full))

    pos_rating = train_df["purchase_item_rating"].dropna()
    if not pos_rating.empty:
        neg_full["purchase_item_rating"] = np.random.choice(pos_rating.values, size=len(neg_full))

    # Timestamps: puedes dejarlos NaT (se dropean en process_df) o muestrear fechas válidas si lo prefieres.

    # Completa columnas de evento de compra para negativos
    neg_full["purchase_id"] = -1
    neg_full["purchase_timestamp"] = pd.NaT
    neg_full["customer_item_views"] = 0
    neg_full["purchase_item_rating"] = np.nan
    neg_full["purchase_device"] = "unknown"

    # Asegura mismas columnas que los positivos (si falta alguna, créala)
    for c in pos_cols:
        if c not in neg_full.columns:
            neg_full[c] = pd.NA
    neg_full = neg_full[pos_cols]

    # Concatena y baraja
    df_all = pd.concat([train_df, neg_full], ignore_index=True)
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df_all


def read_csv_stem(stem: str) -> pd.DataFrame:
    """Lee un CSV por su 'stem' dentro de DATA_DIR (sin .csv)."""
    path = (DATA_DIR / f"{stem}.csv").resolve()
    return pd.read_csv(path)


def save_df(df: pd.DataFrame, filename: str) -> None:
    path = (DATA_DIR / filename).resolve()
    df.to_csv(path, index=False)
    print(f"[save_df] df saved to {path}")

# ------------------ Feature engineering ------------------ #
def extract_customer_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega SOLO agregados por cliente que no dupliquen columnas base."""
    customer_feat = (
        train_df.groupby("customer_id").agg(
            avg_item_price=("item_price", "mean"),
            total_purchases=("item_id", "count"),
            unique_categories=("item_category", "nunique"),
        )
        .reset_index()
    )
    save_df(customer_feat, "customer_features.csv")
    return customer_feat


def process_dates(df: pd.DataFrame, today_str: str = "2025-09-21") -> pd.DataFrame:
    df = df.copy()
    today = pd.to_datetime(today_str)
    # Convierte si existen
    for col in ["customer_date_of_birth", "customer_signup_date", "item_release_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Derivados
    if "customer_date_of_birth" in df.columns:
        df["customer_age"] = ((today - df["customer_date_of_birth"]).dt.days // 365).astype("Int64")
    if "customer_signup_date" in df.columns:
        df["days_since_signup"] = (today - df["customer_signup_date"]).dt.days.astype("Int64")
    if "item_release_date" in df.columns:
        df["days_since_release"] = (today - df["item_release_date"]).dt.days.astype("Int64")
    # Drop fechas crudas
    for c in ["customer_date_of_birth", "customer_signup_date", "item_release_date"]:
        if c in df.columns:
            df.drop(columns=c, inplace=True)
    return df


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return pre


def process_df(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    df = process_dates(df)

    # Selección robusta de columnas existentes
    candidate_num = ["item_price", "customer_age", "days_since_signup", "days_since_release", "customer_item_views", "item_avg_rating", "item_num_ratings"]
    candidate_cat = ["customer_gender", "purchase_device", "item_category"]
    num_cols = [c for c in candidate_num if c in df.columns]
    cat_cols = [c for c in candidate_cat if c in df.columns]

    pre = build_preprocessor(num_cols, cat_cols)

    if training:
        arr = pre.fit_transform(df)
        joblib.dump(pre, SAVEPATH_PREPROCESSOR)
    else:
        pre = joblib.load(SAVEPATH_PREPROCESSOR)
        arr = pre.transform(df)

    # Nombres de columnas resultantes
    cat_names = []
    if cat_cols:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        cat_names = list(ohe.get_feature_names_out(cat_cols))
    all_names = num_cols + cat_names

    out = pd.DataFrame(arr, columns=all_names, index=df.index)
    return out


def read_train_data() -> tuple[pd.DataFrame, pd.Series]:
    train_df = read_csv_stem("customer_purchases_train")

    # 👉 si falta la clase 0, genera negativos
    if "label" not in train_df.columns or train_df["label"].nunique() < 2:
        train_df = _build_train_with_negatives(train_df, n_per_positive=2)

    # features de cliente (si no existen)
    if not FEATURES_CSV.exists():
        extract_customer_features(train_df)

    # merge con features de cliente
    cust = pd.read_csv(FEATURES_CSV)
    dfm = train_df.merge(cust, on="customer_id", how="left")

    X = process_df(dfm, training=True)
    y = dfm["label"].astype(int)
    return X, y

def read_train_raw() -> tuple[pd.DataFrame, pd.Series]:
    train_df = read_csv_stem("customer_purchases_train")

    # Asegura dos clases
    if "label" not in train_df.columns or train_df["label"].nunique() < 2:
        train_df = _build_train_with_negatives(train_df, n_per_positive=2)

    # (re)features de cliente
    if not FEATURES_CSV.exists():
        extract_customer_features(train_df)
    cust = pd.read_csv(FEATURES_CSV)

    dfm = train_df.merge(cust, on="customer_id", how="left")
    y = dfm["label"].astype(int)
    # devolvemos las columnas crudas (sin procesar) y la y
    return dfm.drop(columns=["label"]), y


def read_test_data() -> pd.DataFrame:
    test_df = read_csv_stem("customer_purchases_test")
    # Merge features de cliente ya calculadas
    cust = pd.read_csv(FEATURES_CSV)
    dfm = test_df.merge(cust, on="customer_id", how="left")
    X_test = process_df(dfm, training=False)
    return X_test


if __name__ == "__main__":
    # Info rápida
    train_df = read_csv_stem("customer_purchases_train")
    print(train_df.info())
    test_df = read_csv_stem("customer_purchases_test")
    print(test_df.columns)

    # Genera features si no existen y transforma
    X, y = read_train_data()
    print("Train:", X.shape, y.shape)

    X_test = read_test_data()
    print("Test:", X_test.shape)
