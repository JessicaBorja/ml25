import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from sklearn.feature_extraction.text import CountVectorizer

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../../datasets/customer_purchases/"


def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    df = pd.read_csv(fullfilename)
    return df


def save_df(df, filename: str):
    # Guardar
    save_path = os.path.join(DATA_DIR, filename)
    df.to_csv(save_path, index=False)
    print(f"df saved to {save_path}")


def extract_customer_features(train_df):
    # Consideren: que atributos del cliente siguen disponibles en prueba?
    customer_feat={
        train_df.groupby("customer_id")
        .agg(
            customer_date_of_birth=("customer_date_of_birth", "first"),
            customer_gender=("customer_gender", "first"),
            customer_signup_date=("customer_signup_date", "first"),
            avg_item_price=("item_price", "mean"),
            total_purchases=("item_id", "count"),
            unique_categories=("item_category", "nunique")
        )
        .reset_index()
    }
    save_df(customer_feat, "customer_features.csv")


def process_df(df, training=True):
    """
    Investiga las siguientes funciones de SKlearn y determina si te son útiles
    - OneHotEncoder
    - StandardScaler
    - CountVectorizer
    - ColumnTransformer
    """
    # Ejemplo de codigo para guardar y cargar archivos con pickle
    # savepath = Path(DATA_DIR) / "preprocessor.pkl"
    # if training:
    #     processed_array = preprocessor.fit_transform(df)
    #     joblib.dump(preprocessor, savepath)
    # else:
    #     preprocessor = joblib.load(savepath)
    #     processed_array = preprocessor.transform(df)

    # processed_df = pd.DataFrame(processed_array, columns=[...])
    # return processed_df
    df['customer_date_of_birth'] = pd.to_datetime(df['customer_date_of_birth'], errors='coerce')
    df['customer_signup_date'] = pd.to_datetime(df['customer_signup_date'], errors='coerce')
    df['item_release_date'] = pd.to_datetime(df['item_release_date'], errors='coerce')
    
    today = pd.to_datetime("2025-09-21")
    df['customer_age'] = (today - df['customer_date_of_birth']).dt.days // 365
    df['days_since_signup'] = (today - df['customer_signup_date']).dt.days
    df['days_since_release'] = (today - df['item_release_date']).dt.days
    
    # Drop original date columns
    df = df.drop(columns=['customer_date_of_birth', 'customer_signup_date', 'item_release_date'])


def preprocess(raw_df, training=False):
    """
    Agrega tu procesamiento de datos, considera si necesitas guardar valores de entrenamiento.
    Utiliza la bandera para distinguir entre preprocesamiento de entrenamiento y validación/prueba
    """
    customer_feat = pd.read_csv(Path(DATA_DIR) / "customer_features.csv")
    if "customer_id" in raw_df.columns:
        raw_df = raw_df.merge(customer_feat, on="customer_id", how="left")
    processed_df = process_df(raw_df, training)
    return processed_df


def df_to_numeric(df):
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


def read_train_data():
    train_df = read_csv("customer_purchases_train")
    customer_feat = extract_customer_features(train_df)
    ...
    return X, y


def read_test_data():
    test_df = read_csv("customer_purchases_test")
    customer_feat = read_csv("customer_feat.csv")

    # Cambiar por sus datos procesados
    # Prueba no tiene etiquetas
    X_test = test_df
    return X_test


if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    print(train_df.info())
    test_df = read_csv("customer_purchases_test")
    print(test_df.columns)
