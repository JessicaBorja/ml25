#Hice copia del documento para poder hacer mis propias graficas sin arruinar el otro :)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../datasets/customer_purchases/"


def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    df = pd.read_csv(fullfilename)
    return df

if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    print(train_df.info())
    print(train_df.describe(include="all"))
    test_df = read_csv("customer_purchases_test")
    print(test_df.columns)
    numeric_cols = ["item_price", "item_avg_rating", "item_num_ratings", "customer_item_views", "purchase_item_rating"]
    for col in numeric_cols:
        if col in train_df.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(train_df[col], bins=50)
            plt.title(f"Distribution of {col}")
            plt.show()
    categorical_cols = ["customer_gender", "item_id", "item_category", "item_img_filename", "item_release_date", "purchase_timestamp"]

#relación genero-producto
plt.figure(figsize=(10,6))
sns.countplot(
    x="item_category",
    hue="customer_gender",
    data=train_df,
    order=train_df["item_category"].value_counts().index
)
plt.title("Distribution of Gender by Item Category")
plt.xticks(rotation=45)
plt.show()

#Relación genero-color
plt.figure(figsize=(12,6))
sns.countplot(
    y="item_img_filename",       # filas con colores
    hue="customer_gender",       # separa por género
    data=train_df,
    order=train_df["item_img_filename"].value_counts().index,
    palette={"male" : "green", "female": "purple"}
)

plt.title("Distribution of Gender by Clothing Color")
plt.xlabel("Count")
plt.ylabel("Clothing Color")
plt.legend(title="Gender")
plt.show()

#Estaciones de mayor compra en el año

# Hacer formato datetime
train_df['purchase_timestamp'] = pd.to_datetime(train_df['purchase_timestamp'])
train_df['item_release_date'] = pd.to_datetime(train_df['item_release_date'])

# Datos que quiero: mes y día de la semana
train_df['purchase_by_month'] = train_df['purchase_timestamp'].dt.month
train_df['purchase_dayofweek'] = train_df['purchase_timestamp'].dt.dayofweek

# Compras por mes

plt.figure(figsize=(12,6))
sns.countplot(
    x='purchase_by_month',
    data=train_df,
    order=sorted(train_df['purchase_by_month'].unique())
)
plt.title("Compras por mes")
plt.xlabel("Mes (1 = Enero, 12 = Diciembre)")
plt.ylabel("Número de compras")
plt.show()

#  Compras por día de la semana

plt.figure(figsize=(12,6))
sns.countplot(
    x='purchase_dayofweek',
    data=train_df,
    order=sorted(train_df['purchase_dayofweek'].unique())
)
plt.title("Compras por día de la semana (0 = Lunes)")
plt.xlabel("Día de la semana")
plt.ylabel("Número de compras")
plt.show()

#  Diferencia entre lanzamiento y compra (tiempo hasta la compra)

train_df['days_to_buy_after_release'] = (
    train_df['purchase_timestamp'] - train_df['item_release_date']
).dt.days

plt.figure(figsize=(10,5))
sns.histplot(train_df['days_to_buy_after_release'], bins=50)
plt.title("Días entre lanzamiento y compra")
plt.xlabel("Días desde el lanzamiento")
plt.ylabel("Número de compras")
plt.show()

# Porcentaje de 'early adopters' (compras dentro de 30 días)
early_buyers = (train_df['days_to_buy_after_release'] <= 30).mean()
print(f"Porcentaje de compras dentro de los primeros 30 días del lanzamiento: {early_buyers*100:.2f}%")
