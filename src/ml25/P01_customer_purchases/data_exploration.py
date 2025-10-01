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
    categorical_cols = train_df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(y=train_df[col], order=train_df[col].value_counts().index)
        plt.title(f"Counts of {col}")
        plt.show()
