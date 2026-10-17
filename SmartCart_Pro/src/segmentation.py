"""Reusable, leakage-aware customer segmentation pipeline for SmartCart."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA

SPENDING_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]
REQUIRED_COLS = [
    "Education", "Marital_Status", "Income", "Kidhome", "Teenhome",
    "Dt_Customer", "Recency", *SPENDING_COLS, "NumDealsPurchases",
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth", "Complain", "Response"
]

def load_and_engineer(path_or_df):
    """Load data and create interpretable customer-level features."""
    df = path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    missing_cols = sorted(set(REQUIRED_COLS) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    # Reference date is derived from the dataset, not the computer clock.
    ref_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (ref_date - df["Dt_Customer"]).dt.days
    df["Age"] = ref_date.year - pd.to_numeric(df["Year_Birth"], errors="coerce")
    df["Total_Spending"] = df[SPENDING_COLS].sum(axis=1)
    df["Total_Purchases"] = df[
        ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    ].sum(axis=1)
    df["Total_Children"] = df[["Kidhome", "Teenhome"]].sum(axis=1)
    df["Web_Purchase_Share"] = (
        df["NumWebPurchases"] / df["Total_Purchases"].replace(0, np.nan)
    ).fillna(0)
    df["Deal_Purchase_Share"] = (
        df["NumDealsPurchases"] / df["Total_Purchases"].replace(0, np.nan)
    ).fillna(0)
    df["Avg_Spend_Per_Purchase"] = (
        df["Total_Spending"] / df["Total_Purchases"].replace(0, np.nan)
    ).fillna(0)
    df["Living_With"] = df["Marital_Status"].map({
        "Married": "Partner", "Together": "Partner", "Single": "Alone",
        "Divorced": "Alone", "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
    }).fillna("Unknown")
    df["Education_Group"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"
    }).fillna("Unknown")
    # Filter only clearly implausible values; keep the source rows in the raw data.
    df = df[df["Age"].between(18, 100) & (df["Income"].isna() | df["Income"].between(0, 250000))].copy()
    return df.reset_index(drop=True)

def feature_columns():
    """Features available at segmentation time; exclude outcome/response fields."""
    numeric = [
        "Income", "Age", "Customer_Tenure_Days", "Recency", "Total_Spending",
        "Total_Purchases", "Total_Children", "NumDealsPurchases",
        "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
        "NumWebVisitsMonth", "Web_Purchase_Share", "Deal_Purchase_Share",
        "Avg_Spend_Per_Purchase", *SPENDING_COLS
    ]
    categorical = ["Education_Group", "Living_With"]
    return numeric, categorical

def prepare_matrix(df):
    """Fit preprocessing only on supplied rows and return scaled features + transformer."""
    numeric, categorical = feature_columns()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    # Clip extreme numeric tails before scaling so a single unusual customer does not
    # dominate distance-based clustering. Raw values remain unchanged for reporting.
    model_df = df.copy()
    for col in numeric:
        series = pd.to_numeric(model_df[col], errors="coerce")
        lower, upper = series.quantile(0.01), series.quantile(0.99)
        model_df[col] = series.clip(lower=lower, upper=upper)
    transformer = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ], remainder="drop", verbose_feature_names_out=False)
    matrix = transformer.fit_transform(model_df)
    return matrix, transformer

def fit_segmentation(df, k=4, random_state=42):
    """Fit KMeans and return labels, model, metrics and PCA projection."""
    matrix, transformer = prepare_matrix(df)
    model = KMeans(n_clusters=int(k), random_state=random_state, n_init=20)
    labels = model.fit_predict(matrix)
    metrics = {
        "silhouette": float(silhouette_score(matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
        "inertia": float(model.inertia_),
    }
    pca = PCA(n_components=2, random_state=random_state)
    projection = pca.fit_transform(matrix)
    return {
        "data": df.copy(), "matrix": matrix, "transformer": transformer,
        "model": model, "labels": labels, "metrics": metrics,
        "pca": pca, "projection": projection
    }

def evaluate_k_values(df, k_values=range(2, 9)):
    """Compare candidate K values using complementary clustering metrics."""
    matrix, _ = prepare_matrix(df)
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=int(k), random_state=42, n_init=20)
        labels = model.fit_predict(matrix)
        rows.append({
            "k": int(k),
            "silhouette": float(silhouette_score(matrix, labels)),
            "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
            "inertia": float(model.inertia_)
        })
    return pd.DataFrame(rows)

def make_cluster_profile(df, labels):
    """Business-readable cluster profile; response is described, not used for fitting."""
    data = df.copy()
    data["Cluster"] = labels
    numeric = [
        "Income", "Age", "Total_Spending", "Total_Purchases", "Recency",
        "Customer_Tenure_Days", "Total_Children", "NumDealsPurchases",
        "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
        "NumWebVisitsMonth", "Response"
    ]
    profile = data.groupby("Cluster")[numeric].mean(numeric_only=True).round(2)
    profile.insert(0, "Customers", data.groupby("Cluster").size())
    profile["Customer_Share_%"] = (
        data["Cluster"].value_counts(normalize=True).sort_index() * 100
    ).round(1)
    profile["Campaign_Response_Rate_%"] = (
        data.groupby("Cluster")["Response"].mean() * 100
    ).round(1)
    return profile.reset_index()
