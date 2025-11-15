import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ===========================================================
# 1. Clean and validate Age column
# ===========================================================
def clean_age(age):
    """Convert age to an integer, and set invalid ages (<15 or >100) to NaN.
    The dataset contains unrealistic values (e.g., -2, 322), so we clamp to a reasonable range.
    """
    try:
        a = int(age)
    except Exception:
        return np.nan

    if a < 15 or a > 100:
        return np.nan
    return a


# ===========================================================
# 2. Normalize Gender categories
# ===========================================================
def normalize_gender(x):
    """Consolidate messy and diverse gender labels into standardized categories:
    Male, Female, Trans, Non-binary, Other.
    """
    if not isinstance(x, str):
        return "Other"

    s = x.strip().lower()
    male_tokens = ["m", "male", "man", "cis male", "cis-male", "msle", "mail"]
    female_tokens = ["f", "female", "woman", "cis female", "cis-female"]

    if any(tok in s for tok in male_tokens):
        return "Male"
    if any(tok in s for tok in female_tokens):
        return "Female"
    if "trans" in s:
        return "Trans"
    if "non-binary" in s or s == "nb" or "enby" in s or "genderqueer" in s:
        return "Non-binary"

    return "Other"


# ===========================================================
# 3. Main preprocessing function
# ===========================================================
def preprocess_raw_df(df: pd.DataFrame):
    """Clean and prepare the raw survey dataframe.

    Returns:
        X -> Cleaned feature dataframe
        y -> Binary target (treatment: yes=1, no=0)
        preprocessor -> ColumnTransformer containing imputation,
                        scaling, and one-hot encoding steps
    """
    df = df.copy()

    # -------------------------------------------------------
    # 3.1 Drop columns with high missingness or low value
    # -------------------------------------------------------
    to_drop = []

    if "comments" in df.columns:
        to_drop.append("comments")

    if "state" in df.columns:
        to_drop.append("state")

    if "Timestamp" in df.columns:
        to_drop.append("Timestamp")

    df = df.drop(columns=to_drop, errors="ignore")

    # -------------------------------------------------------
    # 3.2 Clean Age values
    # -------------------------------------------------------
    if "Age" in df.columns:
        df["Age"] = df["Age"].apply(clean_age)

    # -------------------------------------------------------
    # 3.3 Normalize Gender values
    # -------------------------------------------------------
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].apply(normalize_gender)

    # -------------------------------------------------------
    # 3.4 Fill missing values for specific columns
    # -------------------------------------------------------
    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].fillna(
            df["self_employed"].mode().iloc[0]
        )

    if "work_interfere" in df.columns:
        df["work_interfere"] = df["work_interfere"].fillna("Unknown")

    # -------------------------------------------------------
    # 3.5 Create target variable
    # -------------------------------------------------------
    if "treatment" not in df.columns:
        raise ValueError("ERROR: Column 'treatment' not found in dataframe.")

    y = (df["treatment"].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=["treatment"])

    # -------------------------------------------------------
    # 3.6 Identify categorical vs numerical columns
    # -------------------------------------------------------
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # -------------------------------------------------------
    # 3.7 Build preprocessing pipeline
    # -------------------------------------------------------
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    return X, y, preprocessor
