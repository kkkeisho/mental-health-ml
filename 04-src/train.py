# src/train.py

import argparse
import joblib
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Import preprocessing pipeline from src/preprocess.py
from src.preprocess import preprocess_raw_df


def get_feature_names(preprocessor, X_frame: pd.DataFrame):
    """
    Retrieve the final feature names after preprocessing (numeric + one-hot encoded).
    Works for sklearn >= 1.0.
    """

    # Numeric column names come directly from X
    num_cols = preprocessor.transformers_[0][2]
    num_names = list(num_cols)

    # Categorical feature names come from the OneHotEncoder
    cat_cols = preprocessor.transformers_[1][2]
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]

    try:
        # sklearn >=1.0
        cat_names = list(ohe.get_feature_names_out(cat_cols))
    except AttributeError:
        # Fallback for older sklearn versions
        cat_names = []
        for i, col in enumerate(cat_cols):
            categories = ohe.categories_[i]
            cat_names.extend([f"{col}_{c}" for c in categories])

    return num_names + cat_names


def main(args):
    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===============================================================
    # 1) Load raw dataset
    # ===============================================================
    df = pd.read_csv(data_path)

    # ===============================================================
    # 2) Apply preprocessing steps (cleaning, encoding, etc.)
    # ===============================================================
    X, y, preprocessor = preprocess_raw_df(df)

    # ===============================================================
    # 3) Train/test split
    #    If you prefer to split manually, replace this block.
    # ===============================================================
    if args.do_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=42,
            stratify=y
        )
    else:
        # If no split: train on full dataset, skip evaluation
        X_train, y_train = X, y
        X_test = y_test = None

    # ===============================================================
    # 4) Build model pipeline (preprocessor + classifier)
    # ===============================================================
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])

    # ===============================================================
    # 5) MLflow experiment setup + training + evaluation
    # ===============================================================
    mlflow.set_experiment(args.experiment)
    mlflow.sklearn.autolog(log_models=False)  # We save the model manually

    with mlflow.start_run():
        # Model training
        pipe.fit(X_train, y_train)

        # Evaluate only when test data exists
        if X_test is not None:
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

            # Log metrics to MLflow
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            print("[Test metrics]", metrics)

        # ===========================================================
        # 6) Save artifacts: trained model + feature names
        # ===========================================================
        model_path = out_dir / "model.pkl"
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(str(model_path))

        # Save one-hot encoded feature names for debugging/explainability
        feat_names = get_feature_names(preprocessor, X)
        feature_file = out_dir / "feature_names.txt"
        feature_file.write_text("\n".join(feat_names), encoding="utf-8")
        mlflow.log_artifact(str(feature_file))

    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Feature names saved to: {feature_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/survey.csv")
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--experiment", type=str, default="mental-health-train")
    parser.add_argument("--do-split", action="store_true",
                        help="Enable train/test split. If not set, trains on full data.")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    main(args)
