"""
train.py
--------
Trains and compares Logistic Regression, Random Forest and XGBoost models
on the historical_leads.csv dataset to predict `target_call_picked`
(whether a lead is likely to pick up the call).

Run:  python src/train.py
Outputs:
  - models/best_model.pkl   (full sklearn Pipeline: preprocessing + model)
  - models/feature_columns.json
  - outputs/model_comparison.csv
  - outputs/feature_importance.csv
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from preprocessing import run_pipeline

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "target_call_picked"  # primary target (see preprocessing.py docstring for why)

# Features used by the model. We deliberately EXCLUDE: Name, Phone,
# Lead_ID, Assigned_Counsellor (operational/PII, not predictive of intent in
# a generalisable way), Callback_Timestamp / Call_Response (these leak the
# outcome we are trying to predict, or only exist after the call happens).
NUMERIC_FEATURES = [
    "Experience_Years", "Expected_Salary_LPA", "Graduation_Year",
    "message_length", "word_count", "has_career_switch_keywords",
    "has_urgent_keywords", "has_question_mark", "message_is_empty",
    "submission_hour", "submission_day", "submission_month",
    "submission_dayofweek", "submission_is_weekend",
    "preferred_callback_hour", "years_since_graduation",
    "source_quality_score", "career_switcher_signal",
    "weekend_x_source_quality", "is_repeat_enquirer",
    "salary_missing", "invalid_phone",
]
CATEGORICAL_FEATURES = [
    "Program_Interested", "Highest_Qualification", "Current_Status",
    "City", "Lead_Source", "Campaign", "Device",
    "experience_bucket", "preferred_callback_part_of_day",
]


def load_and_prepare():
    hist = pd.read_csv("data/historical_leads.csv")
    df, cmap, report = run_pipeline(hist, is_historical=True)
    print("Cleaning report:", report)
    return df


def build_preprocessor(num_feats, cat_feats):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, num_feats),
        ("cat", categorical_pipeline, cat_feats),
    ])


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }


def main():
    df = load_and_prepare()

    num_feats = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = df[num_feats + cat_feats]
    y = df[TARGET]

    print(f"\nClass balance for {TARGET}:")
    print(y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(num_feats, cat_feats)

    # Class imbalance handling: the target is only mildly imbalanced
    # (~64% / 36%), so rather than oversampling (e.g. SMOTE) which can
    # distort categorical one-hot encodings, we use `class_weight="balanced"`
    # for Logistic Regression / Random Forest, and `scale_pos_weight` for
    # XGBoost. This is a lighter-touch, more interpretable approach that's
    # appropriate for this level of imbalance.
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        )

    results = {}
    fitted_pipelines = {}
    for name, model in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"\n--- {name} ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    results_df.to_csv("outputs/model_comparison.csv")
    print("\nModel comparison:\n", results_df)

    # Model selection: we pick the model with the best ROC-AUC (good overall
    # ranking ability, which matters most for *prioritising* leads), as long
    # as recall is reasonably balanced against precision - missing a likely
    # responder is costlier for the business than calling one extra
    # unlikely lead. XGBoost (if available) or Random Forest typically wins
    # on tabular data like this with non-linear feature interactions.
    best_name = results_df.index[0]
    best_pipe = fitted_pipelines[best_name]
    print(f"\nSelected best model: {best_name}")

    joblib.dump(best_pipe, "models/best_model.pkl")
    with open("models/feature_columns.json", "w") as f:
        json.dump({"numeric": num_feats, "categorical": cat_feats,
                    "target": TARGET, "best_model": best_name}, f, indent=2)

    # Feature importance (tree models) or coefficients (logistic regression)
    try:
        feature_names = (
            num_feats +
            list(best_pipe.named_steps["preprocessor"]
                 .named_transformers_["cat"].named_steps["encoder"]
                 .get_feature_names_out(cat_feats))
        )
        model = best_pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            importances = None

        if importances is not None:
            fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            fi_df = fi_df.sort_values("importance", ascending=False)
            fi_df.to_csv("outputs/feature_importance.csv", index=False)
            print("\nTop 15 features:\n", fi_df.head(15))
    except Exception as e:
        print("Could not compute feature importance:", e)

    print("\nDetailed classification report for best model:")
    print(classification_report(y_test, best_pipe.predict(X_test)))


if __name__ == "__main__":
    main()
