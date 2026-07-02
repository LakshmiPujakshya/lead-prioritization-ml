"""
predict.py
----------
Loads the trained model (models/best_model.pkl) and scores new leads
(default: data/new_incoming_leads.csv), producing a ranked predictions file.

Run:  python src/predict.py
      python src/predict.py --input path/to/other_leads.csv --output path/to/out.csv

Output columns:
  Lead_ID, predicted_probability, priority_score, priority_category,
  recommended_callback_order
"""

import argparse
import json
import joblib
import pandas as pd

from preprocessing import run_pipeline


def load_model(model_path="models/best_model.pkl", feature_path="models/feature_columns.json"):
    model = joblib.load(model_path)
    with open(feature_path) as f:
        feat_info = json.load(f)
    return model, feat_info


def predict_leads(input_csv: str, output_csv: str,
                   model_path="models/best_model.pkl",
                   feature_path="models/feature_columns.json"):
    model, feat_info = load_model(model_path, feature_path)
    num_feats, cat_feats = feat_info["numeric"], feat_info["categorical"]

    raw = pd.read_csv(input_csv)
    df, cmap, report = run_pipeline(raw, is_historical=False)
    print("Cleaning report:", report)

    # Make sure every expected column exists, even if missing in new data
    for col in num_feats + cat_feats:
        if col not in df.columns:
            df[col] = pd.NA

    X = df[num_feats + cat_feats]
    proba = model.predict_proba(X)[:, 1]

    id_col = cmap.get("lead_id", "Lead_ID")
    result = pd.DataFrame({
        "Lead_ID": df[id_col] if id_col in df.columns else range(len(df)),
        "predicted_probability": proba,
    })

    # Priority score: 0-100 scaled version of the predicted probability,
    # easy for non-technical counsellors to read at a glance.
    result["priority_score"] = (result["predicted_probability"] * 100).round(1)

    # Priority category via simple, business-friendly thresholds.
    # These cut points were chosen so that roughly the top third of
    # probability mass becomes "High" priority - tune as call-centre
    # capacity changes.
    def categorize(p):
        if p >= 0.65:
            return "High"
        elif p >= 0.40:
            return "Medium"
        return "Low"

    result["priority_category"] = result["predicted_probability"].apply(categorize)

    # Recommended order: simply rank by probability, descending.
    result = result.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
    result["recommended_callback_order"] = result.index + 1

    result = result[["Lead_ID", "predicted_probability", "priority_score",
                      "priority_category", "recommended_callback_order"]]
    result["predicted_probability"] = result["predicted_probability"].round(4)

    result.to_csv(output_csv, index=False)
    print(f"Saved predictions for {len(result)} leads -> {output_csv}")
    print(result["priority_category"].value_counts())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/new_incoming_leads.csv")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--model", default="models/best_model.pkl")
    parser.add_argument("--features", default="models/feature_columns.json")
    args = parser.parse_args()

    predict_leads(args.input, args.output, args.model, args.features)
