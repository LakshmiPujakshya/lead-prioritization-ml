# Primriq Lead Prioritization

A machine learning solution that predicts which incoming leads are most likely to pick up
a counsellor's call, so that the limited counselling team can focus on the right leads
first.

## Problem Framing

Rather than a vague "priority score," the project formulates two concrete prediction
targets from the historical data:

1. **`target_call_picked`** (primary) — will this lead answer the call? This is the
   cleanest, most actionable target: a call no one answers produces zero business value
   regardless of how "good" the lead looked.
2. **`target_meaningful_engagement`** (secondary) — did the call lead to genuine interest
   (Enrolled / Interested / Requested Callback / Follow-up Required / Needs More
   Information)? In this particular dataset the two targets happen to coincide (every
   picked call in the sample has a positive-leaning outcome), but the pipeline keeps
   them separate since this is unlikely to hold in real production data.

The final deliverable for each new lead is a **predicted probability of answering**,
a **0-100 priority score**, a **High/Medium/Low priority category**, and a
**recommended callback order** — everything a counsellor needs to plan their day.

## Project Structure

```
primriq_project/
├── data/
│   ├── historical_leads.csv          # provided
│   └── new_incoming_leads.csv        # provided
├── notebooks/
│   └── eda_and_modeling.ipynb        # full EDA + modeling walkthrough (executed, with outputs)
├── src/
│   ├── preprocessing.py              # cleaning + feature engineering (shared by everything)
│   ├── train.py                      # trains & compares models, saves best one
│   └── predict.py                    # scores new leads -> predictions.csv
├── models/
│   ├── best_model.pkl                # trained sklearn Pipeline (preprocessing + model)
│   └── feature_columns.json          # feature list + chosen model metadata
├── outputs/
│   ├── predictions.csv               # final predictions for new_incoming_leads.csv
│   ├── model_comparison.csv          # metrics for all candidate models
│   └── feature_importance.csv        # ranked feature importances
├── app.py                            # Streamlit app for interactive scoring
├── requirements.txt
├── README.md                         # this file
├── brief_report.md                   # methodology write-up
└── ai_assisted_development.md        # AI tool usage disclosure
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (reads data/historical_leads.csv)
python src/train.py

# 3. Score new leads (reads data/new_incoming_leads.csv by default)
python src/predict.py
# Output: outputs/predictions.csv

# 4. (Optional) Run the interactive app
streamlit run app.py
```

To score a different file: `python src/predict.py --input path/to/leads.csv --output path/to/out.csv`

## Data Cleaning Decisions (summary)

- **City** — mixed-case values (`Noida` / `NOIDA`) normalised via title-casing.
- **Phone** — non-digit characters stripped; numbers with fewer than 10 digits after
  cleaning flagged as `invalid_phone` rather than imputed.
- **Highest_Qualification** (~3-4% missing) — filled with `"Unknown"`; still informative.
- **Expected_Salary_LPA** (~9-13% missing) — filled with median + `salary_missing` flag
  added so "didn't disclose" remains a usable signal.
- **Message** (~2-3% missing) — filled with empty string so text features default to 0.
- **Callback_Timestamp** (~36% missing, historical only) — left as-is; missing simply
  means the call wasn't returned, which is expected, not an error.
- **Duplicates** — exact full-row duplicates and duplicate `Lead_ID`s are removed;
  repeat enquiries from the same phone number across *different* records are kept and
  instead captured as an `is_repeat_enquirer` feature.

Full reasoning is documented inline in `src/preprocessing.py` and in `brief_report.md`.

## Model

Three models are trained and compared: Logistic Regression (interpretable baseline),
Random Forest, and XGBoost. Class imbalance (~64%/36%) is handled with
`class_weight="balanced"` / `scale_pos_weight` rather than synthetic oversampling, to
avoid distorting the one-hot encoded categorical feature space. The model with the
best ROC-AUC is selected as the final model, since ranking quality (not just binary
accuracy) is what matters for a call-prioritization use case. See
`outputs/model_comparison.csv` for exact metrics from the latest run.

## Limitations & Human Oversight

This is a decision-support tool, not a replacement for counsellor judgement. See
`brief_report.md` §6 for a full discussion of limitations and the situations where a
human should override the model's recommendation (VIP/referral leads, urgent explicit
requests, unusual high-value profiles, etc.).

## Production Notes

See `brief_report.md` §7 for deployment, monitoring, retraining, and drift-detection
recommendations.
