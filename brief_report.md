# Brief Report — Primriq Lead Prioritization

## 1. Business Understanding & Problem Formulation

Primriq's counselling team cannot call every lead immediately. The objective is to rank
incoming leads so counsellors spend time on the leads most likely to engage. I formulated
this as a **supervised binary classification** problem with two related targets built
from the historical data:

- **`target_call_picked`** (primary modelling target): did the lead pick up the call?
  This is the cleanest possible target — it requires no subjective judgement of "interest"
  and directly measures a precondition for any value at all.
- **`target_meaningful_engagement`** (secondary/business target): did the call result in
  a genuinely positive outcome (Enrolled, Interested, Requested Callback, Follow-up
  Required, Needs More Information)?

In this dataset the two targets turned out to be identical — every record where the call
was picked up also has one of the positive outcome labels (no "picked but not interested"
case exists in this sample). I deliberately kept both targets distinct in the code
(`src/preprocessing.py::build_targets`) because in real production data, a lead can pick
up the phone and still say "not interested" — the two questions are conceptually
different even though this particular sample doesn't distinguish them.

The final output for each new lead — predicted probability, a 0-100 priority score, a
High/Medium/Low category, and a recommended call order — is designed to be directly
usable by a non-technical counselling team, not just a data science artifact.

## 2. Data Cleaning

| Issue | Decision | Reasoning |
|---|---|---|
| City mixed case (`Noida`/`NOIDA`) | Title-cased all categorical text columns | Prevents the model from treating the same city as two different categories |
| Phone numbers with <10 digits (~2%) | Stripped non-digits, flagged short numbers as `invalid_phone` rather than dropping rows | Preserves all other valid information in the row; "bad phone number" is itself a useful operational signal |
| `Highest_Qualification` missing (~3.7%) | Filled with `"Unknown"` | Avoids losing rows; "didn't state qualification" can be informative |
| `Expected_Salary_LPA` missing (~9.5-13%) | Filled with median + added `salary_missing` flag | Median is robust to skew; the flag preserves the "didn't disclose" signal separately |
| `Message` missing (~2.5-2.8%) | Filled with empty string | Keeps text-derived features (length, keywords) well-defined (default to 0) instead of crashing |
| `Callback_Timestamp` missing (~36%, historical only) | Left as missing | Expected outcome of calls that were never returned — not a data error |
| Full-row duplicates / duplicate Lead_IDs | Dropped (none found in the provided files, but pipeline checks for both) | A lead should have one enquiry record |
| Same phone, different enquiry records | **Kept**, captured instead via `is_repeat_enquirer` feature | A genuine repeat-enquiry pattern, not a data error to clean away |

All logic lives in `src/preprocessing.py` so the notebook, training script, prediction
script, and Streamlit app all clean data identically.

## 3. Exploratory Data Analysis

Key findings (see `notebooks/eda_and_modeling.ipynb` for full charts):

- **Lead source**: pickup rates vary by source — referral and direct-website leads
  generally show different engagement patterns from cold social-media clicks, motivating
  the `source_quality_score` feature.
- **Programs**: enquiry volume is fairly evenly spread across the five programs offered
  (Generative AI, Machine Learning, Data Science, Data Analytics, Business Analytics).
- **Message length/word count**: leads who write longer, more specific messages have
  visibly different pickup rates than those who write one-line or blank messages — this
  became the single strongest predictive feature.
- **Experience & qualification**: weaker but present relationship with pickup behaviour.
- **Class balance**: the primary target is moderately imbalanced (~64% picked / 36% not
  picked) — not severe enough to require synthetic oversampling.

## 4. Feature Engineering

Implemented in `src/preprocessing.py::engineer_features`:

- **Text features**: `message_length`, `word_count`, `has_career_switch_keywords`,
  `has_urgent_keywords`, `has_question_mark`, `message_is_empty`.
- **Time features**: `submission_hour/day/month/dayofweek`, `submission_is_weekend`,
  `preferred_callback_part_of_day`, `preferred_callback_hour` (a representative hour
  mapped from the stated preference bucket).
- **Experience/qualification**: `experience_bucket` (Fresher / Junior / Mid / Senior /
  Expert), `years_since_graduation`.
- **Source quality**: `source_quality_score`, a domain-informed (not outcome-leaked)
  ranking of lead-source intent strength.
- **Interaction features**: `career_switcher_signal` (Job Seeker status OR career-switch
  keywords in message), `weekend_x_source_quality`.
- **Data-quality-derived features**: `salary_missing`, `invalid_phone`, `is_repeat_enquirer`.

## 5. Machine Learning Approach & Evaluation

Three models were trained and compared with an 80/20 stratified train-test split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0.660 | 0.847 | 0.572 | 0.683 | **0.741** | **0.847** |
| XGBoost | 0.670 | 0.778 | 0.678 | 0.725 | 0.731 | 0.830 |
| Logistic Regression | 0.666 | 0.840 | 0.591 | 0.694 | 0.729 | 0.830 |

*(exact numbers also saved to `outputs/model_comparison.csv` — re-running `train.py` may
shift them slightly run-to-run due to model randomness)*

**Model selection**: Random Forest was selected as the final model based on best
ROC-AUC and PR-AUC — these metrics measure ranking quality across all thresholds, which
is exactly what a lead-prioritization use case needs (we care about getting the *order*
right, not just a single yes/no cutoff). XGBoost showed a better F1/recall trade-off at
the default 0.5 threshold, so it remains a strong alternative if the business prefers to
optimise for catching more true positives even at the cost of some extra wasted calls —
this is a tunable choice via the `priority_category` thresholds in `predict.py`.

**Class imbalance handling**: `class_weight="balanced"` (Logistic Regression, Random
Forest) and `scale_pos_weight` (XGBoost) were used rather than SMOTE/oversampling, since
the imbalance (~64/36) is moderate and this approach avoids distorting the one-hot
encoded categorical feature space that oversampling can corrupt.

**Validation**: stratified train/test split (80/20) preserving target class proportions;
cross-validation could be added for a more robust estimate in a future iteration, noted
as a limitation below.

## 6. Prediction on New Leads & Explainability

`outputs/predictions.csv` contains, for every lead in `new_incoming_leads.csv`:
`Lead_ID`, `predicted_probability`, `priority_score` (0-100), `priority_category`
(High ≥0.65, Medium 0.40-0.65, Low <0.40), and `recommended_callback_order`.

**Top influential features** (from `outputs/feature_importance.csv`): `message_length`
and `word_count` dominate, followed by submission-time features (`submission_day`,
`submission_hour`), `has_urgent_keywords`, `Graduation_Year`/`Expected_Salary_LPA`, and
`has_career_switch_keywords`. Intuitively: leads who write detailed messages and submit
during active hours/days are more reachable and engaged.

**Limitations**:
- Trained on a moderate, single-snapshot historical dataset (2,500 rows) that appears
  synthetic/sample data — real-world distributions may differ and should be re-validated
  once live outcomes accumulate.
- `target_call_picked` and `target_meaningful_engagement` coincide in this sample only;
  the model has not actually learned to distinguish "answered but uninterested" from
  "answered and interested," since no such examples exist here.
- The model cannot see context outside recorded fields (tone, urgency conveyed in a real
  conversation, recent life events, etc.).
- Feature importance reflects correlation, not guaranteed causation.

**When human judgement should override the model**: VIP/referral leads with known
business value, leads with explicit urgent requests not well captured by the keyword
list, unusual but high-potential profiles, and any case where a counsellor has direct
prior context about the lead that the model cannot access.

## 7. Production Considerations

- **Serving**: at current volume, scheduled **batch scoring** (e.g. a daily cron job
  running `src/predict.py` against the day's new leads) is sufficient and simplest to
  operate. For ad-hoc/interactive use, the bundled Streamlit app (`app.py`) supports
  on-demand upload-and-score.
- **Retraining**: schedule periodic retraining (e.g. monthly) on accumulated outcomes
  via `src/train.py`, validating the new model against a held-out set before promoting
  it to replace `models/best_model.pkl`.
- **Monitoring**: track (a) prediction-score distribution drift over time, (b) input
  feature distribution drift (e.g. population stability index on key features like
  `Lead_Source`, `message_length`), and (c) live calibration — does the realised pickup
  rate among "High priority" leads actually stay above the realised rate among "Low
  priority" leads?
- **Versioning**: `models/feature_columns.json` records the exact feature list and
  chosen model name alongside each `best_model.pkl`; in a larger deployment this would
  be extended to a proper model registry (e.g. MLflow) with timestamped versions.
- **Data drift detection**: alert if new-lead feature distributions diverge
  significantly from the training distribution (e.g. a new lead source/program appears,
  or message lengths shift dramatically), since this signals the model may need
  retraining sooner than scheduled.

## 8. Assumptions

- Both provided CSVs share the same logical schema even though `new_incoming_leads.csv`
  naturally lacks the outcome columns.
- "Picked up the call" is treated as the most reliable, least-ambiguous proxy for
  "worth calling first," in line with the assignment's suggested formulations.
- The `source_quality_score` ranking is a domain-reasoned assumption (referral/website >
  search/webinar/LinkedIn > Instagram), not derived from the outcome data itself, to
  avoid leakage.
- Priority category thresholds (0.65 / 0.40) were chosen to produce a roughly balanced
  three-way split for this dataset and should be tuned against actual counsellor
  capacity in production.
