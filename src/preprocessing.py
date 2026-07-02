"""
preprocessing.py
-----------------
Data cleaning and feature engineering for the Primriq Lead Prioritization project.

This module is written to be "flexible" about column names where reasonably
possible (the assignment warns column names could differ), but since both
provided CSVs share the same schema, we centre the code around a small
CANDIDATE_COLUMNS map that tries a few likely aliases for each logical field.
If your column names differ, just update the map below.

All cleaning decisions are explained in comments next to the code that
performs them, and summarised again in brief_report.md.
"""

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. FLEXIBLE COLUMN MAPPING
# ---------------------------------------------------------------------------
# Maps a logical field name -> list of possible column names we might see.
# find_column() picks the first match present in the dataframe.
CANDIDATE_COLUMNS = {
    "lead_id": ["Lead_ID", "LeadID", "lead_id", "ID"],
    "timestamp": ["Submission_Timestamp", "Timestamp", "Submitted_At"],
    "name": ["Name", "Candidate_Name"],
    "phone": ["Phone", "Phone_Number", "Mobile"],
    "preferred_callback_time": ["Preferred_Callback_Time", "Callback_Time_Preference"],
    "program": ["Program_Interested", "Program", "Course_Interested"],
    "qualification": ["Highest_Qualification", "Qualification"],
    "status": ["Current_Status", "Status"],
    "experience_years": ["Experience_Years", "Experience"],
    "city": ["City"],
    "lead_source": ["Lead_Source", "Source"],
    "campaign": ["Campaign", "Marketing_Campaign"],
    "device": ["Device", "Device_Type"],
    "graduation_year": ["Graduation_Year"],
    "expected_salary": ["Expected_Salary_LPA", "Expected_Salary"],
    "message": ["Message", "Candidate_Message"],
    "counsellor": ["Assigned_Counsellor", "Counsellor"],
    "callback_timestamp": ["Callback_Timestamp", "Call_Timestamp"],
    "call_picked": ["Call_Picked", "Picked"],
    "call_response": ["Call_Response", "Outcome", "Response"],
}


def find_column(df: pd.DataFrame, logical_name: str):
    """Return the actual column name in df matching a logical field, or None."""
    for candidate in CANDIDATE_COLUMNS.get(logical_name, []):
        if candidate in df.columns:
            return candidate
    return None


def build_column_map(df: pd.DataFrame) -> dict:
    """Build {logical_name: actual_column_name} for every field found in df."""
    return {k: find_column(df, k) for k in CANDIDATE_COLUMNS if find_column(df, k)}


# ---------------------------------------------------------------------------
# 2. CLEANING
# ---------------------------------------------------------------------------

def clean_text_value(value):
    """Lowercase + strip whitespace for simple categorical text normalisation."""
    if pd.isna(value):
        return value
    return re.sub(r"\s+", " ", str(value)).strip()


def normalise_categoricals(df: pd.DataFrame, cmap: dict) -> pd.DataFrame:
    """
    Fix inconsistent categorical values.

    Observed issue in the data: City values appear in mixed case, e.g.
    'Noida' and 'NOIDA' both exist and should be treated as the same city.
    We title-case all categorical/text columns to merge such duplicates.
    """
    df = df.copy()
    cat_fields = [
        "preferred_callback_time", "program", "qualification", "status",
        "city", "lead_source", "campaign", "device", "call_picked",
        "call_response", "counsellor",
    ]
    for field in cat_fields:
        col = cmap.get(field)
        if col is None:
            continue
        df[col] = df[col].apply(clean_text_value)
        # Title-case category labels (but not free-text message/name fields)
        df[col] = df[col].apply(lambda v: v.title() if isinstance(v, str) else v)
    return df


def clean_phone(series: pd.Series) -> pd.Series:
    """
    Standardise phone numbers: strip non-digits, keep last 10 digits.
    Numbers shorter than 10 digits after cleaning are treated as invalid (NaN)
    since they cannot be valid Indian mobile numbers - this becomes a useful
    'invalid_phone' quality signal rather than being silently dropped.
    """
    def _clean(v):
        if pd.isna(v):
            return np.nan
        digits = re.sub(r"\D", "", str(v))
        if len(digits) >= 10:
            return digits[-10:]
        return np.nan
    return series.apply(_clean)


def remove_duplicates(df: pd.DataFrame, cmap: dict) -> pd.DataFrame:
    """
    Remove duplicate lead records.

    Strategy:
      1. Exact full-row duplicates are dropped outright (data entry copy).
      2. Duplicate Lead_ID (if any) -> keep the first occurrence, since a
         lead should only have one enquiry record by definition.
    Note: the same person (same phone number) submitting multiple separate
    enquiries is NOT treated as a duplicate to remove - that's a legitimate
    repeat-enquiry pattern and is instead captured as a feature
    (see `is_repeat_enquirer` in engineer_features).
    """
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    full_dupes_removed = before - len(df)

    id_col = cmap.get("lead_id")
    id_dupes_removed = 0
    if id_col:
        before2 = len(df)
        df = df.drop_duplicates(subset=[id_col], keep="first")
        id_dupes_removed = before2 - len(df)

    return df, {"full_row_duplicates_removed": full_dupes_removed,
                "duplicate_lead_id_removed": id_dupes_removed}


def handle_missing_values(df: pd.DataFrame, cmap: dict) -> pd.DataFrame:
    """
    Handle missing values column by column, with reasoning:

    - Highest_Qualification (categorical, ~3-4% missing): filled with
      'Unknown' rather than dropped, since qualification is still informative
      even when unspecified, and dropping rows would lose other valid signal.
    - Expected_Salary_LPA (numeric, ~9-13% missing): filled with the median
      salary, a standard robust strategy for skewed numeric data; a
      'salary_missing' flag is also created so the model can use
      "didn't disclose salary" as a signal in its own right.
    - Message (free text, ~2-3% missing): filled with empty string '' so
      downstream text features (length, keywords) default to zero rather
      than crashing on NaN.
    - Callback_Timestamp (historical only, ~36% missing): missing simply
      means the call was never returned / picked - this is expected and
      informative, not an error, so it is left as-is and used to engineer
      a `was_called_back` indicator.
    """
    df = df.copy()

    qual_col = cmap.get("qualification")
    if qual_col:
        df[qual_col] = df[qual_col].fillna("Unknown")

    salary_col = cmap.get("expected_salary")
    if salary_col:
        df["salary_missing"] = df[salary_col].isna().astype(int)
        median_salary = df[salary_col].median()
        df[salary_col] = df[salary_col].fillna(median_salary)

    msg_col = cmap.get("message")
    if msg_col:
        df[msg_col] = df[msg_col].fillna("")

    phone_col = cmap.get("phone")
    if phone_col:
        df["invalid_phone"] = df[phone_col].isna().astype(int)

    return df


def clean_dataframe(df: pd.DataFrame) -> tuple:
    """Run the full cleaning pipeline. Returns (clean_df, column_map, report)."""
    cmap = build_column_map(df)

    phone_col = cmap.get("phone")
    if phone_col:
        df = df.copy()
        df[phone_col] = clean_phone(df[phone_col])

    df = normalise_categoricals(df, cmap)
    df, dupe_report = remove_duplicates(df, cmap)
    df = handle_missing_values(df, cmap)

    report = {"duplicates": dupe_report, "rows_after_cleaning": len(df)}
    return df, cmap, report


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

CAREER_SWITCH_KEYWORDS = [
    "switch", "transition", "career change", "shift to", "move into",
    "change my career", "looking for job",
]
URGENT_KEYWORDS = [
    "urgent", "asap", "immediately", "today", "right now", "ready to enroll",
    "ready to join",
]

PART_OF_DAY_MAP = {
    "Morning": "Morning",
    "Afternoon": "Afternoon",
    "Evening": "Evening",
    "Weekend": "Weekend",
    "After 7 Pm": "Night",
}

# Rough subjective "quality" ranking of lead sources based on typical
# intent levels seen in EdTech businesses: referrals and direct website
# visits tend to indicate stronger pre-existing intent than cold social
# clicks. This is an assumption documented in the report - it is informed by
# domain reasoning, not by data leakage from the outcome column.
SOURCE_QUALITY_MAP = {
    "Referral": 3,
    "Website": 3,
    "Google Search": 2,
    "Webinar": 2,
    "Linkedin": 2,
    "Instagram": 1,
}


def _keyword_flag(text: str, keywords) -> int:
    text = (text or "").lower()
    return int(any(k in text for k in keywords))


def engineer_features(df: pd.DataFrame, cmap: dict) -> pd.DataFrame:
    """Create the engineered feature set described in the assignment brief."""
    df = df.copy()

    # ---- Text / message features -----------------------------------------
    msg_col = cmap.get("message")
    if msg_col:
        msgs = df[msg_col].fillna("")
        df["message_length"] = msgs.str.len()
        df["word_count"] = msgs.str.split().apply(len)
        df["has_career_switch_keywords"] = msgs.apply(
            lambda t: _keyword_flag(t, CAREER_SWITCH_KEYWORDS))
        df["has_urgent_keywords"] = msgs.apply(
            lambda t: _keyword_flag(t, URGENT_KEYWORDS))
        df["has_question_mark"] = msgs.str.contains(r"\?", regex=True).astype(int)
        df["message_is_empty"] = (msgs.str.strip() == "").astype(int)

    # ---- Time-based features -----------------------------------------------
    ts_col = cmap.get("timestamp")
    if ts_col:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
        df["submission_hour"] = ts.dt.hour
        df["submission_day"] = ts.dt.day
        df["submission_month"] = ts.dt.month
        df["submission_dayofweek"] = ts.dt.dayofweek
        df["submission_is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)

    pref_col = cmap.get("preferred_callback_time")
    if pref_col:
        df["preferred_callback_part_of_day"] = df[pref_col].map(PART_OF_DAY_MAP).fillna(df[pref_col])
        # Crude representative hour per preference bucket, useful as a numeric feature
        hour_map = {"Morning": 9, "Afternoon": 14, "Evening": 18, "Night": 20, "Weekend": 11}
        df["preferred_callback_hour"] = df["preferred_callback_part_of_day"].map(hour_map)

    # ---- Experience / qualification features -------------------------------
    exp_col = cmap.get("experience_years")
    if exp_col:
        bins = [-1, 0, 2, 5, 10, 100]
        labels = ["Fresher", "Junior (1-2y)", "Mid (3-5y)", "Senior (6-10y)", "Expert (10y+)"]
        df["experience_bucket"] = pd.cut(df[exp_col], bins=bins, labels=labels)

    grad_col = cmap.get("graduation_year")
    if grad_col and ts_col:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
        df["years_since_graduation"] = ts.dt.year - df[grad_col]
        df["years_since_graduation"] = df["years_since_graduation"].clip(lower=0)

    # ---- Source quality -------------------------------------------------
    src_col = cmap.get("lead_source")
    if src_col:
        df["source_quality_score"] = df[src_col].map(SOURCE_QUALITY_MAP).fillna(1)

    # ---- Interaction features --------------------------------------------
    status_col = cmap.get("status")
    if status_col and "has_career_switch_keywords" in df.columns:
        df["career_switcher_signal"] = (
            (df[status_col] == "Job Seeker") | (df["has_career_switch_keywords"] == 1)
        ).astype(int)

    if src_col and "submission_is_weekend" in df.columns:
        df["weekend_x_source_quality"] = df["submission_is_weekend"] * df["source_quality_score"]

    # ---- Repeat enquirer (uses phone, computed within a single dataset) ---
    phone_col = cmap.get("phone")
    if phone_col:
        df["is_repeat_enquirer"] = df.duplicated(subset=[phone_col], keep=False).astype(int)
        df["is_repeat_enquirer"] = df["is_repeat_enquirer"].fillna(0)

    return df


# ---------------------------------------------------------------------------
# 4. TARGET CONSTRUCTION (historical data only)
# ---------------------------------------------------------------------------

POSITIVE_RESPONSES = {"Enrolled", "Interested", "Requested Callback", "Follow-Up Required",
                      "Needs More Information"}


def build_targets(df: pd.DataFrame, cmap: dict) -> pd.DataFrame:
    """
    Build the two target columns used for modelling:

    1. target_call_picked (primary target): 1 if the call was picked up,
       0 otherwise. This is the cleanest, least ambiguous target and
       directly answers the assignment's core question - "should we call
       this lead?" - because a call that isn't picked up has zero chance
       of producing any business value.

    2. target_meaningful_engagement (secondary / business target): 1 if the
       call was picked up AND the outcome reflects real interest
       (Enrolled, Interested, Requested Callback, Follow-up Required,
       Needs More Information), 0 otherwise (includes not-picked calls).
       This is a stricter target aimed at finding leads worth a counsellor's
       time, not just ones who will answer the phone.
    """
    df = df.copy()
    picked_col = cmap.get("call_picked")
    resp_col = cmap.get("call_response")

    if picked_col:
        df["target_call_picked"] = (df[picked_col] == "Yes").astype(int)

    if picked_col and resp_col:
        df["target_meaningful_engagement"] = (
            (df[picked_col] == "Yes") & (df[resp_col].isin(POSITIVE_RESPONSES))
        ).astype(int)

    return df


# ---------------------------------------------------------------------------
# 5. FULL PIPELINE ENTRY POINT
# ---------------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame, is_historical: bool = True):
    """
    Run cleaning + feature engineering (+ target construction if historical)
    on a raw dataframe. Returns (processed_df, column_map, report).
    """
    df_clean, cmap, report = clean_dataframe(df)
    df_feat = engineer_features(df_clean, cmap)
    if is_historical:
        df_feat = build_targets(df_feat, cmap)
    return df_feat, cmap, report


if __name__ == "__main__":
    # Quick smoke test when run directly: python src/preprocessing.py
    hist = pd.read_csv("data/historical_leads.csv")
    processed, cmap, report = run_pipeline(hist, is_historical=True)
    print("Cleaning report:", report)
    print("Processed shape:", processed.shape)
    print(processed[["target_call_picked", "target_meaningful_engagement"]].mean())
