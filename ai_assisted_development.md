# AI-Assisted Development

## Tools Used

### 1. Claude (claude.ai / Anthropic)
**Primary assistant** for this project.

#### How it assisted:
- Helped structure the overall project layout (src/, notebooks/, models/, outputs/ convention)
  and talked through the trade-offs of different problem formulations (binary classification
  vs ranking vs regression) before I committed to `target_call_picked` as the primary target.
- Generated first drafts of `src/preprocessing.py`, `src/train.py`, `src/predict.py`, and
  `app.py`, which I then reviewed, adjusted, and extended.
- Discussed the pros and cons of SMOTE/oversampling vs `class_weight="balanced"` for the
  moderate class imbalance found in this dataset — the recommendation to use the simpler
  weight-based approach was well reasoned and I adopted it after verifying the logic.
- Suggested `source_quality_score` as a feature and helped me think through why it should be
  a domain-reasoned assumption rather than derived from outcome data (to avoid target leakage).
- Helped draft docstrings, inline comments, and all three markdown documentation files
  (README.md, brief_report.md, this file) based on my findings from actually running the code.

#### How I validated its outputs:
- Ran every generated Python file directly and inspected stdout for correctness
  (`python src/preprocessing.py`, `python src/train.py`, `python src/predict.py`).
- Executed the full notebook cell-by-cell and verified chart outputs matched the underlying
  data distributions I had seen in manual exploration with `pd.value_counts()` etc.
- Checked that feature importances made intuitive sense: `message_length` / `word_count`
  ranking highest is plausible (detailed messages indicate engaged leads), while
  `salary_missing` and `invalid_phone` appearing lower-but-nonzero is also reasonable.
- Verified that predictions.csv contained a lead for every row of new_incoming_leads.csv and
  that the priority score distribution was not degenerate (all the same value, etc.).

#### Limitations / incorrect suggestions encountered:
- Claude initially suggested using SMOTE for class imbalance handling, without checking whether
  the pipeline was mixing categorical (one-hot encoded) and numeric features. I pushed back and
  it agreed that `class_weight="balanced"` is safer for this pipeline structure.
- First draft of the notebook ran `train.py` via subprocess, which needed a path adjustment
  depending on whether the notebook is run from the repo root or the `notebooks/` directory —
  I patched this after seeing the path error.
- The `source_quality_score` feature was originally sourced by Claude from outcome-column
  statistics, which would have introduced target leakage. I caught this during review and
  required it to be a hard-coded domain assumption instead.

---

### 2. GitHub Copilot (used incidentally in VS Code for boilerplate)
**Minor assist** for repeated patterns: sklearn `Pipeline` boilerplate, argparse setup in
`predict.py`, and standard matplotlib subplot layouts. All suggestions were reviewed and
accepted, modified, or rejected inline — no block was used without reading it.

---

## Overall AI Usage Summary

| Task | AI involvement | Validation method |
|---|---|---|
| Problem formulation | High — discussed options, chose together | Manual business reasoning check |
| Data exploration | Low — ran my own pandas commands, AI helped interpret | Directly inspected outputs |
| Cleaning decisions | Medium — AI suggested initial strategy, I confirmed | Checked value_counts before/after |
| Feature engineering | Medium — co-designed feature list | Reviewed importances + sanity check |
| Model training code | High — AI drafted, I ran and tuned | Checked all metrics, compared models |
| Explainability section | Medium — AI drafted text, I verified against actual feature_importance.csv | Cross-referenced file contents |
| Streamlit app | High — AI drafted, I tested locally | Ran `streamlit run app.py` |
| Documentation | High — AI drafted from my findings, I edited | Read every word before accepting |

The generated code was treated as a starting point, not a final answer. Every file was
read, tested, and where needed corrected before inclusion in the submission.
