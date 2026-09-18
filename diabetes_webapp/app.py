"""
Diabetes Early Detection System - Web Application

A Flask application that serves a patient-screening interface backed by a
trained XGBoost classifier.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained artefacts once, at startup
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_DIR / "XGBoost_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

with open(MODEL_DIR / "feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)  # exact column order the model/scaler expect

with open(MODEL_DIR / "categorical_options.json") as f:
    CATEGORICAL_OPTIONS = json.load(f)  # dropdown choices, taken from the training data

with open(MODEL_DIR / "numeric_ranges.json") as f:
    NUMERIC_RANGES = json.load(f)  # min/max seen in training data, for input hints

CATEGORICAL_COLS = list(CATEGORICAL_OPTIONS.keys())
NUMERIC_COLS = list(NUMERIC_RANGES.keys())

FIELD_LABELS = {
    "Gender": "Gender",
    "Age": "Age (years)",
    "Physical Activity": "Physical activity level",
    "Smoking Status": "Smoking status",
    "Alcohol Intake": "Alcohol intake",
    "Glucose": "Glucose (mg/dL)",
    "Blood Pressure": "Blood pressure (mm Hg)",
    "Skin Thickness": "Skin thickness (mm)",
    "Insulin": "Insulin (mu U/mL)",
    "BMI": "Body mass index (BMI)",
    "Cholesterol": "Cholesterol (mg/dL)",
    "Diabetes Pedigree Function": "Diabetes pedigree function",
    "Family History": "Family history of diabetes",
    "Hypertension": "Hypertension",
}

# Fields grouped for the form, in clinical-intake order
FIELD_GROUPS = [
    {"title": "Patient details", "fields": ["Gender", "Age"]},
    {
        "title": "Clinical measurements",
        "fields": [
            "Glucose", "Blood Pressure", "Skin Thickness", "Insulin",
            "BMI", "Cholesterol", "Diabetes Pedigree Function",
        ],
    },
    {
        "title": "Lifestyle & history",
        "fields": [
            "Physical Activity", "Smoking Status", "Alcohol Intake",
            "Family History", "Hypertension",
        ],
    },
]

# Team shown on the Home page
TEAM_MEMBERS = [
    {"name": "Preeti", "id": "S20250496"},
    {"name": "Shoaibuddin Mohammad", "id": "S20250574"},
    {"name": "Parth Munjal", "id": "S20250149"},
    {"name": "Vinit", "id": "S20250433"},
    {"name": "Ajay Kumar", "id": ""},
]


def build_feature_row(form_values: dict) -> pd.DataFrame:
    """Turn raw form input into the exact one-hot-encoded, ordered row the
    model was trained on (matches training-time preprocessing: pd.get_dummies
    with drop_first=True, then StandardScaler)."""
    raw = {}
    for col in NUMERIC_COLS:
        raw[col] = float(form_values[col])
    for col in CATEGORICAL_COLS:
        raw[col] = form_values[col]

    row_df = pd.DataFrame([raw])
    encoded = pd.get_dummies(row_df, columns=CATEGORICAL_COLS, drop_first=True)

    # Align to the training-time column set/order; any dummy column not
    # produced by this single row (because that category wasn't chosen)
    # is filled with 0, exactly as it would be in the full training matrix.
    encoded = encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return encoded


def validate(form_values: dict) -> list:
    """Basic input validation."""
    errors = []
    for col in NUMERIC_COLS:
        val = form_values.get(col, "").strip()
        if val == "":
            errors.append(f"{FIELD_LABELS[col]} is required.")
            continue
        try:
            float(val)
        except ValueError:
            errors.append(f"{FIELD_LABELS[col]} must be a number.")
    for col in CATEGORICAL_COLS:
        val = form_values.get(col, "")
        if val not in CATEGORICAL_OPTIONS[col]:
            errors.append(f"{FIELD_LABELS[col]} is required.")
    return errors


@app.route("/")
def home():
    return render_template("home.html", team_members=TEAM_MEMBERS)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    errors = []
    submitted_values = {}

    if request.method == "POST":
        submitted_values = request.form.to_dict()
        errors = validate(submitted_values)

        if not errors:
            X_row = build_feature_row(submitted_values)
            X_scaled = scaler.transform(X_row)

            prediction = int(model.predict(X_scaled)[0])
            probability = float(model.predict_proba(X_scaled)[0, 1])

            result = {
                "label": "Diabetic indicators detected" if prediction == 1 else "No diabetic indicators detected",
                "is_positive": prediction == 1,
                "probability_pct": round(probability * 100, 1),
            }

    return render_template(
        "predict.html",
        field_groups=FIELD_GROUPS,
        field_labels=FIELD_LABELS,
        categorical_options=CATEGORICAL_OPTIONS,
        numeric_ranges=NUMERIC_RANGES,
        numeric_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        result=result,
        errors=errors,
        values=submitted_values,
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
