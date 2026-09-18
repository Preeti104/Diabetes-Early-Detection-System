# Diabetes Early Detection System — Web Application

A Flask + HTML5/CSS3 web interface for the trained XGBoost diabetes-screening
model.

## What's included

```
diabetes_webapp/
├── app.py                       # Flask backend: preprocessing + prediction
├── requirements.txt
├── model/
│   ├── XGBoost_model.pkl        # trained model (from your notebook)
│   ├── scaler.pkl               # fitted StandardScaler (reproduced)
│   ├── feature_columns.json     # exact one-hot column order the model expects
│   ├── categorical_options.json # dropdown choices, taken from the training data
│   └── numeric_ranges.json      # min/max per numeric field, for input hints
├── templates/
│   └── index.html               # patient intake form + result panel
└── static/
    └── style.css
```

## Setup

```bash
cd diabetes_webapp
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.

## How it works

1. The patient intake form (`templates/index.html`) collects the same 14 raw
   fields present in `Diabetes_Dataset.xlsx` (Gender, Age, Physical Activity,
   Smoking Status, Alcohol Intake, Glucose, Blood Pressure, Skin Thickness,
   Insulin, BMI, Cholesterol, Diabetes Pedigree Function, Family History,
   Hypertension).
2. On submit, `app.py` validates the input (FR2/FR6), one-hot encodes the
   categorical fields to match the training-time column layout, applies the
   same `StandardScaler`, and calls the trained XGBoost model.
3. The result panel shows the predicted class and the model's probability
   for the "Diabetic" class, with a clear "not a diagnosis" disclaimer per
   the proposal's Ethical Considerations (Section 12).

## Mapping to the proposal

| Proposal requirement | Where it's implemented |
|---|---|
| FR1 — accept patient input attributes | `templates/index.html` intake form |
| FR2 — validate submitted input values | `validate()` in `app.py` |
| FR3 — preprocess submitted data | `build_feature_row()` in `app.py` |
| FR4 — apply trained ML model | `model.predict` / `predict_proba` in `app.py` |
| FR5 — display detection output | Result panel in `templates/index.html` |
| FR6 — handle invalid/incomplete input | Validation banner with field-level messages |
| NFR1 — usability | Grouped form sections, plain-language labels |
| NFR6 — privacy | No patient data is persisted; each request is stateless |
