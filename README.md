# Flasdia

Flasdia is an educational Flask web application for exploring diabetes risk prediction with classical machine-learning models. It was created as a graduation project and is maintained as an open-source learning project for students, ML practitioners, and health-informatics hobbyists who want a small end-to-end example of a model-backed web form.

> **Medical disclaimer:** Flasdia is not a medical device and does not diagnose, treat, or prevent diabetes. Its output is an educational risk estimate only and must not replace advice from a qualified clinician.

## Features

- Bilingual English/Russian web interface.
- Patient-style input form for age, gender, BMI, HbA1c, blood glucose, hypertension, heart disease, and smoking history.
- Model selection across Random Forest, AdaBoost, Decision Tree, Logistic Regression, and a linear SVM.
- Risk output with a probability-style score and Low/Medium/High risk bands.
- Feature-importance and personalized feature-contribution explanations for the selected model.
- JSON API endpoints for general and personalized feature importance.
- Optional AI-generated advice only when the user explicitly consents and `NVIDIA_API_KEY` is configured.

## Models and data

The Flask app loads bundled pickle model files from `models/` at startup. The included `train_svm_model.py` script demonstrates how a linear SVM can be trained on a synthetic 1,000-row dataset with the same feature layout used by the web form.

The bundled models and synthetic training script are for demonstration and education. They are not clinically validated, and this repository does not claim diagnostic accuracy. A production or research-grade version should document the real dataset source, inclusion criteria, train/test split, calibration, fairness checks, and external validation before any health-related use.

Only load pickle files from sources you trust. Python pickle files can execute code during loading.

## Quick start

```bash
git clone https://github.com/hazem18977/flasdia.git
cd flasdia
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Then open `http://127.0.0.1:5000`.

## Configuration

Copy `.env.example` to `.env` for local settings, or set the variables in your deployment environment:

| Variable | Required | Purpose |
| --- | --- | --- |
| `FLASK_SECRET_KEY` | Yes for deployment | Secret key used by Flask sessions. Use a long random value outside local development. |
| `FLASK_DEBUG` | No | Set to `1` only for local debugging. |
| `NVIDIA_API_KEY` | No | Enables optional AI-generated advice. Leave unset to disable that feature. |

Never commit real API keys or secrets to Git. If a key was ever committed, revoke or rotate it in the provider dashboard; removing it from the latest file does not remove it from Git history.

## API

The web form posts to `/predict`. Feature-importance endpoints are also available:

```text
GET  /api/feature_importance/<model_type>?lang=en&top_n=5
POST /api/feature_importance/<model_type>/personalized
GET  /feature_importance/<model_type>
POST /feature_contribution/<model_type>
```

Supported `model_type` values are `random_forest`, `adaboost`, `decision_tree`, `logistic_regression`, and `svm`.

Personalized feature-importance requests expect a `features` array in this order:

```text
age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level,
gender_Female, gender_Male, smoking_history_No Info, smoking_history_current,
smoking_history_ever, smoking_history_former, smoking_history_never,
smoking_history_not current
```

## Project structure

```text
app.py                         Flask app, prediction routes, explanations, optional AI advice
train_svm_model.py             Synthetic-data SVM training example
models/                        Bundled demo model files loaded by the app
templates/                     Jinja templates for login and prediction pages
static/                        CSS and images
test_feature_importance.py     Manual feature-importance checks
test_api_endpoints.py          Manual API smoke test for a running server
```

## Checks

Run a syntax check without importing the app or loading model files:

```bash
python -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in ['app.py', 'train_svm_model.py', 'test_feature_importance.py', 'test_api_endpoints.py']]"
```

The manual feature-importance script imports `app.py` and loads the bundled models:

```bash
python test_feature_importance.py
```

The API smoke test requires the Flask server to be running and the `requests` package to be available in your environment:

```bash
python app.py
python test_api_endpoints.py
```

## Deployment

The included `Procfile` starts the app with Gunicorn:

```text
web: gunicorn app:app
```

Set `FLASK_SECRET_KEY` in the deployment environment. Set `NVIDIA_API_KEY` only if you want to enable the optional AI advice feature.

## Roadmap

- Document a real, licensed dataset and publish reproducible evaluation metrics.
- Add a model card with intended use, limitations, and bias/fairness notes.
- Convert manual checks into automated tests and CI.
- Remove generated artifacts from version control and keep model provenance explicit.
- Improve accessibility and add screenshots or a hosted demo.

## Contributing

Contributions are welcome. Please read `CONTRIBUTING.md`, keep the medical disclaimer intact, and do not add real patient data, credentials, or unreviewed model files.

## License

MIT. See `LICENSE`.
