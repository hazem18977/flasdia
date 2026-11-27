# Contributing to Flasdia

Thank you for helping improve Flasdia.

## Ground rules

- Keep the project educational. Do not present Flasdia as a diagnostic or clinical tool.
- Do not add real patient data, credentials, API keys, or private datasets.
- Treat bundled pickle models as untrusted unless their provenance is documented. Only load model files from trusted sources.
- Keep the medical disclaimer visible in user-facing changes.

## Development checks

Run the syntax check before opening a pull request:

```bash
python -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in ['app.py', 'train_svm_model.py', 'test_feature_importance.py', 'test_api_endpoints.py']]"
```

If your change affects predictions or explanations, also run the relevant manual checks described in `README.md` and explain the result in the pull request.

## Pull requests

Please keep pull requests focused. Describe what changed, why it helps, and how you checked it. For model or data changes, include the dataset source, license, feature order, evaluation method, and limitations.
