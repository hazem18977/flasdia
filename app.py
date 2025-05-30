from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from matplotlib.figure import Figure
from openai import OpenAI
import json
import shap
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# DeepSeek API Configuration
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def generate_health_advice(data, predictions, lang='en'):
    """Generate personalized health advice using DeepSeek API"""
    # Combine predictions from all models
    majority_prediction = 1 if sum(pred[0] for pred in predictions.values()) >= len(predictions)/2 else 0

    # Define concise prompts based on risk level
    if lang == 'en':
        prompt = f"""Patient data:
- Age: {data['age']}
- Gender: {data['gender']}
- BMI: {data['bmi']}
- HbA1c: {data['HbA1c_level']}%
- Blood Glucose: {data['blood_glucose_level']} mg/dL (Normal: <140, Prediabetes: 140–199, Diabetes: ≥200)
- Hypertension: {'Yes' if int(data['hypertension']) == 1 else 'No'}
- Heart Disease: {'Yes' if int(data['heart_disease']) == 1 else 'No'}
- Smoking History: {'Current' if data['smoking_history'].lower() == 'current' else 'Former' if data['smoking_history'].lower() == 'former' else 'Never' if data['smoking_history'].lower() == 'never' else 'Ever' if data['smoking_history'].lower() == 'ever' else 'Not Current' if data['smoking_history'].lower() == 'not current' else 'No Info'}
- Risk Level: {'High risk' if majority_prediction == 1 else 'Lower risk'} of diabetes"""
        system_message = "You are a concise medical assistant specializing in diabetes and chronic conditions. Based on the patient's provided information (age, gender, BMI, HbA1c, blood glucose, hypertension, heart disease, smoking status), generate exactly five personalized medical advice sentences, each starting with 'The medical advice is:'. For critical conditions (e.g., HbA1c ≥9% or glucose ≥300 mg/dL, severe hypertension, or critical cardiac issues), include urgent advice to consult a doctor immediately. For prediabetes (HbA1c 5.7–6.4% or glucose 140–199 mg/dL), recommend specific prevention steps (e.g., reduce sugar intake) and consulting a doctor within a week. For diabetes (HbA1c ≥6.5%), advise daily glucose monitoring and dietary changes. For obesity (BMI ≥30), specify a weight loss goal (5-10% of body weight), calorie reduction (500-1000 kcal/day), and exercise (30 min/day). For overweight (BMI 25–29.9), recommend moderate weight loss (3-5%) and exercise (20-30 min/day). For hypertension, advise reducing salt to <2g/day and weekly blood pressure monitoring. For heart disease, recommend a low-saturated-fat diet and cardiac check-ups every 6 months. If the patient is a current smoker, urge quitting smoking and seeking medical support. If a former smoker, encourage continued abstinence and periodic lung/heart screenings. If never smoked, acknowledge as a positive factor. Ensure all advice is practical, actionable, and tailored to the patient's data, avoiding generic phrases like 'maintain a healthy lifestyle'. Do not include any extra explanations or text outside the five advice sentences."
    else:
        prompt = f"""Данные пациента:
- Возраст: {data['age']}
- Пол: {'Женский' if data['gender'] == 'Female' else 'Мужской' if data['gender'] == 'Male' else 'Неизвестно'}
- ИМТ: {data['bmi']}
- HbA1c: {data['HbA1c_level']}%
- Глюкоза: {data['blood_glucose_level']} мг/дл (Норма: <140, Преддиабет: 140–199, Диабет: ≥200)
- Гипертония: {'Да' if int(data['hypertension']) == 1 else 'Нет'}
- Сердечные заболевания: {'Да' if int(data['heart_disease']) == 1 else 'Нет'}
- История курения: {'Текущий курильщик' if data['smoking_history'].lower() == 'current' else 'Бывший курильщик' if data['smoking_history'].lower() == 'former' else 'Никогда не курил' if data['smoking_history'].lower() == 'never' else 'Курил ранее' if data['smoking_history'].lower() == 'ever' else 'Не курит сейчас' if data['smoking_history'].lower() == 'not current' else 'Нет информации'}
- Уровень риска: {'Высокий риск' if majority_prediction == 1 else 'Низкий риск'} диабета"""
        system_message = "detailed thinking off. Вы лаконичный медицинский ассистент, специализирующийся на диабете и хронических заболеваниях. На основе предоставленных данных пациента (возраст, пол, ИМТ, HbA1c, уровень глюкозы в крови, гипертония, сердечные заболевания, статус курения) сгенерируйте ровно пять персонализированных медицинских рекомендаций, каждая из которых начинается с «Медицинская рекомендация:». Для критических состояний (например, HbA1c ≥9% или глюкоза ≥300 мг/дл, тяжелая гипертония или критические сердечные проблемы) включите срочный совет немедленно обратиться к врачу. Для предиабета (HbA1c 5.7–6.4% или глюкоза 140–199 мг/дл) рекомендуйте конкретные меры профилактики (например, снижение потребления сахара) и консультацию с врачом в течение недели. Для диабета (HbA1c ≥6.5%) советуйте ежедневный мониторинг глюкозы и изменения в питании. Для ожирения (ИМТ ≥30) укажите цель снижения веса (5–10% массы тела), сокращение калорий (500–1000 ккал/день) и физические упражнения (30 мин/день). Для избыточного веса (ИМТ 25–29.9) рекомендуйте умеренное снижение веса (3–5%) и упражнения (20–30 мин/день). Для гипертонии советуйте сократить потребление соли до <2 г/день и еженедельный мониторинг артериального давления. Для сердечных заболеваний рекомендуйте диету с низким содержанием насыщенных жиров и кардиологические осмотры каждые 6 месяцев. Если пациент курит в настоящее время, настоятельно советуйте бросить курить и обратиться за медицинской поддержкой. Если пациент бывший курильщик, поощряйте продолжение воздержания и периодические обследования легких/сердца. Если пациент никогда не курил, отметьте это как положительный фактор. Убедитесь, что все рекомендации практичны, конкретны и адаптированы к данным пациента, избегая общих фраз, таких как «вести здоровый образ жизни». Не включайте никаких дополнительных объяснений или текста, кроме пяти рекомендательных предложений.."

    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
            messages=[{
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": prompt
            }],
            temperature=0.1,
            top_p=0.7,
            max_tokens=500,
            stream=False
        )

        advice = completion.choices[0].message.content.strip()
        return advice.rstrip('.')

    except Exception as e:
        print(f"Error generating health advice: {str(e)}")
        return "Consult your doctor for personalized advice." if lang == 'en' else "Проконсультируйтесь с врачом для получения персональных рекомендаций."

# Global variables for models
models = {
    'random_forest': None,
    'adaboost': None,
    'decision_tree': None,
    'logistic_regression': None,
    'svm': None
}

# Model file paths with a more reliable directory structure
MODEL_PATHS = {
    'random_forest': os.path.join('models', 'SR_Random_Forest.pkl'),
    'adaboost': os.path.join('models', 'SR_AdaBoost.pkl'),
    'decision_tree': os.path.join('models', 'SR_Decision_Tree.pkl'),
    'logistic_regression': os.path.join('models', 'SR_Logistic_Regression.pkl'),
    'svm': os.path.join('models', 'svm_linear_pickle.pkl')
}

# Expected features for the models (14 features)
EXPECTED_FEATURES = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'gender_Female', 'gender_Male', 'smoking_history_No Info', 'smoking_history_current',
    'smoking_history_ever', 'smoking_history_former', 'smoking_history_never',
    'smoking_history_not current'
]

# Human-readable feature names for display
FEATURE_DISPLAY_NAMES = {
    'age': {'en': 'Age', 'ru': 'Возраст'},
    'hypertension': {'en': 'Hypertension', 'ru': 'Гипертония'},
    'heart_disease': {'en': 'Heart Disease', 'ru': 'Сердечные заболевания'},
    'bmi': {'en': 'BMI', 'ru': 'ИМТ'},
    'HbA1c_level': {'en': 'HbA1c Level', 'ru': 'Уровень HbA1c'},
    'blood_glucose_level': {'en': 'Blood Glucose', 'ru': 'Уровень глюкозы'},
    'gender_Female': {'en': 'Female Gender', 'ru': 'Женский пол'},
    'gender_Male': {'en': 'Male Gender', 'ru': 'Мужской пол'},
    'smoking_history_No Info': {'en': 'No Smoking Info', 'ru': 'Нет данных о курении'},
    'smoking_history_current': {'en': 'Current Smoker', 'ru': 'Курит сейчас'},
    'smoking_history_ever': {'en': 'Ever Smoked', 'ru': 'Курил когда-либо'},
    'smoking_history_former': {'en': 'Former Smoker', 'ru': 'Бывший курильщик'},
    'smoking_history_never': {'en': 'Never Smoked', 'ru': 'Никогда не курил'},
    'smoking_history_not current': {'en': 'Not Current Smoker', 'ru': 'Не курит сейчас'}
}

# Error messages in both languages
ERROR_MESSAGES = {
    'en': {
        'model_not_loaded': 'Model is not loaded correctly',
        'model_not_selected': 'Please select a model',
        'all_fields_required': 'All fields are required',
        'invalid_numeric': 'Please enter valid numeric values',
        'age_range': 'Age must be between 0 and 120',
        'bmi_range': 'BMI must be between 15 and 75',
        'hba1c_range': 'HbA1c level must be between 3 and 20',
        'svm_feature_error': 'One or more input values are outside the expected range for the SVM model.',
        'svm_convergence_warning': 'The SVM model had difficulty making a confident prediction with the given input values.'
    },
    'ru': {
        'model_not_loaded': 'Модель не загружена корректно',
        'model_not_selected': 'Пожалуйста, выберите модель',
        'all_fields_required': 'Все поля обязательны для заполнения',
        'invalid_numeric': 'Пожалуйста, введите корректные числовые значения',
        'age_range': 'Возраст должен быть от 0 до 120',
        'bmi_range': 'ИМТ должен быть от 15 до 75',
        'hba1c_range': 'Уровень HbA1c должен быть от 3 до 20',
        'glucose_range': 'Уровень глюкозы в крови должен быть от 50 до 400',
        'processing_error': 'Произошла ошибка при обработке данных',
        'svm_prediction_error': 'Модель SVM столкнулась с ошибкой при обработке ваших входных значений. Это может быть связано с неожиданными диапазонами значений.',
        'svm_feature_error': 'Одно или несколько входных значений находятся за пределами ожидаемого диапазона для модели SVM.',
        'svm_convergence_warning': 'Модель SVM испытывала трудности с уверенным прогнозом при заданных входных значениях.'
    }
}

def load_models():
    """Load all models with enhanced error handling and debugging"""
    models_loaded = {model_name: False for model_name in models.keys()}
    models_status = []

    # Get absolute path to the models directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for model_name, rel_path in MODEL_PATHS.items():
        abs_path = os.path.join(base_dir, rel_path)
        try:
            # Verify file exists
            if not os.path.exists(abs_path):
                models_status.append(f"Model file not found: {abs_path}")
                continue

            # Check if file is readable
            if not os.access(abs_path, os.R_OK):
                models_status.append(f"Model file not readable: {abs_path}")
                continue

            # Load and verify the model
            with open(abs_path, 'rb') as file:
                try:
                    model = pickle.load(file)
                    if not hasattr(model, 'predict'):
                        models_status.append(f"Invalid model format: {model_name} does not have predict method")
                        continue

                    models[model_name] = model
                    models_loaded[model_name] = True
                    models_status.append(f"{model_name} model loaded successfully!")
                except (pickle.UnpicklingError, EOFError) as e:
                    models_status.append(f"Error unpickling {model_name} model: {str(e)}")
                    continue

        except Exception as e:
            models_status.append(f"Error loading {model_name} model: {str(e)}")
            models[model_name] = None

    # Print detailed status report
    print("\nModel Loading Status Report:")
    print("=" * 50)
    for status in models_status:
        print(status)
    print("=" * 50)

    if not any(models_loaded.values()):
        print("\nCRITICAL: No models were loaded successfully!")
        print("Checked paths:")
        for model_name, rel_path in MODEL_PATHS.items():
            print(f"- {os.path.join(base_dir, rel_path)}")
        return False

    print(f"\nSuccessfully loaded {sum(models_loaded.values())}/{len(models_loaded)} models")
    return True

class FeatureImportanceCalculator:
    """
    Comprehensive feature importance calculation system that handles both general model importance
    and personalized feature contributions for different model types.
    """

    def __init__(self, models_dict, expected_features, feature_display_names):
        self.models = models_dict
        self.expected_features = expected_features
        self.feature_display_names = feature_display_names

    def clean_feature_name(self, feature_name, lang='en'):
        """Clean and format feature names for display"""
        try:
            # Get display name from mapping
            display_name = self.feature_display_names.get(feature_name, {}).get(lang, feature_name)

            # If no mapping found, clean the raw feature name
            if display_name == feature_name:
                # Remove underscores and capitalize
                cleaned = feature_name.replace('_', ' ').title()
                # Handle special cases
                cleaned = cleaned.replace('Hba1c', 'HbA1c')
                cleaned = cleaned.replace('Bmi', 'BMI')
                return cleaned

            return display_name
        except Exception as e:
            print(f"Error cleaning feature name {feature_name}: {str(e)}")
            return feature_name

    def calculate_general_importance(self, model_type, lang='en', top_n=None):
        """
        Calculate general model importance (not personalized to specific input).

        For linear models: Use absolute values of model coefficients
        For tree models: Use feature_importances_ attribute
        """
        try:
            if model_type not in self.models or self.models[model_type] is None:
                print(f"Model {model_type} not found or not loaded")
                return []

            model = self.models[model_type]

            # Get importance values based on model type
            if model_type in ['logistic_regression', 'svm']:
                # For linear models: Use absolute values of coefficients
                if not hasattr(model, 'coef_'):
                    print(f"Linear model {model_type} doesn't have coef_ attribute")
                    return []
                importance_values = np.abs(model.coef_[0])
                print(f"Using absolute coefficients for {model_type}")

            elif model_type in ['random_forest', 'adaboost', 'decision_tree']:
                # For tree models: Use feature_importances_
                if not hasattr(model, 'feature_importances_'):
                    print(f"Tree model {model_type} doesn't have feature_importances_ attribute")
                    return []
                importance_values = model.feature_importances_
                print(f"Using feature_importances_ for {model_type}")

            else:
                print(f"Unknown model type: {model_type}")
                return []

            # Validate importance values
            if len(importance_values) != len(self.expected_features):
                print(f"Mismatch: {len(importance_values)} importance values vs {len(self.expected_features)} features")
                return []

            # Create feature importance data
            feature_data = []
            for i, feature_name in enumerate(self.expected_features):
                importance_value = float(importance_values[i])

                # Skip features with zero or very small importance
                if abs(importance_value) < 1e-6:
                    continue

                feature_data.append({
                    'feature': feature_name,
                    'importance': importance_value,
                    'abs_importance': abs(importance_value),
                    'display_name': self.clean_feature_name(feature_name, lang),
                    'direction': 'neutral'  # General importance doesn't have direction
                })

            # Sort by importance in descending order
            feature_data.sort(key=lambda x: x['abs_importance'], reverse=True)

            # Limit to top N if specified
            if top_n:
                feature_data = feature_data[:top_n]

            # Calculate percentages (normalize to 100%)
            total_importance = sum(item['abs_importance'] for item in feature_data)

            if total_importance > 0:
                for item in feature_data:
                    item['percentage'] = round((item['abs_importance'] / total_importance) * 100, 1)
            else:
                for item in feature_data:
                    item['percentage'] = 0.0

            print(f"Calculated general importance for {len(feature_data)} features")
            return feature_data

        except Exception as e:
            print(f"Error calculating general importance for {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def calculate_personalized_contributions(self, model_type, features, lang='en', top_n=None):
        """
        Calculate personalized feature contributions for specific input.

        For linear models: Multiply coefficients by feature values
        For tree models: Calculate contribution by comparing predictions with/without each feature
        """
        try:
            if model_type not in self.models or self.models[model_type] is None:
                print(f"Model {model_type} not found or not loaded")
                return []

            model = self.models[model_type]
            features_array = np.array(features).reshape(1, -1)

            # Validate input features
            if len(features) != len(self.expected_features):
                print(f"Mismatch: {len(features)} input features vs {len(self.expected_features)} expected")
                return []

            # Calculate contributions based on model type
            if model_type in ['logistic_regression', 'svm']:
                # For linear models: Multiply coefficients by feature values
                if not hasattr(model, 'coef_'):
                    print(f"Linear model {model_type} doesn't have coef_ attribute")
                    return []
                contributions = model.coef_[0] * features_array[0]
                print(f"Using coef_ * features for {model_type}")

            elif model_type in ['random_forest', 'adaboost', 'decision_tree']:
                # For tree models: Compare predictions with/without each feature
                contributions = self._calculate_tree_contributions(model, features_array[0])
                print(f"Calculated tree contributions for {model_type}")

            else:
                print(f"Unknown model type: {model_type}")
                return []

            # Create feature contribution data
            feature_data = []
            for i, feature_name in enumerate(self.expected_features):
                contribution_value = float(contributions[i])

                # Skip features with zero or very small contribution
                if abs(contribution_value) < 1e-6:
                    continue

                feature_data.append({
                    'feature': feature_name,
                    'importance': contribution_value,
                    'abs_importance': abs(contribution_value),
                    'display_name': self.clean_feature_name(feature_name, lang),
                    'direction': 'positive' if contribution_value > 0 else 'negative'
                })

            # Sort by absolute importance in descending order
            feature_data.sort(key=lambda x: x['abs_importance'], reverse=True)

            # Limit to top N if specified
            if top_n:
                feature_data = feature_data[:top_n]

            # Calculate percentages (normalize to 100%)
            total_abs_importance = sum(item['abs_importance'] for item in feature_data)

            if total_abs_importance > 0:
                for item in feature_data:
                    item['percentage'] = round((item['abs_importance'] / total_abs_importance) * 100, 1)
            else:
                for item in feature_data:
                    item['percentage'] = 0.0

            print(f"Calculated personalized contributions for {len(feature_data)} features")
            return feature_data

        except Exception as e:
            print(f"Error calculating personalized contributions for {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _calculate_tree_contributions(self, model, features):
        """
        Calculate feature contributions for tree-based models by comparing
        predictions with and without each feature.
        """
        try:
            # Get baseline prediction with all features
            baseline_pred = model.predict_proba(features.reshape(1, -1))[0][1]

            contributions = np.zeros(len(features))

            # For each feature, calculate contribution by setting it to zero
            for i in range(len(features)):
                # Create modified feature array with feature i set to 0
                modified_features = features.copy()
                modified_features[i] = 0

                # Get prediction without this feature
                modified_pred = model.predict_proba(modified_features.reshape(1, -1))[0][1]

                # Contribution is the difference
                contributions[i] = baseline_pred - modified_pred

            return contributions

        except Exception as e:
            print(f"Error calculating tree contributions: {str(e)}")
            # Fallback to feature_importances_ * feature_values
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_ * features
            else:
                return np.zeros(len(features))

# Initialize the feature importance calculator
feature_calculator = FeatureImportanceCalculator(models, EXPECTED_FEATURES, FEATURE_DISPLAY_NAMES)

def get_feature_importance(model_type, features=None, lang='en', top_n=6, importance_type='personalized'):
    """
    Unified function to get feature importance with proper error handling.

    Args:
        model_type: Type of model ('random_forest', 'logistic_regression', etc.)
        features: Feature values for personalized importance (None for general importance)
        lang: Language for display names ('en' or 'ru')
        top_n: Number of top features to return
        importance_type: 'general' for model-wide importance, 'personalized' for input-specific

    Returns:
        List of dictionaries with feature importance data
    """
    try:
        if importance_type == 'general' or features is None:
            return feature_calculator.calculate_general_importance(model_type, lang, top_n)
        else:
            return feature_calculator.calculate_personalized_contributions(model_type, features, lang, top_n)
    except Exception as e:
        print(f"Error in get_feature_importance: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def validate_feature_importance_input(model_type, features=None):
    """
    Validate inputs for feature importance calculation.

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check if model exists and is loaded
        if model_type not in models or models[model_type] is None:
            return False, f"Model {model_type} not found or not loaded"

        # If features provided, validate them
        if features is not None:
            if not isinstance(features, (list, np.ndarray)):
                return False, "Features must be a list or numpy array"

            if len(features) != len(EXPECTED_FEATURES):
                return False, f"Expected {len(EXPECTED_FEATURES)} features, got {len(features)}"

            # Check for valid numeric values
            try:
                features_array = np.array(features, dtype=float)
                if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                    return False, "Features contain invalid values (NaN or Inf)"
            except (ValueError, TypeError):
                return False, "Features must be numeric values"

        return True, ""

    except Exception as e:
        return False, f"Validation error: {str(e)}"

# Load models when the app starts
if not load_models():
    print("Failed to load ANY models! Application cannot function without at least one model.")
    exit(1)
else:
    print("Application starting with available models...")

def create_feature_importance_plot(model_type):
    """Create a feature importance plot using the new feature calculator"""
    try:
        # Use the new feature calculator to get general importance
        importance_data = feature_calculator.calculate_general_importance(
            model_type, lang='en', top_n=None
        )

        if not importance_data:
            return None

        # Convert to DataFrame for plotting
        feature_importance = pd.DataFrame(importance_data)
        feature_importance = feature_importance.sort_values('importance', ascending=True)

        # Create matplotlib figure with larger size
        plt.figure(figsize=(12, 8))

        # Create barplot with improved styling
        ax = sns.barplot(data=feature_importance, y='display_name', x='importance', palette='magma')

        # Customize the plot
        plt.title(f'Feature Importance - {model_type.replace("_", " ").title()}', pad=20, fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)

        # Add value labels to the bars
        for i, v in enumerate(feature_importance['importance']):
            ax.text(v, i, f'{v:.3f}', va='center', fontsize=10)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)
        return buf

    except Exception as e:
        print(f"Error creating feature importance plot: {str(e)}")
        return None

def create_feature_contribution_plot(model_type, features):
    """Create a feature contribution plot using the new feature calculator"""
    try:
        # Use the new feature calculator to get contributions
        contributions_data = feature_calculator.calculate_personalized_contributions(
            model_type, features, lang='en', top_n=None
        )

        if not contributions_data:
            return None

        # Convert to DataFrame for plotting
        contribution_df = pd.DataFrame(contributions_data)
        contribution_df = contribution_df.sort_values('importance', ascending=True)

        # Create matplotlib figure with larger size
        plt.figure(figsize=(12, 8))

        # Create barplot with improved styling
        colors = ['#d73027' if x < 0 else '#4575b4' for x in contribution_df['importance']]
        ax = sns.barplot(data=contribution_df, y='display_name', x='importance', palette=colors)

        # Customize the plot
        plt.title(f'Feature Contribution to Prediction - {model_type.replace("_", " ").title()}', pad=20, fontsize=14)
        plt.xlabel('Contribution to Prediction', fontsize=12)
        plt.ylabel('Feature', fontsize=12)

        # Add value labels to the bars
        for i, v in enumerate(contribution_df['importance']):
            ax.text(v, i, f'{v:.3f}', va='center', fontsize=10)

        # Add a vertical line at x=0
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)
        return buf

    except Exception as e:
        print(f"Error creating feature contribution plot: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def home():
    if 'patient_name' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', patient_name=session['patient_name'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        patient_name = request.form.get('patient_name')
        if patient_name:
            session['patient_name'] = patient_name
            return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the selected model
        model_type = request.form.get('model_type', 'random_forest')
        if model_type not in models:
            return jsonify({
                'error': ERROR_MESSAGES[request.form.get('lang', 'en')]['model_not_selected'],
                'risk_level': 'Error'
            })

        model = models[model_type]
        if model is None:
            return jsonify({
                'error': ERROR_MESSAGES[request.form.get('lang', 'en')]['model_not_loaded'],
                'risk_level': 'Error'
            })

        lang = request.form.get('lang', 'en')
        data = {
            'age': request.form.get('age'),
            'gender': request.form.get('gender'),
            'hypertension': request.form.get('hypertension'),
            'heart_disease': request.form.get('heart_disease'),
            'smoking_history': request.form.get('smoking_history'),
            'bmi': request.form.get('bmi'),
            'HbA1c_level': request.form.get('HbA1c_level'),
            'blood_glucose_level': request.form.get('blood_glucose_level')
        }

        # Validate inputs
        if not all(value is not None and value != '' for value in data.values()):
            return jsonify({'error': ERROR_MESSAGES[lang]['all_fields_required'], 'risk_level': 'Error'})

        try:
            # Convert and validate numeric inputs with additional error handling
            age = float(data['age'])
            bmi = float(data['bmi'])
            HbA1c_level = float(data['HbA1c_level'])
            blood_glucose_level = float(data['blood_glucose_level'])
            hypertension = int(data['hypertension'])
            heart_disease = int(data['heart_disease'])

            if not (0 <= age <= 120):
                return jsonify({'error': ERROR_MESSAGES[lang]['age_range'], 'risk_level': 'Error'})
            if not (15 <= bmi <= 75):
                return jsonify({'error': ERROR_MESSAGES[lang]['bmi_range'], 'risk_level': 'Error'})
            if not (3 <= HbA1c_level <= 20):
                return jsonify({'error': ERROR_MESSAGES[lang]['hba1c_range'], 'risk_level': 'Error'})
            if not (50 <= blood_glucose_level <= 530):
                return jsonify({'error': ERROR_MESSAGES[lang]['glucose_range'], 'risk_level': 'Error'})
        except ValueError:
            return jsonify({'error': ERROR_MESSAGES[lang]['invalid_numeric'], 'risk_level': 'Error'})

        # Process categorical variables
        gender_Female = 1 if data['gender'] == 'Female' else 0
        gender_Male = 1 if data['gender'] == 'Male' else 0

        smoking_history = data['smoking_history'].lower()
        smoking_history_No_Info = 1 if smoking_history == 'no info' else 0
        smoking_history_current = 1 if smoking_history == 'current' else 0
        smoking_history_ever = 1 if smoking_history == 'ever' else 0
        smoking_history_former = 1 if smoking_history == 'former' else 0
        smoking_history_never = 1 if smoking_history == 'never' else 0
        smoking_history_not_current = 1 if smoking_history == 'not current' else 0

        # Create feature array ensuring correct data type and order
        features = np.array([
            [
                age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level,
                gender_Female, gender_Male, smoking_history_No_Info, smoking_history_current,
                smoking_history_ever, smoking_history_former, smoking_history_never,
                smoking_history_not_current
            ]
        ])

        # Get predictions from all models
        all_predictions = {}
        for model_name, model_instance in models.items():
            if model_instance is not None:
                try:
                    prediction = model_instance.predict(features)
                    all_predictions[model_name] = prediction
                except Exception as e:
                    print(f"Error with {model_name} prediction: {str(e)}")

        # Get prediction and probability for the selected model
        try:
            if model_type == 'svm':
                # For SVM, get binary prediction and assign probability as 1.0 or 0.0
                prediction = model.predict(features)
                # Assign probability as 1.0 if prediction == 1 (positive class), else 0.0
                proba = 1.0 if prediction[0] == 1 else 0.0
            else:
                # For all other models, use predict_proba
                prediction = model.predict(features)
                proba = model.predict_proba(features)[0][1]

            if proba < 0.3:
                risk_level = 'Low'
            elif proba < 0.6:
                risk_level = 'Medium'
            else:
                risk_level = 'High'

            # Only generate health advice if user has given consent
            ai_advice_consent = request.form.get('ai_advice_consent') == 'true'
            health_advice = generate_health_advice(data, all_predictions, lang) if ai_advice_consent else None

            # Calculate personalized feature importance using the new unified function
            print(f"Calculating feature importance for model: {model_type}")
            print(f"Features array shape: {features.shape}")
            print(f"Features: {features[0]}")

            # Validate inputs before calculation
            is_valid, error_msg = validate_feature_importance_input(model_type, features[0])
            if not is_valid:
                print(f"Feature importance validation failed: {error_msg}")
                feature_importance = []
            else:
                feature_importance = get_feature_importance(
                    model_type=model_type,
                    features=features[0],
                    lang=lang,
                    top_n=6,
                    importance_type='personalized'
                )

            print(f"Feature importance result: {feature_importance}")
            print(f"Number of features returned: {len(feature_importance)}")

            response_data = {
                'prediction': int(prediction[0]),
                'probability': float(proba),
                'risk_level': risk_level,
                'model_type': model_type,
                'feature_importance': feature_importance
            }

            # Only include health advice in response if consent was given
            if health_advice:
                response_data['health_advice'] = health_advice

            return jsonify(response_data)

        except Exception as model_error:
            print(f"Model prediction error: {str(model_error)}")
            return jsonify({
                'error': ERROR_MESSAGES[lang]['processing_error'],
                'risk_level': 'Error'
            })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': ERROR_MESSAGES[request.form.get('lang', 'en')]['processing_error'],
            'risk_level': 'Error'
        })

@app.route('/feature_importance/<model_type>')
def get_feature_importance_plot(model_type):
    """Return a static feature importance plot as PNG"""
    if (model_type not in models or models[model_type] is None):
        return 'Model not found', 404

    plot_buffer = create_feature_importance_plot(model_type)
    if plot_buffer is None:
        return 'Could not create plot', 500

    return send_file(
        plot_buffer,
        mimetype='image/png',
        as_attachment=False,
        download_name=f'feature_importance_{model_type}.png'
    )

@app.route('/feature_contribution/<model_type>', methods=['POST'])
def get_feature_contribution(model_type):
    """Return a feature contribution plot for the current prediction"""
    if model_type not in models or models[model_type] is None:
        return 'Model not found', 404

    # Get feature values from the request
    try:
        features = request.get_json()
        if not features or 'features' not in features:
            return 'No feature data provided', 400

        feature_array = np.array(features['features'])
        plot_buffer = create_feature_contribution_plot(model_type, feature_array)

        if plot_buffer is None:
            return 'Could not create plot', 500

        return send_file(
            plot_buffer,
            mimetype='image/png',
            as_attachment=False,
            download_name=f'feature_contribution_{model_type}.png'
        )
    except Exception as e:
        print(f"Error creating feature contribution plot: {str(e)}")
        return 'Error creating plot', 500

@app.route('/api/feature_importance/<model_type>')
def api_get_general_feature_importance(model_type):
    """API endpoint to get general feature importance as JSON"""
    try:
        lang = request.args.get('lang', 'en')
        top_n = request.args.get('top_n', None)
        if top_n:
            top_n = int(top_n)

        # Validate inputs
        is_valid, error_msg = validate_feature_importance_input(model_type)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        # Get general feature importance
        importance_data = get_feature_importance(
            model_type=model_type,
            features=None,
            lang=lang,
            top_n=top_n,
            importance_type='general'
        )

        return jsonify({
            'model_type': model_type,
            'importance_type': 'general',
            'features': importance_data,
            'total_features': len(importance_data)
        })

    except Exception as e:
        print(f"Error in API get general feature importance: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/feature_importance/<model_type>/personalized', methods=['POST'])
def api_get_personalized_feature_importance(model_type):
    """API endpoint to get personalized feature importance as JSON"""
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'No feature data provided'}), 400

        features = data['features']
        lang = data.get('lang', 'en')
        top_n = data.get('top_n', None)

        # Validate inputs
        is_valid, error_msg = validate_feature_importance_input(model_type, features)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        # Get personalized feature importance
        importance_data = get_feature_importance(
            model_type=model_type,
            features=features,
            lang=lang,
            top_n=top_n,
            importance_type='personalized'
        )

        return jsonify({
            'model_type': model_type,
            'importance_type': 'personalized',
            'features': importance_data,
            'total_features': len(importance_data)
        })

    except Exception as e:
        print(f"Error in API get personalized feature importance: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
