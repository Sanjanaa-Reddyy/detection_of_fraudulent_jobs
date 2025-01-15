import joblib
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from collections import Counter
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from Preprocessing_P import FillMissingValues
from Preprocessing_P import CreateMissingIndicators
from Preprocessing_P import VectorizeDescriptiveText
from Preprocessing_P import CollapseHighCardinality
from Preprocessing_P import OneHotEncodeLowCardinality
from Preprocessing_P import ConvertToIntegers
from Preprocessing_P import FeatureSelection
from Preprocessing_P import RandomForest

# Load model and preprocessing pipeline
model_path = './model/random_forest_model.pkl'
preprocessing_path = './model/preprocessing_pipeline.pkl'

with open(model_path, 'rb') as model_file:
    model = joblib.load(model_file)

with open(preprocessing_path, 'rb') as preprocessing_file:
    preprocessor = joblib.load(preprocessing_file)

app = Flask(__name__)

def none_if_empty(arg):
    return None if arg == '' else arg

@app.route('/')
def home():
    return render_template('code.html')  # Serve the HTML form

@app.route('/predict', methods=['GET'])
def predict():
    # Extract the query parameters from the GET request
    title = request.args.get('job_title')
    print(f"job_title: {title}")  # Print statement for job_title

    description = none_if_empty(request.args.get('job-description'))
    print(f"description: {description}")  # Print statement for description

    salary_range = none_if_empty(request.args.get('salary-range'))
    print(f"salary_range: {salary_range}")  # Print statement for salary_range

    location = none_if_empty(request.args.get('job-location'))
    print(f"job_location: {location}")  # Print statement for job_location

    requirements = none_if_empty(request.args.get('job-requirements'))
    print(f"requirements: {requirements}")  # Print statement for requirements

    department = none_if_empty(request.args.get('department'))
    print(f"department: {department}")  # Print statement for department

    company_profile = none_if_empty(request.args.get('company-profile'))
    print(f"company_profile: {company_profile}")  # Print statement for company_profile

    benefits = none_if_empty(request.args.get('benefits'))
    print(f"benefits: {benefits}")  # Print statement for benefits

    required_education = none_if_empty(request.args.get('required-education'))
    print(f"required_education: {required_education}")  # Print statement for required_education

    required_experience = none_if_empty(request.args.get('required-experience'))
    print(f"required_experience: {required_experience}")  # Print statement for required_experience

    industry = none_if_empty(request.args.get('industry'))
    print(f"industry: {industry}")  # Print statement for industry

    function = none_if_empty(request.args.get('domain'))
    print(f"domain: {function}")  # Print statement for domain

    employment_type = none_if_empty(request.args.get('employment-type'))
    print(f"employment_type: {employment_type}")  # Print statement for employment_type

    telecommuting = none_if_empty(request.args.get('telecommuting'))
    print(f"telecommuting_positions: {telecommuting}")  # Print statement for telecommuting_positions

    has_company_logo = none_if_empty(request.args.get('logo'))
    print(f"logo: {has_company_logo}")  # Print statement for logo

    has_questions = none_if_empty(request.args.get('screening-questions'))
    print(f"screening_questions: {has_questions}")  # Print statement for screening_questions


    # List of inputs to be processed
    inputs = [
        title, description, salary_range, location, requirements, department,
        company_profile, benefits, required_education, required_experience, industry, function,
        employment_type, telecommuting, has_company_logo, has_questions
    ]

    # Replace empty inputs with NaN
    inputs = [x if x else np.nan for x in inputs]

    # Column names
    column_names = [
        'title', 'description', 'salary_range', 'location', 'requirements', 'department',
        'company_profile', 'benefits', 'required_education', 'required_experience', 'industry', 'function',
        'employment_type', 'telecommuting', 'has_company_logo', 'has_questions'
    ]
    
    # Create DataFrame
    df = pd.DataFrame([inputs], columns=column_names)

    print(df)
    print(df.columns)

    processed_features = preprocessor.transform(df)

    # Predict the result using the XGBoost model
    prediction = model.predict(processed_features)

    print(f"prediction: {prediction}")

    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
