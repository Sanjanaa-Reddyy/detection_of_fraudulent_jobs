from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.tree import DecisionTreeClassifier
import json
high_cardinality_cols = ['title', 'location', 'department', 'industry','salary_range','function']

# Descriptive columns to process
descriptive_columns = ['company_profile', 'description', 'requirements', 'benefits']

subset_features=['company_profile_encoded',
 'has_questions',
 'benefits_encoded',
 'location_US, TX, Houston',
 'required_experience_Mid-Senior level',
 "required_education_Bachelor's Degree",
 'has_company_logo',
 'department_Other',
 'required_experience_Entry level',
 'salary_range_Other',
 'function_Other',
 'industry_Other',
 'department_Unknown',
 'function_Unknown',
 'function_Engineering',
 'salary_range_Unknown',
 'required_experience_Unknown',
 'requirements_encoded',
 'industry_Hospital & Health Care',
 'required_education_High School or equivalent',
 'location_Other',
 'employment_type_Unknown',
 'industry_Unknown',
 'required_education_Some High School Coursework',
 'employment_type_Full-time',
 'industry_Marketing and Advertising',
 'employment_type_Part-time',
 'function_Customer Service',
 'function_Sales',
 'required_education_Unknown',
 'telecommuting',
 'required_education_Unspecified',
 'industry_Information Technology and Services',
 'location_US, , ',
 'industry_Financial Services',
 'required_experience_Director',
 'clients',
 'required_experience_Not Applicable',
 'position',
 'required_experience_Internship',
 'location_US, CA, San Francisco',
 'industry_Consumer Services',
 "required_education_Master's Degree",
 'department_Sales',
 'process',
 'required',
 'department_Information Technology',
 'required_education_Certification',
 'amp',
 'location_Unknown',
 'location_US, NY, New York',
 'office',
 'employment_type_Other',
 'function_Information Technology',
 'required_experience_Executive',
 'salary_range_30000-40000',
 'title_Other',
 'department_Engineering',
 'function_Health Care Provider',
 'required_education_Some College Coursework Completed']

common_words={'new', 'including', 'sales', '-', '&amp;', 'development', 'company', 'project', 'job', 'people', 'support', 'solutions', 'design', 'work', 'customer', 'looking', 'service', 'data', 'services', 'knowledge', 'high', 'business', 'product', 'skills', 'technology', 'unknown', 'team', 'help', 'working', 'ability', 'quality', 'time', 'technical', 'management', 'provide', 'experience', 'years', 'strong', 'communication'}

# Step 1: Fill missing values (for non-numeric columns)
class FillMissingValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting required for filling missing values
        return self
    
    def transform(self, X):
        return X.fillna('Unknown')

# Step 2: Create missing indicators for descriptive columns
class CreateMissingIndicators(BaseEstimator, TransformerMixin):
    def __init__(self, descriptive_columns):
        self.descriptive_columns = descriptive_columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for col in self.descriptive_columns:
            X[f'{col}_encoded'] = X[col].apply(lambda x: 1 if x != 'Unknown' else 0)
        return X

# Step 3: Vectorize descriptive text columns using TF-IDF
class VectorizeDescriptiveText(BaseEstimator, TransformerMixin):
    def __init__(self, descriptive_columns, max_features=20, common_words=None):
        self.descriptive_columns = descriptive_columns
        self.max_features = max_features
        self.common_words = common_words
        self.vectorizer = None  # To store the fitted vectorizer

    def fit(self, X, y=None):
        # Combine all descriptive columns into one text column
        X['combined_text'] = X[self.descriptive_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        
        # Prepare the list of stop words
        exclude_words = list(ENGLISH_STOP_WORDS.union(self.common_words)) if self.common_words else list(ENGLISH_STOP_WORDS)
        exclude_words.append('industry')
        
        # Initialize the vectorizer and fit it on the training data's combined_text column
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words=exclude_words)
        self.vectorizer.fit(X['combined_text'])
        
        return self

    def transform(self, X):
        if self.vectorizer is None:
            raise ValueError("fit has not been called before transform")
        
        # Combine all descriptive columns into one text column
        X['combined_text'] = X[self.descriptive_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        
        # Perform TF-IDF vectorization using the fitted vectorizer
        text_vectors = self.vectorizer.transform(X['combined_text'])
        
        # Convert sparse matrix to dense and create a DataFrame
        text_vectors_dense = text_vectors.toarray()
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(text_vectors_dense, columns=feature_names)
        
        # Drop the combined text column and original descriptive columns
        X = X.drop(columns=self.descriptive_columns + ['combined_text'])
        
        # Reset index and concatenate the TF-IDF features with the original features
        X_reset = X.reset_index(drop=True)
        tfidf_df_reset = tfidf_df.reset_index(drop=True)
        Ndf = pd.concat([X_reset, tfidf_df_reset], axis=1)
        
        return Ndf

    def fit_transform(self, X, y=None):
        # Combine fit and transform in one step
        return self.fit(X, y).transform(X)

class CollapseHighCardinality(BaseEstimator, TransformerMixin):
    def __init__(self, columns, top_n=8):
        self.columns = columns
        self.top_n = top_n
    
    def fit(self, X, y=None):
        self.top_values = {}

        for column in self.columns:
            self.top_values[column] = list(X[column].value_counts().nlargest(self.top_n).index)

        return self
    
    def transform(self, X):
        for column in self.columns:
            X[column] = X[column].apply(lambda x: x if x in self.top_values[column] else 'Other')
        return X
      


      

class OneHotEncodeLowCardinality(BaseEstimator, TransformerMixin):
    def __init__(self, max_categories=20):
        self.max_categories = max_categories
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')  # sparse=False returns dense array
    
    def fit(self, X, y=None):
        # Identify low cardinality categorical columns
        self.low_cardinality_cols = [col for col in X.columns if X[col].nunique() > 2 and X[col].dtype == 'object']
        
        # Fit the encoder only on these columns
        self.encoder.fit(X[self.low_cardinality_cols])
        return self
    
    def transform(self, X):
        # Apply OneHotEncoder on the selected columns
        encoded = self.encoder.transform(X[self.low_cardinality_cols])
        
        # Convert the encoded array to DataFrame and merge with the original DataFrame
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.low_cardinality_cols))
        
        # Drop the original categorical columns and append the encoded columns
        X = X.drop(columns=self.low_cardinality_cols)
        X = pd.concat([X, encoded_df], axis=1)
        
        return X


# Step 6: Convert all columns to integers
class ConvertToIntegers(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

# Step 7: Feature Selection
class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features):
        self.selected_features = selected_features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.selected_features]


# Combine all transformers into one pipeline
preprocessing_pipeline = Pipeline([
    ('fill_missing_values', FillMissingValues()),  # Step 1: Fill missing values
    ('create_missing_indicators', CreateMissingIndicators(descriptive_columns)),  # Step 2: Create missing indicators
    ('vectorize_text', VectorizeDescriptiveText(descriptive_columns, max_features=20, common_words=common_words)),  # Step 3: TF-IDF Vectorization
    ('collapse_high_cardinality', CollapseHighCardinality(columns=high_cardinality_cols , top_n=9)),  # Step 4: Collapse high cardinality columns
    ('one_hot_encode', OneHotEncodeLowCardinality(max_categories=20)),  # Step 5: One-hot encode low cardinality columns
    ('convert_to_integers', ConvertToIntegers()),  # Step 6: Convert all columns to integers
    ('feature_selection', FeatureSelection(subset_features))  # Step 7: Feature Selection
])

class RandomForest:
    def __init__(self, n_estimators=200, max_depth=20, random_state=50):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        self.feature_importances_ = np.zeros(X.shape[1])  # Initialize feature importances
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X.iloc[indices]
            y_bootstrap = y.iloc[indices]

            # Train a Decision Tree on the bootstrap sample
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

            # Update feature importances from this tree
            self.feature_importances_ += tree.feature_importances_

        # Normalize the feature importances
        self.feature_importances_ /= self.n_estimators

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        majority_votes = [np.bincount(pred).argmax() for pred in predictions.T]
        return np.array(majority_votes)

    def save_model(self, filename='random_forest_model.json'):
        """ Save the model trees and feature importances in a readable format (JSON-like) """
        model_data = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'feature_importances': self.feature_importances_.tolist(),
            'trees': [self.tree_to_dict(tree) for tree in self.trees]
        }
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=4)
        
        print(f"Model saved to {filename}")

    def tree_to_dict(self, tree):
        """ Convert a trained Decision Tree into a dictionary format """
        tree_ = tree.tree_
        feature_name = [f"feature_{i}" for i in range(len(tree_.feature))]
        tree_dict = {
            "node_count": tree_.node_count,
            "children_left": tree_.children_left.tolist(),
            "children_right": tree_.children_right.tolist(),
            "feature": [feature_name[i] for i in tree_.feature.tolist()],
            "threshold": tree_.threshold.tolist(),
            "value": tree_.value.tolist()
        }
        return tree_dict

    def load_model(self, filename='random_forest_model.json'):
        """ Load the model from a saved JSON-like file """
        with open(filename, 'r') as f:
            model_data = json.load(f)

        self.n_estimators = model_data['n_estimators']
        self.max_depth = model_data['max_depth']
        self.feature_importances_ = np.array(model_data['feature_importances'])

        # Recreate trees from the saved data
        self.trees = []
        for tree_dict in model_data['trees']:
            tree = self.create_tree_from_dict(tree_dict)
            self.trees.append(tree)
    
    def create_tree_from_dict(self, tree_dict):
        """ Rebuild a tree from a dictionary format """
        from sklearn.tree import DecisionTreeClassifier
        
        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        
        return tree