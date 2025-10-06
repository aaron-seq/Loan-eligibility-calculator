"""
Enhanced ML Model Training Script
Trains and saves the loan eligibility prediction model
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanEligibilityModelTrainer:
    """Enhanced model trainer with hyperparameter optimization"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_and_prepare_data(self, data_path: str = 'data/train.csv'):
        """Load and prepare training data with feature engineering"""
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} records from {data_path}")
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Feature engineering
            df = self.engineer_features(df)
            
            # Prepare target variable
            df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
            
            return df
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill missing values with appropriate strategies
        df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna('0', inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
        
        # Add CIBIL score (synthetic feature for enhanced model)
        np.random.seed(42)
        df['CIBIL_Score'] = np.random.randint(300, 850, size=len(df))
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for better prediction"""
        # Calculate total income
        df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        # Loan to income ratio
        df['Loan_to_Income_Ratio'] = df['LoanAmount'] / df['Total_Income']
        df['Loan_to_Income_Ratio'].replace([np.inf, -np.inf], 0, inplace=True)
        
        # EMI-like feature
        df['EMI_feature'] = df['LoanAmount'] / df['Loan_Amount_Term']
        df['EMI_feature'].replace([np.inf, -np.inf], 0, inplace=True)
        
        # Log transformations for skewed features
        df['ApplicantIncome_log'] = np.log(df['ApplicantIncome'] + 1)
        df['CoapplicantIncome_log'] = np.log(df['CoapplicantIncome'] + 1)
        df['LoanAmount_log'] = np.log(df['LoanAmount'] + 1)
        df['Total_Income_log'] = np.log(df['Total_Income'] + 1)
        
        # Encode categorical variables
        df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 
                                        'Self_Employed', 'Property_Area'], drop_first=True)
        
        return df
    
    def prepare_features_and_target(self, df: pd.DataFrame):
        """Prepare feature matrix and target vector"""
        # Define feature columns (excluding target and ID columns)
        feature_columns = [col for col in df.columns if col not in ['Loan_ID', 'Loan_Status']]
        self.feature_columns = feature_columns
        
        X = df[feature_columns]
        y = df['Loan_Status']
        
        logger.info(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        
        return X, y
    
    def train_model_with_optimization(self, X, y):
        """Train model with hyperparameter optimization"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        # Grid search with cross-validation
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        logger.info("Starting hyperparameter optimization...")
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        return test_accuracy
    
    def save_models(self, model_dir: str = 'models'):
        """Save trained model and scaler"""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'loan_eligibility_model.pkl')
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        features_path = os.path.join(model_dir, 'feature_columns.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        logger.info(f"Models saved to {model_dir}")

def main():
    """Main training pipeline"""
    trainer = LoanEligibilityModelTrainer()
    
    # Load and prepare data
    df = trainer.load_and_prepare_data()
    
    # Prepare features and target
    X, y = trainer.prepare_features_and_target(df)
    
    # Train model
    accuracy = trainer.train_model_with_optimization(X, y)
    
    # Save models
    trainer.save_models()
    
    logger.info(f"Model training completed with accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
