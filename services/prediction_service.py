import os
import pickle
import logging
import asyncio
from typing import Dict, Any
import numpy as np
import pandas as pd
from config import Config

logger = logging.getLogger(__name__)

class PredictionService:
    """ML model service with caching and async support."""
    
    def __init__(self):
        self.config = Config()
        self._model = None
        self._scaler = None
        self._model_loaded = False
        self._load_lock = asyncio.Lock()
    
    @property
    def is_model_available(self) -> bool:
        return self._model_loaded and self._model is not None and self._scaler is not None
    
    def _load_models(self) -> bool:
        if self._model_loaded:
            return self.is_model_available
        try:
            model_path = os.path.join(self.config.MODEL_PATH, 'loan_eligibility_model.pkl')
            scaler_path = os.path.join(self.config.MODEL_PATH, 'feature_scaler.pkl')
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger.warning("ML model files not found")
                return False
            with open(model_path, 'rb') as f:
                self._model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self._scaler = pickle.load(f)
            self._model_loaded = True
            logger.info("ML models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            return False
    
    async def load_models_async(self) -> bool:
        async with self._load_lock:
            return self._load_models()
    
    def engineer_features(self, loan_data: Dict[str, Any]) -> pd.DataFrame:
        applicant_income = float(loan_data['applicant_income'])
        coapplicant_income = float(loan_data['coapplicant_income'])
        loan_amount = float(loan_data['loan_amount'])
        loan_tenure = float(loan_data['loan_tenure'])
        cibil_score = int(loan_data['cibil_score'])
        total_income = applicant_income + coapplicant_income
        loan_to_income_ratio = loan_amount / total_income if total_income > 0 else 0
        emi_feature = loan_amount / loan_tenure if loan_tenure > 0 else 0
        applicant_income_log = np.log(applicant_income + 1)
        coapplicant_income_log = np.log(coapplicant_income + 1)
        loan_amount_log = np.log(loan_amount + 1)
        total_income_log = np.log(total_income + 1)
        return pd.DataFrame({
            'Loan_Amount_Term': [loan_tenure],
            'Credit_History': [float(loan_data.get('credit_history', 1.0))],
            'CIBIL_Score': [cibil_score],
            'Loan_to_Income_Ratio': [loan_to_income_ratio],
            'EMI_feature': [emi_feature],
            'ApplicantIncome_log': [applicant_income_log],
            'CoapplicantIncome_log': [coapplicant_income_log],
            'LoanAmount_log': [loan_amount_log],
            'Total_Income_log': [total_income_log],
            'Gender_Male': [1 if loan_data.get('gender') == 1 else 0],
            'Married_Yes': [1 if loan_data.get('marital_status') == 1 else 0],
            'Dependents_1': [1 if loan_data.get('dependents_count') == 1 else 0],
            'Dependents_2': [1 if loan_data.get('dependents_count') == 2 else 0],
            'Dependents_3+': [1 if loan_data.get('dependents_count', 0) >= 3 else 0],
            'Education_Not Graduate': [1 if loan_data.get('education_level') == 1 else 0],
            'Self_Employed_Yes': [1 if loan_data.get('employment_type') == 1 else 0],
            'Property_Area_Semiurban': [1 if loan_data.get('property_location') == 1 else 0],
            'Property_Area_Urban': [1 if loan_data.get('property_location') == 2 else 0]
        })
    
    def predict(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self._load_models():
            raise RuntimeError("ML models not available")
        feature_vector = self.engineer_features(loan_data)
        scaled = self._scaler.transform(feature_vector)
        pred = self._model.predict(scaled)
        proba = self._model.predict_proba(scaled)
        decision = "APPROVED" if pred[0] == 1 else "REJECTED"
        confidence = float(proba[0][pred[0]] * 100)
        return {
            'decision': decision,
            'confidence': round(confidence, 2),
            'prediction_class': int(pred[0])
        }
    
    async def predict_async(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        await self.load_models_async()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict, loan_data)

prediction_service = PredictionService()
