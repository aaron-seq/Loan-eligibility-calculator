"""ML Model Loader - Singleton Pattern

Handles loading and caching of trained ML models.
Uses singleton pattern to ensure models are loaded once.
"""

import os
import pickle
import threading
from typing import Optional
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.exceptions import ModelLoadError

logger = get_logger(__name__)


class ModelLoader:
    """Singleton model loader for ML models
    
    Ensures models are loaded once and cached in memory.
    Thread-safe implementation.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_path: str):
        """Implement singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str):
        """Initialize model loader
        
        Args:
            model_path: Path to models directory
        """
        if self._initialized:
            return

        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self._model_loaded = False

        # Load models on initialization
        self._load_models()
        self._initialized = True

    def _load_models(self) -> None:
        """Load ML models from disk
        
        Raises:
            ModelLoadError: If models cannot be loaded
        """
        try:
            model_file = os.path.join(self.model_path, "loan_eligibility_model.pkl")
            scaler_file = os.path.join(self.model_path, "feature_scaler.pkl")
            columns_file = os.path.join(self.model_path, "feature_columns.pkl")

            # Check if files exist
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}")
            if not os.path.exists(scaler_file):
                raise FileNotFoundError(f"Scaler file not found: {scaler_file}")

            # Load model
            with open(model_file, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {model_file}")

            # Load scaler
            with open(scaler_file, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler loaded successfully from {scaler_file}")

            # Load feature columns if available
            if os.path.exists(columns_file):
                with open(columns_file, "rb") as f:
                    self.feature_columns = pickle.load(f)
                logger.info("Feature columns loaded successfully")

            self._model_loaded = True
            logger.info("All ML components loaded successfully")

        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            self._model_loaded = False
            # Don't raise - allow app to start without models for other features

        except Exception as e:
            logger.error(f"Error loading ML models: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load ML models: {str(e)}")

    def is_model_loaded(self) -> bool:
        """Check if models are loaded
        
        Returns:
            True if models are available
        """
        return self._model_loaded and self.model is not None and self.scaler is not None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate prediction
        
        Args:
            features: Scaled feature dataframe
            
        Returns:
            Prediction array
            
        Raises:
            ModelLoadError: If model is not loaded
        """
        if not self.is_model_loaded():
            raise ModelLoadError("Model is not loaded")

        try:
            return self.model.predict(features)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ModelLoadError(f"Prediction failed: {str(e)}")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities
        
        Args:
            features: Scaled feature dataframe
            
        Returns:
            Probability array
            
        Raises:
            ModelLoadError: If model is not loaded
        """
        if not self.is_model_loaded():
            raise ModelLoadError("Model is not loaded")

        try:
            return self.model.predict_proba(features)
        except Exception as e:
            logger.error(f"Probability prediction error: {e}")
            raise ModelLoadError(f"Probability prediction failed: {str(e)}")

    def scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Scale features using loaded scaler
        
        Args:
            features: Raw feature dataframe
            
        Returns:
            Scaled feature dataframe
            
        Raises:
            ModelLoadError: If scaler is not loaded
        """
        if not self.is_model_loaded():
            raise ModelLoadError("Scaler is not loaded")

        try:
            # Ensure feature order matches training
            if self.feature_columns is not None:
                features = features[self.feature_columns]

            scaled = self.scaler.transform(features)
            return scaled

        except Exception as e:
            logger.error(f"Feature scaling error: {e}")
            raise ModelLoadError(f"Feature scaling failed: {str(e)}")

    def get_model_info(self) -> dict:
        """Get information about loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            "loaded": self._model_loaded,
            "model_type": type(self.model).__name__ if self.model else None,
            "scaler_type": type(self.scaler).__name__ if self.scaler else None,
            "num_features": len(self.feature_columns) if self.feature_columns else None,
        }
