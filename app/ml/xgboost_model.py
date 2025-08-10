import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import optuna

from app.core.config import settings
from app.core.logging import logger
from app.ml.feature_extractors import feature_extractor


class XGBoostSentimentModel:
    """
    XGBoost model for options sentiment analysis optimized for speed and accuracy.
    
    Features:
    - Fast inference (<50ms)
    - Handles mixed feature types (text + numerical + temporal)
    - Hyperparameter optimization with Optuna
    - Time-series aware validation
    - Feature importance analysis
    - Model persistence and versioning
    """
    
    def __init__(self, model_version: str = "v1"):
        self.model = None
        self.feature_scaler = StandardScaler()
        self.model_version = model_version
        self.feature_names = []
        self.is_trained = False
        
        # Model paths
        self.model_dir = Path(settings.data_dir) / "models" / "xgboost"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Default hyperparameters (will be optimized)
        self.params = {
            'objective': 'reg:squarederror',  # For regression (-1 to +1 sentiment)
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Performance tracking
        self.training_history = []
        self.feature_importance = {}
        
    def prepare_features(self, 
                        articles_data: List[Dict],
                        include_tfidf: bool = True,
                        include_market_data: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix from articles data.
        
        Args:
            articles_data: List of article dictionaries with content and metadata
            include_tfidf: Whether to include TF-IDF features
            include_market_data: Whether to include market-related features
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        all_features = []
        feature_names = []
        
        logger.info(f"Preparing features for {len(articles_data)} articles")
        
        # Extract features for each article
        for article in articles_data:
            article_features = {}
            
            # Text features
            text_content = article.get('body_markdown', '') or article.get('body_html', '')
            title = article.get('title', '')
            
            text_features = feature_extractor.extract_text_features(text_content, title)
            article_features.update(text_features)
            
            # Temporal features
            published_at = article.get('published_at_utc')
            if published_at:
                if isinstance(published_at, str):
                    published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                
                temporal_features = feature_extractor.extract_temporal_features(published_at)
                article_features.update(temporal_features)
            
            # Multimodal features (from images)
            images_data = article.get('images', [])
            vision_analysis = article.get('vision_analysis', {})
            
            multimodal_features = feature_extractor.extract_multimodal_features(
                images_data, vision_analysis
            )
            article_features.update(multimodal_features)
            
            # Market context features (if available)
            if include_market_data and 'market_data' in article:
                market_features = self._extract_market_features(article['market_data'])
                article_features.update(market_features)
            
            # TF-IDF features (if requested and vectorizer is fitted)
            if include_tfidf and feature_extractor.tfidf_vectorizer:
                tfidf_features = feature_extractor.get_tfidf_features(text_content)
                tfidf_feature_names = [f'tfidf_{i}' for i in range(len(tfidf_features))]
                
                # Add TF-IDF features to article features
                for i, value in enumerate(tfidf_features):
                    article_features[tfidf_feature_names[i]] = value
            
            all_features.append(article_features)
        
        # Convert to DataFrame for easier handling
        features_df = pd.DataFrame(all_features)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Store feature names
        feature_names = list(features_df.columns)
        
        # Convert to numpy array
        feature_matrix = features_df.values.astype(np.float32)
        
        logger.info(f"Prepared feature matrix: {feature_matrix.shape}")
        
        return feature_matrix, feature_names
    
    def _extract_market_features(self, market_data: Dict) -> Dict[str, float]:
        """Extract features from market data."""
        features = {}
        
        for asset in ['BTC', 'ETH']:
            if asset in market_data:
                asset_data = market_data[asset]
                base_price = asset_data.get('base_price', {})
                
                if base_price:
                    features[f'{asset.lower()}_price'] = base_price.get('price', 0)
                    features[f'{asset.lower()}_timestamp_diff'] = (
                        base_price.get('time_difference_seconds', 0) / 3600  # Convert to hours
                    )
                
                # Forward returns (if available)
                forward_returns = asset_data.get('forward_returns', {})
                for return_period, return_value in forward_returns.items():
                    if return_value is not None:
                        features[f'{asset.lower()}_{return_period}'] = return_value
        
        return features
    
    def train(self, 
              articles_data: List[Dict],
              target_values: List[float],
              validation_split: float = 0.2,
              optimize_hyperparams: bool = True,
              n_trials: int = 50) -> Dict[str, float]:
        """
        Train the XGBoost model on articles data.
        
        Args:
            articles_data: List of article dictionaries
            target_values: Target sentiment scores (-1 to +1)
            validation_split: Proportion of data for validation
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            
        Returns:
            Training metrics dictionary
        """
        try:
            logger.info(f"Training XGBoost model on {len(articles_data)} articles")
            
            # Prepare features
            X, feature_names = self.prepare_features(articles_data)
            y = np.array(target_values, dtype=np.float32)
            
            # Store feature names
            self.feature_names = feature_names
            
            # Time-series split for validation (respects temporal order)
            tscv = TimeSeriesSplit(n_splits=3)
            split_indices = list(tscv.split(X))
            
            train_idx, val_idx = split_indices[-1]  # Use last split for final validation
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit feature scaler
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            # Hyperparameter optimization
            if optimize_hyperparams:
                logger.info("Optimizing hyperparameters with Optuna")
                best_params = self._optimize_hyperparameters(
                    X_train_scaled, y_train, n_trials=n_trials
                )
                self.params.update(best_params)
            
            # Train final model
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            
            # Calculate metrics
            train_pred = self.model.predict(X_train_scaled)
            val_pred = self.model.predict(X_val_scaled)
            
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'train_mae': np.mean(np.abs(y_train - train_pred)),
                'val_mae': np.mean(np.abs(y_val - val_pred)),
                'train_r2': self.model.score(X_train_scaled, y_train),
                'val_r2': self.model.score(X_val_scaled, y_val),
                'n_features': len(feature_names),
                'n_samples': len(X)
            }
            
            # Store feature importance
            self.feature_importance = dict(zip(
                feature_names,
                self.model.feature_importances_
            ))
            
            # Update training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'params': self.params.copy(),
                'feature_count': len(feature_names)
            }
            self.training_history.append(training_record)
            
            self.is_trained = True
            
            logger.info(f"Training completed - Val RMSE: {metrics['val_rmse']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error("XGBoost training failed", error=str(e))
            raise
    
    def _optimize_hyperparameters(self, 
                                 X: np.ndarray, 
                                 y: np.ndarray,
                                 n_trials: int = 50) -> Dict:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'objective': 'reg:squarederror',
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            # Cross-validation with time series split
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_fold_train, y_fold_train, verbose=False)
                
                pred = model.predict(X_fold_val)
                score = np.sqrt(mean_squared_error(y_fold_val, pred))
                scores.append(score)
            
            return np.mean(scores)
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        return study.best_params
    
    def predict(self, 
                articles_data: List[Dict],
                return_probabilities: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict sentiment scores for articles.
        
        Args:
            articles_data: List of article dictionaries
            return_probabilities: Whether to return prediction confidence
            
        Returns:
            Sentiment predictions (-1 to +1) and optionally confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare features
            X, _ = self.prepare_features(articles_data)
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            # Clip predictions to [-1, 1] range
            predictions = np.clip(predictions, -1.0, 1.0)
            
            if return_probabilities:
                # Calculate confidence based on prediction magnitude and feature importance
                confidence_scores = self._calculate_confidence(X_scaled, predictions)
                return predictions, confidence_scores
            
            return predictions
            
        except Exception as e:
            logger.error("XGBoost prediction failed", error=str(e))
            raise
    
    def _calculate_confidence(self, X: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        try:
            # Method 1: Use prediction magnitude (extreme values are more confident)
            magnitude_confidence = np.abs(predictions)
            
            # Method 2: Use feature completeness (more features = higher confidence)
            feature_completeness = np.mean(X != 0, axis=1)
            
            # Method 3: Use ensemble variance if available
            # For now, use a simple combination
            confidence_scores = 0.6 * magnitude_confidence + 0.4 * feature_completeness
            
            # Normalize to [0, 1]
            confidence_scores = np.clip(confidence_scores, 0.0, 1.0)
            
            return confidence_scores
            
        except Exception as e:
            logger.error("Confidence calculation failed", error=str(e))
            return np.ones(len(predictions)) * 0.5  # Default to medium confidence
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N most important features."""
        if not self.feature_importance:
            return {}
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features[:top_n])
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            model_name: Custom model name (default: auto-generated)
            
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            if model_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"xgboost_sentiment_{self.model_version}_{timestamp}"
            
            model_path = self.model_dir / f"{model_name}.joblib"
            
            # Prepare model data for saving
            model_data = {
                'model': self.model,
                'feature_scaler': self.feature_scaler,
                'feature_names': self.feature_names,
                'params': self.params,
                'feature_importance': self.feature_importance,
                'training_history': self.training_history,
                'model_version': self.model_version,
                'saved_at': datetime.now().isoformat()
            }
            
            # Save model
            joblib.dump(model_data, model_path)
            
            logger.info(f"Model saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error("Failed to save model", error=str(e))
            raise
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            True if loading succeeded
        """
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.feature_scaler = model_data['feature_scaler']
            self.feature_names = model_data['feature_names']
            self.params = model_data['params']
            self.feature_importance = model_data.get('feature_importance', {})
            self.training_history = model_data.get('training_history', [])
            self.model_version = model_data.get('model_version', 'unknown')
            
            self.is_trained = True
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error("Failed to load model", model_path=model_path, error=str(e))
            return False
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        if not self.is_trained:
            return {'status': 'untrained'}
        
        latest_training = self.training_history[-1] if self.training_history else {}
        
        return {
            'status': 'trained',
            'model_version': self.model_version,
            'feature_count': len(self.feature_names),
            'last_training': latest_training.get('timestamp'),
            'last_metrics': latest_training.get('metrics', {}),
            'parameters': self.params,
            'top_features': self.get_feature_importance(10)
        }
    
    def benchmark_inference_speed(self, 
                                 articles_data: List[Dict],
                                 n_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model inference speed."""
        if not self.is_trained:
            raise ValueError("Model must be trained before benchmarking")
        
        import time
        
        # Prepare features once
        X, _ = self.prepare_features(articles_data)
        X_scaled = self.feature_scaler.transform(X)
        
        # Warm up
        for _ in range(10):
            self.model.predict(X_scaled[:1])
        
        # Benchmark
        times = []
        for _ in range(n_iterations):
            start_time = time.time()
            self.model.predict(X_scaled)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        per_sample_time = avg_time / len(X_scaled) * 1000  # ms per sample
        
        return {
            'avg_batch_time_ms': avg_time * 1000,
            'per_sample_time_ms': per_sample_time,
            'throughput_samples_per_sec': len(X_scaled) / avg_time,
            'batch_size': len(X_scaled),
            'n_iterations': n_iterations
        }


# Global XGBoost model instance
xgboost_model = XGBoostSentimentModel()