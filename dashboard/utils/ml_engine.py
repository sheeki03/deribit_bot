"""
Machine Learning Engine for Options Analysis Dashboard
Implements relevant ML models for predicting price movements from article analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class MLEngine:
    """Machine Learning engine for article-based price prediction."""
    
    def __init__(self, data_processor):
        """Initialize ML engine with data processor."""
        self.data_processor = data_processor
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = []
        
        # Model performance tracking
        self.model_performance = {}
        self.feature_importance = {}
        
        # Default temporal features (computed from training data when available)
        self.DEFAULT_DATE_FEATURES = {
            'weekday': 2,  # Tuesday (median weekday)
            'month': 6,    # June (median month)
            'quarter': 2   # Q2 (median quarter)
        }
        
    def prepare_ml_dataset(self, target_days: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare dataset for ML training with engineered features."""
        logger.info(f"Preparing ML dataset for {target_days}-day prediction horizon")
        
        # Get correlation data which has article features + forward returns
        correlation_data = self.data_processor.get_correlation_data({})
        
        if correlation_data.empty:
            raise ValueError("No correlation data available for ML training")
        
        # Filter for specific target horizon
        ml_data = correlation_data[correlation_data['days_forward'] == target_days].copy()
        
        if ml_data.empty:
            raise ValueError(f"No data available for {target_days}-day horizon")
        
        logger.info(f"Base dataset size: {len(ml_data)} samples")
        
        # Feature engineering
        ml_data = self._engineer_features(ml_data)
        
        # Create target variables
        ml_data = self._create_targets(ml_data)
        
        # Split features and targets
        feature_cols = [col for col in ml_data.columns if col.startswith('feat_')]
        target_cols = [col for col in ml_data.columns if col.startswith('target_')]
        
        if not feature_cols or not target_cols:
            raise ValueError("No features or targets found after engineering")
        
        X = ml_data[feature_cols]
        y = ml_data[target_cols]
        
        self.feature_columns = feature_cols
        self.target_columns = target_cols
        
        # Compute default temporal features from training data
        self._compute_default_temporal_features(ml_data)
        
        logger.info(f"Final dataset: {len(X)} samples, {len(feature_cols)} features, {len(target_cols)} targets")
        
        return X, y
    
    def _compute_default_temporal_features(self, data: pd.DataFrame):
        """Compute default temporal features from training data distribution."""
        try:
            if 'article_date' in data.columns:
                dates = pd.to_datetime(data['article_date'], errors='coerce')
                valid_dates = dates.dropna()
                
                if len(valid_dates) > 0:
                    # Compute median values from training distribution
                    weekdays = valid_dates.dt.weekday
                    months = valid_dates.dt.month
                    quarters = valid_dates.dt.quarter
                    
                    self.DEFAULT_DATE_FEATURES = {
                        'weekday': int(weekdays.median()),
                        'month': int(months.median()),
                        'quarter': int(quarters.median())
                    }
                    logger.info(f"Updated default temporal features from training data: {self.DEFAULT_DATE_FEATURES}")
        except Exception as e:
            logger.warning(f"Could not compute temporal defaults from training data: {e}")
            # Keep the hardcoded defaults
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw article data."""
        
        # Numerical features (already available)
        data['feat_signal_strength'] = data['signal_strength']
        data['feat_extraction_confidence'] = data['extraction_confidence']
        data['feat_period_volatility'] = data['period_volatility']
        
        # Categorical features - one-hot encode
        # Directional bias
        bias_dummies = pd.get_dummies(data['directional_bias'], prefix='feat_bias')
        data = pd.concat([data, bias_dummies], axis=1)
        
        # Primary theme  
        theme_dummies = pd.get_dummies(data['primary_theme'], prefix='feat_theme')
        data = pd.concat([data, theme_dummies], axis=1)
        
        # Asset
        asset_dummies = pd.get_dummies(data['asset'], prefix='feat_asset')
        data = pd.concat([data, asset_dummies], axis=1)
        
        # Market period
        if 'market_period' in data.columns:
            period_dummies = pd.get_dummies(data['market_period'], prefix='feat_period')
            data = pd.concat([data, period_dummies], axis=1)
        
        # Derived features
        # Confidence-signal interaction
        data['feat_confidence_signal_interaction'] = data['signal_strength'] * data['extraction_confidence']
        
        # Volatility-signal ratio (with robust division guard)
        eps = 1e-6
        # Preserve sign while ensuring denominator is not zero
        sign = np.sign(data['signal_strength'])
        sign = np.where(sign == 0, 1, sign)  # Replace zero signs with 1
        magnitude = np.maximum(np.abs(data['signal_strength']), eps)
        signal_strength_clipped = sign * magnitude
        data['feat_volatility_signal_ratio'] = data['period_volatility'] / signal_strength_clipped
        
        # Time-based features
        data['feat_article_month'] = pd.to_datetime(data['article_date']).dt.month
        data['feat_article_weekday'] = pd.to_datetime(data['article_date']).dt.weekday
        
        # Quarter encoding
        data['feat_article_quarter'] = pd.to_datetime(data['article_date']).dt.quarter
        
        return data
    
    def _create_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction."""
        
        # Classification target: direction (up/down/neutral)
        data['target_direction'] = np.where(data['period_return'] > 0.02, 1,  # Up > 2%
                                   np.where(data['period_return'] < -0.02, -1, 0))  # Down < -2%, else neutral
        
        # Binary classification: profitable or not
        data['target_profitable'] = (data['period_return'] > 0).astype(int)
        
        # Regression target: return magnitude
        data['target_return'] = data['period_return']
        
        # Risk-adjusted return (if volatility available)
        data['target_risk_adjusted_return'] = data['period_return'] / (data['period_volatility'] + 1e-8)
        
        return data
    
    def train_models(self, X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Train multiple ML models and evaluate performance."""
        
        # Time-aware train-test split (important for financial data)
        # Check if we have temporal data for proper time-series split
        correlation_data = self.data_processor.get_correlation_data({})
        if 'article_date' in correlation_data.columns:
            # Time-based split - sort by date and split chronologically
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            # No temporal data available - raise error to prevent data leakage
            raise ValueError(
                "No temporal column 'article_date' found in correlation data. "
                "Time-series split cannot be performed safely. "
                "Please ensure the dataset includes article_date column for proper temporal splitting."
            )
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Scale features for some models
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        # Store scaler for prediction use
        self.scalers['feature_scaler'] = scaler
        
        results = {}
        
        # 1. XGBoost - Main predictive model
        results.update(self._train_xgboost(X_train, X_test, y_train, y_test))
        
        # 2. Random Forest - Comparison and feature importance
        results.update(self._train_random_forest(X_train, X_test, y_train, y_test))
        
        # 3. Logistic Regression - Interpretable baseline
        results.update(self._train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test))
        
        # Store results
        self.model_performance = results
        
        return results
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """Train XGBoost models for classification and regression."""
        
        results = {}
        
        # XGBoost Classifier for direction prediction
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        xgb_clf.fit(X_train, y_train['target_profitable'])
        clf_pred = xgb_clf.predict(X_test)
        clf_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]
        
        # Classification metrics
        clf_accuracy = (clf_pred == y_test['target_profitable']).mean()
        clf_precision = np.sum((clf_pred == 1) & (y_test['target_profitable'] == 1)) / np.sum(clf_pred == 1) if np.sum(clf_pred == 1) > 0 else 0
        clf_recall = np.sum((clf_pred == 1) & (y_test['target_profitable'] == 1)) / np.sum(y_test['target_profitable'] == 1) if np.sum(y_test['target_profitable'] == 1) > 0 else 0
        
        # Financial metrics for classification
        actual_returns = y_test['target_return'].values
        predicted_positions = clf_pred  # 1 for long, 0 for neutral/short
        
        # Calculate strategy returns
        strategy_returns = actual_returns * predicted_positions
        hit_rate = np.sum((predicted_positions == 1) & (actual_returns > 0)) / np.sum(predicted_positions == 1) if np.sum(predicted_positions == 1) > 0 else 0
        
        self.models['xgb_classifier'] = xgb_clf
        self.feature_importance['xgb_classifier'] = dict(zip(X_train.columns, xgb_clf.feature_importances_))
        
        results['xgb_classifier'] = {
            'accuracy': clf_accuracy,
            'precision': clf_precision,
            'recall': clf_recall,
            'hit_rate': hit_rate,
            'avg_strategy_return': np.mean(strategy_returns),
            'total_strategy_return': np.sum(strategy_returns),
            'predictions': clf_pred,
            'probabilities': clf_pred_proba
        }
        
        # XGBoost Regressor for return magnitude
        xgb_reg = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        xgb_reg.fit(X_train, y_train['target_return'])
        reg_pred = xgb_reg.predict(X_test)
        
        # Regression metrics
        mae = mean_absolute_error(y_test['target_return'], reg_pred)
        mse = mean_squared_error(y_test['target_return'], reg_pred)
        r2 = r2_score(y_test['target_return'], reg_pred)
        
        # Directional accuracy from regression
        reg_direction_pred = (reg_pred > 0).astype(int)
        reg_direction_accuracy = (reg_direction_pred == y_test['target_profitable']).mean()
        
        self.models['xgb_regressor'] = xgb_reg
        self.feature_importance['xgb_regressor'] = dict(zip(X_train.columns, xgb_reg.feature_importances_))
        
        results['xgb_regressor'] = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'direction_accuracy': reg_direction_accuracy,
            'predictions': reg_pred
        }
        
        logger.info(f"XGBoost Classifier - Accuracy: {clf_accuracy:.3f}, Hit Rate: {hit_rate:.3f}")
        logger.info(f"XGBoost Regressor - MAE: {mae:.4f}, R²: {r2:.3f}")
        
        return results
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """Train Random Forest models."""
        
        results = {}
        
        # Random Forest Classifier
        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        
        rf_clf.fit(X_train, y_train['target_profitable'])
        clf_pred = rf_clf.predict(X_test)
        clf_pred_proba = rf_clf.predict_proba(X_test)[:, 1]
        
        # Metrics
        clf_accuracy = (clf_pred == y_test['target_profitable']).mean()
        
        # Strategy performance
        actual_returns = y_test['target_return'].values
        strategy_returns = actual_returns * clf_pred
        hit_rate = np.sum((clf_pred == 1) & (actual_returns > 0)) / np.sum(clf_pred == 1) if np.sum(clf_pred == 1) > 0 else 0
        
        self.models['rf_classifier'] = rf_clf
        self.feature_importance['rf_classifier'] = dict(zip(X_train.columns, rf_clf.feature_importances_))
        
        results['rf_classifier'] = {
            'accuracy': clf_accuracy,
            'hit_rate': hit_rate,
            'avg_strategy_return': np.mean(strategy_returns),
            'predictions': clf_pred,
            'probabilities': clf_pred_proba
        }
        
        logger.info(f"Random Forest - Accuracy: {clf_accuracy:.3f}, Hit Rate: {hit_rate:.3f}")
        
        return results
    
    def _train_logistic_regression(self, X_train_scaled, X_test_scaled, y_train, y_test) -> Dict[str, Any]:
        """Train Logistic Regression for interpretable baseline."""
        
        lr_clf = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        
        lr_clf.fit(X_train_scaled, y_train['target_profitable'])
        clf_pred = lr_clf.predict(X_test_scaled)
        clf_pred_proba = lr_clf.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        clf_accuracy = (clf_pred == y_test['target_profitable']).mean()
        
        # Strategy performance
        actual_returns = y_test['target_return'].values
        strategy_returns = actual_returns * clf_pred
        hit_rate = np.sum((clf_pred == 1) & (actual_returns > 0)) / np.sum(clf_pred == 1) if np.sum(clf_pred == 1) > 0 else 0
        
        self.models['logistic_regression'] = lr_clf
        self.feature_importance['logistic_regression'] = dict(zip(X_train_scaled.columns, np.abs(lr_clf.coef_[0])))
        
        results = {
            'logistic_regression': {
                'accuracy': clf_accuracy,
                'hit_rate': hit_rate,
                'avg_strategy_return': np.mean(strategy_returns),
                'predictions': clf_pred,
                'probabilities': clf_pred_proba,
                'coefficients': dict(zip(X_train_scaled.columns, lr_clf.coef_[0]))
            }
        }
        
        logger.info(f"Logistic Regression - Accuracy: {clf_accuracy:.3f}, Hit Rate: {hit_rate:.3f}")
        
        return results
    
    def predict(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make production predictions using trained models."""
        
        if not self.models:
            raise ValueError("Models not trained yet. Please run train_models() first.")
        
        # Create feature vector from article data
        features = self._create_feature_vector(article_data)
        
        if features is None:
            raise ValueError("Could not create feature vector from article data")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name in ['xgb_classifier', 'rf_classifier', 'logistic_regression']:
                    # Classification models - predict probability of profitable trade
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features.reshape(1, -1))
                        probability = proba[0][1]  # Probability of positive class
                        prediction = model.predict(features.reshape(1, -1))[0]
                        
                        predictions[model_name] = {
                            'type': 'classification',
                            'probability': float(probability),
                            'prediction': int(prediction),
                            'confidence': float(max(proba[0])),  # Max probability as confidence
                            'signal': 'BULLISH' if prediction == 1 else 'BEARISH'
                        }
                    
                elif model_name in ['xgb_regressor', 'rf_regressor']:
                    # Regression models - predict expected return
                    predicted_return = model.predict(features.reshape(1, -1))[0]
                    
                    predictions[model_name] = {
                        'type': 'regression',
                        'predicted_return': float(predicted_return),
                        'signal': 'BULLISH' if predicted_return > 0 else 'BEARISH',
                        'magnitude': abs(float(predicted_return))
                    }
                    
            except Exception as e:
                logger.error(f"Error making prediction with {model_name}: {e}")
                predictions[model_name] = {
                    'error': str(e),
                    'type': 'error'
                }
        
        return predictions
    
    def is_trained(self) -> bool:
        """Check if models are trained and ready for predictions."""
        return len(self.models) > 0 and 'feature_scaler' in self.scalers
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models."""
        if not self.is_trained():
            return {"status": "not_trained", "models": []}
        
        model_info = {
            "status": "trained",
            "models": list(self.models.keys()),
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "performance": self.model_performance
        }
        return model_info
    
    def _create_feature_vector(self, article_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Create feature vector from article data for prediction."""
        
        try:
            # Extract features that were used in training
            features = []
            
            # Signal strength features
            signal_strength = article_data.get('signal_strength', 0.5)
            features.extend([
                signal_strength,
                signal_strength ** 2,  # Non-linear signal strength
                np.log(signal_strength + 1e-6)  # Log-transformed signal
            ])
            
            # Confidence features  
            confidence = article_data.get('extraction_confidence', 0.5)
            features.extend([
                confidence,
                confidence * signal_strength,  # Interaction term
            ])
            
            # Directional bias encoding
            bias = article_data.get('directional_bias', 'neutral')
            features.extend([
                1.0 if bias == 'bullish' else 0.0,
                1.0 if bias == 'bearish' else 0.0,
                1.0 if bias == 'neutral' else 0.0
            ])
            
            # Theme encoding (top themes)
            theme = article_data.get('primary_theme', 'other')
            common_themes = ['gamma_squeeze', 'volatility_play', 'earnings_trade', 'technical_analysis', 'market_structure']
            for t in common_themes:
                features.append(1.0 if theme == t else 0.0)
            
            # Asset encoding
            asset = article_data.get('asset', 'BTC')
            features.extend([
                1.0 if asset == 'BTC' else 0.0,
                1.0 if asset == 'ETH' else 0.0
            ])
            
            # Temporal features (if date available)
            if 'article_date' in article_data:
                date = pd.to_datetime(article_data['article_date'], errors='coerce')
                if pd.isna(date):
                    # Use default values if date parsing fails
                    features.extend([
                        self.DEFAULT_DATE_FEATURES['weekday'],
                        self.DEFAULT_DATE_FEATURES['month'],
                        self.DEFAULT_DATE_FEATURES['quarter']
                    ])
                else:
                    # Date parsed successfully - extract temporal features
                    try:
                        features.extend([
                            date.weekday(),  # Day of week [0-6]
                            date.month,      # Month [1-12]
                            date.quarter     # Quarter [1-4]
                        ])
                    except (AttributeError, TypeError) as e:
                        # Handle unexpected date object issues
                        print(f"Warning: Error extracting date features: {e}")
                        features.extend([
                            self.DEFAULT_DATE_FEATURES['weekday'],
                            self.DEFAULT_DATE_FEATURES['month'],
                            self.DEFAULT_DATE_FEATURES['quarter']
                        ])
            else:
                # Use consistent defaults when no date provided
                features.extend([
                    self.DEFAULT_DATE_FEATURES['weekday'],
                    self.DEFAULT_DATE_FEATURES['month'],
                    self.DEFAULT_DATE_FEATURES['quarter']
                ])
            
            # Market context features (if available)
            market_context = article_data.get('market_context', {})
            features.extend([
                market_context.get('volatility_regime_score', 0.5),
                market_context.get('trend_strength', 0.5),
                1.0 if market_context.get('market_period') == 'bull' else 0.0,
                1.0 if market_context.get('market_period') == 'bear' else 0.0
            ])
            
            # Convert to array and validate against training features
            feature_array = np.array(features)
            
            # Validate feature dimensions match training data
            if self.feature_columns:
                expected_length = len(self.feature_columns)
                if len(feature_array) != expected_length:
                    raise ValueError(
                        f"Feature vector length mismatch: got {len(feature_array)}, "
                        f"expected {expected_length} to match training data"
                    )
            
            # Scale features if scaler is available and trained
            if 'feature_scaler' in self.scalers and self.feature_columns:
                # Ensure we have the right shape for the scaler
                try:
                    feature_array = self.scalers['feature_scaler'].transform(feature_array.reshape(1, -1))[0]
                except ValueError as e:
                    logger.error(f"Scaling failed: {e}")
                    raise ValueError(
                        f"Feature scaling failed - ensure features match training format. "
                        f"Feature vector shape: {feature_array.shape}, "
                        f"Expected features: {len(self.feature_columns) if self.feature_columns else 'unknown'}"
                    )
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            return None
    
    def get_feature_importance_chart(self, model_name: str, top_n: int = 15) -> go.Figure:
        """Create feature importance chart for specified model."""
        
        if model_name not in self.feature_importance:
            raise ValueError(f"Model {model_name} not found in feature importance")
        
        importance = self.feature_importance[model_name]
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        features, values = zip(*sorted_features)
        
        # Clean feature names for display
        clean_features = [f.replace('feat_', '').replace('_', ' ').title() for f in features]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=clean_features[::-1],  # Reverse for horizontal bar
            x=list(values)[::-1],
            orientation='h',
            text=[f"{v:.3f}" for v in values][::-1],
            textposition='auto',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f"Feature Importance - {model_name.replace('_', ' ').title()}",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(features) * 25)
        )
        
        return fig
    
    def get_model_comparison_chart(self) -> go.Figure:
        """Create model comparison chart."""
        
        models = []
        accuracies = []
        hit_rates = []
        avg_returns = []
        
        for model_name, metrics in self.model_performance.items():
            if 'classifier' in model_name:
                models.append(model_name.replace('_', ' ').title())
                accuracies.append(metrics['accuracy'])
                hit_rates.append(metrics['hit_rate'])
                avg_returns.append(metrics.get('avg_strategy_return', 0))
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Accuracy', 'Hit Rate', 'Avg Strategy Return'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(go.Bar(x=models, y=accuracies, name='Accuracy', marker_color='blue'), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=hit_rates, name='Hit Rate', marker_color='green'), row=1, col=2)
        fig.add_trace(go.Bar(x=models, y=avg_returns, name='Avg Return', marker_color='red'), row=1, col=3)
        
        fig.update_layout(
            title="Model Performance Comparison",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def get_prediction_analysis_chart(self, model_name: str, y_test: pd.DataFrame) -> go.Figure:
        """Create prediction vs actual analysis chart."""
        
        if model_name not in self.model_performance:
            raise ValueError(f"Model {model_name} not found")
        
        metrics = self.model_performance[model_name]
        
        if 'regressor' in model_name:
            # Regression: prediction vs actual scatter plot
            predictions = metrics['predictions']
            actual = y_test['target_return'].values
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=actual,
                y=predictions,
                mode='markers',
                name='Predictions',
                text=[f"Actual: {a:.3f}<br>Predicted: {p:.3f}" for a, p in zip(actual, predictions)],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            # Perfect prediction line
            min_val, max_val = min(min(actual), min(predictions)), max(max(actual), max(predictions))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            fig.update_layout(
                title=f"Prediction vs Actual - {model_name.replace('_', ' ').title()}",
                xaxis_title="Actual Returns",
                yaxis_title="Predicted Returns"
            )
        
        else:
            # Classification: probability distribution
            probabilities = metrics.get('probabilities', [])
            actual = y_test['target_profitable'].values
            
            if len(probabilities) > 0:
                fig = go.Figure()
                
                # Probability distribution for each class
                fig.add_trace(go.Histogram(
                    x=probabilities[actual == 0],
                    name='Actual Negative',
                    opacity=0.7,
                    nbinsx=20
                ))
                
                fig.add_trace(go.Histogram(
                    x=probabilities[actual == 1],
                    name='Actual Positive',
                    opacity=0.7,
                    nbinsx=20
                ))
                
                fig.update_layout(
                    title=f"Probability Distribution - {model_name.replace('_', ' ').title()}",
                    xaxis_title="Predicted Probability",
                    yaxis_title="Count",
                    barmode='overlay'
                )
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text="Probability data not available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        
        return fig
    
    def predict_new_data(self, X_new: pd.DataFrame, model_name: str = 'xgb_classifier') -> np.ndarray:
        """Make predictions on new data."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        
        # Scale if necessary
        if model_name == 'logistic_regression':
            scaler = self.scalers['feature_scaler']
            X_new_scaled = pd.DataFrame(
                scaler.transform(X_new), 
                columns=X_new.columns, 
                index=X_new.index
            )
            return model.predict(X_new_scaled)
        
        return model.predict(X_new)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model performance summary."""
        
        summary = {
            'total_models': len(self.models),
            'best_accuracy_model': None,
            'best_hit_rate_model': None,
            'best_return_model': None,
            'feature_count': len(self.feature_columns),
            'training_samples': None
        }
        
        if self.model_performance:
            classifier_models = {k: v for k, v in self.model_performance.items() if 'classifier' in k}
            
            if classifier_models:
                # Best accuracy
                best_acc_model = max(classifier_models.items(), key=lambda x: x[1]['accuracy'])
                summary['best_accuracy_model'] = {
                    'name': best_acc_model[0],
                    'score': best_acc_model[1]['accuracy']
                }
                
                # Best hit rate
                best_hit_model = max(classifier_models.items(), key=lambda x: x[1]['hit_rate'])
                summary['best_hit_rate_model'] = {
                    'name': best_hit_model[0],
                    'score': best_hit_model[1]['hit_rate']
                }
                
                # Best average return
                best_return_model = max(classifier_models.items(), key=lambda x: x[1].get('avg_strategy_return', 0))
                summary['best_return_model'] = {
                    'name': best_return_model[0],
                    'score': best_return_model[1].get('avg_strategy_return', 0)
                }
        
        return summary