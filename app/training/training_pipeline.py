import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

from app.core.config import settings
from app.core.logging import logger
from app.scrapers.bulletproof_scraper import bulletproof_scraper
from app.market_data.coingecko_client import coingecko_client
from app.ml.xgboost_model import xgboost_model
from app.ml.finbert_model import finbert_model
from app.ml.feature_extractors import feature_extractor
from app.ml.ensemble_scorer import ensemble_scorer
from app.scoring.flow_scorer import flow_scorer


class TrainingPipeline:
    """
    Complete training pipeline for the option flows sentiment analysis system.
    
    Features:
    - Historical data collection and validation
    - Ground truth generation from forward returns
    - Time-series aware cross-validation
    - Model training and evaluation
    - Performance monitoring and model selection
    """
    
    def __init__(self):
        self.training_dir = Path(settings.data_dir) / "training"
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.config = {
            'lookback_days': 365,  # How far back to collect training data
            'min_articles': 50,    # Minimum articles needed for training
            'validation_split': 0.2,
            'test_split': 0.1,
            'forward_return_hours': [24, 72],  # Which returns to use for labels
            'sentiment_threshold': 0.02,  # Return threshold for sentiment labels
            'min_confidence_score': 0.1   # Minimum score to include in training
        }
        
        # Training data storage
        self.training_data = None
        self.validation_metrics = {}
        
    async def collect_historical_data(self, 
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Collect historical Deribit option flows articles and market data.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with articles and market data
        """
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            start_date = end_date - timedelta(days=self.config['lookback_days'])
        
        logger.info(f"Collecting historical data from {start_date} to {end_date}")
        
        try:
            # Step 1: Discover historical articles
            logger.info("Discovering historical articles...")
            
            historical_articles = []
            
            # Use bulletproof scraper to get historical articles
            async with bulletproof_scraper:
                discovered_articles = await bulletproof_scraper.discover_all_articles()
                
                # Filter by date range
                for article in discovered_articles:
                    pub_date = article.get('published_at')
                    if pub_date and start_date <= pub_date <= end_date:
                        
                        # Scrape full content for each article
                        logger.info(f"Scraping: {article['title']}")
                        
                        full_content = await bulletproof_scraper.scrape_article_content(
                            article['url'], method='auto'
                        )
                        
                        if full_content:
                            # Merge article metadata with content
                            complete_article = {**article, **full_content}
                            historical_articles.append(complete_article)
                        
                        # Rate limiting
                        await asyncio.sleep(2)
            
            logger.info(f"Collected {len(historical_articles)} historical articles")
            
            if len(historical_articles) < self.config['min_articles']:
                logger.warning(f"Insufficient articles for training: {len(historical_articles)}")
                return pd.DataFrame()
            
            # Step 2: Enrich with market data
            logger.info("Enriching with market data...")
            
            enriched_articles = []
            for article in historical_articles:
                try:
                    # Get market data for this article
                    market_data = await coingecko_client.get_price_data_for_articles([article])
                    
                    if market_data:
                        article['market_data'] = market_data.get(
                            article.get('article_id', article['url']), {}
                        )
                    
                    enriched_articles.append(article)
                    
                except Exception as e:
                    logger.error(f"Market data enrichment failed for {article['url']}: {e}")
                    continue
            
            # Step 3: Convert to DataFrame
            df = pd.DataFrame(enriched_articles)
            
            # Clean and validate data
            df = self._clean_training_data(df)
            
            # Save raw training data
            training_file = self.training_dir / f"raw_data_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.parquet"
            df.to_parquet(training_file)
            
            logger.info(f"Historical data collection completed: {len(df)} articles")
            logger.info(f"Saved to: {training_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Historical data collection failed: {e}")
            return pd.DataFrame()
    
    def _clean_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate training data."""
        original_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['url'])
        
        # Remove articles without content
        df = df.dropna(subset=['body_markdown', 'body_html'], how='all')
        
        # Remove articles with very short content
        df = df[df.apply(lambda row: len(str(row.get('body_markdown', '') or row.get('body_html', ''))), axis=1) > 200]
        
        # Remove articles without publication date
        df = df.dropna(subset=['published_at_utc'])
        
        # Sort by publication date
        df = df.sort_values('published_at_utc')
        
        logger.info(f"Data cleaning: {original_count} → {len(df)} articles")
        
        return df
    
    def generate_training_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate training labels from forward returns.
        
        Uses future price movements to create sentiment labels:
        - Positive returns → Bullish sentiment (+1)
        - Negative returns → Bearish sentiment (-1)  
        - Small returns → Neutral sentiment (0)
        """
        logger.info("Generating training labels from forward returns")
        
        labeled_data = []
        
        for _, article in df.iterrows():
            market_data = article.get('market_data', {})
            
            for asset in ['BTC', 'ETH']:
                if asset not in market_data:
                    continue
                
                asset_data = market_data[asset]
                forward_returns = asset_data.get('forward_returns', {})
                
                # Use 24h and 72h returns for labeling
                returns_24h = forward_returns.get('ret_24h')
                returns_72h = forward_returns.get('ret_72h')
                
                if returns_24h is None and returns_72h is None:
                    continue
                
                # Combine returns with different weights
                if returns_24h is not None and returns_72h is not None:
                    combined_return = 0.7 * returns_24h + 0.3 * returns_72h
                else:
                    combined_return = returns_24h or returns_72h
                
                # Generate sentiment label
                threshold = self.config['sentiment_threshold']
                
                if combined_return > threshold:
                    sentiment_label = 1.0  # Bullish
                elif combined_return < -threshold:
                    sentiment_label = -1.0  # Bearish
                else:
                    sentiment_label = 0.0  # Neutral
                
                # Create training sample
                training_sample = {
                    'article_id': article.get('article_id', article['url']),
                    'url': article['url'],
                    'title': article.get('title', ''),
                    'body_markdown': article.get('body_markdown', ''),
                    'body_html': article.get('body_html', ''),
                    'published_at_utc': article['published_at_utc'],
                    'images': article.get('images', []),
                    'market_data': {asset: asset_data},
                    'asset': asset,
                    'forward_return_24h': returns_24h,
                    'forward_return_72h': returns_72h,
                    'combined_return': combined_return,
                    'sentiment_label': sentiment_label
                }
                
                labeled_data.append(training_sample)
        
        labeled_df = pd.DataFrame(labeled_data)
        
        if len(labeled_df) > 0:
            # Remove samples with very small absolute returns (noise)
            min_abs_return = 0.005  # 0.5%
            labeled_df = labeled_df[np.abs(labeled_df['combined_return']) >= min_abs_return]
            
            # Log label distribution
            label_dist = labeled_df['sentiment_label'].value_counts()
            logger.info(f"Label distribution: {dict(label_dist)}")
            
        logger.info(f"Generated {len(labeled_df)} labeled training samples")
        
        return labeled_df
    
    async def prepare_training_features(self, labeled_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and labels for training.
        
        Returns:
            Tuple of (features, labels, feature_names)
        """
        logger.info("Preparing training features")
        
        # Convert to list of dictionaries for feature extraction
        articles_data = labeled_df.to_dict('records')
        
        # Extract features using our feature extractor
        X, feature_names = xgboost_model.prepare_features(
            articles_data, 
            include_tfidf=True,
            include_market_data=True
        )
        
        # Get labels
        y = labeled_df['sentiment_label'].values.astype(np.float32)
        
        # Fit TF-IDF vectorizer on training texts
        texts = [
            (row.get('body_markdown', '') or row.get('body_html', '')) 
            for row in articles_data
        ]
        feature_extractor.fit_tfidf_vectorizer(texts, max_features=500)
        
        logger.info(f"Training features prepared: {X.shape}, Labels: {len(y)}")
        
        return X, y, feature_names
    
    async def train_models(self, 
                          labeled_df: pd.DataFrame,
                          optimize_hyperparams: bool = True) -> Dict[str, Dict]:
        """
        Train all models in the ensemble.
        
        Returns:
            Dictionary with training results for each model
        """
        logger.info("Starting model training")
        
        # Prepare features
        X, y, feature_names = await self.prepare_training_features(labeled_df)
        
        if len(X) < self.config['min_articles']:
            raise ValueError(f"Insufficient training data: {len(X)} samples")
        
        training_results = {}
        
        # 1. Train XGBoost model
        logger.info("Training XGBoost model...")
        try:
            xgboost_metrics = xgboost_model.train(
                labeled_df.to_dict('records'),
                y.tolist(),
                optimize_hyperparams=optimize_hyperparams,
                n_trials=30 if optimize_hyperparams else 0
            )
            
            training_results['xgboost'] = {
                'status': 'success',
                'metrics': xgboost_metrics,
                'model_path': xgboost_model.save_model()
            }
            
            logger.info("XGBoost training completed successfully")
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            training_results['xgboost'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # 2. Load/Fine-tune FinBERT model
        logger.info("Loading FinBERT model...")
        try:
            finbert_success = finbert_model.load_model()
            
            if finbert_success:
                # Test FinBERT on sample data
                sample_texts = labeled_df['body_markdown'].fillna('').head(10).tolist()
                finbert_results = finbert_model.predict_batch(sample_texts)
                
                # Calculate FinBERT accuracy on training data
                finbert_accuracy = self._evaluate_finbert_on_training(labeled_df)
                
                training_results['finbert'] = {
                    'status': 'success',
                    'accuracy': finbert_accuracy,
                    'sample_predictions': len(finbert_results)
                }
            else:
                training_results['finbert'] = {
                    'status': 'failed',
                    'error': 'Failed to load FinBERT model'
                }
            
        except Exception as e:
            logger.error(f"FinBERT setup failed: {e}")
            training_results['finbert'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # 3. Save training configuration
        training_config = {
            'timestamp': datetime.now().isoformat(),
            'training_samples': len(labeled_df),
            'feature_count': len(feature_names),
            'label_distribution': labeled_df['sentiment_label'].value_counts().to_dict(),
            'config': self.config,
            'results': training_results
        }
        
        config_file = self.training_dir / f"training_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(config_file, 'w') as f:
            json.dump(training_config, f, indent=2, default=str)
        
        logger.info(f"Model training completed. Config saved: {config_file}")
        
        return training_results
    
    def _evaluate_finbert_on_training(self, labeled_df: pd.DataFrame) -> float:
        """Evaluate FinBERT accuracy on training data."""
        try:
            # Sample subset for evaluation (expensive operation)
            sample_size = min(100, len(labeled_df))
            sample_df = labeled_df.sample(n=sample_size, random_state=42)
            
            # Get FinBERT predictions
            texts = sample_df['body_markdown'].fillna('').tolist()
            predictions = finbert_model.predict_batch(texts, batch_size=8)
            
            # Compare with labels
            correct_predictions = 0
            total_predictions = 0
            
            for i, pred in enumerate(predictions):
                true_label = sample_df.iloc[i]['sentiment_label']
                pred_score = pred['sentiment_score']
                
                # Convert to categorical for comparison
                if pred_score > 0.1:
                    pred_category = 1.0
                elif pred_score < -0.1:
                    pred_category = -1.0
                else:
                    pred_category = 0.0
                
                if pred_category == true_label:
                    correct_predictions += 1
                
                total_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            logger.info(f"FinBERT training accuracy: {accuracy:.3f}")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"FinBERT evaluation failed: {e}")
            return 0.0
    
    async def validate_ensemble_performance(self, labeled_df: pd.DataFrame) -> Dict[str, float]:
        """
        Validate ensemble performance using time-series cross-validation.
        """
        logger.info("Validating ensemble performance")
        
        try:
            # Use time-series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            validation_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(labeled_df)):
                logger.info(f"Validation fold {fold + 1}/3")
                
                val_data = labeled_df.iloc[val_idx]
                
                # Get ensemble predictions
                predictions = []
                true_labels = []
                
                # Process in small batches
                batch_size = 10
                for i in range(0, len(val_data), batch_size):
                    batch = val_data.iloc[i:i + batch_size]
                    
                    for _, article in batch.iterrows():
                        try:
                            # Convert to format expected by flow scorer
                            article_data = {
                                'url': article['url'],
                                'title': article['title'],
                                'body_markdown': article['body_markdown'],
                                'published_at_utc': article['published_at_utc'],
                                'images': article.get('images', []),
                                'market_data': article.get('market_data', {})
                            }
                            
                            # Get ensemble prediction
                            asset_scores = await flow_scorer.score_article(
                                article_data, 
                                assets=[article['asset']],
                                enrich_market_data=False
                            )
                            
                            if article['asset'] in asset_scores:
                                pred_score = asset_scores[article['asset']].final_score
                                predictions.append(pred_score)
                                true_labels.append(article['sentiment_label'])
                            
                        except Exception as e:
                            logger.warning(f"Validation prediction failed: {e}")
                            continue
                
                # Calculate fold metrics
                if len(predictions) > 0:
                    fold_metrics = {
                        'mae': mean_absolute_error(true_labels, predictions),
                        'mse': mean_squared_error(true_labels, predictions),
                        'r2': r2_score(true_labels, predictions)
                    }
                    validation_scores.append(fold_metrics)
                    
                    logger.info(f"Fold {fold + 1} metrics: MAE={fold_metrics['mae']:.4f}, R2={fold_metrics['r2']:.4f}")
            
            # Average metrics across folds
            if validation_scores:
                avg_metrics = {
                    metric: np.mean([fold[metric] for fold in validation_scores])
                    for metric in validation_scores[0].keys()
                }
                
                logger.info(f"Average validation metrics: {avg_metrics}")
                return avg_metrics
            else:
                logger.warning("No validation scores computed")
                return {}
                
        except Exception as e:
            logger.error(f"Ensemble validation failed: {e}")
            return {}
    
    async def run_complete_training_pipeline(self, 
                                           start_date: Optional[datetime] = None,
                                           end_date: Optional[datetime] = None) -> Dict:
        """
        Run the complete training pipeline from data collection to model validation.
        """
        logger.info("Starting complete training pipeline")
        
        pipeline_results = {
            'status': 'started',
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        try:
            # Stage 1: Collect historical data
            logger.info("Stage 1: Historical data collection")
            
            historical_df = await self.collect_historical_data(start_date, end_date)
            
            if historical_df.empty:
                pipeline_results['status'] = 'failed'
                pipeline_results['error'] = 'No historical data collected'
                return pipeline_results
            
            pipeline_results['stages']['data_collection'] = {
                'status': 'success',
                'articles_collected': len(historical_df)
            }
            
            # Stage 2: Generate training labels
            logger.info("Stage 2: Label generation")
            
            labeled_df = self.generate_training_labels(historical_df)
            
            if labeled_df.empty:
                pipeline_results['status'] = 'failed'
                pipeline_results['error'] = 'No labeled training data generated'
                return pipeline_results
            
            pipeline_results['stages']['label_generation'] = {
                'status': 'success',
                'training_samples': len(labeled_df),
                'label_distribution': labeled_df['sentiment_label'].value_counts().to_dict()
            }
            
            # Stage 3: Train models
            logger.info("Stage 3: Model training")
            
            training_results = await self.train_models(labeled_df, optimize_hyperparams=True)
            
            pipeline_results['stages']['model_training'] = training_results
            
            # Stage 4: Validate ensemble
            logger.info("Stage 4: Ensemble validation")
            
            validation_metrics = await self.validate_ensemble_performance(labeled_df)
            
            pipeline_results['stages']['ensemble_validation'] = {
                'status': 'success' if validation_metrics else 'failed',
                'metrics': validation_metrics
            }
            
            # Save training data for future use
            self.training_data = labeled_df
            
            pipeline_results['status'] = 'completed'
            logger.info("Training pipeline completed successfully")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            return pipeline_results
    
    def get_training_summary(self) -> Dict:
        """Get summary of training data and model performance."""
        if self.training_data is None:
            return {'status': 'no_training_data'}
        
        df = self.training_data
        
        summary = {
            'data_summary': {
                'total_samples': len(df),
                'assets': df['asset'].value_counts().to_dict(),
                'date_range': {
                    'start': df['published_at_utc'].min().isoformat(),
                    'end': df['published_at_utc'].max().isoformat()
                },
                'label_distribution': df['sentiment_label'].value_counts().to_dict(),
                'return_stats': {
                    'mean_return_24h': df['forward_return_24h'].mean(),
                    'std_return_24h': df['forward_return_24h'].std(),
                    'mean_return_72h': df['forward_return_72h'].mean(),
                    'std_return_72h': df['forward_return_72h'].std()
                }
            },
            'model_status': {
                'xgboost_trained': xgboost_model.is_trained,
                'finbert_loaded': finbert_model.is_loaded,
            }
        }
        
        return summary


# Global training pipeline instance
training_pipeline = TrainingPipeline()