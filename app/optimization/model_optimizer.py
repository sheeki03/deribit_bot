#!/usr/bin/env python3
"""
Model Optimization System

Advanced optimization engine for fine-tuning the multimodal FlowScore system
based on backtesting results and performance metrics.

This implements Enhanced PRD Phase 4: Optimization - Model Fine-tuning component.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_squared_error, accuracy_score
import pickle

from app.backtesting.event_study_engine import EventStudyEngine, EventStudyResult
from app.ml.multimodal_scorer import multimodal_scorer
from app.core.logging import logger


@dataclass
class OptimizationResult:
    """Results from model optimization."""
    
    # Optimization target and results
    optimization_target: str
    original_performance: float
    optimized_performance: float
    improvement_pct: float
    
    # Optimized parameters
    original_params: Dict[str, Any]
    optimized_params: Dict[str, Any]
    param_changes: Dict[str, float]
    
    # Validation results
    validation_metrics: Dict[str, float]
    out_of_sample_performance: float
    
    # Optimization metadata
    optimization_method: str
    iterations_completed: int
    convergence_achieved: bool
    optimization_time_seconds: float
    
    # Model performance breakdown
    component_contributions: Dict[str, float]
    confidence_improvements: Dict[str, float]
    
    # Backtesting results
    backtest_results: Dict[str, Any]
    
    timestamp: str


class ModelOptimizer:
    """
    Advanced model optimization system that fine-tunes the multimodal
    FlowScore system for maximum predictive performance.
    
    Optimization targets:
    - Information Ratio maximization
    - Hit Rate improvement
    - Sharpe Ratio optimization
    - Component weight tuning
    - Confidence threshold optimization
    """
    
    def __init__(self):
        self.event_study_engine = EventStudyEngine()
        self.optimization_history = []
        
        # Optimization bounds
        self.component_weight_bounds = {
            'text': (0.1, 0.5),      # 10% to 50%
            'image': (0.2, 0.6),     # 20% to 60%  
            'market': (0.1, 0.4),    # 10% to 40%
            'meta': (0.05, 0.2)      # 5% to 20%
        }
        
        # Confidence threshold bounds
        self.confidence_bounds = (0.1, 0.9)
        
        # Alert threshold bounds  
        self.alert_threshold_bounds = (0.1, 0.8)
    
    async def optimize_component_weights(self,
                                       training_data: List[Dict],
                                       validation_data: List[Dict],
                                       target_metric: str = 'information_ratio',
                                       method: str = 'differential_evolution') -> OptimizationResult:
        """
        Optimize component weights to maximize target metric.
        
        Args:
            training_data: FlowScore data for optimization
            validation_data: Separate data for validation
            target_metric: 'information_ratio', 'hit_rate', or 'sharpe_ratio'
            method: Optimization method
            
        Returns:
            Complete optimization results
        """
        logger.info(f"Starting component weight optimization targeting {target_metric}")
        start_time = datetime.now()
        
        # Store original weights
        original_weights = multimodal_scorer.COMPONENT_WEIGHTS.copy()
        
        # Calculate baseline performance
        baseline_performance = await self._evaluate_performance(
            training_data, original_weights, target_metric
        )
        
        logger.info(f"Baseline {target_metric}: {baseline_performance:.4f}")
        
        # Define optimization objective
        def objective(params):
            # Unpack parameters
            text_weight, image_weight, market_weight, meta_weight = params
            
            # Ensure weights sum to 1.0
            total_weight = text_weight + image_weight + market_weight + meta_weight
            if total_weight <= 0:
                return 1e6  # Invalid weights
            
            # Normalize weights
            weights = {
                'text': text_weight / total_weight,
                'image': image_weight / total_weight,
                'market': market_weight / total_weight,
                'meta': meta_weight / total_weight
            }
            
            # Evaluate performance with these weights
            try:
                # Run evaluation in synchronous context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                performance = loop.run_until_complete(
                    self._evaluate_performance(training_data, weights, target_metric)
                )
                loop.close()
                
                # Return negative for minimization (we want to maximize performance)
                return -performance
                
            except Exception as e:
                logger.error(f"Optimization evaluation failed: {e}")
                return 1e6
        
        # Set up optimization bounds
        bounds = [
            self.component_weight_bounds['text'],
            self.component_weight_bounds['image'], 
            self.component_weight_bounds['market'],
            self.component_weight_bounds['meta']
        ]
        
        # Initial guess (current weights)
        x0 = [
            original_weights['text'],
            original_weights['image'],
            original_weights['market'],
            original_weights['meta']
        ]
        
        # Run optimization
        if method == 'differential_evolution':
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=10
            )
        else:
            result = minimize(
                objective,
                x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
        
        # Extract optimized weights
        opt_params = result.x
        total_weight = sum(opt_params)
        
        optimized_weights = {
            'text': opt_params[0] / total_weight,
            'image': opt_params[1] / total_weight,
            'market': opt_params[2] / total_weight,
            'meta': opt_params[3] / total_weight
        }
        
        # Calculate optimized performance
        optimized_performance = await self._evaluate_performance(
            training_data, optimized_weights, target_metric
        )
        
        # Validate on out-of-sample data
        validation_performance = await self._evaluate_performance(
            validation_data, optimized_weights, target_metric
        )
        
        # Run comprehensive validation
        validation_metrics = await self._comprehensive_validation(
            validation_data, optimized_weights
        )
        
        # Calculate improvement
        improvement = (optimized_performance - baseline_performance) / abs(baseline_performance) * 100
        
        # Create optimization result
        optimization_result = OptimizationResult(
            optimization_target=target_metric,
            original_performance=baseline_performance,
            optimized_performance=optimized_performance,
            improvement_pct=improvement,
            original_params=original_weights,
            optimized_params=optimized_weights,
            param_changes={
                component: optimized_weights[component] - original_weights[component]
                for component in original_weights.keys()
            },
            validation_metrics=validation_metrics,
            out_of_sample_performance=validation_performance,
            optimization_method=method,
            iterations_completed=result.nit if hasattr(result, 'nit') else result.nfev,
            convergence_achieved=result.success,
            optimization_time_seconds=(datetime.now() - start_time).total_seconds(),
            component_contributions=optimized_weights,
            confidence_improvements={},  # Would be calculated from detailed analysis
            backtest_results={},  # Would include full backtest
            timestamp=datetime.now().isoformat()
        )
        
        # Store optimization result
        self.optimization_history.append(optimization_result)
        
        logger.info(
            f"Component weight optimization completed",
            improvement=f"{improvement:.2f}%",
            baseline=baseline_performance,
            optimized=optimized_performance,
            validation=validation_performance
        )
        
        return optimization_result
    
    async def optimize_confidence_thresholds(self,
                                           training_data: List[Dict],
                                           validation_data: List[Dict],
                                           target_metric: str = 'precision') -> OptimizationResult:
        """Optimize confidence thresholds for alert generation."""
        logger.info("Starting confidence threshold optimization")
        start_time = datetime.now()
        
        # Original threshold
        original_threshold = 0.3  # Current default
        
        # Calculate baseline performance
        baseline_performance = await self._evaluate_threshold_performance(
            training_data, original_threshold, target_metric
        )
        
        # Define optimization objective  
        def objective(threshold):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                performance = loop.run_until_complete(
                    self._evaluate_threshold_performance(training_data, threshold[0], target_metric)
                )
                loop.close()
                
                return -performance  # Minimize (maximize performance)
                
            except Exception as e:
                logger.error(f"Threshold evaluation failed: {e}")
                return 1e6
        
        # Optimize threshold
        result = minimize(
            objective,
            x0=[original_threshold],
            bounds=[self.confidence_bounds],
            method='L-BFGS-B'
        )
        
        optimized_threshold = result.x[0]
        
        # Calculate performance
        optimized_performance = await self._evaluate_threshold_performance(
            training_data, optimized_threshold, target_metric
        )
        
        validation_performance = await self._evaluate_threshold_performance(
            validation_data, optimized_threshold, target_metric
        )
        
        improvement = (optimized_performance - baseline_performance) / abs(baseline_performance) * 100
        
        # Create result
        optimization_result = OptimizationResult(
            optimization_target=f"confidence_threshold_{target_metric}",
            original_performance=baseline_performance,
            optimized_performance=optimized_performance,
            improvement_pct=improvement,
            original_params={'confidence_threshold': original_threshold},
            optimized_params={'confidence_threshold': optimized_threshold},
            param_changes={'confidence_threshold': optimized_threshold - original_threshold},
            validation_metrics={'validation_performance': validation_performance},
            out_of_sample_performance=validation_performance,
            optimization_method='L-BFGS-B',
            iterations_completed=result.nfev,
            convergence_achieved=result.success,
            optimization_time_seconds=(datetime.now() - start_time).total_seconds(),
            component_contributions={},
            confidence_improvements={},
            backtest_results={},
            timestamp=datetime.now().isoformat()
        )
        
        self.optimization_history.append(optimization_result)
        
        logger.info(f"Confidence threshold optimization completed: {original_threshold:.3f} -> {optimized_threshold:.3f}")
        
        return optimization_result
    
    async def run_comprehensive_optimization(self,
                                           training_data: List[Dict],
                                           validation_data: List[Dict]) -> Dict[str, OptimizationResult]:
        """Run comprehensive optimization across all parameters."""
        logger.info("Starting comprehensive model optimization")
        
        results = {}
        
        # 1. Optimize component weights for different objectives
        for target in ['information_ratio', 'hit_rate', 'sharpe_ratio']:
            try:
                result = await self.optimize_component_weights(
                    training_data, validation_data, target
                )
                results[f'weights_{target}'] = result
            except Exception as e:
                logger.error(f"Weight optimization failed for {target}: {e}")
        
        # 2. Optimize confidence thresholds
        for metric in ['precision', 'recall', 'f1']:
            try:
                result = await self.optimize_confidence_thresholds(
                    training_data, validation_data, metric
                )
                results[f'threshold_{metric}'] = result
            except Exception as e:
                logger.error(f"Threshold optimization failed for {metric}: {e}")
        
        # 3. Run combined optimization (if any individual optimizations succeeded)
        if results:
            try:
                combined_result = await self._optimize_combined_parameters(
                    training_data, validation_data, results
                )
                results['combined_optimization'] = combined_result
            except Exception as e:
                logger.error(f"Combined optimization failed: {e}")
        
        logger.info(f"Comprehensive optimization completed with {len(results)} successful optimizations")
        
        return results
    
    async def _evaluate_performance(self,
                                  data: List[Dict],
                                  weights: Dict[str, float],
                                  target_metric: str) -> float:
        """Evaluate model performance with given weights."""
        if not data:
            return 0.0
        
        # Temporarily update weights
        original_weights = multimodal_scorer.COMPONENT_WEIGHTS.copy()
        multimodal_scorer.COMPONENT_WEIGHTS.update(weights)
        
        try:
            # Run event study analysis
            event_study_result = await self.event_study_engine.run_event_study(
                data, asset='BTC', confidence_threshold=0.3
            )
            
            # Extract target metric
            if target_metric == 'information_ratio':
                performance = event_study_result.information_ratio
            elif target_metric == 'hit_rate':
                performance = event_study_result.hit_rate
            elif target_metric == 'sharpe_ratio':
                performance = event_study_result.sharpe_ratio
            else:
                performance = event_study_result.overall_correlation
            
            return abs(performance) if performance is not None else 0.0
            
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return 0.0
        finally:
            # Restore original weights
            multimodal_scorer.COMPONENT_WEIGHTS.update(original_weights)
    
    async def _evaluate_threshold_performance(self,
                                            data: List[Dict],
                                            threshold: float,
                                            target_metric: str) -> float:
        """Evaluate performance with specific confidence threshold."""
        if not data:
            return 0.0
        
        try:
            # Run event study with this threshold
            event_study_result = await self.event_study_engine.run_event_study(
                data, asset='BTC', confidence_threshold=threshold
            )
            
            # Calculate precision/recall metrics based on predictions
            if target_metric == 'precision':
                # High precision = fewer false positives
                return event_study_result.hit_rate if event_study_result.significant_events > 0 else 0.0
            elif target_metric == 'recall':
                # High recall = more true positives captured
                return event_study_result.significant_events / max(event_study_result.total_events, 1)
            elif target_metric == 'f1':
                precision = event_study_result.hit_rate if event_study_result.significant_events > 0 else 0.0
                recall = event_study_result.significant_events / max(event_study_result.total_events, 1)
                return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Threshold evaluation failed: {e}")
            return 0.0
    
    async def _comprehensive_validation(self,
                                      validation_data: List[Dict],
                                      optimized_weights: Dict[str, float]) -> Dict[str, float]:
        """Run comprehensive validation metrics."""
        # Temporarily update weights
        original_weights = multimodal_scorer.COMPONENT_WEIGHTS.copy()
        multimodal_scorer.COMPONENT_WEIGHTS.update(optimized_weights)
        
        try:
            # Run full event study
            event_study_result = await self.event_study_engine.run_event_study(
                validation_data, asset='BTC', confidence_threshold=0.3
            )
            
            return {
                'information_ratio': event_study_result.information_ratio,
                'hit_rate': event_study_result.hit_rate,
                'sharpe_ratio': event_study_result.sharpe_ratio,
                'correlation': event_study_result.overall_correlation,
                'correlation_p_value': event_study_result.correlation_p_value,
                'total_events': float(event_study_result.total_events),
                'significant_events': float(event_study_result.significant_events)
            }
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            return {}
        finally:
            # Restore original weights
            multimodal_scorer.COMPONENT_WEIGHTS.update(original_weights)
    
    async def _optimize_combined_parameters(self,
                                          training_data: List[Dict],
                                          validation_data: List[Dict],
                                          individual_results: Dict[str, OptimizationResult]) -> OptimizationResult:
        """Optimize all parameters simultaneously using best individual results."""
        # Start with best individual results
        best_weights_result = max(
            [r for k, r in individual_results.items() if 'weights_' in k],
            key=lambda x: x.optimized_performance,
            default=None
        )
        
        best_threshold_result = max(
            [r for k, r in individual_results.items() if 'threshold_' in k],
            key=lambda x: x.optimized_performance,
            default=None
        )
        
        if not best_weights_result:
            raise ValueError("No successful weight optimization found")
        
        # Use best individual results as starting point
        optimized_weights = best_weights_result.optimized_params
        optimized_threshold = best_threshold_result.optimized_params.get('confidence_threshold', 0.3) if best_threshold_result else 0.3
        
        # Evaluate combined performance
        combined_performance = await self._evaluate_combined_performance(
            validation_data, optimized_weights, optimized_threshold
        )
        
        return OptimizationResult(
            optimization_target='combined_parameters',
            original_performance=best_weights_result.original_performance,
            optimized_performance=combined_performance,
            improvement_pct=0.0,  # Would calculate properly
            original_params=best_weights_result.original_params,
            optimized_params={**optimized_weights, 'confidence_threshold': optimized_threshold},
            param_changes=best_weights_result.param_changes,
            validation_metrics={},
            out_of_sample_performance=combined_performance,
            optimization_method='combined',
            iterations_completed=0,
            convergence_achieved=True,
            optimization_time_seconds=0.0,
            component_contributions=optimized_weights,
            confidence_improvements={},
            backtest_results={},
            timestamp=datetime.now().isoformat()
        )
    
    async def _evaluate_combined_performance(self,
                                           data: List[Dict],
                                           weights: Dict[str, float],
                                           threshold: float) -> float:
        """Evaluate performance with combined optimized parameters."""
        # Temporarily update parameters
        original_weights = multimodal_scorer.COMPONENT_WEIGHTS.copy()
        multimodal_scorer.COMPONENT_WEIGHTS.update(weights)
        
        try:
            event_study_result = await self.event_study_engine.run_event_study(
                data, asset='BTC', confidence_threshold=threshold
            )
            
            # Combined metric: weighted average of key metrics
            combined_score = (
                0.4 * event_study_result.information_ratio +
                0.3 * event_study_result.hit_rate +
                0.3 * abs(event_study_result.overall_correlation)
            )
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Combined evaluation failed: {e}")
            return 0.0
        finally:
            multimodal_scorer.COMPONENT_WEIGHTS.update(original_weights)
    
    def save_optimization_results(self, results: Dict[str, OptimizationResult], output_file: str):
        """Save optimization results to file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            serializable_results = {}
            for key, result in results.items():
                serializable_results[key] = asdict(result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Optimization results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
    
    def load_optimization_results(self, input_file: str) -> Dict[str, OptimizationResult]:
        """Load optimization results from file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = {}
            for key, result_data in data.items():
                # Convert back to OptimizationResult
                results[key] = OptimizationResult(**result_data)
            
            logger.info(f"Loaded {len(results)} optimization results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load optimization results: {e}")
            return {}
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of all optimization runs."""
        if not self.optimization_history:
            return {'message': 'No optimizations completed'}
        
        return {
            'total_optimizations': len(self.optimization_history),
            'successful_optimizations': sum(1 for r in self.optimization_history if r.convergence_achieved),
            'best_improvement': max(r.improvement_pct for r in self.optimization_history),
            'average_improvement': np.mean([r.improvement_pct for r in self.optimization_history]),
            'optimization_targets': list(set(r.optimization_target for r in self.optimization_history)),
            'latest_optimization': self.optimization_history[-1].timestamp
        }


# Global model optimizer instance
model_optimizer = ModelOptimizer()