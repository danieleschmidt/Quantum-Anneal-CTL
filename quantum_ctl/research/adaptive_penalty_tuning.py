"""
Adaptive penalty tuning using machine learning for quantum annealing.

This module implements advanced penalty parameter optimization using
Bayesian optimization and machine learning to automatically tune
QUBO penalty weights for optimal solution quality.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
import asyncio
from abc import ABC, abstractmethod

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from scipy.optimize import minimize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..optimization.quantum_solver import QuantumSolution


@dataclass
class PenaltyTuningResult:
    """Result of penalty parameter tuning."""
    optimal_weights: Dict[str, float]
    solution_quality: float
    constraint_violations: Dict[str, float]
    tuning_iterations: int
    convergence_achieved: bool
    tuning_time: float


@dataclass
class TuningHistory:
    """History of penalty tuning attempts."""
    weights: Dict[str, float]
    quality_score: float
    constraint_violations: Dict[str, float]
    iteration: int
    timestamp: str


class BayesianPenaltyOptimizer:
    """
    Bayesian optimization for penalty parameter tuning.
    
    Uses Gaussian Process regression to model the relationship between
    penalty weights and solution quality, enabling efficient exploration
    of the penalty parameter space.
    """
    
    def __init__(
        self,
        acquisition_function: str = "expected_improvement",
        kernel: str = "rbf",
        exploration_ratio: float = 0.1
    ):
        self.acquisition_function = acquisition_function
        self.kernel_type = kernel
        self.exploration_ratio = exploration_ratio
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gaussian Process model
        if SKLEARN_AVAILABLE:
            if kernel == "rbf":
                kernel_obj = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            elif kernel == "matern":
                kernel_obj = Matern(length_scale=1.0, nu=2.5)
            else:
                kernel_obj = RBF(length_scale=1.0)
                
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel_obj,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10,
                random_state=42
            )
        else:
            self.gp_model = None
            self.logger.warning("Scikit-learn not available, using simplified optimization")
            
        # Optimization history
        self.tuning_history: List[TuningHistory] = []
        self.X_observed = []  # Parameter combinations tried
        self.y_observed = []  # Observed objective values
        
    async def optimize_penalties(
        self,
        constraint_types: List[str],
        objective_function: Callable,
        bounds: Dict[str, Tuple[float, float]] = None,
        max_iterations: int = 50,
        convergence_tolerance: float = 1e-4
    ) -> PenaltyTuningResult:
        """
        Optimize penalty parameters using Bayesian optimization.
        
        Args:
            constraint_types: List of constraint types to optimize
            objective_function: Function that evaluates solution quality
            bounds: Bounds for each penalty parameter
            max_iterations: Maximum optimization iterations
            convergence_tolerance: Convergence tolerance
            
        Returns:
            Optimized penalty parameters and results
        """
        
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting Bayesian penalty optimization for {len(constraint_types)} constraints")
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = {ct: (1.0, 1000.0) for ct in constraint_types}
            
        # Initialize with random samples
        await self._initialize_random_samples(constraint_types, bounds, objective_function)
        
        best_weights = None
        best_quality = float('-inf')
        convergence_achieved = False
        
        for iteration in range(max_iterations):
            # Fit Gaussian Process to observed data
            if SKLEARN_AVAILABLE and len(self.X_observed) > 0:
                self.gp_model.fit(self.X_observed, self.y_observed)
                
            # Find next candidate using acquisition function
            next_weights = await self._acquire_next_candidate(
                constraint_types, bounds, iteration
            )
            
            # Evaluate objective at candidate point
            quality_score, violations = await objective_function(next_weights)
            
            # Update observations
            weights_vector = [next_weights[ct] for ct in constraint_types]
            self.X_observed.append(weights_vector)
            self.y_observed.append(quality_score)
            
            # Record in history
            self.tuning_history.append(TuningHistory(
                weights=next_weights.copy(),
                quality_score=quality_score,
                constraint_violations=violations,
                iteration=iteration,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            ))
            
            # Update best solution
            if quality_score > best_quality:
                best_quality = quality_score
                best_weights = next_weights.copy()
                
            # Check convergence
            if iteration > 10 and self._check_convergence(convergence_tolerance):
                convergence_achieved = True
                self.logger.info(f"Convergence achieved at iteration {iteration}")
                break
                
            self.logger.debug(f"Iteration {iteration}: quality={quality_score:.6f}, weights={next_weights}")
            
        tuning_time = time.time() - start_time
        
        self.logger.info(f"Bayesian optimization completed: best_quality={best_quality:.6f}, time={tuning_time:.2f}s")
        
        return PenaltyTuningResult(
            optimal_weights=best_weights or {ct: 100.0 for ct in constraint_types},
            solution_quality=best_quality,
            constraint_violations=violations,
            tuning_iterations=len(self.tuning_history),
            convergence_achieved=convergence_achieved,
            tuning_time=tuning_time
        )
        
    async def _initialize_random_samples(
        self,
        constraint_types: List[str],
        bounds: Dict[str, Tuple[float, float]],
        objective_function: Callable,
        n_samples: int = 5
    ) -> None:
        """Initialize with random samples for GP training."""
        
        self.logger.debug(f"Initializing with {n_samples} random samples")
        
        for _ in range(n_samples):
            # Generate random weights within bounds
            random_weights = {}
            for ct in constraint_types:
                lower, upper = bounds.get(ct, (1.0, 1000.0))
                random_weights[ct] = np.random.uniform(lower, upper)
                
            # Evaluate objective
            quality_score, violations = await objective_function(random_weights)
            
            # Store observations
            weights_vector = [random_weights[ct] for ct in constraint_types]
            self.X_observed.append(weights_vector)
            self.y_observed.append(quality_score)
            
    async def _acquire_next_candidate(
        self,
        constraint_types: List[str],
        bounds: Dict[str, Tuple[float, float]],
        iteration: int
    ) -> Dict[str, float]:
        """Acquire next candidate using acquisition function."""
        
        if not SKLEARN_AVAILABLE or len(self.X_observed) < 2:
            # Fallback to random sampling
            return self._random_candidate(constraint_types, bounds)
            
        # Define acquisition function
        def acquisition(x):
            x_reshaped = np.array(x).reshape(1, -1)
            
            if self.acquisition_function == "expected_improvement":
                return self._expected_improvement(x_reshaped)
            elif self.acquisition_function == "upper_confidence_bound":
                return self._upper_confidence_bound(x_reshaped, iteration)
            else:
                return self._probability_of_improvement(x_reshaped)
                
        # Optimize acquisition function
        best_candidate = None
        best_acquisition_value = float('-inf')
        
        # Multi-start optimization of acquisition function
        for _ in range(10):
            # Random starting point
            x0 = [np.random.uniform(*bounds.get(ct, (1.0, 1000.0))) for ct in constraint_types]
            
            # Bounds for scipy optimize
            scipy_bounds = [bounds.get(ct, (1.0, 1000.0)) for ct in constraint_types]
            
            try:
                result = minimize(
                    lambda x: -acquisition(x),  # Minimize negative for maximization
                    x0,
                    bounds=scipy_bounds,
                    method='L-BFGS-B'
                )
                
                if result.success and -result.fun > best_acquisition_value:
                    best_acquisition_value = -result.fun
                    best_candidate = result.x
                    
            except Exception as e:
                self.logger.warning(f"Acquisition optimization failed: {e}")
                continue
                
        if best_candidate is not None:
            # Convert back to dictionary
            return {ct: best_candidate[i] for i, ct in enumerate(constraint_types)}
        else:
            # Fallback to random
            return self._random_candidate(constraint_types, bounds)
            
    def _expected_improvement(self, x: np.ndarray, xi: float = 0.01) -> float:
        """Calculate expected improvement acquisition function."""
        
        try:
            mu, sigma = self.gp_model.predict(x, return_std=True)
            
            best_y = np.max(self.y_observed)
            
            with np.errstate(divide='warn'):
                improvement = mu - best_y - xi
                Z = improvement / sigma
                ei = improvement * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
                
            return ei[0] if len(ei) == 1 else ei
            
        except Exception:
            return np.random.random()
            
    def _upper_confidence_bound(self, x: np.ndarray, iteration: int, kappa: float = 2.0) -> float:
        """Calculate upper confidence bound acquisition function."""
        
        try:
            mu, sigma = self.gp_model.predict(x, return_std=True)
            
            # Decrease exploration over time
            exploration_factor = kappa * np.sqrt(np.log(iteration + 1))
            
            ucb = mu + exploration_factor * sigma
            
            return ucb[0] if len(ucb) == 1 else ucb
            
        except Exception:
            return np.random.random()
            
    def _probability_of_improvement(self, x: np.ndarray, xi: float = 0.01) -> float:
        """Calculate probability of improvement acquisition function."""
        
        try:
            mu, sigma = self.gp_model.predict(x, return_std=True)
            
            best_y = np.max(self.y_observed)
            
            with np.errstate(divide='warn'):
                Z = (mu - best_y - xi) / sigma
                pi = stats.norm.cdf(Z)
                
            return pi[0] if len(pi) == 1 else pi
            
        except Exception:
            return np.random.random()
            
    def _random_candidate(
        self,
        constraint_types: List[str],
        bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """Generate random candidate within bounds."""
        
        candidate = {}
        for ct in constraint_types:
            lower, upper = bounds.get(ct, (1.0, 1000.0))
            candidate[ct] = np.random.uniform(lower, upper)
            
        return candidate
        
    def _check_convergence(self, tolerance: float) -> bool:
        """Check if optimization has converged."""
        
        if len(self.y_observed) < 10:
            return False
            
        # Check if recent improvements are small
        recent_values = self.y_observed[-5:]
        improvement = max(recent_values) - min(recent_values)
        
        return improvement < tolerance
        
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization process."""
        
        if not self.tuning_history:
            return {'status': 'no_optimization_run'}
            
        best_result = max(self.tuning_history, key=lambda h: h.quality_score)
        
        return {
            'total_iterations': len(self.tuning_history),
            'best_quality': best_result.quality_score,
            'best_weights': best_result.weights,
            'convergence_curve': [h.quality_score for h in self.tuning_history],
            'final_violations': best_result.constraint_violations,
            'gp_model_available': self.gp_model is not None
        }


class MLPenaltyTuner:
    """
    Machine learning-based penalty tuning using ensemble methods.
    
    Uses random forest and other ML techniques to learn the relationship
    between problem characteristics and optimal penalty parameters.
    """
    
    def __init__(self, ensemble_size: int = 10):
        self.ensemble_size = ensemble_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models if available
        if SKLEARN_AVAILABLE:
            self.models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=ensemble_size,
                    random_state=42,
                    n_jobs=-1
                )
            }
        else:
            self.models = {}
            
        # Training data
        self.problem_features = []
        self.penalty_targets = []
        self.quality_scores = []
        
        # Model performance tracking
        self.model_scores = {}
        
    async def learn_from_history(
        self,
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Learn penalty tuning patterns from historical optimization runs.
        
        Args:
            historical_data: List of historical optimization results
            
        Returns:
            Learning summary and model performance
        """
        
        self.logger.info(f"Learning from {len(historical_data)} historical optimization runs")
        
        # Extract features and targets from historical data
        for data in historical_data:
            problem_features = self._extract_problem_features(data)
            penalty_weights = data.get('optimal_weights', {})
            quality_score = data.get('solution_quality', 0.0)
            
            if problem_features and penalty_weights:
                self.problem_features.append(problem_features)
                self.penalty_targets.append(list(penalty_weights.values()))
                self.quality_scores.append(quality_score)
                
        if len(self.problem_features) < 5:
            self.logger.warning("Insufficient historical data for ML training")
            return {'status': 'insufficient_data', 'data_points': len(self.problem_features)}
            
        # Train models
        training_results = {}
        
        if SKLEARN_AVAILABLE:
            X = np.array(self.problem_features)
            y = np.array(self.penalty_targets)
            
            for model_name, model in self.models.items():
                try:
                    # Train model
                    model.fit(X, y)
                    
                    # Evaluate with cross-validation
                    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                    cv_score = -scores.mean()
                    
                    self.model_scores[model_name] = cv_score
                    
                    training_results[model_name] = {
                        'cv_score': cv_score,
                        'feature_importance': self._get_feature_importance(model, model_name),
                        'trained': True
                    }
                    
                    self.logger.info(f"Trained {model_name}: CV score = {cv_score:.6f}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to train {model_name}: {e}")
                    training_results[model_name] = {'trained': False, 'error': str(e)}
                    
        return {
            'status': 'training_complete',
            'data_points': len(self.problem_features),
            'models': training_results,
            'best_model': min(self.model_scores, key=self.model_scores.get) if self.model_scores else None
        }
        
    async def predict_optimal_penalties(
        self,
        problem_characteristics: Dict[str, Any],
        constraint_types: List[str]
    ) -> Dict[str, float]:
        """
        Predict optimal penalty parameters for a new problem.
        
        Args:
            problem_characteristics: Features of the optimization problem
            constraint_types: List of constraint types to predict for
            
        Returns:
            Predicted optimal penalty weights
        """
        
        if not self.models or not self.problem_features:
            # Fallback to heuristic-based penalties
            return self._heuristic_penalty_prediction(problem_characteristics, constraint_types)
            
        # Extract features for the new problem
        features = self._extract_problem_features(problem_characteristics)
        
        if not features:
            self.logger.warning("Could not extract features for penalty prediction")
            return {ct: 100.0 for ct in constraint_types}
            
        # Get predictions from all available models
        predictions = {}
        
        if SKLEARN_AVAILABLE:
            features_array = np.array(features).reshape(1, -1)
            
            for model_name, model in self.models.items():
                if hasattr(model, 'predict'):
                    try:
                        prediction = model.predict(features_array)[0]
                        predictions[model_name] = prediction
                    except Exception as e:
                        self.logger.warning(f"Prediction failed for {model_name}: {e}")
                        
        # Ensemble prediction (average of all models)
        if predictions:
            ensemble_prediction = np.mean(list(predictions.values()), axis=0)
        else:
            # Fallback
            return {ct: 100.0 for ct in constraint_types}
            
        # Convert prediction to penalty dictionary
        penalty_dict = {}
        for i, ct in enumerate(constraint_types):
            if i < len(ensemble_prediction):
                penalty_dict[ct] = max(1.0, ensemble_prediction[i])  # Ensure positive
            else:
                penalty_dict[ct] = 100.0  # Default
                
        self.logger.info(f"Predicted penalties: {penalty_dict}")
        return penalty_dict
        
    def _extract_problem_features(self, problem_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features from problem characteristics."""
        
        features = []
        
        # Problem size features
        features.append(problem_data.get('problem_size', 0))
        features.append(problem_data.get('horizon', 0))
        features.append(problem_data.get('zones', 0))
        
        # Problem density/complexity
        constraints = problem_data.get('constraints', {})
        features.append(len(constraints))
        
        # Energy and comfort weight features
        objectives = problem_data.get('objectives', {})
        features.append(objectives.get('weights', {}).get('energy', 0.6))
        features.append(objectives.get('weights', {}).get('comfort', 0.3))
        features.append(objectives.get('weights', {}).get('carbon', 0.1))
        
        # Thermal characteristics
        thermal_mass = problem_data.get('thermal_mass', [])
        if isinstance(thermal_mass, (list, np.ndarray)) and len(thermal_mass) > 0:
            features.append(np.mean(thermal_mass))
            features.append(np.std(thermal_mass))
        else:
            features.extend([1000.0, 100.0])  # Default values
            
        # Power limits
        power_limits = problem_data.get('power_limits', {})
        max_power = power_limits.get('max_power', [])
        if isinstance(max_power, (list, np.ndarray)) and len(max_power) > 0:
            features.append(np.mean(max_power))
            features.append(np.max(max_power))
        else:
            features.extend([10.0, 15.0])  # Default values
            
        # Weather variability
        weather = problem_data.get('weather_profile', {})
        temp_profile = weather.get('temperature', [])
        if isinstance(temp_profile, (list, np.ndarray)) and len(temp_profile) > 0:
            features.append(np.mean(temp_profile))
            features.append(np.std(temp_profile))
        else:
            features.extend([20.0, 5.0])  # Default values
            
        return features
        
    def _heuristic_penalty_prediction(
        self,
        problem_characteristics: Dict[str, Any],
        constraint_types: List[str]
    ) -> Dict[str, float]:
        """Heuristic-based penalty prediction as fallback."""
        
        problem_size = problem_characteristics.get('problem_size', 100)
        horizon = problem_characteristics.get('horizon', 24)
        
        # Base penalties scale with problem complexity
        base_penalty = 100.0 * (1 + np.log10(max(1, problem_size / 100.0)))
        
        penalty_dict = {}
        
        for ct in constraint_types:
            if 'dynamics' in ct.lower():
                # Dynamics constraints are critical - high penalty
                penalty_dict[ct] = base_penalty * 10.0
            elif 'comfort' in ct.lower():
                # Comfort constraints are important - medium penalty
                penalty_dict[ct] = base_penalty * 2.0
            elif 'power' in ct.lower() or 'energy' in ct.lower():
                # Power/energy constraints - medium penalty
                penalty_dict[ct] = base_penalty * 1.5
            else:
                # Other constraints - base penalty
                penalty_dict[ct] = base_penalty
                
        return penalty_dict
        
    def _get_feature_importance(self, model, model_name: str) -> Dict[str, float]:
        """Get feature importance from trained model."""
        
        feature_names = [
            'problem_size', 'horizon', 'zones', 'n_constraints',
            'energy_weight', 'comfort_weight', 'carbon_weight',
            'avg_thermal_mass', 'std_thermal_mass',
            'avg_max_power', 'max_max_power',
            'avg_temperature', 'std_temperature'
        ]
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return {name: float(imp) for name, imp in zip(feature_names, importances)}
            else:
                return {name: 1.0/len(feature_names) for name in feature_names}
                
        except Exception:
            return {name: 1.0/len(feature_names) for name in feature_names}
            
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance."""
        
        return {
            'models_available': list(self.models.keys()),
            'training_data_points': len(self.problem_features),
            'model_scores': self.model_scores,
            'best_model': min(self.model_scores, key=self.model_scores.get) if self.model_scores else None,
            'sklearn_available': SKLEARN_AVAILABLE
        }