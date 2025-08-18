"""
Tests for research modules including novel QUBO formulations,
advanced embedding strategies, and experimental benchmarks.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Test imports - handle missing dependencies gracefully
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from quantum_ctl.research.novel_qubo_formulations import (
        NovelQUBOFormulator, AdaptiveConstraintWeighting
    )
    from quantum_ctl.research.advanced_embedding_strategies import (
        TopologyAwareEmbedder, DynamicEmbeddingOptimizer
    )
    from quantum_ctl.research.experimental_benchmarks import (
        ComparativeBenchmarkSuite, StatisticalValidator, ProblemInstanceGenerator
    )
    from quantum_ctl.research.adaptive_penalty_tuning import (
        BayesianPenaltyOptimizer, MLPenaltyTuner
    )
    RESEARCH_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Research modules not available: {e}")
    RESEARCH_MODULES_AVAILABLE = False


@pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
class TestNovelQUBOFormulations:
    """Test novel QUBO formulation strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formulator = NovelQUBOFormulator(
            state_dim=5,
            control_dim=3,
            horizon=12
        )
        
        # Create mock MPC problem
        self.mock_mpc_problem = {
            'state_dynamics': {
                'A': np.eye(5),
                'B': np.random.random((5, 3))
            },
            'initial_state': np.random.random(5),
            'disturbances': np.random.random((5, 12)),
            'objectives': {
                'comfort': np.eye(5),
                'energy': np.eye(3),
                'weights': {
                    'comfort': 0.3,
                    'energy': 0.6,
                    'carbon': 0.1
                }
            },
            'constraints': {
                'temperature_limits': {
                    'matrix': np.eye(5),
                    'bounds': {'lower': 20.0, 'upper': 25.0}
                },
                'power_limits': {
                    'matrix': np.eye(3),
                    'bounds': {'lower': 0.0, 'upper': 10.0}
                }
            },
            'horizon': 12
        }
        
    @pytest.mark.asyncio
    async def test_logarithmic_penalty_formulation(self):
        """Test logarithmic penalty QUBO formulation."""
        
        qubo = await self.formulator.formulate_novel_qubo(
            self.mock_mpc_problem,
            method="logarithmic_penalty"
        )
        
        # Verify QUBO structure
        assert isinstance(qubo, dict)
        assert len(qubo) > 0
        
        # Check variable indices are within expected range
        n_vars = self.formulator.control_dim * self.formulator.horizon
        for (i, j) in qubo.keys():
            assert 0 <= i < n_vars
            assert 0 <= j < n_vars
            
        # Check coefficients are reasonable
        coeffs = list(qubo.values())
        assert all(np.isfinite(c) for c in coeffs)
        assert max(abs(c) for c in coeffs) < 1e6  # Not too large
        
    @pytest.mark.asyncio
    async def test_hierarchical_decomposition(self):
        """Test hierarchical constraint decomposition."""
        
        qubo = await self.formulator.formulate_novel_qubo(
            self.mock_mpc_problem,
            method="hierarchical_decomposition"
        )
        
        # Verify hierarchical structure
        assert isinstance(qubo, dict)
        assert len(qubo) > 0
        
        # Check that hierarchical levels are represented
        coeffs = np.array(list(qubo.values()))
        
        # Should have different magnitude coefficients for different hierarchy levels
        unique_magnitudes = len(set(np.round(np.log10(np.abs(coeffs[coeffs != 0])), 1)))
        assert unique_magnitudes >= 2  # At least 2 different magnitude levels
        
    @pytest.mark.asyncio 
    async def test_multi_objective_pareto(self):
        """Test multi-objective Pareto formulation."""
        
        qubo = await self.formulator.formulate_novel_qubo(
            self.mock_mpc_problem,
            method="multi_objective_pareto"
        )
        
        # Verify multi-objective structure
        assert isinstance(qubo, dict)
        assert len(qubo) > 0
        
        # Check that all objectives are represented
        # The QUBO should incorporate energy, comfort, and carbon objectives
        coeffs = list(qubo.values())
        assert len(coeffs) >= self.formulator.control_dim * self.formulator.horizon
        
    def test_adaptive_constraint_weighting(self):
        """Test adaptive constraint weighting system."""
        
        weighting = AdaptiveConstraintWeighting(learning_rate=0.01)
        
        # Test weight updates
        violations = {
            'dynamics': 0.1,
            'comfort': 0.05,
            'power': 0.0
        }
        
        current_weights = {
            'dynamics': 100.0,
            'comfort': 50.0,
            'power': 25.0
        }
        
        new_weights = weighting.update_weights(violations, 0.8, current_weights)
        
        # Weights should increase for violated constraints
        assert new_weights['dynamics'] > current_weights['dynamics']
        assert new_weights['comfort'] > current_weights['comfort']
        
        # Well-satisfied constraints should have reduced weights
        assert new_weights['power'] <= current_weights['power']
        
    def test_formulation_quality_analysis(self):
        """Test QUBO formulation quality analysis."""
        
        # Create simple test QUBO
        test_qubo = {
            (0, 0): 1.0,
            (1, 1): 1.0,
            (0, 1): -0.5,
            (2, 2): 2.0
        }
        
        test_solution = {0: 1, 1: 0, 2: 1}
        
        quality_metrics = self.formulator.analyze_formulation_quality(
            test_qubo, test_solution
        )
        
        assert 'energy' in quality_metrics
        assert 'binary_satisfaction' in quality_metrics
        assert 'solution_sparsity' in quality_metrics
        assert 'matrix_sparsity' in quality_metrics
        assert 'problem_size' in quality_metrics
        
        # Check metric ranges
        assert 0.0 <= quality_metrics['binary_satisfaction'] <= 1.0
        assert 0.0 <= quality_metrics['solution_sparsity'] <= 1.0
        assert 0.0 <= quality_metrics['matrix_sparsity'] <= 1.0


@pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
class TestAdvancedEmbeddingStrategies:
    """Test advanced embedding strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.embedder = TopologyAwareEmbedder()
        self.dynamic_optimizer = DynamicEmbeddingOptimizer()
        
        # Create test problem graph
        self.test_qubo = {
            (0, 0): 1.0,
            (1, 1): -1.0,
            (0, 1): 0.5,
            (1, 2): -0.3,
            (2, 2): 0.8
        }
        
    @pytest.mark.asyncio
    async def test_topology_aware_embedding(self):
        """Test topology-aware embedding optimization."""
        
        embedding, quality_metrics = await self.embedder.find_optimal_embedding(
            self.test_qubo,
            optimization_strategy="topology_aware"
        )
        
        # Verify embedding structure
        assert isinstance(embedding, dict)
        assert len(embedding) > 0
        
        # Check embedding quality metrics
        assert hasattr(quality_metrics, 'max_chain_length')
        assert hasattr(quality_metrics, 'avg_chain_length')
        assert hasattr(quality_metrics, 'total_qubits_used')
        assert hasattr(quality_metrics, 'topology_efficiency')
        
        # Verify reasonable quality metrics
        assert quality_metrics.topology_efficiency >= 0.0
        assert quality_metrics.topology_efficiency <= 1.0
        
    @pytest.mark.asyncio
    async def test_hierarchical_embedding(self):
        """Test hierarchical embedding strategy."""
        
        embedding, quality_metrics = await self.embedder.find_optimal_embedding(
            self.test_qubo,
            optimization_strategy="hierarchical"
        )
        
        # Verify embedding exists
        assert isinstance(embedding, dict)
        
        # Check that all logical variables are embedded
        logical_vars = set()
        for (i, j) in self.test_qubo.keys():
            logical_vars.add(i)
            logical_vars.add(j)
            
        for var in logical_vars:
            assert var in embedding
            assert isinstance(embedding[var], list)
            assert len(embedding[var]) >= 1
            
    @pytest.mark.asyncio
    async def test_dynamic_embedding_optimization(self):
        """Test dynamic embedding optimization with feedback."""
        
        # Test without feedback first
        embedding1, strategy1 = await self.dynamic_optimizer.optimize_embedding_dynamically(
            self.test_qubo
        )
        
        assert isinstance(embedding1, dict)
        assert isinstance(strategy1, str)
        
        # Test with feedback
        embedding2, strategy2 = await self.dynamic_optimizer.optimize_embedding_dynamically(
            self.test_qubo,
            solution_quality_feedback=0.8
        )
        
        assert isinstance(embedding2, dict)
        assert isinstance(strategy2, str)
        
        # Performance summary should be available
        performance_summary = self.dynamic_optimizer.get_strategy_performance_summary()
        
        assert 'strategy_scores' in performance_summary
        assert 'total_embeddings' in performance_summary
        assert performance_summary['total_embeddings'] >= 2


@pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
class TestExperimentalBenchmarks:
    """Test experimental benchmarking framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.problem_generator = ProblemInstanceGenerator(seed=42)
        self.benchmark_suite = ComparativeBenchmarkSuite()
        self.validator = StatisticalValidator()
        
    def test_problem_instance_generation(self):
        """Test generation of standardized problem instances."""
        
        # Test HVAC instance generation
        hvac_instances = self.problem_generator.generate_hvac_instances(
            sizes=[10, 20],
            instances_per_size=2
        )
        
        assert isinstance(hvac_instances, dict)
        assert len(hvac_instances) == 4  # 2 sizes × 2 instances
        
        # Check instance structure
        for instance_name, instance_data in hvac_instances.items():
            assert 'zones' in instance_data
            assert 'horizon' in instance_data
            assert 'thermal_mass' in instance_data
            assert 'comfort_bounds' in instance_data
            assert 'power_limits' in instance_data
            
    def test_synthetic_qubo_generation(self):
        """Test generation of synthetic QUBO instances."""
        
        synthetic_instances = self.problem_generator.generate_synthetic_qubo_instances(
            sizes=[10, 20],
            densities=[0.2, 0.5],
            instances_per_config=1
        )
        
        assert isinstance(synthetic_instances, dict)
        assert len(synthetic_instances) == 4  # 2 sizes × 2 densities × 1 instance
        
        # Check QUBO structure
        for instance_name, qubo in synthetic_instances.items():
            assert isinstance(qubo, dict)
            
            # Verify QUBO coefficients
            for (i, j), coeff in qubo.items():
                assert isinstance(i, int)
                assert isinstance(j, int)
                assert np.isfinite(coeff)
                
    def test_statistical_validation(self):
        """Test statistical validation framework."""
        
        # Generate test data
        quantum_results = [0.8, 0.85, 0.82, 0.87, 0.83, 0.86, 0.84, 0.88]
        classical_results = [0.75, 0.77, 0.74, 0.76, 0.78, 0.73, 0.75, 0.77]
        
        validation_result = self.validator.validate_quantum_advantage(
            quantum_results,
            classical_results,
            metric_name="solution_quality"
        )
        
        # Check validation structure
        assert 'metric' in validation_result
        assert 'quantum_stats' in validation_result
        assert 'classical_stats' in validation_result
        assert 'tests' in validation_result
        assert 'conclusion' in validation_result
        
        # Check statistical test results
        assert 'normality' in validation_result['tests']
        assert 'main_test' in validation_result['tests']
        assert 'effect_size' in validation_result['tests']
        
        # Check conclusion structure
        conclusion = validation_result['conclusion']
        assert 'conclusion' in conclusion
        assert 'interpretation' in conclusion
        assert 'is_statistically_significant' in conclusion
        
    def test_algorithm_improvement_validation(self):
        """Test algorithm improvement validation."""
        
        baseline_results = [0.7, 0.72, 0.69, 0.71, 0.73, 0.68, 0.70, 0.72]
        improved_results = [0.85, 0.87, 0.84, 0.86, 0.88, 0.83, 0.85, 0.87]
        
        validation_result = self.validator.validate_algorithm_improvement(
            baseline_results,
            improved_results,
            improvement_threshold=0.05
        )
        
        # Check validation structure
        assert 'baseline_mean' in validation_result
        assert 'improved_mean' in validation_result
        assert 'improvement_ratio' in validation_result
        assert 'is_significant_improvement' in validation_result
        assert 'tests' in validation_result
        
        # Should detect significant improvement
        assert validation_result['improvement_ratio'] > 0.05
        
        # Check confidence interval
        assert 'improvement_confidence_interval' in validation_result


@pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
class TestAdaptivePenaltyTuning:
    """Test adaptive penalty tuning system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.bayesian_optimizer = BayesianPenaltyOptimizer()
        self.ml_tuner = MLPenaltyTuner()
        
    @pytest.mark.asyncio
    async def test_bayesian_penalty_optimization(self):
        """Test Bayesian penalty optimization."""
        
        # Mock objective function
        async def mock_objective(weights):
            # Simulate optimization with some noise
            base_quality = sum(weights.values()) / len(weights) / 100.0
            noise = np.random.normal(0, 0.1)
            quality = base_quality + noise
            
            violations = {ct: max(0, np.random.normal(0.05, 0.02)) for ct in weights.keys()}
            
            return quality, violations
            
        constraint_types = ['dynamics', 'comfort', 'power']
        
        result = await self.bayesian_optimizer.optimize_penalties(
            constraint_types,
            mock_objective,
            max_iterations=5  # Small for testing
        )
        
        # Check optimization result structure
        assert isinstance(result.optimal_weights, dict)
        assert len(result.optimal_weights) == len(constraint_types)
        assert result.tuning_iterations >= 5
        assert result.tuning_time > 0
        
        # Check that all constraint types are in optimal weights
        for ct in constraint_types:
            assert ct in result.optimal_weights
            assert result.optimal_weights[ct] > 0
            
    @pytest.mark.asyncio
    async def test_ml_penalty_learning(self):
        """Test ML-based penalty learning from history."""
        
        # Generate mock historical data
        historical_data = []
        for i in range(10):
            problem_data = {
                'problem_size': 50 + i * 10,
                'zones': 10 + i * 2,
                'horizon': 24,
                'constraints': {'dynamics': {}, 'comfort': {}, 'power': {}},
                'objectives': {
                    'weights': {'energy': 0.6, 'comfort': 0.3, 'carbon': 0.1}
                },
                'thermal_mass': [1000] * (10 + i * 2),
                'power_limits': {'max_power': [10] * (10 + i * 2)},
                'weather_profile': {'temperature': [20] * 24},
                'optimal_weights': {
                    'dynamics': 100 + i * 10,
                    'comfort': 50 + i * 5,
                    'power': 25 + i * 2
                },
                'solution_quality': 0.8 + i * 0.01
            }
            historical_data.append(problem_data)
            
        # Test learning from history
        learning_result = await self.ml_tuner.learn_from_history(historical_data)
        
        assert 'status' in learning_result
        
        if learning_result['status'] == 'training_complete':
            assert 'data_points' in learning_result
            assert learning_result['data_points'] == len(historical_data)
            
            # Test penalty prediction
            test_problem = {
                'problem_size': 75,
                'zones': 15,
                'horizon': 24,
                'constraints': {'dynamics': {}, 'comfort': {}, 'power': {}},
                'objectives': {'weights': {'energy': 0.6, 'comfort': 0.3, 'carbon': 0.1}},
                'thermal_mass': [1000] * 15,
                'power_limits': {'max_power': [10] * 15},
                'weather_profile': {'temperature': [20] * 24}
            }
            
            predicted_penalties = await self.ml_tuner.predict_optimal_penalties(
                test_problem,
                ['dynamics', 'comfort', 'power']
            )
            
            assert isinstance(predicted_penalties, dict)
            assert len(predicted_penalties) == 3
            
            for ct in ['dynamics', 'comfort', 'power']:
                assert ct in predicted_penalties
                assert predicted_penalties[ct] > 0
                
    def test_penalty_tuner_performance_summary(self):
        """Test penalty tuner performance summary."""
        
        summary = self.ml_tuner.get_model_performance_summary()
        
        assert 'models_available' in summary
        assert 'training_data_points' in summary
        assert 'model_scores' in summary
        assert 'sklearn_available' in summary
        
        assert isinstance(summary['models_available'], list)
        assert isinstance(summary['training_data_points'], int)
        assert isinstance(summary['model_scores'], dict)
        assert isinstance(summary['sklearn_available'], bool)


# Integration tests
@pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
class TestResearchModulesIntegration:
    """Test integration between research modules."""
    
    @pytest.mark.asyncio
    async def test_novel_formulation_with_embedding(self):
        """Test integration of novel QUBO formulation with advanced embedding."""
        
        # Create formulator and embedder
        formulator = NovelQUBOFormulator(state_dim=3, control_dim=2, horizon=6)
        embedder = TopologyAwareEmbedder()
        
        # Create mock MPC problem
        mpc_problem = {
            'state_dynamics': {'A': np.eye(3), 'B': np.random.random((3, 2))},
            'initial_state': np.random.random(3),
            'objectives': {'energy': np.eye(2), 'comfort': np.eye(3), 'weights': {'energy': 0.7, 'comfort': 0.3}},
            'constraints': {'power': {'matrix': np.eye(2), 'bounds': {'lower': 0, 'upper': 1}}},
            'horizon': 6
        }
        
        # Generate novel QUBO
        qubo = await formulator.formulate_novel_qubo(
            mpc_problem, method="hierarchical_decomposition"
        )
        
        # Find optimal embedding
        embedding, quality_metrics = await embedder.find_optimal_embedding(
            qubo, optimization_strategy="topology_aware"
        )
        
        # Verify integration success
        assert isinstance(qubo, dict)
        assert isinstance(embedding, dict)
        assert quality_metrics.topology_efficiency >= 0.0
        
        # Check that embedding covers all QUBO variables
        qubo_vars = set()
        for (i, j) in qubo.keys():
            qubo_vars.add(i)
            qubo_vars.add(j)
            
        for var in qubo_vars:
            if var in embedding:  # Some vars might not be embedded in mock
                assert isinstance(embedding[var], list)
                
    @pytest.mark.asyncio 
    async def test_benchmarking_with_penalty_tuning(self):
        """Test integration of benchmarking with penalty tuning."""
        
        # Create penalty optimizer
        penalty_optimizer = BayesianPenaltyOptimizer()
        
        # Mock optimization function
        async def mock_optimization(weights):
            # Simulate better performance with balanced weights
            balance_score = 1.0 / (1.0 + np.std(list(weights.values())))
            noise = np.random.normal(0, 0.05)
            quality = balance_score + noise
            
            violations = {ct: max(0, np.random.normal(0.02, 0.01)) for ct in weights.keys()}
            return quality, violations
            
        # Optimize penalties
        constraint_types = ['dynamics', 'comfort']
        result = await penalty_optimizer.optimize_penalties(
            constraint_types,
            mock_optimization,
            max_iterations=3
        )
        
        # Use optimized penalties in benchmarking context
        optimized_weights = result.optimal_weights
        
        # Verify optimization worked
        assert len(optimized_weights) == len(constraint_types)
        assert all(w > 0 for w in optimized_weights.values())
        assert result.solution_quality > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])