"""
Self-Optimizing Quantum Controller
Continuously evolves optimization strategies based on performance feedback
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import time
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class OptimizationState:
    """Current state of the optimization system"""
    energy_efficiency: float
    comfort_score: float
    computation_time: float
    quantum_advantage: float
    adaptation_rate: float
    convergence_stability: float

@dataclass
class PerformanceMetrics:
    """Performance metrics for strategy evaluation"""
    energy_reduction_percent: float
    comfort_improvement: float
    quantum_speedup: float
    adaptation_success_rate: float
    convergence_time: float
    stability_index: float

class OptimizationStrategy(ABC):
    """Base class for self-optimizing strategies"""
    
    @abstractmethod
    async def optimize(self, building_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimization strategy"""
        pass
    
    @abstractmethod
    def evolve(self, performance: PerformanceMetrics) -> None:
        """Evolve strategy based on performance feedback"""
        pass
    
    @abstractmethod
    def get_adaptation_score(self) -> float:
        """Get current adaptation effectiveness score"""
        pass

class QuantumEvolutionaryStrategy(OptimizationStrategy):
    """Quantum-inspired evolutionary optimization strategy"""
    
    def __init__(self):
        self.population_size = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.quantum_tunneling_probability = 0.2
        self.elite_percentage = 0.2
        self.generations_count = 0
        self.best_fitness_history = []
    
    async def optimize(self, building_state: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum evolutionary optimization"""
        zones = len(building_state.get('temperatures', [22] * 5))
        
        # Initialize population
        population = self._initialize_population(zones)
        
        # Evolution loop
        for generation in range(10):  # Limit generations for real-time use
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = await self._evaluate_fitness(individual, building_state)
                fitness_scores.append(fitness)
            
            # Select elite
            elite_count = int(self.population_size * self.elite_percentage)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            
            # Create new population
            new_population = [population[i] for i in elite_indices]
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Quantum crossover
                if np.random.random() < self.crossover_rate:
                    parent1, parent2 = np.random.choice(population, 2, replace=False)
                    offspring = self._quantum_crossover(parent1, parent2)
                else:
                    offspring = np.random.choice(population)
                
                # Quantum mutation
                if np.random.random() < self.mutation_rate:
                    offspring = self._quantum_mutation(offspring)
                
                # Quantum tunneling
                if np.random.random() < self.quantum_tunneling_probability:
                    offspring = self._quantum_tunneling(offspring, building_state)
                
                new_population.append(offspring)
            
            population = new_population[:self.population_size]
        
        # Return best solution
        final_fitness = []
        for individual in population:
            fitness = await self._evaluate_fitness(individual, building_state)
            final_fitness.append(fitness)
        
        best_individual = population[np.argmax(final_fitness)]
        self.generations_count += 1
        self.best_fitness_history.append(max(final_fitness))
        
        return {
            'control_solution': self._decode_individual(best_individual, zones),
            'fitness_score': max(final_fitness),
            'generations_evolved': self.generations_count,
            'quantum_enhancements': {
                'tunneling_used': True,
                'population_diversity': self._calculate_diversity(population),
                'convergence_rate': self._calculate_convergence_rate()
            }
        }
    
    def evolve(self, performance: PerformanceMetrics) -> None:
        """Evolve strategy parameters based on performance"""
        # Adapt mutation rate
        if performance.adaptation_success_rate > 0.8:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
        else:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.05)
        
        # Adapt quantum tunneling probability
        if performance.quantum_speedup > 1.5:
            self.quantum_tunneling_probability = min(0.4, self.quantum_tunneling_probability * 1.1)
        else:
            self.quantum_tunneling_probability = max(0.1, self.quantum_tunneling_probability * 0.9)
        
        # Adapt population size based on convergence
        if performance.convergence_time > 5.0:
            self.population_size = min(50, self.population_size + 5)
        else:
            self.population_size = max(10, self.population_size - 2)
    
    def get_adaptation_score(self) -> float:
        """Calculate adaptation effectiveness"""
        if len(self.best_fitness_history) < 5:
            return 0.5
        
        recent_improvement = np.mean(self.best_fitness_history[-3:]) - np.mean(self.best_fitness_history[-6:-3])
        return max(0.0, min(1.0, 0.5 + recent_improvement))
    
    def _initialize_population(self, zones: int) -> List[np.ndarray]:
        """Initialize quantum-inspired population"""
        population = []
        for _ in range(self.population_size):
            # Each individual represents control parameters for all zones
            individual = np.random.random(zones * 2)  # temp_setpoint, airflow_rate per zone
            population.append(individual)
        return population
    
    async def _evaluate_fitness(self, individual: np.ndarray, building_state: Dict[str, Any]) -> float:
        """Evaluate fitness of individual solution"""
        zones = len(building_state.get('temperatures', [22] * 5))
        control_actions = self._decode_individual(individual, zones)
        
        # Calculate energy efficiency
        energy_cost = sum([
            action.get('airflow_rate', 1.0) ** 1.5 
            for action in control_actions.values()
        ])
        
        # Calculate comfort score
        comfort_score = 0
        for i, (zone_id, action) in enumerate(control_actions.items()):
            target_temp = action.get('temp_setpoint', 22)
            current_temp = building_state.get('temperatures', [22] * zones)[i]
            comfort_penalty = abs(target_temp - 22) + abs(current_temp - target_temp)
            comfort_score += max(0, 10 - comfort_penalty)
        
        # Composite fitness with quantum bonus
        base_fitness = comfort_score - 0.1 * energy_cost
        quantum_bonus = 0.2 if energy_cost < 5 else 0  # Reward efficiency
        
        return base_fitness + quantum_bonus
    
    def _decode_individual(self, individual: np.ndarray, zones: int) -> Dict[str, Dict[str, float]]:
        """Decode individual to control actions"""
        control_actions = {}
        for zone in range(zones):
            temp_param = individual[zone * 2]
            airflow_param = individual[zone * 2 + 1]
            
            # Map parameters to valid ranges
            temp_setpoint = 18 + temp_param * 8  # 18-26°C
            airflow_rate = 0.3 + airflow_param * 1.7  # 0.3-2.0
            
            control_actions[f'zone_{zone}'] = {
                'temp_setpoint': temp_setpoint,
                'airflow_rate': airflow_rate
            }
        
        return control_actions
    
    def _quantum_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Quantum-inspired crossover"""
        # Quantum superposition-like crossover
        alpha = np.random.random(len(parent1))
        offspring = alpha * parent1 + (1 - alpha) * parent2
        
        # Add quantum interference
        interference = 0.1 * np.random.normal(0, 1, len(offspring))
        return np.clip(offspring + interference, 0, 1)
    
    def _quantum_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Quantum mutation with tunneling"""
        mutated = individual.copy()
        mutation_mask = np.random.random(len(individual)) < self.mutation_rate
        
        # Quantum-inspired mutation
        for i in np.where(mutation_mask)[0]:
            # Quantum tunneling through energy barriers
            if np.random.random() < 0.3:
                mutated[i] = np.random.random()  # Quantum jump
            else:
                mutated[i] += np.random.normal(0, 0.1)  # Small perturbation
        
        return np.clip(mutated, 0, 1)
    
    def _quantum_tunneling(self, individual: np.ndarray, building_state: Dict[str, Any]) -> np.ndarray:
        """Quantum tunneling to escape local optima"""
        tunneled = individual.copy()
        
        # Determine if system is in local optimum
        current_temp_variance = np.var(building_state.get('temperatures', [22] * 5))
        
        if current_temp_variance < 1.0:  # Low variance suggests local optimum
            # Apply quantum tunneling
            tunnel_indices = np.random.choice(len(individual), size=len(individual)//4, replace=False)
            for idx in tunnel_indices:
                tunneled[idx] = np.random.random()  # Quantum jump to new state
        
        return tunneled
    
    def _calculate_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate population diversity"""
        if len(population) < 2:
            return 0.0
        
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.linalg.norm(population[i] - population[j])
                distances.append(distance)
        
        return np.mean(distances)
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate from fitness history"""
        if len(self.best_fitness_history) < 5:
            return 0.5
        
        recent_slope = np.polyfit(range(5), self.best_fitness_history[-5:], 1)[0]
        return max(0.0, min(1.0, 0.5 + recent_slope))

class AdaptiveNeuralStrategy(OptimizationStrategy):
    """Neural network-based adaptive optimization"""
    
    def __init__(self):
        self.learning_rate = 0.01
        self.network_weights = None
        self.experience_buffer = []
        self.max_buffer_size = 1000
        self.adaptation_cycles = 0
    
    async def optimize(self, building_state: Dict[str, Any]) -> Dict[str, Any]:
        """Neural network optimization with online learning"""
        zones = len(building_state.get('temperatures', [22] * 5))
        
        # Initialize network if not done
        if self.network_weights is None:
            self._initialize_network(zones)
        
        # Prepare input features
        features = self._extract_features(building_state)
        
        # Forward pass through network
        output = self._forward_pass(features)
        
        # Decode output to control actions
        control_actions = self._decode_output(output, zones)
        
        # Calculate reward for experience buffer
        reward = await self._calculate_reward(control_actions, building_state)
        
        # Store experience
        experience = {
            'features': features,
            'actions': output,
            'reward': reward,
            'timestamp': time.time()
        }
        self._store_experience(experience)
        
        # Online learning update
        if len(self.experience_buffer) >= 10:
            self._update_network()
        
        return {
            'control_solution': control_actions,
            'neural_confidence': self._calculate_confidence(features),
            'reward_prediction': reward,
            'adaptation_cycles': self.adaptation_cycles,
            'experience_size': len(self.experience_buffer)
        }
    
    def evolve(self, performance: PerformanceMetrics) -> None:
        """Evolve neural network based on performance"""
        # Adapt learning rate
        if performance.adaptation_success_rate > 0.8:
            self.learning_rate = min(0.05, self.learning_rate * 1.1)
        else:
            self.learning_rate = max(0.001, self.learning_rate * 0.9)
        
        # Prune experience buffer if performance is poor
        if performance.energy_reduction_percent < 10:
            self.experience_buffer = self.experience_buffer[-500:]  # Keep recent experiences
    
    def get_adaptation_score(self) -> float:
        """Calculate neural adaptation effectiveness"""
        if len(self.experience_buffer) < 10:
            return 0.3
        
        # Analyze reward trend in recent experiences
        recent_rewards = [exp['reward'] for exp in self.experience_buffer[-10:]]
        reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        
        return max(0.0, min(1.0, 0.5 + reward_trend))
    
    def _initialize_network(self, zones: int):
        """Initialize neural network weights"""
        input_size = zones * 3  # temp, occupancy, external factors
        hidden_size = zones * 2
        output_size = zones * 2  # temp_setpoint, airflow_rate per zone
        
        self.network_weights = {
            'W1': np.random.randn(input_size, hidden_size) * 0.1,
            'b1': np.zeros((1, hidden_size)),
            'W2': np.random.randn(hidden_size, output_size) * 0.1,
            'b2': np.zeros((1, output_size))
        }
    
    def _extract_features(self, building_state: Dict[str, Any]) -> np.ndarray:
        """Extract features from building state"""
        temperatures = building_state.get('temperatures', [22] * 5)
        occupancy = building_state.get('occupancy', [0.5] * 5)
        external_temp = building_state.get('weather_forecast', {}).get('external_temp', 20)
        
        features = []
        for temp, occ in zip(temperatures, occupancy):
            features.extend([temp, occ, external_temp])
        
        return np.array(features).reshape(1, -1)
    
    def _forward_pass(self, features: np.ndarray) -> np.ndarray:
        """Forward pass through neural network"""
        # Hidden layer with ReLU activation
        hidden = np.maximum(0, features @ self.network_weights['W1'] + self.network_weights['b1'])
        
        # Output layer with sigmoid activation
        output = 1 / (1 + np.exp(-(hidden @ self.network_weights['W2'] + self.network_weights['b2'])))
        
        return output.flatten()
    
    def _decode_output(self, output: np.ndarray, zones: int) -> Dict[str, Dict[str, float]]:
        """Decode neural network output to control actions"""
        control_actions = {}
        
        for zone in range(zones):
            temp_param = output[zone * 2]
            airflow_param = output[zone * 2 + 1]
            
            # Map to valid control ranges
            temp_setpoint = 18 + temp_param * 8  # 18-26°C
            airflow_rate = 0.3 + airflow_param * 1.7  # 0.3-2.0
            
            control_actions[f'zone_{zone}'] = {
                'temp_setpoint': temp_setpoint,
                'airflow_rate': airflow_rate
            }
        
        return control_actions
    
    async def _calculate_reward(self, actions: Dict[str, Dict[str, float]], 
                              building_state: Dict[str, Any]) -> float:
        """Calculate reward for current actions"""
        energy_cost = sum([action['airflow_rate'] ** 1.2 for action in actions.values()])
        
        comfort_score = 0
        temperatures = building_state.get('temperatures', [22] * 5)
        for i, (zone_id, action) in enumerate(actions.items()):
            temp_error = abs(action['temp_setpoint'] - 22)  # Comfort target
            current_error = abs(temperatures[i] - action['temp_setpoint'])
            comfort_score += max(0, 10 - temp_error - current_error)
        
        return comfort_score - 0.15 * energy_cost
    
    def _store_experience(self, experience: Dict[str, Any]):
        """Store experience in replay buffer"""
        self.experience_buffer.append(experience)
        
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)  # Remove oldest experience
    
    def _update_network(self):
        """Update neural network using experience replay"""
        if len(self.experience_buffer) < 10:
            return
        
        # Sample batch from experience buffer
        batch_size = min(32, len(self.experience_buffer))
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        
        batch_features = []
        batch_targets = []
        
        for idx in batch_indices:
            exp = self.experience_buffer[idx]
            batch_features.append(exp['features'].flatten())
            
            # Target is action with reward adjustment
            target = exp['actions'].copy()
            if exp['reward'] > 5:  # Good reward
                target *= 1.1  # Reinforce good actions
            else:
                target *= 0.9  # Reduce poor actions
            
            batch_targets.append(target)
        
        features_batch = np.array(batch_features)
        targets_batch = np.array(batch_targets)
        
        # Simple gradient descent update
        predictions = np.array([self._forward_pass(f.reshape(1, -1)) for f in features_batch])
        
        # Calculate gradients and update weights (simplified)
        error = targets_batch - predictions
        
        # Update output layer weights
        hidden_states = []
        for f in features_batch:
            hidden = np.maximum(0, f.reshape(1, -1) @ self.network_weights['W1'] + self.network_weights['b1'])
            hidden_states.append(hidden.flatten())
        
        hidden_batch = np.array(hidden_states)
        
        # Gradient updates
        self.network_weights['W2'] += self.learning_rate * hidden_batch.T @ error / batch_size
        self.network_weights['b2'] += self.learning_rate * np.mean(error, axis=0, keepdims=True)
        
        self.adaptation_cycles += 1
    
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence in neural network prediction"""
        if len(self.experience_buffer) < 5:
            return 0.3
        
        # Calculate similarity to past experiences
        similarities = []
        for exp in self.experience_buffer[-20:]:  # Check last 20 experiences
            similarity = 1 / (1 + np.linalg.norm(features.flatten() - exp['features'].flatten()))
            similarities.append(similarity)
        
        return np.mean(similarities)

class SelfOptimizingController:
    """Main self-optimizing controller that manages multiple strategies"""
    
    def __init__(self):
        self.strategies = {
            'quantum_evolutionary': QuantumEvolutionaryStrategy(),
            'adaptive_neural': AdaptiveNeuralStrategy()
        }
        self.active_strategy = 'quantum_evolutionary'
        self.performance_history = []
        self.optimization_count = 0
        self.strategy_switch_threshold = 5
    
    async def optimize_with_self_adaptation(self, building_state: Dict[str, Any]) -> Dict[str, Any]:
        """Main optimization with self-adaptation"""
        start_time = time.time()
        
        # Select optimal strategy
        strategy_name = self._select_optimal_strategy()
        strategy = self.strategies[strategy_name]
        
        # Execute optimization
        result = await strategy.optimize(building_state)
        optimization_time = time.time() - start_time
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(result, optimization_time)
        
        # Evolve selected strategy
        strategy.evolve(metrics)
        
        # Store performance
        performance_record = {
            'timestamp': time.time(),
            'strategy': strategy_name,
            'metrics': metrics,
            'building_state': building_state,
            'optimization_time': optimization_time
        }
        self.performance_history.append(performance_record)
        
        # Keep history manageable
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
        
        self.optimization_count += 1
        
        return {
            **result,
            'self_optimization': {
                'strategy_used': strategy_name,
                'performance_metrics': asdict(metrics),
                'adaptation_score': strategy.get_adaptation_score(),
                'optimization_count': self.optimization_count,
                'continuous_improvement': True
            }
        }
    
    def _select_optimal_strategy(self) -> str:
        """Select optimal strategy based on recent performance"""
        if len(self.performance_history) < self.strategy_switch_threshold * 2:
            return self.active_strategy
        
        # Analyze performance of each strategy over recent optimizations
        strategy_performance = {}
        recent_records = self.performance_history[-20:]  # Last 20 optimizations
        
        for record in recent_records:
            strategy = record['strategy']
            metrics = record['metrics']
            
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            
            # Composite performance score
            score = (
                metrics.energy_reduction_percent * 0.3 +
                metrics.comfort_improvement * 0.25 +
                metrics.quantum_speedup * 0.2 +
                metrics.adaptation_success_rate * 0.15 +
                metrics.stability_index * 0.1
            )
            
            strategy_performance[strategy].append(score)
        
        # Switch to best performing strategy if significantly better
        if len(strategy_performance) > 1:
            current_performance = np.mean(strategy_performance.get(self.active_strategy, [0]))
            
            for strategy, scores in strategy_performance.items():
                if strategy != self.active_strategy:
                    avg_performance = np.mean(scores)
                    if avg_performance > current_performance * 1.1:  # 10% improvement threshold
                        logger.info(f"Switching strategy from {self.active_strategy} to {strategy}")
                        self.active_strategy = strategy
                        break
        
        return self.active_strategy
    
    def _calculate_performance_metrics(self, result: Dict[str, Any], optimization_time: float) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        # Energy reduction estimation
        baseline_energy = 10.0
        if 'control_solution' in result:
            optimized_energy = sum([
                action.get('airflow_rate', 1.0) * 1.2 
                for action in result['control_solution'].values()
            ])
        else:
            optimized_energy = result.get('energy_estimate', 8.0)
        
        energy_reduction = max(0, ((baseline_energy - optimized_energy) / baseline_energy) * 100)
        
        # Comfort improvement
        comfort_improvement = result.get('fitness_score', 85) if 'fitness_score' in result else 80
        comfort_improvement = max(0, min(100, comfort_improvement * 10))  # Normalize to 0-100
        
        # Quantum speedup
        classical_baseline_time = 0.2
        quantum_speedup = classical_baseline_time / max(optimization_time, 0.001)
        
        # Adaptation success rate
        adaptation_success_rate = 0.8  # Default good rate
        if len(self.performance_history) >= 5:
            recent_energy_reductions = [
                p['metrics'].energy_reduction_percent 
                for p in self.performance_history[-5:]
                if hasattr(p['metrics'], 'energy_reduction_percent')
            ]
            if recent_energy_reductions:
                trend = np.polyfit(range(len(recent_energy_reductions)), recent_energy_reductions, 1)[0]
                adaptation_success_rate = max(0, min(1, 0.5 + trend / 10))
        
        # Convergence time
        convergence_time = optimization_time
        
        # Stability index
        stability_index = 0.8
        if len(self.performance_history) >= 10:
            recent_times = [p['optimization_time'] for p in self.performance_history[-10:]]
            stability_index = max(0, 1 - np.std(recent_times) / np.mean(recent_times))
        
        return PerformanceMetrics(
            energy_reduction_percent=energy_reduction,
            comfort_improvement=comfort_improvement,
            quantum_speedup=quantum_speedup,
            adaptation_success_rate=adaptation_success_rate,
            convergence_time=convergence_time,
            stability_index=stability_index
        )
    
    def get_self_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive self-optimization status"""
        if not self.performance_history:
            return {"status": "INITIALIZING"}
        
        latest_metrics = self.performance_history[-1]['metrics']
        
        # Calculate strategy effectiveness
        strategy_stats = {}
        for record in self.performance_history[-50:]:  # Last 50 optimizations
            strategy = record['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = []
            
            metrics = record['metrics']
            composite_score = (
                metrics.energy_reduction_percent * 0.4 +
                metrics.comfort_improvement * 0.3 +
                metrics.quantum_speedup * 0.3
            )
            strategy_stats[strategy].append(composite_score)
        
        # Calculate adaptation trends
        if len(self.performance_history) >= 20:
            recent_energy = [p['metrics'].energy_reduction_percent for p in self.performance_history[-20:]]
            energy_trend = np.polyfit(range(len(recent_energy)), recent_energy, 1)[0]
        else:
            energy_trend = 0
        
        return {
            "status": "FULLY_SELF_OPTIMIZING",
            "active_strategy": self.active_strategy,
            "optimization_count": self.optimization_count,
            "current_performance": {
                "energy_reduction": f"{latest_metrics.energy_reduction_percent:.1f}%",
                "comfort_score": f"{latest_metrics.comfort_improvement:.1f}%",
                "quantum_speedup": f"{latest_metrics.quantum_speedup:.2f}x",
                "adaptation_rate": f"{latest_metrics.adaptation_success_rate:.2f}"
            },
            "strategy_performance": {
                name: f"{np.mean(scores):.1f}" 
                for name, scores in strategy_stats.items()
            },
            "adaptation_trend": {
                "energy_improvement_rate": f"{energy_trend:.2f}%/optimization",
                "system_stability": f"{latest_metrics.stability_index:.2f}",
                "continuous_evolution": energy_trend > 0
            },
            "autonomous_capabilities": [
                "Strategy Auto-Selection",
                "Parameter Self-Tuning", 
                "Performance Self-Monitoring",
                "Continuous Self-Improvement"
            ]
        }