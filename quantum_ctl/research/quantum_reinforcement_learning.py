"""
Quantum Reinforcement Learning for Adaptive HVAC Control

This module implements cutting-edge quantum reinforcement learning algorithms
for adaptive HVAC control that learn optimal policies from real building data
and quantum environment interactions.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from abc import ABC, abstractmethod
import time
from collections import deque

try:
    import dimod
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False

logger = logging.getLogger(__name__)


class QuantumRLAlgorithm(Enum):
    """Quantum reinforcement learning algorithms."""
    QUANTUM_Q_LEARNING = "quantum_q_learning"
    VARIATIONAL_QUANTUM_ACTOR_CRITIC = "vqac"
    QUANTUM_POLICY_GRADIENT = "quantum_policy_gradient"
    QUANTUM_ADVANTAGE_ACTOR_CRITIC = "qa2c"


@dataclass
class QuantumCircuitConfig:
    """Configuration for quantum circuits in RL."""
    num_qubits: int = 8
    circuit_depth: int = 6
    entangling_gates: str = "cx"  # cx, cz, iswap
    measurement_basis: str = "computational"  # computational, x, y
    noise_model: Optional[Dict[str, float]] = None


@dataclass
class HVACState:
    """State representation for HVAC RL environment."""
    zone_temperatures: np.ndarray
    outdoor_temperature: float
    occupancy: np.ndarray
    time_of_day: float
    day_of_week: int
    energy_price: float
    system_status: np.ndarray
    weather_forecast: np.ndarray
    
    def to_vector(self) -> np.ndarray:
        """Convert state to flat vector for quantum encoding."""
        state_vector = np.concatenate([
            self.zone_temperatures,
            [self.outdoor_temperature],
            self.occupancy,
            [self.time_of_day],
            [self.day_of_week / 7.0],  # Normalize
            [self.energy_price / 100.0],  # Normalize
            self.system_status,
            self.weather_forecast
        ])
        return state_vector
    
    def normalize(self) -> 'HVACState':
        """Normalize state values for quantum processing."""
        normalized = HVACState(
            zone_temperatures=(self.zone_temperatures - 20) / 10,  # Center around 20°C
            outdoor_temperature=(self.outdoor_temperature - 15) / 20,
            occupancy=self.occupancy,  # Already 0-1
            time_of_day=self.time_of_day / 24.0,
            day_of_week=self.day_of_week,
            energy_price=np.clip(self.energy_price / 200.0, 0, 1),
            system_status=self.system_status,
            weather_forecast=(self.weather_forecast - 15) / 20
        )
        return normalized


@dataclass
class HVACAction:
    """Action representation for HVAC control."""
    setpoints: np.ndarray          # Temperature setpoints for each zone
    fan_speeds: np.ndarray         # Fan speed controls (0-1)
    damper_positions: np.ndarray   # Damper positions (0-1)
    system_mode: int               # 0=off, 1=heat, 2=cool, 3=auto
    
    def to_vector(self) -> np.ndarray:
        """Convert action to flat vector."""
        action_vector = np.concatenate([
            (self.setpoints - 20) / 10,  # Normalize around 20°C
            self.fan_speeds,
            self.damper_positions,
            [self.system_mode / 3.0]  # Normalize mode
        ])
        return action_vector


@dataclass
class QuantumRLConfig:
    """Configuration for quantum RL training."""
    algorithm: QuantumRLAlgorithm
    circuit_config: QuantumCircuitConfig
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    batch_size: int = 32
    memory_size: int = 10000
    target_update_frequency: int = 100
    quantum_advantage_threshold: float = 0.05


class QuantumStateEncoder:
    """Encodes classical HVAC states into quantum states."""
    
    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        self.max_features = 2 ** num_qubits
        
    def encode_state(self, state: HVACState) -> np.ndarray:
        """Encode HVAC state as quantum amplitudes."""
        state_vector = state.normalize().to_vector()
        
        # Pad or truncate to fit quantum register
        if len(state_vector) > self.max_features:
            state_vector = state_vector[:self.max_features]
        else:
            state_vector = np.pad(state_vector, (0, self.max_features - len(state_vector)))
        
        # Normalize for quantum state (amplitudes must sum to 1)
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            quantum_state = state_vector / norm
        else:
            quantum_state = np.ones(self.max_features) / np.sqrt(self.max_features)
        
        return quantum_state
    
    def encode_batch(self, states: List[HVACState]) -> np.ndarray:
        """Encode batch of states for quantum processing."""
        encoded_states = []
        for state in states:
            encoded_states.append(self.encode_state(state))
        return np.array(encoded_states)


class QuantumActionSelector:
    """Selects actions using quantum circuits."""
    
    def __init__(self, num_actions: int, circuit_config: QuantumCircuitConfig):
        self.num_actions = num_actions
        self.circuit_config = circuit_config
        self.quantum_circuit = self._build_action_circuit()
        
    def _build_action_circuit(self) -> Dict[str, Any]:
        """Build quantum circuit for action selection."""
        # This would use a quantum computing framework like Qiskit or Cirq
        # For now, we'll simulate the quantum circuit behavior
        
        circuit_info = {
            "num_qubits": self.circuit_config.num_qubits,
            "depth": self.circuit_config.circuit_depth,
            "parameters": np.random.random(self.circuit_config.circuit_depth * self.circuit_config.num_qubits)
        }
        
        return circuit_info
    
    def select_action(self, quantum_state: np.ndarray, 
                     exploration_rate: float = 0.0) -> HVACAction:
        """Select action using quantum circuit."""
        
        # Simulate quantum circuit execution
        action_probabilities = self._execute_quantum_circuit(quantum_state)
        
        # Apply exploration
        if np.random.random() < exploration_rate:
            # Quantum exploration: sample from circuit output
            action_idx = np.random.choice(len(action_probabilities), p=action_probabilities)
        else:
            # Exploitation: choose highest probability action
            action_idx = np.argmax(action_probabilities)
        
        # Convert action index to HVAC action
        return self._decode_action(action_idx)
    
    def _execute_quantum_circuit(self, quantum_state: np.ndarray) -> np.ndarray:
        """Execute quantum circuit and return action probabilities."""
        # Simulate quantum circuit execution
        # In real implementation, this would run on quantum hardware or simulator
        
        # Apply parameterized quantum gates
        processed_state = quantum_state.copy()
        for i in range(self.circuit_config.circuit_depth):
            # Simulate rotation gates
            processed_state = self._apply_rotation_layer(processed_state, i)
            # Simulate entangling gates
            processed_state = self._apply_entangling_layer(processed_state)
        
        # Measure to get action probabilities
        action_probs = np.abs(processed_state[:self.num_actions]) ** 2
        action_probs /= np.sum(action_probs)  # Normalize
        
        return action_probs
    
    def _apply_rotation_layer(self, state: np.ndarray, layer_idx: int) -> np.ndarray:
        """Simulate rotation gate layer."""
        # Apply parameterized rotations
        angles = self.quantum_circuit["parameters"][layer_idx::self.circuit_config.circuit_depth]
        
        modified_state = state.copy()
        for i, angle in enumerate(angles[:len(state)]):
            # Simulate rotation effect
            modified_state[i] *= np.exp(1j * angle)
        
        return modified_state
    
    def _apply_entangling_layer(self, state: np.ndarray) -> np.ndarray:
        """Simulate entangling gate layer."""
        # Simple entanglement simulation
        entangled_state = state.copy()
        for i in range(0, len(state) - 1, 2):
            # Simulate CNOT-like entanglement
            entangled_state[i], entangled_state[i+1] = (
                (entangled_state[i] + entangled_state[i+1]) / np.sqrt(2),
                (entangled_state[i] - entangled_state[i+1]) / np.sqrt(2)
            )
        
        return entangled_state
    
    def _decode_action(self, action_idx: int) -> HVACAction:
        """Decode action index to HVAC action."""
        # Simple decoding scheme - would be more sophisticated in practice
        num_zones = 4  # Assume 4 zones for simplicity
        
        # Extract components from action index
        setpoint_base = 20 + (action_idx % 8) - 4  # 16-24°C range
        fan_speed = (action_idx % 4) / 3.0  # 0-1 range
        damper_pos = ((action_idx // 4) % 4) / 3.0  # 0-1 range
        system_mode = (action_idx // 16) % 4  # 0-3 modes
        
        return HVACAction(
            setpoints=np.full(num_zones, setpoint_base),
            fan_speeds=np.full(num_zones, fan_speed),
            damper_positions=np.full(num_zones, damper_pos),
            system_mode=system_mode
        )


class QuantumValueNetwork:
    """Quantum neural network for value function approximation."""
    
    def __init__(self, circuit_config: QuantumCircuitConfig):
        self.circuit_config = circuit_config
        self.parameters = np.random.random(circuit_config.num_qubits * circuit_config.circuit_depth * 3)
        self.parameter_momentum = np.zeros_like(self.parameters)
        
    def evaluate(self, quantum_state: np.ndarray) -> float:
        """Evaluate state value using quantum circuit."""
        # Simulate quantum value network
        processed_state = self._process_through_circuit(quantum_state)
        
        # Extract value from quantum state
        value = np.real(np.sum(processed_state * np.conj(processed_state)))
        
        # Scale to reasonable value range
        return value * 1000 - 500  # Center around 0
    
    def _process_through_circuit(self, state: np.ndarray) -> np.ndarray:
        """Process state through parameterized quantum circuit."""
        processed = state.copy()
        
        param_idx = 0
        for layer in range(self.circuit_config.circuit_depth):
            # Rotation layer
            for qubit in range(min(len(processed), self.circuit_config.num_qubits)):
                angle_x = self.parameters[param_idx]
                angle_y = self.parameters[param_idx + 1]
                angle_z = self.parameters[param_idx + 2]
                param_idx += 3
                
                # Simulate rotation gates
                processed[qubit] *= np.exp(1j * (angle_x + angle_y + angle_z))
            
            # Entangling layer
            for i in range(0, min(len(processed) - 1, self.circuit_config.num_qubits - 1)):
                # CNOT-like entanglement
                control = processed[i]
                target = processed[i + 1]
                processed[i] = control
                processed[i + 1] = target * control / (np.abs(control) + 1e-8)
        
        return processed
    
    def update_parameters(self, gradient: np.ndarray, learning_rate: float = 0.01):
        """Update quantum circuit parameters using gradient."""
        # Apply momentum
        momentum = 0.9
        self.parameter_momentum = momentum * self.parameter_momentum + learning_rate * gradient
        
        # Update parameters
        self.parameters += self.parameter_momentum
        
        # Keep parameters in reasonable range
        self.parameters = np.clip(self.parameters, -2*np.pi, 2*np.pi)


class HVACQuantumEnvironment:
    """Quantum-enhanced HVAC simulation environment."""
    
    def __init__(self, building_config: Dict[str, Any]):
        self.building_config = building_config
        self.num_zones = building_config.get('num_zones', 4)
        self.current_state = None
        self.time_step = 0
        self.episode_length = 288  # 24 hours in 5-minute steps
        
        # Environment parameters
        self.comfort_range = (20, 24)  # °C
        self.energy_price_base = 100  # $/MWh
        
        # Quantum noise simulation
        self.quantum_noise_level = 0.05
        
    def reset(self) -> HVACState:
        """Reset environment to initial state."""
        self.time_step = 0
        
        # Initialize state
        self.current_state = HVACState(
            zone_temperatures=np.random.uniform(18, 26, self.num_zones),
            outdoor_temperature=np.random.uniform(10, 35),
            occupancy=np.random.randint(0, 2, self.num_zones),
            time_of_day=0.0,
            day_of_week=0,
            energy_price=self.energy_price_base,
            system_status=np.ones(self.num_zones),
            weather_forecast=np.random.uniform(10, 35, 8)  # 8-hour forecast
        )
        
        return self.current_state
    
    def step(self, action: HVACAction) -> Tuple[HVACState, float, bool, Dict[str, Any]]:
        """Take environment step with quantum noise effects."""
        
        # Apply quantum noise to action (simulating quantum hardware noise)
        noisy_action = self._apply_quantum_noise(action)
        
        # Update building thermal dynamics
        new_temperatures = self._update_thermal_dynamics(noisy_action)
        
        # Update time and other state variables
        self.time_step += 1
        new_time_of_day = (self.time_step * 5 / 60) % 24  # 5-minute steps
        
        # Update state
        new_state = HVACState(
            zone_temperatures=new_temperatures,
            outdoor_temperature=self.current_state.outdoor_temperature + np.random.normal(0, 0.5),
            occupancy=self._update_occupancy(new_time_of_day),
            time_of_day=new_time_of_day,
            day_of_week=self.current_state.day_of_week,
            energy_price=self._update_energy_price(new_time_of_day),
            system_status=self.current_state.system_status,
            weather_forecast=self.current_state.weather_forecast
        )
        
        # Compute reward
        reward = self._compute_reward(new_state, noisy_action)
        
        # Check if episode is done
        done = self.time_step >= self.episode_length
        
        # Additional info
        info = {
            'comfort_violations': self._count_comfort_violations(new_state),
            'energy_consumption': self._compute_energy_consumption(noisy_action),
            'quantum_noise_applied': True
        }
        
        self.current_state = new_state
        return new_state, reward, done, info
    
    def _apply_quantum_noise(self, action: HVACAction) -> HVACAction:
        """Apply quantum hardware noise to action."""
        noise_std = self.quantum_noise_level
        
        noisy_action = HVACAction(
            setpoints=action.setpoints + np.random.normal(0, noise_std, len(action.setpoints)),
            fan_speeds=np.clip(action.fan_speeds + np.random.normal(0, noise_std, len(action.fan_speeds)), 0, 1),
            damper_positions=np.clip(action.damper_positions + np.random.normal(0, noise_std, len(action.damper_positions)), 0, 1),
            system_mode=action.system_mode  # Discrete, no noise
        )
        
        return noisy_action
    
    def _update_thermal_dynamics(self, action: HVACAction) -> np.ndarray:
        """Update zone temperatures based on HVAC action."""
        current_temps = self.current_state.zone_temperatures
        outdoor_temp = self.current_state.outdoor_temperature
        
        # Simple thermal model
        new_temps = current_temps.copy()
        
        for i in range(self.num_zones):
            # Heat transfer with outdoor
            outdoor_influence = 0.1 * (outdoor_temp - current_temps[i])
            
            # HVAC influence
            hvac_influence = 0.0
            if action.system_mode == 1:  # Heating
                hvac_influence = action.fan_speeds[i] * (action.setpoints[i] - current_temps[i]) * 0.3
            elif action.system_mode == 2:  # Cooling
                hvac_influence = action.fan_speeds[i] * (action.setpoints[i] - current_temps[i]) * 0.3
            
            # Internal heat gains
            occupancy_heat = self.current_state.occupancy[i] * 2.0  # 2°C per person
            
            # Update temperature
            new_temps[i] += outdoor_influence + hvac_influence + occupancy_heat * 0.1
            
            # Add some randomness
            new_temps[i] += np.random.normal(0, 0.1)
        
        return new_temps
    
    def _update_occupancy(self, time_of_day: float) -> np.ndarray:
        """Update occupancy based on time of day."""
        # Simple occupancy model (higher during work hours)
        if 8 <= time_of_day <= 18:
            occupancy_prob = 0.8
        else:
            occupancy_prob = 0.2
        
        return np.random.binomial(1, occupancy_prob, self.num_zones)
    
    def _update_energy_price(self, time_of_day: float) -> float:
        """Update energy price based on time of day."""
        # Peak pricing during day
        if 12 <= time_of_day <= 18:
            return self.energy_price_base * 1.5
        elif 6 <= time_of_day <= 12 or 18 <= time_of_day <= 22:
            return self.energy_price_base * 1.2
        else:
            return self.energy_price_base * 0.8
    
    def _compute_reward(self, state: HVACState, action: HVACAction) -> float:
        """Compute reward for current state and action."""
        # Comfort reward
        comfort_reward = 0
        for temp in state.zone_temperatures:
            if self.comfort_range[0] <= temp <= self.comfort_range[1]:
                comfort_reward += 10  # Reward for comfort
            else:
                violation = min(abs(temp - self.comfort_range[0]), abs(temp - self.comfort_range[1]))
                comfort_reward -= violation * 5  # Penalty for violation
        
        # Energy cost penalty
        energy_consumption = self._compute_energy_consumption(action)
        energy_cost = energy_consumption * state.energy_price / 1000  # Normalize
        energy_penalty = -energy_cost
        
        # Efficiency reward (reward for lower fan speeds when appropriate)
        efficiency_reward = 10 - np.sum(action.fan_speeds) * 2
        
        total_reward = comfort_reward + energy_penalty + efficiency_reward
        return total_reward
    
    def _compute_energy_consumption(self, action: HVACAction) -> float:
        """Compute energy consumption for action."""
        base_consumption = 50  # kW base load
        
        # HVAC consumption
        hvac_consumption = 0
        for i in range(self.num_zones):
            hvac_consumption += action.fan_speeds[i] * 10  # kW per zone
        
        return base_consumption + hvac_consumption
    
    def _count_comfort_violations(self, state: HVACState) -> int:
        """Count comfort violations in current state."""
        violations = 0
        for temp in state.zone_temperatures:
            if not (self.comfort_range[0] <= temp <= self.comfort_range[1]):
                violations += 1
        return violations


class QuantumQLearningAgent:
    """Quantum Q-Learning agent for HVAC control."""
    
    def __init__(self, config: QuantumRLConfig, num_actions: int = 64):
        self.config = config
        self.num_actions = num_actions
        
        # Quantum components
        self.state_encoder = QuantumStateEncoder(config.circuit_config.num_qubits)
        self.action_selector = QuantumActionSelector(num_actions, config.circuit_config)
        self.q_network = QuantumValueNetwork(config.circuit_config)
        self.target_network = QuantumValueNetwork(config.circuit_config)
        
        # Experience replay
        self.memory = deque(maxlen=config.memory_size)
        self.update_counter = 0
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'quantum_advantage': [],
            'comfort_violations': [],
            'energy_consumption': []
        }
        
        logger.info(f"Initialized Quantum Q-Learning Agent with {num_actions} actions")
    
    def select_action(self, state: HVACState, exploration_rate: float = None) -> HVACAction:
        """Select action using quantum policy."""
        if exploration_rate is None:
            exploration_rate = self.config.exploration_rate
        
        # Encode state to quantum representation
        quantum_state = self.state_encoder.encode_state(state)
        
        # Select action using quantum circuit
        action = self.action_selector.select_action(quantum_state, exploration_rate)
        
        return action
    
    def store_experience(self, state: HVACState, action: HVACAction, 
                        reward: float, next_state: HVACState, done: bool):
        """Store experience in replay buffer."""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def train(self) -> Dict[str, float]:
        """Train the quantum Q-network."""
        if len(self.memory) < self.config.batch_size:
            return {}
        
        # Sample batch from memory
        batch_indices = np.random.choice(len(self.memory), self.config.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        
        # Prepare training data
        states = [exp[0] for exp in batch]
        actions = [exp[1] for exp in batch]
        rewards = np.array([exp[2] for exp in batch])
        next_states = [exp[3] for exp in batch]
        dones = np.array([exp[4] for exp in batch])
        
        # Encode states to quantum representation
        quantum_states = self.state_encoder.encode_batch(states)
        quantum_next_states = self.state_encoder.encode_batch(next_states)
        
        # Compute current Q-values
        current_q_values = np.array([
            self.q_network.evaluate(q_state) for q_state in quantum_states
        ])
        
        # Compute target Q-values
        next_q_values = np.array([
            self.target_network.evaluate(q_state) for q_state in quantum_next_states
        ])
        
        target_q_values = rewards + self.config.discount_factor * next_q_values * (1 - dones)
        
        # Compute loss and gradients
        loss = np.mean((current_q_values - target_q_values) ** 2)
        
        # Compute gradients (simplified - would use automatic differentiation in practice)
        gradients = self._compute_gradients(quantum_states, current_q_values, target_q_values)
        
        # Update Q-network parameters
        self.q_network.update_parameters(gradients, self.config.learning_rate)
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.config.target_update_frequency == 0:
            self._update_target_network()
        
        return {
            'loss': loss,
            'q_values_mean': np.mean(current_q_values),
            'q_values_std': np.std(current_q_values)
        }
    
    def _compute_gradients(self, quantum_states: np.ndarray, 
                          current_q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
        """Compute gradients for quantum network parameters."""
        # Simplified gradient computation
        # In practice, would use parameter-shift rule or finite differences
        
        num_params = len(self.q_network.parameters)
        gradients = np.zeros(num_params)
        
        eps = 0.01  # Small perturbation for finite differences
        
        for i in range(num_params):
            # Perturb parameter
            self.q_network.parameters[i] += eps
            
            # Compute perturbed Q-values
            perturbed_q = np.array([
                self.q_network.evaluate(q_state) for q_state in quantum_states
            ])
            
            # Compute gradient
            loss_plus = np.mean((perturbed_q - target_q) ** 2)
            
            # Restore parameter and perturb in opposite direction
            self.q_network.parameters[i] -= 2 * eps
            
            perturbed_q = np.array([
                self.q_network.evaluate(q_state) for q_state in quantum_states
            ])
            
            loss_minus = np.mean((perturbed_q - target_q) ** 2)
            
            # Finite difference gradient
            gradients[i] = (loss_plus - loss_minus) / (2 * eps)
            
            # Restore parameter
            self.q_network.parameters[i] += eps
        
        return gradients
    
    def _update_target_network(self):
        """Update target network parameters."""
        # Copy parameters from main network to target network
        self.target_network.parameters = self.q_network.parameters.copy()
        
        logger.info("Updated target network parameters")
    
    def compute_quantum_advantage(self, classical_performance: float, 
                                quantum_performance: float) -> float:
        """Compute quantum advantage metric."""
        if classical_performance == 0:
            return 0.0
        
        advantage = (quantum_performance - classical_performance) / abs(classical_performance)
        return advantage
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training performance metrics."""
        return {
            'total_episodes': len(self.training_metrics['episode_rewards']),
            'average_reward': np.mean(self.training_metrics['episode_rewards'][-100:]) if self.training_metrics['episode_rewards'] else 0,
            'average_quantum_advantage': np.mean(self.training_metrics['quantum_advantage'][-100:]) if self.training_metrics['quantum_advantage'] else 0,
            'memory_usage': len(self.memory),
            'exploration_rate': self.config.exploration_rate,
            'parameters_norm': np.linalg.norm(self.q_network.parameters)
        }


async def train_quantum_hvac_agent(building_config: Dict[str, Any],
                                 episodes: int = 1000,
                                 save_frequency: int = 100) -> QuantumQLearningAgent:
    """Train quantum RL agent for HVAC control."""
    
    # Create environment and agent
    env = HVACQuantumEnvironment(building_config)
    
    config = QuantumRLConfig(
        algorithm=QuantumRLAlgorithm.QUANTUM_Q_LEARNING,
        circuit_config=QuantumCircuitConfig(num_qubits=8, circuit_depth=6),
        learning_rate=0.001,
        exploration_rate=0.1
    )
    
    agent = QuantumQLearningAgent(config)
    
    logger.info(f"Starting quantum RL training for {episodes} episodes")
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_violations = 0
        episode_energy = 0
        
        while True:
            # Select action
            action = agent.select_action(state, config.exploration_rate)
            
            # Take environment step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            training_metrics = agent.train()
            
            # Update metrics
            episode_reward += reward
            episode_violations += info['comfort_violations']
            episode_energy += info['energy_consumption']
            
            state = next_state
            
            if done:
                break
        
        # Store episode metrics
        agent.training_metrics['episode_rewards'].append(episode_reward)
        agent.training_metrics['comfort_violations'].append(episode_violations)
        agent.training_metrics['energy_consumption'].append(episode_energy)
        
        # Decay exploration
        config.exploration_rate *= config.exploration_decay
        config.exploration_rate = max(config.exploration_rate, 0.01)
        
        # Log progress
        if episode % 50 == 0:
            avg_reward = np.mean(agent.training_metrics['episode_rewards'][-50:])
            logger.info(f"Episode {episode}: Average Reward = {avg_reward:.2f}, "
                       f"Exploration Rate = {config.exploration_rate:.3f}")
    
    logger.info("Quantum RL training completed")
    return agent


# Convenience functions
def create_quantum_hvac_agent(building_config: Dict[str, Any]) -> QuantumQLearningAgent:
    """Create a quantum RL agent for HVAC control."""
    
    config = QuantumRLConfig(
        algorithm=QuantumRLAlgorithm.QUANTUM_Q_LEARNING,
        circuit_config=QuantumCircuitConfig(
            num_qubits=min(8, max(4, building_config.get('num_zones', 4))),
            circuit_depth=6
        ),
        learning_rate=0.001,
        exploration_rate=0.1
    )
    
    num_actions = 4 ** building_config.get('num_zones', 4)  # Exponential action space
    num_actions = min(num_actions, 256)  # Cap at reasonable size
    
    return QuantumQLearningAgent(config, num_actions)