"""
Unit tests for MPC to QUBO transformation.
"""

import pytest
import numpy as np
from typing import Dict, Tuple

from quantum_ctl.optimization.mpc_to_qubo import MPCToQUBO, VariableEncoding


class TestVariableEncoding:
    """Test VariableEncoding class."""
    
    def test_variable_encoding_creation(self):
        """Test variable encoding creation."""
        encoding = VariableEncoding(
            name="test_var",
            min_val=0.0,
            max_val=1.0,
            precision=4,
            offset=0
        )
        
        assert encoding.name == "test_var"
        assert encoding.min_val == 0.0
        assert encoding.max_val == 1.0
        assert encoding.precision == 4
        assert encoding.offset == 0
        assert encoding.num_bits == 4
    
    def test_encode_value(self):
        """Test continuous value encoding to binary."""
        encoding = VariableEncoding("test", 0.0, 1.0, 4, 0)
        
        # Test boundary values
        bits_0 = encoding.encode_value(0.0)
        bits_1 = encoding.encode_value(1.0)
        
        assert bits_0 == [0, 0, 0, 0]  # All zeros for minimum
        assert bits_1 == [1, 1, 1, 1]  # All ones for maximum
        
        # Test middle value
        bits_mid = encoding.encode_value(0.5)
        assert len(bits_mid) == 4
        assert all(bit in [0, 1] for bit in bits_mid)
    
    def test_decode_bits(self):
        """Test binary bits decoding to continuous value."""
        encoding = VariableEncoding("test", 0.0, 1.0, 4, 0)
        
        # Test boundary values
        val_0 = encoding.decode_bits([0, 0, 0, 0])
        val_1 = encoding.decode_bits([1, 1, 1, 1])
        
        assert abs(val_0 - 0.0) < 1e-10
        assert abs(val_1 - 1.0) < 1e-10
        
        # Test known bit pattern
        val_mid = encoding.decode_bits([0, 0, 0, 1])  # MSB only
        expected = encoding.min_val + (8/15) * (encoding.max_val - encoding.min_val)
        assert abs(val_mid - expected) < 1e-10
    
    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip accuracy."""
        encoding = VariableEncoding("test", -2.0, 5.0, 6, 0)
        
        test_values = [-2.0, -1.0, 0.0, 2.5, 5.0]
        
        for original_val in test_values:
            bits = encoding.encode_value(original_val)
            decoded_val = encoding.decode_bits(bits)
            
            # Should be within discretization error
            max_error = (encoding.max_val - encoding.min_val) / (2**encoding.precision)
            assert abs(decoded_val - original_val) <= max_error
    
    def test_value_clamping(self):
        """Test value clamping to bounds."""
        encoding = VariableEncoding("test", 0.0, 1.0, 4, 0)
        
        # Values outside bounds should be clamped
        bits_low = encoding.encode_value(-0.5)
        bits_high = encoding.encode_value(1.5)
        
        val_low = encoding.decode_bits(bits_low)
        val_high = encoding.decode_bits(bits_high)
        
        assert abs(val_low - 0.0) < 1e-10
        assert abs(val_high - 1.0) < 1e-10


class TestMPCToQUBO:
    """Test MPCToQUBO class."""
    
    def test_mpc_to_qubo_creation(self):
        """Test MPC to QUBO converter creation."""
        converter = MPCToQUBO(
            state_dim=5,
            control_dim=3,
            horizon=4,
            precision_bits=4
        )
        
        assert converter.state_dim == 5
        assert converter.control_dim == 3
        assert converter.horizon == 4
        assert converter.precision_bits == 4
    
    def test_create_variable_encodings(self):
        """Test variable encoding creation."""
        converter = MPCToQUBO(3, 2, 3, 4)
        
        # Mock MPC problem
        mcp_problem = {
            'constraints': {
                'control_limits': [
                    {'min': 0.0, 'max': 1.0},
                    {'min': 0.2, 'max': 0.8}
                ]
            }
        }
        
        converter._create_variable_encodings(mcp_problem)
        
        # Should have control variables for each time step
        control_encodings = [enc for enc in converter._variable_encodings 
                           if enc.name.startswith('u_')]
        
        expected_control_vars = converter.horizon * converter.control_dim
        # Note: actual count includes slack variables too
        assert len(control_encodings) == expected_control_vars
    
    def test_get_control_encoding(self):
        """Test control variable encoding retrieval."""
        converter = MPCToQUBO(2, 2, 2, 4)
        
        # Create encodings
        mcp_problem = {'constraints': {'control_limits': []}}
        converter._create_variable_encodings(mcp_problem)
        
        # Get specific control encoding
        encoding = converter._get_control_encoding(0, 1)  # Time 0, control 1
        
        assert encoding is not None
        assert encoding.name == "u_0_1"
    
    def test_to_qubo_basic(self):
        """Test basic QUBO matrix generation."""
        converter = MPCToQUBO(2, 1, 2, 3)  # Small problem
        
        # Simple MPC problem
        mcp_problem = {
            'state_dynamics': {
                'A': np.eye(2),
                'B': np.ones((2, 1))
            },
            'initial_state': np.array([1.0, 2.0]),
            'objectives': {
                'weights': {
                    'energy': 0.7,
                    'comfort': 0.3,
                    'carbon': 0.0
                }
            },
            'constraints': {
                'control_limits': [{'min': 0.0, 'max': 1.0}],
                'comfort_bounds': [{'temp_min': 20.0, 'temp_max': 24.0}],
                'power_limits': [{'heating_max': 10.0, 'cooling_max': -8.0}]
            }
        }
        
        Q = converter.to_qubo(mcp_problem)
        
        assert isinstance(Q, dict)
        assert len(Q) > 0
        
        # Check QUBO structure
        for (i, j), coeff in Q.items():
            assert isinstance(i, int)
            assert isinstance(j, int)
            assert isinstance(coeff, (int, float))
            assert i <= j  # Upper triangular form
    
    def test_decode_solution(self):
        """Test solution decoding from QUBO result."""
        converter = MPCToQUBO(2, 1, 2, 3)
        
        # Create variable encodings
        mcp_problem = {'constraints': {'control_limits': []}}
        converter._create_variable_encodings(mcp_problem)
        
        # Mock solution (all variables set to 1)
        solution = {}
        for encoding in converter._variable_encodings:
            if encoding.name.startswith('u_'):  # Control variables only
                for bit_i in range(encoding.precision):
                    solution[encoding.offset + bit_i] = 1
        
        # Decode solution
        control_schedule = converter.decode_solution(solution)
        
        expected_length = converter.horizon * converter.control_dim
        assert len(control_schedule) == expected_length
        
        # All control values should be near maximum (1.0)
        assert np.all(control_schedule >= 0.9)
        assert np.all(control_schedule <= 1.0)
    
    def test_get_problem_size(self):
        """Test problem size metrics."""
        converter = MPCToQUBO(5, 3, 4, 4)
        
        # Create encodings first
        mcp_problem = {'constraints': {'control_limits': []}}
        converter._create_variable_encodings(mcp_problem)
        
        size_info = converter.get_problem_size()
        
        assert 'total_qubits' in size_info
        assert 'control_variables' in size_info
        assert 'state_dimension' in size_info
        assert 'horizon_steps' in size_info
        
        assert size_info['control_variables'] == 4 * 3  # horizon * control_dim
        assert size_info['state_dimension'] == 5
        assert size_info['horizon_steps'] == 4
    
    def test_validate_solution(self):
        """Test solution validation."""
        converter = MPCToQUBO(2, 1, 2, 4)
        
        # Valid control schedule
        valid_schedule = np.array([0.5, 0.7])  # 2 time steps, 1 control each
        
        mcp_problem = {
            'constraints': {
                'control_limits': [{'min': 0.0, 'max': 1.0}]
            }
        }
        
        violations = converter.validate_solution(valid_schedule, mcp_problem)
        
        assert 'control_bounds' in violations
        assert 'comfort_violations' in violations
        assert 'energy_limit' in violations
        
        # Should have no violations for valid schedule
        assert len(violations['control_bounds']) == 0
    
    def test_validate_solution_violations(self):
        """Test solution validation with violations."""
        converter = MPCToQUBO(2, 1, 2, 4)
        
        # Invalid control schedule (values outside bounds)
        invalid_schedule = np.array([1.5, -0.2])  # Outside [0,1] bounds
        
        mcp_problem = {
            'constraints': {
                'control_limits': [{'min': 0.0, 'max': 1.0}]
            }
        }
        
        violations = converter.validate_solution(invalid_schedule, mcp_problem)
        
        # Should detect bound violations
        assert len(violations['control_bounds']) > 0
        
        # Check violation details
        violation = violations['control_bounds'][0]
        assert 'time_step' in violation
        assert 'control' in violation
        assert 'value' in violation
        assert 'bounds' in violation
    
    def test_energy_objective_terms(self):
        """Test energy objective QUBO terms."""
        converter = MPCToQUBO(2, 1, 2, 3)
        
        mcp_problem = {'constraints': {'control_limits': []}}
        converter._create_variable_encodings(mcp_problem)
        
        Q = {}
        converter._add_energy_objective(Q, mcp_problem, weight=1.0)
        
        # Should have added terms to Q
        assert len(Q) > 0
        
        # Energy objective should create positive diagonal terms
        for (i, j), coeff in Q.items():
            if i == j:  # Diagonal terms
                assert coeff >= 0  # Energy penalties should be positive
    
    def test_comfort_objective_terms(self):
        """Test comfort objective QUBO terms."""
        converter = MPCToQUBO(2, 1, 2, 3)
        
        mcp_problem = {
            'state_dynamics': {
                'A': np.eye(2),
                'B': np.ones((2, 1))
            },
            'initial_state': np.array([22.0, 21.0]),
            'constraints': {'control_limits': []}
        }
        converter._create_variable_encodings(mcp_problem)
        
        Q = {}
        converter._add_comfort_objective(Q, mcp_problem, weight=1.0)
        
        # Should have added comfort penalty terms
        assert len(Q) > 0
    
    def test_dynamics_constraints(self):
        """Test dynamics constraint penalties."""
        converter = MPCToQUBO(2, 1, 2, 3)
        
        mcp_problem = {
            'state_dynamics': {
                'A': np.eye(2),
                'B': np.ones((2, 1))
            },
            'constraints': {'control_limits': []}
        }
        converter._create_variable_encodings(mcp_problem)
        
        Q = {}
        converter._add_dynamics_constraints(Q, mcp_problem, penalty_weight=1000.0)
        
        # Should add smoothness constraints between time steps
        # Check for cross-time coupling terms
        cross_time_terms = [(i, j) for (i, j) in Q.keys() if i != j]
        assert len(cross_time_terms) > 0
    
    def test_large_problem_scaling(self):
        """Test scaling to larger problems."""
        # Larger problem
        converter = MPCToQUBO(10, 5, 8, 4)
        
        mcp_problem = {
            'state_dynamics': {
                'A': np.eye(10),
                'B': np.random.randn(10, 5)
            },
            'initial_state': np.random.randn(10),
            'objectives': {
                'weights': {'energy': 0.6, 'comfort': 0.4, 'carbon': 0.0}
            },
            'constraints': {
                'control_limits': [{'min': 0.0, 'max': 1.0}] * 5,
                'comfort_bounds': [{'temp_min': 20.0, 'temp_max': 24.0}] * 5,
                'power_limits': [{'heating_max': 10.0, 'cooling_max': -8.0}] * 5
            }
        }
        
        Q = converter.to_qubo(mcp_problem)
        
        # Should handle larger problems
        assert len(Q) > 0
        
        size_info = converter.get_problem_size()
        assert size_info['total_qubits'] > 100  # Should be substantial problem
        
        # Test solution decoding still works
        mock_solution = {i: 0 for i in range(size_info['total_qubits'])}
        control_schedule = converter.decode_solution(mock_solution)
        
        expected_length = converter.horizon * converter.control_dim
        assert len(control_schedule) == expected_length