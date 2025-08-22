"""
Classical fallback solver for quantum optimization problems.
Provides graceful degradation when quantum resources are unavailable.
"""

import numpy as np
import scipy.optimize
import time
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClassicalSolution:
    """Solution from classical optimization."""
    x: np.ndarray
    cost: float
    success: bool
    message: str
    iterations: int
    computation_time: float


class ClassicalFallbackSolver:
    """Classical optimization solver for HVAC control problems."""
    
    def __init__(self):
        self.solver_name = "classical_fallback"
        self.logger = logger
    
    def solve_qubo(self, Q: Dict[Tuple[int, int], float], **kwargs) -> ClassicalSolution:
        """
        Solve QUBO problem using classical optimization.
        
        Args:
            Q: QUBO matrix as dict of (i,j): coefficient
            **kwargs: Additional solver parameters
        
        Returns:
            ClassicalSolution with optimization results
        """
        start_time = time.time()
        
        try:
            # Convert QUBO dict to matrix
            if not Q:
                return ClassicalSolution(
                    x=np.array([]),
                    cost=0.0,
                    success=True,
                    message="Empty problem",
                    iterations=0,
                    computation_time=0.0
                )
            
            # Determine problem size
            max_var = max(max(i, j) for (i, j) in Q.keys())
            n_vars = max_var + 1
            
            # Build QUBO matrix
            Q_matrix = np.zeros((n_vars, n_vars))
            for (i, j), coeff in Q.items():
                if i == j:
                    Q_matrix[i, i] = coeff
                else:
                    Q_matrix[i, j] = coeff / 2
                    Q_matrix[j, i] = coeff / 2
            
            # Define objective function for binary variables
            def objective(x):
                x_binary = (x > 0.5).astype(int)
                return x_binary.T @ Q_matrix @ x_binary
            
            # Initial guess - random binary
            x0 = np.random.rand(n_vars)
            
            # Use basin hopping for global optimization
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "bounds": [(0, 1) for _ in range(n_vars)]
            }
            
            result = scipy.optimize.basinhopping(
                objective,
                x0,
                niter=kwargs.get('max_iter', 100),
                minimizer_kwargs=minimizer_kwargs,
                stepsize=0.5,
                seed=kwargs.get('seed', 42)
            )
            
            # Convert to binary solution
            x_binary = (result.x > 0.5).astype(int)
            final_cost = objective(x_binary)
            
            computation_time = time.time() - start_time
            
            return ClassicalSolution(
                x=x_binary,
                cost=final_cost,
                success=result.success,
                message=result.message if hasattr(result, 'message') else "Optimization complete",
                iterations=result.nfev,
                computation_time=computation_time
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"Classical optimization failed: {e}")
            
            return ClassicalSolution(
                x=np.zeros(n_vars if 'n_vars' in locals() else 1),
                cost=float('inf'),
                success=False,
                message=f"Optimization failed: {str(e)}",
                iterations=0,
                computation_time=computation_time
            )
    
    def solve_mpc_problem(self, 
                         A: np.ndarray,
                         B: np.ndarray, 
                         Q_cost: np.ndarray,
                         R_cost: np.ndarray,
                         x0: np.ndarray,
                         horizon: int,
                         constraints: Optional[Dict] = None) -> ClassicalSolution:
        """
        Solve MPC problem directly using classical methods.
        
        Args:
            A: State transition matrix
            B: Control input matrix  
            Q_cost: State cost matrix
            R_cost: Control cost matrix
            x0: Initial state
            horizon: Prediction horizon
            constraints: Optional constraints dict
        
        Returns:
            ClassicalSolution with optimal control sequence
        """
        start_time = time.time()
        
        try:
            n_states = A.shape[0]
            n_controls = B.shape[1]
            
            # Decision variables: [x1, u0, x2, u1, ..., xN, uN-1]
            n_vars = horizon * (n_states + n_controls)
            
            # Quadratic cost matrix
            P = np.zeros((n_vars, n_vars))
            q = np.zeros(n_vars)
            
            # Build cost function
            for k in range(horizon):
                x_idx = k * (n_states + n_controls)
                u_idx = x_idx + n_states
                
                # State cost Q
                if k < horizon:
                    P[x_idx:x_idx+n_states, x_idx:x_idx+n_states] = Q_cost
                
                # Control cost R  
                if k < horizon - 1:
                    P[u_idx:u_idx+n_controls, u_idx:u_idx+n_controls] = R_cost
            
            # Equality constraints for dynamics
            A_eq = []
            b_eq = []
            
            for k in range(horizon - 1):
                x_k_idx = k * (n_states + n_controls)
                u_k_idx = x_k_idx + n_states
                x_k1_idx = (k + 1) * (n_states + n_controls)
                
                # x_{k+1} = A*x_k + B*u_k
                constraint = np.zeros(n_vars)
                constraint[x_k1_idx:x_k1_idx+n_states] = np.eye(n_states)
                constraint[x_k_idx:x_k_idx+n_states] = -A
                constraint[u_k_idx:u_k_idx+n_controls] = -B
                
                A_eq.append(constraint)
                b_eq.append(np.zeros(n_states))
            
            # Initial state constraint
            initial_constraint = np.zeros(n_vars)
            initial_constraint[:n_states] = np.eye(n_states)
            A_eq.append(initial_constraint)
            b_eq.append(x0)
            
            if A_eq:
                A_eq = np.vstack(A_eq)
                b_eq = np.concatenate(b_eq)
            else:
                A_eq = None
                b_eq = None
            
            # Solve quadratic program
            result = scipy.optimize.minimize(
                fun=lambda x: 0.5 * x.T @ P @ x + q.T @ x,
                x0=np.zeros(n_vars),
                method='SLSQP',
                constraints={'type': 'eq', 'fun': lambda x: A_eq @ x - b_eq} if A_eq is not None else [],
                options={'maxiter': 1000}
            )
            
            computation_time = time.time() - start_time
            
            if result.success:
                # Extract control sequence
                u_sequence = []
                for k in range(horizon - 1):
                    u_idx = k * (n_states + n_controls) + n_states
                    u_k = result.x[u_idx:u_idx+n_controls]
                    u_sequence.append(u_k)
                
                return ClassicalSolution(
                    x=np.array(u_sequence),
                    cost=result.fun,
                    success=True,
                    message="MPC optimization successful",
                    iterations=result.nfev,
                    computation_time=computation_time
                )
            else:
                return ClassicalSolution(
                    x=np.zeros((horizon-1, n_controls)),
                    cost=float('inf'),
                    success=False,
                    message=f"MPC optimization failed: {result.message}",
                    iterations=result.nfev,
                    computation_time=computation_time
                )
                
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"MPC optimization failed: {e}")
            
            return ClassicalSolution(
                x=np.zeros((horizon-1, n_controls if 'n_controls' in locals() else 1)),
                cost=float('inf'),
                success=False,
                message=f"MPC optimization failed: {str(e)}",
                iterations=0,
                computation_time=computation_time
            )


# Make available for import
__all__ = ['ClassicalFallbackSolver', 'ClassicalSolution']