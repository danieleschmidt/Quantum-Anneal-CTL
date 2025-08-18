"""
Experimental benchmarking framework for quantum annealing research.

This module provides comprehensive benchmarking capabilities for comparing
quantum annealing approaches against classical methods, with statistical
validation and publication-ready analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

from ..optimization.quantum_solver import QuantumSolver, QuantumSolution
from ..optimization.mpc_to_qubo import MPCToQUBO
from .novel_qubo_formulations import NovelQUBOFormulator


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    method_name: str
    problem_instance: str
    problem_size: int
    solution_quality: float
    solve_time: float
    energy: float
    constraint_violations: int
    additional_metrics: Dict[str, Any]
    timestamp: str
    

@dataclass
class StatisticalSummary:
    """Statistical summary of benchmark results."""
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    confidence_interval_95: Tuple[float, float]
    sample_size: int


@dataclass
class ComparisonResult:
    """Result of statistical comparison between methods."""
    method_1: str
    method_2: str
    metric: str
    statistical_test: str
    p_value: float
    effect_size: float
    significance_level: float
    is_significant: bool
    interpretation: str


class ProblemInstanceGenerator:
    """Generate standardized problem instances for benchmarking."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger(__name__)
        
    def generate_hvac_instances(
        self,
        sizes: List[int] = [20, 50, 100, 200],
        instances_per_size: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """Generate HVAC optimization problem instances."""
        
        instances = {}
        
        for size in sizes:
            for instance_id in range(instances_per_size):
                instance_name = f"hvac_{size}zones_{instance_id:02d}"
                
                instance_data = {
                    'zones': size,
                    'horizon': 24,  # 24-hour optimization horizon
                    'control_interval': 15,  # 15-minute intervals
                    'thermal_mass': self.rng.uniform(800, 1200, size),  # kJ/K per zone
                    'heat_transfer_matrix': self._generate_heat_transfer_matrix(size),
                    'comfort_bounds': {
                        'lower': self.rng.uniform(20, 22, size),  # Celsius
                        'upper': self.rng.uniform(24, 26, size)
                    },
                    'power_limits': {
                        'min_power': np.zeros(size),
                        'max_power': self.rng.uniform(5, 15, size)  # kW per zone
                    },
                    'energy_prices': self._generate_energy_price_profile(),
                    'weather_profile': self._generate_weather_profile(),
                    'occupancy_schedule': self._generate_occupancy_schedule(size),
                    'carbon_intensity': self._generate_carbon_intensity_profile()
                }
                
                instances[instance_name] = instance_data
                
        self.logger.info(f"Generated {len(instances)} HVAC problem instances")
        return instances
        
    def generate_synthetic_qubo_instances(
        self,
        sizes: List[int] = [50, 100, 200, 500],
        densities: List[float] = [0.1, 0.3, 0.5],
        instances_per_config: int = 5
    ) -> Dict[str, Dict[Tuple[int, int], float]]:
        """Generate synthetic QUBO instances with controlled properties."""
        
        instances = {}
        
        for size in sizes:
            for density in densities:
                for instance_id in range(instances_per_config):
                    instance_name = f"synthetic_{size}vars_{density:.1f}density_{instance_id:02d}"
                    
                    # Generate QUBO with specified density
                    n_edges = int(density * size * (size - 1) / 2)
                    
                    Q = {}
                    
                    # Add diagonal terms (linear)
                    for i in range(size):
                        Q[(i, i)] = self.rng.uniform(-2, 2)
                        
                    # Add off-diagonal terms (quadratic)
                    edges = self.rng.choice(
                        [(i, j) for i in range(size) for j in range(i+1, size)],
                        size=n_edges,
                        replace=False
                    )
                    
                    for (i, j) in edges:
                        Q[(i, j)] = self.rng.uniform(-1, 1)
                        
                    instances[instance_name] = Q
                    
        self.logger.info(f"Generated {len(instances)} synthetic QUBO instances")
        return instances
        
    def _generate_heat_transfer_matrix(self, size: int) -> np.ndarray:
        """Generate realistic heat transfer coupling matrix."""
        
        # Start with diagonal dominance
        H = np.diag(self.rng.uniform(0.8, 1.2, size))
        
        # Add adjacent zone coupling
        for i in range(size - 1):
            coupling = self.rng.uniform(0.1, 0.3)
            H[i, i+1] = -coupling
            H[i+1, i] = -coupling
            
        # Add some random long-range coupling
        n_long_range = max(1, size // 10)
        for _ in range(n_long_range):
            i, j = self.rng.choice(size, 2, replace=False)
            coupling = self.rng.uniform(0.05, 0.15)
            H[i, j] = -coupling
            H[j, i] = -coupling
            
        return H
        
    def _generate_energy_price_profile(self, hours: int = 24) -> np.ndarray:
        """Generate realistic energy price profile."""
        
        base_price = 0.12  # $/kWh
        peak_multiplier = 2.0
        
        prices = np.full(hours, base_price)
        
        # Peak hours (6-9 AM, 5-8 PM)
        morning_peak = [6, 7, 8]
        evening_peak = [17, 18, 19]
        
        for hour in morning_peak + evening_peak:
            prices[hour] = base_price * peak_multiplier
            
        # Add some randomness
        prices += self.rng.normal(0, base_price * 0.1, hours)
        prices = np.maximum(prices, base_price * 0.5)  # Floor price
        
        return prices
        
    def _generate_weather_profile(self, hours: int = 24) -> Dict[str, np.ndarray]:
        """Generate realistic weather profile."""
        
        # Temperature profile (sinusoidal with noise)
        base_temp = 20  # Celsius
        daily_variation = 8
        
        hours_array = np.arange(hours)
        temperature = base_temp + daily_variation * np.sin(2 * np.pi * (hours_array - 6) / 24)
        temperature += self.rng.normal(0, 1, hours)
        
        # Humidity (inversely related to temperature)
        humidity = 70 - 0.5 * (temperature - base_temp) + self.rng.normal(0, 5, hours)
        humidity = np.clip(humidity, 30, 90)
        
        # Solar radiation (daytime peak)
        solar = np.maximum(0, 800 * np.sin(np.pi * hours_array / 12)) * (hours_array >= 6) * (hours_array <= 18)
        solar += self.rng.normal(0, 50, hours)
        solar = np.maximum(solar, 0)
        
        return {
            'temperature': temperature,
            'humidity': humidity,
            'solar_radiation': solar
        }
        
    def _generate_occupancy_schedule(self, zones: int, hours: int = 24) -> np.ndarray:
        """Generate realistic occupancy schedule."""
        
        occupancy = np.zeros((zones, hours))
        
        for zone in range(zones):
            # Different occupancy patterns for different zone types
            if zone < zones // 2:  # Office zones
                # High occupancy during work hours
                work_hours = list(range(8, 17))
                for hour in work_hours:
                    occupancy[zone, hour] = self.rng.uniform(0.6, 1.0)
                    
            else:  # Common area zones
                # More distributed occupancy
                for hour in range(6, 22):
                    occupancy[zone, hour] = self.rng.uniform(0.2, 0.8)
                    
        return occupancy
        
    def _generate_carbon_intensity_profile(self, hours: int = 24) -> np.ndarray:
        """Generate carbon intensity profile."""
        
        # Higher intensity during peak hours
        base_intensity = 0.5  # kg CO2/kWh
        
        intensity = np.full(hours, base_intensity)
        
        # Peak carbon hours (typically evening when coal/gas plants ramp up)
        peak_hours = [18, 19, 20, 21]
        for hour in peak_hours:
            intensity[hour] = base_intensity * 1.5
            
        # Low carbon hours (night when renewables dominate)
        low_hours = [2, 3, 4, 5]
        for hour in low_hours:
            intensity[hour] = base_intensity * 0.3
            
        return intensity


class ComparativeBenchmarkSuite:
    """
    Comprehensive benchmarking suite for quantum vs classical comparison.
    
    Compares quantum annealing methods against classical optimizers with
    statistical validation and publication-ready analysis.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize problem generator
        self.problem_generator = ProblemInstanceGenerator()
        
        # Results storage
        self.all_results = []
        self.comparison_results = []
        
        # Solvers to benchmark
        self.quantum_solvers = {}
        self.classical_solvers = {}
        
    def add_quantum_solver(self, name: str, solver: QuantumSolver) -> None:
        """Add quantum solver to benchmark."""
        self.quantum_solvers[name] = solver
        self.logger.info(f"Added quantum solver: {name}")
        
    def add_classical_solver(self, name: str, solver_func: Callable) -> None:
        """Add classical solver to benchmark."""
        self.classical_solvers[name] = solver_func
        self.logger.info(f"Added classical solver: {name}")
        
    async def run_comprehensive_benchmark(
        self,
        problem_sizes: List[int] = [20, 50, 100, 200],
        runs_per_instance: int = 5,
        timeout_seconds: float = 300.0
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across all solvers and problem instances.
        
        Args:
            problem_sizes: List of problem sizes to test
            runs_per_instance: Number of runs per problem instance
            timeout_seconds: Timeout for each solver run
            
        Returns:
            Comprehensive benchmark results
        """
        
        self.logger.info(f"Starting comprehensive benchmark with {len(self.quantum_solvers)} quantum and {len(self.classical_solvers)} classical solvers")
        
        # Generate problem instances
        hvac_instances = self.problem_generator.generate_hvac_instances(
            sizes=problem_sizes,
            instances_per_size=5
        )
        
        synthetic_instances = self.problem_generator.generate_synthetic_qubo_instances(
            sizes=problem_sizes[:3],  # Smaller sizes for synthetic
            instances_per_config=3
        )
        
        # Run benchmarks
        await self._run_hvac_benchmarks(hvac_instances, runs_per_instance, timeout_seconds)
        await self._run_synthetic_benchmarks(synthetic_instances, runs_per_instance, timeout_seconds)
        
        # Perform statistical analysis
        statistical_analysis = self._perform_statistical_analysis()
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(statistical_analysis)
        
        # Save results
        self._save_results(report)
        
        return report
        
    async def _run_hvac_benchmarks(
        self,
        instances: Dict[str, Dict[str, Any]],
        runs_per_instance: int,
        timeout_seconds: float
    ) -> None:
        """Run benchmarks on HVAC problem instances."""
        
        self.logger.info(f"Running HVAC benchmarks on {len(instances)} instances")
        
        tasks = []
        
        for instance_name, instance_data in instances.items():
            # Convert HVAC instance to MPC problem
            mpc_problem = self._hvac_to_mpc_problem(instance_data)
            
            # Convert MPC to QUBO
            formulator = MPCToQUBO(
                state_dim=instance_data['zones'],
                control_dim=instance_data['zones'],
                horizon=instance_data['horizon']
            )
            
            qubo = formulator.to_qubo(mpc_problem)
            
            # Benchmark all solvers on this instance
            for run in range(runs_per_instance):
                # Quantum solvers
                for solver_name, solver in self.quantum_solvers.items():
                    task = self._benchmark_single_run(
                        f"{solver_name}_quantum",
                        solver.solve,
                        qubo,
                        instance_name,
                        instance_data['zones'],
                        timeout_seconds
                    )
                    tasks.append(task)
                    
                # Classical solvers
                for solver_name, solver_func in self.classical_solvers.items():
                    task = self._benchmark_single_run(
                        f"{solver_name}_classical",
                        solver_func,
                        qubo,
                        instance_name,
                        instance_data['zones'],
                        timeout_seconds
                    )
                    tasks.append(task)
                    
        # Execute all benchmark runs
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, BenchmarkResult):
                self.all_results.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Benchmark run failed: {result}")
                
    async def _run_synthetic_benchmarks(
        self,
        instances: Dict[str, Dict[Tuple[int, int], float]],
        runs_per_instance: int,
        timeout_seconds: float
    ) -> None:
        """Run benchmarks on synthetic QUBO instances."""
        
        self.logger.info(f"Running synthetic benchmarks on {len(instances)} instances")
        
        tasks = []
        
        for instance_name, qubo in instances.items():
            problem_size = len(set(i for (i, j) in qubo.keys()) | set(j for (i, j) in qubo.keys()))
            
            for run in range(runs_per_instance):
                # Quantum solvers
                for solver_name, solver in self.quantum_solvers.items():
                    task = self._benchmark_single_run(
                        f"{solver_name}_quantum",
                        solver.solve,
                        qubo,
                        instance_name,
                        problem_size,
                        timeout_seconds
                    )
                    tasks.append(task)
                    
                # Classical solvers  
                for solver_name, solver_func in self.classical_solvers.items():
                    task = self._benchmark_single_run(
                        f"{solver_name}_classical",
                        solver_func,
                        qubo,
                        instance_name,
                        problem_size,
                        timeout_seconds
                    )
                    tasks.append(task)
                    
        # Execute all benchmark runs
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, BenchmarkResult):
                self.all_results.append(result)
                
    async def _benchmark_single_run(
        self,
        method_name: str,
        solver_func: Callable,
        qubo: Dict[Tuple[int, int], float],
        instance_name: str,
        problem_size: int,
        timeout_seconds: float
    ) -> BenchmarkResult:
        """Run single benchmark and collect metrics."""
        
        start_time = time.time()
        
        try:
            # Run solver with timeout
            if asyncio.iscoroutinefunction(solver_func):
                solution = await asyncio.wait_for(
                    solver_func(qubo),
                    timeout=timeout_seconds
                )
            else:
                # Run synchronous solver in thread pool
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(solver_func, qubo)
                    solution = await asyncio.wait_for(
                        asyncio.wrap_future(future),
                        timeout=timeout_seconds
                    )
                    
            solve_time = time.time() - start_time
            
            # Extract metrics from solution
            if isinstance(solution, QuantumSolution):
                energy = solution.energy
                additional_metrics = {
                    'chain_break_fraction': solution.chain_break_fraction,
                    'num_occurrences': solution.num_occurrences,
                    'solver_type': 'quantum'
                }
            else:
                # Classical solution
                energy = self._evaluate_qubo_energy(solution, qubo)
                additional_metrics = {
                    'solver_type': 'classical'
                }
                
            # Calculate constraint violations (simplified)
            violations = self._count_constraint_violations(solution, qubo)
            
            # Solution quality (normalized energy)
            quality = self._calculate_solution_quality(energy, problem_size)
            
            return BenchmarkResult(
                method_name=method_name,
                problem_instance=instance_name,
                problem_size=problem_size,
                solution_quality=quality,
                solve_time=solve_time,
                energy=energy,
                constraint_violations=violations,
                additional_metrics=additional_metrics,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except asyncio.TimeoutError:
            solve_time = timeout_seconds
            return BenchmarkResult(
                method_name=method_name,
                problem_instance=instance_name,
                problem_size=problem_size,
                solution_quality=0.0,  # Failed
                solve_time=solve_time,
                energy=float('inf'),
                constraint_violations=9999,
                additional_metrics={'status': 'timeout', 'solver_type': 'unknown'},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except Exception as e:
            solve_time = time.time() - start_time
            return BenchmarkResult(
                method_name=method_name,
                problem_instance=instance_name,
                problem_size=problem_size,
                solution_quality=0.0,  # Failed
                solve_time=solve_time,
                energy=float('inf'),
                constraint_violations=9999,
                additional_metrics={'status': 'error', 'error': str(e), 'solver_type': 'unknown'},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis of benchmark results."""
        
        self.logger.info("Performing statistical analysis")
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame([asdict(result) for result in self.all_results])
        
        analysis = {
            'summary_statistics': {},
            'comparative_analysis': [],
            'scalability_analysis': {},
            'significance_tests': []
        }
        
        # Summary statistics by method
        for method in df['method_name'].unique():
            method_data = df[df['method_name'] == method]
            
            analysis['summary_statistics'][method] = {
                'solution_quality': self._calculate_statistical_summary(method_data['solution_quality']),
                'solve_time': self._calculate_statistical_summary(method_data['solve_time']),
                'energy': self._calculate_statistical_summary(method_data['energy']),
                'constraint_violations': self._calculate_statistical_summary(method_data['constraint_violations']),
                'success_rate': (method_data['solution_quality'] > 0).mean()
            }
            
        # Comparative analysis between quantum and classical
        quantum_methods = [m for m in df['method_name'].unique() if 'quantum' in m]
        classical_methods = [m for m in df['method_name'].unique() if 'classical' in m]
        
        for q_method in quantum_methods:
            for c_method in classical_methods:
                comparison = self._compare_methods(df, q_method, c_method)
                analysis['comparative_analysis'].append(comparison)
                
        # Scalability analysis
        analysis['scalability_analysis'] = self._analyze_scalability(df)
        
        # Statistical significance tests
        analysis['significance_tests'] = self._perform_significance_tests(df)
        
        return analysis
        
    def _calculate_statistical_summary(self, data: pd.Series) -> StatisticalSummary:
        """Calculate statistical summary for a data series."""
        
        finite_data = data[np.isfinite(data)]
        
        if len(finite_data) == 0:
            return StatisticalSummary(0, 0, 0, 0, 0, (0, 0), 0)
            
        mean_val = finite_data.mean()
        std_val = finite_data.std()
        median_val = finite_data.median()
        min_val = finite_data.min()
        max_val = finite_data.max()
        
        # 95% confidence interval
        ci = stats.t.interval(0.95, len(finite_data)-1, loc=mean_val, scale=stats.sem(finite_data))
        
        return StatisticalSummary(
            mean=mean_val,
            std=std_val,
            median=median_val,
            min_val=min_val,
            max_val=max_val,
            confidence_interval_95=ci,
            sample_size=len(finite_data)
        )
        
    def _compare_methods(self, df: pd.DataFrame, method1: str, method2: str) -> Dict[str, Any]:
        """Compare two methods statistically."""
        
        m1_data = df[df['method_name'] == method1]
        m2_data = df[df['method_name'] == method2]
        
        # Compare on common instances
        common_instances = set(m1_data['problem_instance']) & set(m2_data['problem_instance'])
        
        m1_common = m1_data[m1_data['problem_instance'].isin(common_instances)]
        m2_common = m2_data[m2_data['problem_instance'].isin(common_instances)]
        
        comparison = {
            'method_1': method1,
            'method_2': method2,
            'common_instances': len(common_instances),
            'metrics': {}
        }
        
        # Compare each metric
        for metric in ['solution_quality', 'solve_time', 'energy']:
            if len(m1_common) > 0 and len(m2_common) > 0:
                m1_values = m1_common[metric]
                m2_values = m2_common[metric]
                
                # Remove infinite values
                m1_finite = m1_values[np.isfinite(m1_values)]
                m2_finite = m2_values[np.isfinite(m2_values)]
                
                if len(m1_finite) > 0 and len(m2_finite) > 0:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(m1_finite, m2_finite)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(m1_finite)-1)*m1_finite.var() + (len(m2_finite)-1)*m2_finite.var()) / (len(m1_finite)+len(m2_finite)-2))
                    cohens_d = (m1_finite.mean() - m2_finite.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    comparison['metrics'][metric] = {
                        'method_1_mean': m1_finite.mean(),
                        'method_2_mean': m2_finite.mean(),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'effect_size': cohens_d,
                        'is_significant': p_value < 0.05,
                        'winner': method1 if m1_finite.mean() > m2_finite.mean() else method2
                    }
                    
        return comparison
        
    def _save_results(self, report: Dict[str, Any]) -> None:
        """Save benchmark results and analysis."""
        
        # Save detailed results as JSON
        with open(self.output_dir / "benchmark_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Save results as CSV for further analysis
        df = pd.DataFrame([asdict(result) for result in self.all_results])
        df.to_csv(self.output_dir / "benchmark_results.csv", index=False)
        
        # Generate visualizations
        self._generate_visualizations(df)
        
        self.logger.info(f"Benchmark results saved to {self.output_dir}")
        
    def _generate_visualizations(self, df: pd.DataFrame) -> None:
        """Generate visualization plots for benchmark results."""
        
        plt.style.use('seaborn-v0_8')
        
        # Solution quality comparison
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='method_name', y='solution_quality')
        plt.xticks(rotation=45)
        plt.title('Solution Quality Comparison')
        plt.tight_layout()
        plt.savefig(self.output_dir / "solution_quality_comparison.png", dpi=300)
        plt.close()
        
        # Solve time comparison
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df[df['solve_time'] < 1000], x='method_name', y='solve_time')
        plt.xticks(rotation=45)
        plt.title('Solve Time Comparison (< 1000s)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(self.output_dir / "solve_time_comparison.png", dpi=300)
        plt.close()
        
        # Scalability analysis
        plt.figure(figsize=(12, 8))
        for method in df['method_name'].unique():
            method_data = df[df['method_name'] == method]
            avg_by_size = method_data.groupby('problem_size')['solve_time'].mean()
            plt.plot(avg_by_size.index, avg_by_size.values, marker='o', label=method)
            
        plt.xlabel('Problem Size')
        plt.ylabel('Average Solve Time (s)')
        plt.title('Scalability Analysis')
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(self.output_dir / "scalability_analysis.png", dpi=300)
        plt.close()
        
        self.logger.info("Visualizations saved")


class StatisticalValidator:
    """
    Statistical validation framework for quantum annealing research.
    
    Provides rigorous statistical testing and validation for research claims
    about quantum advantage and algorithmic improvements.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
        
    def validate_quantum_advantage(
        self,
        quantum_results: List[float],
        classical_results: List[float],
        metric_name: str = "solution_quality"
    ) -> Dict[str, Any]:
        """
        Statistically validate claimed quantum advantage.
        
        Args:
            quantum_results: Performance metric values for quantum solver
            classical_results: Performance metric values for classical solver
            metric_name: Name of the metric being compared
            
        Returns:
            Validation results with statistical tests
        """
        
        validation = {
            'metric': metric_name,
            'quantum_stats': self._calculate_descriptive_stats(quantum_results),
            'classical_stats': self._calculate_descriptive_stats(classical_results),
            'tests': {},
            'conclusion': {}
        }
        
        # Normality tests
        quantum_normal = self._test_normality(quantum_results)
        classical_normal = self._test_normality(classical_results)
        
        validation['tests']['normality'] = {
            'quantum_is_normal': quantum_normal,
            'classical_is_normal': classical_normal
        }
        
        # Choose appropriate statistical test
        if quantum_normal and classical_normal:
            # Use t-test for normal data
            test_result = self._perform_t_test(quantum_results, classical_results)
            validation['tests']['main_test'] = test_result
        else:
            # Use non-parametric test for non-normal data
            test_result = self._perform_mann_whitney_test(quantum_results, classical_results)
            validation['tests']['main_test'] = test_result
            
        # Effect size calculation
        effect_size = self._calculate_effect_size(quantum_results, classical_results)
        validation['tests']['effect_size'] = effect_size
        
        # Power analysis
        power_analysis = self._perform_power_analysis(quantum_results, classical_results)
        validation['tests']['power_analysis'] = power_analysis
        
        # Statistical conclusion
        validation['conclusion'] = self._draw_statistical_conclusion(
            test_result, effect_size, power_analysis
        )
        
        return validation
        
    def validate_algorithm_improvement(
        self,
        baseline_results: List[float],
        improved_results: List[float],
        improvement_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Validate claimed algorithmic improvement with proper statistical testing."""
        
        # Calculate improvement ratio
        baseline_mean = np.mean(baseline_results)
        improved_mean = np.mean(improved_results)
        improvement_ratio = (improved_mean - baseline_mean) / baseline_mean
        
        validation = {
            'baseline_mean': baseline_mean,
            'improved_mean': improved_mean,
            'improvement_ratio': improvement_ratio,
            'improvement_threshold': improvement_threshold,
            'tests': {},
            'is_significant_improvement': False
        }
        
        # Statistical significance test
        if self._test_normality(baseline_results) and self._test_normality(improved_results):
            test_result = self._perform_paired_t_test(baseline_results, improved_results)
        else:
            test_result = self._perform_wilcoxon_test(baseline_results, improved_results)
            
        validation['tests']['significance_test'] = test_result
        
        # Practical significance (improvement threshold)
        practical_significant = improvement_ratio > improvement_threshold
        statistical_significant = test_result['p_value'] < self.significance_level
        
        validation['is_significant_improvement'] = (
            statistical_significant and practical_significant
        )
        
        # Confidence interval for improvement
        improvement_ci = self._calculate_improvement_confidence_interval(
            baseline_results, improved_results
        )
        validation['improvement_confidence_interval'] = improvement_ci
        
        return validation
        
    def _calculate_descriptive_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics."""
        
        finite_data = [x for x in data if np.isfinite(x)]
        
        if not finite_data:
            return {'count': 0, 'mean': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0}
            
        return {
            'count': len(finite_data),
            'mean': np.mean(finite_data),
            'std': np.std(finite_data, ddof=1),
            'median': np.median(finite_data),
            'min': np.min(finite_data),
            'max': np.max(finite_data)
        }
        
    def _test_normality(self, data: List[float]) -> bool:
        """Test if data follows normal distribution."""
        
        finite_data = [x for x in data if np.isfinite(x)]
        
        if len(finite_data) < 3:
            return False
            
        # Shapiro-Wilk test for normality
        _, p_value = stats.shapiro(finite_data)
        
        return p_value > self.significance_level
        
    def _perform_t_test(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Perform independent t-test."""
        
        finite_g1 = [x for x in group1 if np.isfinite(x)]
        finite_g2 = [x for x in group2 if np.isfinite(x)]
        
        t_stat, p_value = stats.ttest_ind(finite_g1, finite_g2)
        
        return {
            'test_name': 'independent_t_test',
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_of_freedom': len(finite_g1) + len(finite_g2) - 2,
            'is_significant': p_value < self.significance_level
        }
        
    def _perform_mann_whitney_test(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Perform Mann-Whitney U test (non-parametric)."""
        
        finite_g1 = [x for x in group1 if np.isfinite(x)]
        finite_g2 = [x for x in group2 if np.isfinite(x)]
        
        u_stat, p_value = stats.mannwhitneyu(finite_g1, finite_g2, alternative='two-sided')
        
        return {
            'test_name': 'mann_whitney_u_test',
            'u_statistic': u_stat,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level
        }
        
    def _calculate_effect_size(self, group1: List[float], group2: List[float]) -> Dict[str, float]:
        """Calculate effect size (Cohen's d)."""
        
        finite_g1 = np.array([x for x in group1 if np.isfinite(x)])
        finite_g2 = np.array([x for x in group2 if np.isfinite(x)])
        
        if len(finite_g1) == 0 or len(finite_g2) == 0:
            return {'cohens_d': 0.0, 'interpretation': 'no_data'}
            
        # Calculate Cohen's d
        pooled_std = np.sqrt(((len(finite_g1) - 1) * finite_g1.var() + 
                             (len(finite_g2) - 1) * finite_g2.var()) / 
                            (len(finite_g1) + len(finite_g2) - 2))
                            
        if pooled_std == 0:
            cohens_d = 0.0
        else:
            cohens_d = (finite_g1.mean() - finite_g2.mean()) / pooled_std
            
        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = 'negligible'
        elif abs_d < 0.5:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'
            
        return {
            'cohens_d': cohens_d,
            'absolute_effect_size': abs_d,
            'interpretation': interpretation
        }
        
    def _perform_power_analysis(self, group1: List[float], group2: List[float]) -> Dict[str, float]:
        """Perform statistical power analysis."""
        
        finite_g1 = np.array([x for x in group1 if np.isfinite(x)])
        finite_g2 = np.array([x for x in group2 if np.isfinite(x)])
        
        if len(finite_g1) == 0 or len(finite_g2) == 0:
            return {'power': 0.0, 'interpretation': 'insufficient_data'}
            
        # Calculate effect size
        effect_size = self._calculate_effect_size(group1, group2)['cohens_d']
        
        # Estimate power using sample size and effect size
        n1, n2 = len(finite_g1), len(finite_g2)
        
        # Simplified power calculation (for more precise calculation, use statsmodels.stats.power)
        # This is an approximation
        ncp = abs(effect_size) * np.sqrt((n1 * n2) / (n1 + n2))  # Non-centrality parameter
        
        # Power is approximately the probability that |t| > t_critical given non-centrality
        t_critical = stats.t.ppf(1 - self.significance_level/2, n1 + n2 - 2)
        power = 1 - stats.nct.cdf(t_critical, n1 + n2 - 2, ncp) + stats.nct.cdf(-t_critical, n1 + n2 - 2, ncp)
        
        # Interpret power
        if power < 0.6:
            interpretation = 'low_power'
        elif power < 0.8:
            interpretation = 'moderate_power'
        else:
            interpretation = 'high_power'
            
        return {
            'power': power,
            'interpretation': interpretation,
            'recommended_min_power': 0.8
        }
        
    def _draw_statistical_conclusion(
        self,
        test_result: Dict[str, Any],
        effect_size: Dict[str, float],
        power_analysis: Dict[str, float]
    ) -> Dict[str, Any]:
        """Draw overall statistical conclusion."""
        
        is_significant = test_result['is_significant']
        effect_interpretation = effect_size['interpretation']
        power_interpretation = power_analysis['interpretation']
        
        # Overall conclusion
        if is_significant and effect_interpretation in ['medium', 'large'] and power_interpretation == 'high_power':
            conclusion = 'strong_evidence'
            interpretation = "Strong statistical evidence supports the claimed improvement"
        elif is_significant and effect_interpretation == 'small':
            conclusion = 'weak_evidence'
            interpretation = "Statistically significant but small effect size"
        elif not is_significant and power_interpretation == 'low_power':
            conclusion = 'insufficient_power'
            interpretation = "Insufficient statistical power to detect differences"
        elif not is_significant and power_interpretation in ['moderate_power', 'high_power']:
            conclusion = 'no_evidence'
            interpretation = "No statistical evidence of improvement"
        else:
            conclusion = 'inconclusive'
            interpretation = "Results are inconclusive"
            
        return {
            'conclusion': conclusion,
            'interpretation': interpretation,
            'is_statistically_significant': is_significant,
            'effect_size_category': effect_interpretation,
            'power_category': power_interpretation,
            'p_value': test_result['p_value'],
            'effect_size': effect_size['cohens_d'],
            'power': power_analysis['power']
        }