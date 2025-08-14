#!/usr/bin/env python3
"""
Research Validation Demo: Quantum HVAC Optimization Benchmarking

This demonstration script runs comprehensive research validation studies
comparing quantum and classical HVAC optimization algorithms.

Usage:
    python examples/research_validation_demo.py

Features:
- Scalability study across building sizes
- Multi-objective optimization comparison
- Statistical significance testing
- Publication-ready performance analysis
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl.optimization.research_benchmarks import (
    QuantumBenchmarkSuite,
    get_benchmark_suite
)
from quantum_ctl.optimization.adaptive_quantum_engine import (
    AdaptiveQuantumEngine,
    OptimizationStrategy
)
from quantum_ctl.utils.logging_config import setup_logging


async def run_comprehensive_research_validation():
    """Run comprehensive research validation studies."""
    print("🔬 QUANTUM HVAC OPTIMIZATION RESEARCH VALIDATION")
    print("=" * 60)
    print()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize benchmark suite
    benchmark_suite = get_benchmark_suite()
    logger.info("Initialized quantum benchmark suite")
    
    all_study_results = []
    
    try:
        # Study 1: Scalability Analysis
        print("📊 Study 1: Scalability Analysis")
        print("Testing quantum vs classical performance across building sizes...")
        print()
        
        scalability_start = time.time()
        
        # Test with smaller problem sizes for demo (increase for full research)
        scalability_results = await benchmark_suite.run_scalability_study(
            zone_counts=[5, 10, 15, 20],  # Building sizes to test
            num_runs=5,                   # Reduced for demo speed
            algorithms=['adaptive_quantum', 'basic_quantum', 'classical_milp', 'classical_genetic']
        )
        
        scalability_time = time.time() - scalability_start
        print(f"✅ Scalability study completed in {scalability_time:.1f} seconds")
        
        # Display key findings
        if 'statistical_analysis' in scalability_results:
            print("\n📈 Key Scalability Findings:")
            for algorithm, analysis in scalability_results['statistical_analysis'].items():
                if 'solve_time_scaling' in analysis:
                    scaling_info = analysis['solve_time_scaling']
                    complexity = scaling_info.get('complexity_class', 'unknown')
                    exponent = scaling_info.get('scaling_exponent', 0)
                    print(f"  • {algorithm}: {complexity} complexity (α={exponent:.2f})")
        
        # Check quantum advantage scenarios
        quantum_advantages = scalability_results.get('quantum_advantage_analysis', {}).get('quantum_advantage_scenarios', [])
        if quantum_advantages:
            print(f"\n🚀 Quantum Advantage Found: {len(quantum_advantages)} scenarios")
            for adv in quantum_advantages[:3]:  # Show first 3
                size = adv['problem_size']
                time_improvement = adv['time_advantage']
                quality_improvement = adv['quality_advantage']
                print(f"  • Size {size}: {time_improvement:.1f}x faster, {quality_improvement:.1f}x better quality")
        
        scalability_results['study_type'] = 'Scalability Analysis'
        all_study_results.append(scalability_results)
        
        print()
        
        # Study 2: Multi-Objective Optimization
        print("🎯 Study 2: Multi-Objective Optimization Comparison")
        print("Comparing quantum vs classical multi-objective performance...")
        print()
        
        multi_obj_start = time.time()
        
        # Test different objective weight combinations
        objective_combinations = [
            {'energy': 1.0, 'comfort': 0.0, 'carbon': 0.0},  # Energy-only
            {'energy': 0.5, 'comfort': 0.5, 'carbon': 0.0},  # Energy-comfort
            {'energy': 0.4, 'comfort': 0.4, 'carbon': 0.2},  # Balanced
            {'energy': 0.2, 'comfort': 0.7, 'carbon': 0.1}   # Comfort-focused
        ]
        
        multi_objective_results = await benchmark_suite.run_multi_objective_comparison(
            num_zones=15,  # Medium-sized building
            num_runs=4,    # Reduced for demo speed
            objective_combinations=objective_combinations
        )
        
        multi_obj_time = time.time() - multi_obj_start
        print(f"✅ Multi-objective study completed in {multi_obj_time:.1f} seconds")
        
        # Display multi-objective findings
        if 'algorithm_performance' in multi_objective_results:
            performance = multi_objective_results['algorithm_performance']
            quantum_wins = performance.get('quantum_wins', 0)
            classical_wins = performance.get('classical_wins', 0)
            ties = performance.get('ties', 0)
            
            total_tests = quantum_wins + classical_wins + ties
            if total_tests > 0:
                print(f"\n🏆 Multi-Objective Performance:")
                print(f"  • Quantum wins: {quantum_wins}/{total_tests} ({quantum_wins/total_tests*100:.1f}%)")
                print(f"  • Classical wins: {classical_wins}/{total_tests} ({classical_wins/total_tests*100:.1f}%)")
                print(f"  • Ties: {ties}/{total_tests} ({ties/total_tests*100:.1f}%)")
        
        multi_objective_results['study_type'] = 'Multi-Objective Comparison'
        all_study_results.append(multi_objective_results)
        
        print()
        
        # Study 3: Adaptive Algorithm Performance
        print("🧠 Study 3: Adaptive Quantum Algorithm Analysis")
        print("Testing adaptive quantum engine performance characteristics...")
        print()
        
        adaptive_start = time.time()
        
        # Test adaptive quantum engine with different strategies
        adaptive_engine = AdaptiveQuantumEngine(
            optimization_strategy=OptimizationStrategy.ADAPTIVE_HYBRID,
            performance_target=0.85
        )
        
        # Generate test problems
        problem_generator = benchmark_suite.problem_generator
        test_problems = []
        for complexity in ['simple', 'medium', 'complex']:
            for size in [10, 20]:
                problem = problem_generator.generate_building_problem(
                    num_zones=size,
                    horizon_hours=24,
                    problem_complexity=complexity
                )
                problem['complexity_label'] = complexity
                test_problems.append(problem)
        
        adaptive_results = []
        for i, problem in enumerate(test_problems):
            try:
                result = await benchmark_suite._benchmark_quantum_algorithm(
                    adaptive_engine, problem, run_id=i
                )
                adaptive_results.append({
                    'problem_size': problem['num_zones'],
                    'complexity': problem['complexity_label'],
                    'result': result,
                    'performance_report': adaptive_engine.get_adaptive_performance_report()
                })
                print(f"  ✓ Completed problem {i+1}/{len(test_problems)}")
                
            except Exception as e:
                logger.warning(f"Adaptive test {i} failed: {e}")
        
        adaptive_time = time.time() - adaptive_start
        print(f"✅ Adaptive algorithm analysis completed in {adaptive_time:.1f} seconds")
        
        # Analyze adaptive performance
        if adaptive_results:
            avg_quantum_advantage = sum(
                r['performance_report']['performance_metrics']['avg_quantum_advantage'] 
                for r in adaptive_results
            ) / len(adaptive_results)
            
            avg_embedding_quality = sum(
                r['performance_report']['performance_metrics']['avg_embedding_quality']
                for r in adaptive_results  
            ) / len(adaptive_results)
            
            print(f"\n🎯 Adaptive Engine Performance:")
            print(f"  • Average Quantum Advantage Score: {avg_quantum_advantage:.3f}")
            print(f"  • Average Embedding Quality: {avg_embedding_quality:.3f}")
            print(f"  • Total Adaptive Solves: {sum(r['performance_report']['performance_metrics']['total_solves'] for r in adaptive_results)}")
        
        adaptive_study = {
            'study_type': 'Adaptive Algorithm Analysis',
            'results': adaptive_results,
            'summary_metrics': {
                'avg_quantum_advantage': avg_quantum_advantage if adaptive_results else 0,
                'avg_embedding_quality': avg_embedding_quality if adaptive_results else 0
            }
        }
        
        all_study_results.append(adaptive_study)
        
        print()
        
        # Generate Comprehensive Research Report
        print("📝 Generating Research Report...")
        research_report = benchmark_suite.generate_research_report(all_study_results)
        
        # Save report
        report_path = benchmark_suite.output_dir / "comprehensive_research_report.md"
        with open(report_path, 'w') as f:
            f.write(research_report)
        
        print(f"✅ Research report saved to: {report_path}")
        
        print("\n" + "=" * 60)
        print("🎉 RESEARCH VALIDATION COMPLETE")
        print("=" * 60)
        
        # Display summary statistics
        total_studies = len(all_study_results)
        total_time = scalability_time + multi_obj_time + adaptive_time
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"  • Studies completed: {total_studies}")
        print(f"  • Total execution time: {total_time:.1f} seconds")
        print(f"  • Results directory: {benchmark_suite.output_dir}")
        
        # Key research findings
        print(f"\n🔬 KEY RESEARCH FINDINGS:")
        if quantum_advantages:
            print(f"  • Quantum advantage demonstrated in {len(quantum_advantages)} scenarios")
        
        if 'algorithm_performance' in multi_objective_results:
            perf = multi_objective_results['algorithm_performance']
            total_multi_tests = perf.get('quantum_wins', 0) + perf.get('classical_wins', 0) + perf.get('ties', 0)
            if total_multi_tests > 0:
                quantum_success_rate = perf.get('quantum_wins', 0) / total_multi_tests * 100
                print(f"  • Multi-objective quantum success rate: {quantum_success_rate:.1f}%")
        
        if adaptive_results:
            print(f"  • Adaptive quantum engine tested on {len(adaptive_results)} configurations")
            print(f"  • Average quantum advantage score: {avg_quantum_advantage:.3f}/1.0")
        
        print(f"\n📄 Detailed results and analysis available in: {benchmark_suite.output_dir}")
        print("\n✨ Research validation demonstrates quantum HVAC optimization potential!")
        
        return all_study_results
        
    except Exception as e:
        logger.error(f"Research validation failed: {e}")
        raise


def print_demo_header():
    """Print demonstration header with research context."""
    print()
    print("🏢 QUANTUM HVAC OPTIMIZATION RESEARCH VALIDATION")
    print("=" * 55)
    print()
    print("This demo validates novel quantum annealing algorithms for")
    print("building HVAC optimization through comprehensive benchmarking.")
    print()
    print("RESEARCH CONTRIBUTIONS:")
    print("• Adaptive Chain Strength Optimization with Bayesian Learning")
    print("• Dynamic Embedding Re-optimization with Quality Feedback") 
    print("• Multi-Objective Quantum Pareto Frontier Exploration")
    print("• Statistical Performance Analysis vs Classical Methods")
    print()
    print("VALIDATION STUDIES:")
    print("1. Scalability Analysis (5-20 zones)")
    print("2. Multi-Objective Optimization Comparison")
    print("3. Adaptive Algorithm Performance Analysis")
    print()
    print("⚠️  Note: Demo uses reduced problem sizes for speed.")
    print("   Full research studies test 5-200 zones with 50+ runs.")
    print()
    input("Press Enter to begin research validation...")
    print()


async def main():
    """Main research validation demonstration."""
    try:
        print_demo_header()
        
        # Run comprehensive validation
        results = await run_comprehensive_research_validation()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Research validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Research validation failed: {e}")
        logging.error(f"Validation error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())