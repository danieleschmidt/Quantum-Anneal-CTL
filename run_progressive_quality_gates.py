#!/usr/bin/env python3
"""
Progressive Quality Gates Execution Script
Autonomous SDLC Quality Validation System
"""

import asyncio
import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_ctl.quality_gates import QualityGateRunner, QualityGateConfig
from quantum_ctl.quality_gates.intelligent_caching import SmartCacheManager
from quantum_ctl.quality_gates.performance_profiler import PerformanceProfiler
from quantum_ctl.quality_gates.distributed_gates import LoadBalancedGateRunner

def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('quality_gates_execution.log', mode='a')
        ]
    )

async def run_progressive_quality_gates():
    """Execute progressive quality gates with full autonomous features"""
    
    print("🔬 PROGRESSIVE QUALITY GATES - AUTONOMOUS EXECUTION")
    print("=" * 80)
    print("🧠 Intelligent Analysis + 🚀 Progressive Enhancement + ⚡ Auto-Scaling")
    print("=" * 80)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration with production-ready settings
    config = QualityGateConfig(
        min_test_coverage=75.0,
        max_api_response_time_ms=200,
        max_memory_usage_mb=1024,
        security_scan_enabled=True,
        fail_fast=False,  # Continue through all gates for comprehensive analysis
        parallel_execution=True,
        generate_html_report=True,
        report_output_dir="quality_reports"
    )
    
    print(f"📋 Configuration: {config.min_test_coverage}% coverage, {config.max_api_response_time_ms}ms response limit")
    print(f"🔧 Features: Security scan, HTML reports, parallel execution")
    print()
    
    # Initialize smart caching system
    cache_manager = SmartCacheManager({
        'hot_cache_size_mb': 100,
        'cold_cache_size_mb': 200,
        'hot_cache_ttl': 1800,  # 30 minutes
        'cold_cache_ttl': 7200  # 2 hours
    })
    
    # Initialize performance profiler
    profiler = PerformanceProfiler()
    
    # Initialize load-balanced runner
    scaling_config = {
        'max_workers': min(8, max(2, asyncio.cpu_count() or 2)),
        'min_workers': 2,
        'scale_threshold': 0.7
    }
    
    load_balanced_runner = LoadBalancedGateRunner(config, scaling_config)
    
    try:
        # Start intelligent systems
        print("🚀 Starting intelligent caching system...")
        await cache_manager.start()
        
        execution_context = {
            "project_root": str(Path.cwd()),
            "execution_timestamp": datetime.utcnow().isoformat(),
            "autonomous_mode": True,
            "progressive_enhancement": True
        }
        
        print("⚡ Executing quality gates with auto-scaling...")
        
        # Profile the entire execution
        async with profiler.profile_execution(
            sample_interval=0.5,
            enable_function_profiling=True,
            track_memory_leaks=True
        ) as metrics:
            
            # Execute with load balancing and auto-scaling
            result = await load_balanced_runner.run_with_auto_scaling(execution_context)
            
        # Generate comprehensive performance report
        performance_report = profiler.generate_performance_report(metrics)
        
        # Print results
        print("\n" + "="*80)
        print("📊 PROGRESSIVE QUALITY GATES RESULTS")
        print("="*80)
        
        overall_passed = result.get('passed', False)
        overall_score = result.get('score', 0.0)
        summary = result.get('summary', {})
        
        status_icon = "✅" if overall_passed else "❌"
        status_text = "PASSED" if overall_passed else "FAILED"
        
        print(f"\n{status_icon} OVERALL STATUS: {status_text}")
        print(f"📈 Quality Score: {overall_score:.1f}/100")
        print(f"⏱️  Execution Time: {result.get('execution_time_ms', 0):.1f}ms")
        print(f"🎯 Gates Passed: {summary.get('passed_gates', 0)}/{summary.get('total_gates', 0)}")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        perf_summary = performance_report.get('summary', {})
        print(f"   Peak Memory: {perf_summary.get('peak_memory_mb', 0):.1f}MB")
        print(f"   Average CPU: {perf_summary.get('avg_cpu_percent', 0):.1f}%")
        print(f"   Function Calls: {perf_summary.get('function_calls', 0):,}")
        print(f"   Memory Leaks: {perf_summary.get('memory_leaks', 0)}")
        print(f"   Performance Grade: {performance_report.get('grade', 'N/A')}")
        
        # Scaling information
        scaling_info = result.get('scaling_info', {})
        if scaling_info:
            print(f"\n🔄 SCALING METRICS:")
            print(f"   Workers Used: {scaling_info.get('workers_used', 1)}")
            print(f"   Execution Mode: {scaling_info.get('execution_mode', 'local')}")
            
            resource_usage = scaling_info.get('resource_usage', {})
            if resource_usage:
                print(f"   Avg CPU Usage: {resource_usage.get('avg_cpu_percent', 0):.1f}%")
                print(f"   Max Memory: {resource_usage.get('max_memory_percent', 0):.1f}%")
        
        # Individual gate results
        print(f"\n📋 INDIVIDUAL GATE RESULTS:")
        print("-" * 80)
        
        for gate in result.get('gates', []):
            gate_icon = "✅" if gate.get('passed', False) else "❌"
            gate_name = gate.get('gate_name', 'Unknown').replace('_', ' ').title()
            gate_score = gate.get('score', 0)
            gate_time = gate.get('execution_time_ms', 0)
            
            print(f"{gate_icon} {gate_name:<20} Score: {gate_score:>6.1f}/100   Time: {gate_time:>6.1f}ms")
            
            # Show key messages
            messages = gate.get('messages', [])
            if messages:
                for msg in messages[:2]:  # Show first 2 messages
                    print(f"   └─ {msg}")
        
        # Cache statistics
        cache_stats = await cache_manager.get_combined_statistics()
        print(f"\n🧠 INTELLIGENT CACHE PERFORMANCE:")
        print(f"   Hit Rate: {cache_stats.get('combined_hit_rate', 0)*100:.1f}%")
        print(f"   Total Entries: {cache_stats.get('total_entries', 0)}")
        print(f"   Cache Size: {cache_stats.get('total_size_mb', 0):.1f}MB")
        
        # Recommendations
        if not overall_passed:
            print(f"\n💡 IMPROVEMENT RECOMMENDATIONS:")
            print("-" * 80)
            
            recommendations = performance_report.get('recommendations', [])
            bottlenecks = performance_report.get('bottlenecks', [])
            
            if bottlenecks:
                print("   Performance Bottlenecks:")
                for bottleneck in bottlenecks:
                    print(f"   • {bottleneck}")
            
            if recommendations:
                print("   Optimization Suggestions:")
                for rec in recommendations[:5]:  # Show top 5
                    print(f"   • {rec}")
        
        # Save comprehensive results
        comprehensive_results = {
            'quality_gates': result,
            'performance_metrics': performance_report,
            'cache_statistics': cache_stats,
            'execution_metadata': {
                'autonomous_mode': True,
                'progressive_enhancement': True,
                'generation': 'all_3_completed',
                'scaling_enabled': True,
                'intelligent_caching': True
            }
        }
        
        results_file = f"progressive_quality_gates_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\n📄 Comprehensive results saved to: {results_file}")
        print("=" * 80)
        
        # Final autonomous decision
        if overall_passed:
            print("🎉 AUTONOMOUS QUALITY GATES: ALL SYSTEMS GO!")
            print("✅ Code is ready for production deployment")
            return 0
        else:
            print("⚠️  AUTONOMOUS QUALITY GATES: IMPROVEMENTS NEEDED")
            print("❌ Address quality issues before production")
            return 1
            
    except Exception as e:
        logger.error(f"Progressive quality gates execution failed: {e}")
        print(f"\n❌ EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 2
        
    finally:
        # Cleanup
        try:
            await cache_manager.stop()
            print("\n🔄 Intelligent systems shut down gracefully")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


async def main():
    """Main entry point"""
    try:
        exit_code = await run_progressive_quality_gates()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Quality gates execution interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # Handle both sync and async execution
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # Fallback for environments where asyncio is already running
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(main())
        else:
            raise