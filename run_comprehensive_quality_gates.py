#!/usr/bin/env python3
"""
Comprehensive Quality Gates Runner for Autonomous SDLC
Validates all system components and quality metrics.
"""

import sys
import time
import traceback
import asyncio
import numpy as np
from pathlib import Path

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent))

def print_banner(title: str):
    """Print a banner for test sections."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_result(test_name: str, passed: bool, message: str = ""):
    """Print test result with formatting."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} {test_name}")
    if message and not passed:
        print(f"     {message}")

def run_quality_gate_1_imports():
    """Quality Gate 1: Test all critical imports."""
    print_banner("QUALITY GATE 1: SYSTEM IMPORTS")
    
    tests = []
    
    # Test 1.1: Core system imports
    try:
        from quantum_ctl import HVACController, Building
        from quantum_ctl.models.building import BuildingState, ZoneConfig
        from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
        tests.append(("Core system imports", True))
    except Exception as e:
        tests.append(("Core system imports", False, str(e)))
    
    # Test 1.2: Generation 1 (Make it Work) imports
    try:
        from quantum_ctl.utils.graceful_fallback import get_solver_type, DWAVE_AVAILABLE
        from quantum_ctl.optimization.classical_fallback_solver import ClassicalFallbackSolver
        tests.append(("Generation 1 imports", True))
    except Exception as e:
        tests.append(("Generation 1 imports", False, str(e)))
    
    # Test 1.3: Generation 2 (Make it Robust) imports
    try:
        from quantum_ctl.utils.enhanced_monitoring import get_health_monitor
        from quantum_ctl.utils.intelligent_recovery import get_recovery_manager
        from quantum_ctl.utils.comprehensive_validation import validate_system
        tests.append(("Generation 2 imports", True))
    except Exception as e:
        tests.append(("Generation 2 imports", False, str(e)))
    
    # Test 1.4: Generation 3 (Make it Scale) imports  
    try:
        from quantum_ctl.optimization.intelligent_caching_v2 import get_intelligent_cache
        from quantum_ctl.optimization.adaptive_load_balancer import get_load_balancer
        tests.append(("Generation 3 imports", True))
    except Exception as e:
        tests.append(("Generation 3 imports", False, str(e)))
    
    # Print results
    for test in tests:
        if len(test) == 3:
            print_result(test[0], test[1], test[2])
        else:
            print_result(test[0], test[1])
    
    return all(t[1] for t in tests)

def run_quality_gate_2_basic_functionality():
    """Quality Gate 2: Test basic system functionality."""
    print_banner("QUALITY GATE 2: BASIC FUNCTIONALITY")
    
    tests = []
    
    # Test 2.1: Graceful fallback system
    try:
        from quantum_ctl.utils.graceful_fallback import get_solver_type, DWAVE_AVAILABLE
        solver_type = get_solver_type()
        expected_fallback = not DWAVE_AVAILABLE
        test_passed = solver_type == "classical_fallback" if expected_fallback else True
        tests.append(("Graceful fallback system", test_passed))
    except Exception as e:
        tests.append(("Graceful fallback system", False, str(e)))
    
    # Test 2.2: Basic controller creation
    try:
        from quantum_ctl import HVACController, Building
        from quantum_ctl.models.building import ZoneConfig
        from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
        
        zones = [ZoneConfig(
            zone_id="qg_test",
            area=100.0,
            volume=300.0, 
            thermal_mass=250.0,
            max_heating_power=10.0,
            max_cooling_power=8.0
        )]
        building = Building(building_id="quality_gate", zones=zones)
        config = OptimizationConfig(solver="classical_fallback")
        controller = HVACController(building, config, ControlObjectives())
        
        tests.append(("Basic controller creation", controller is not None))
    except Exception as e:
        tests.append(("Basic controller creation", False, str(e)))
    
    # Test 2.3: Health monitoring system
    try:
        from quantum_ctl.utils.enhanced_monitoring import get_health_monitor
        monitor = get_health_monitor()
        status = monitor.get_system_status()
        tests.append(("Health monitoring system", status is not None))
    except Exception as e:
        tests.append(("Health monitoring system", False, str(e)))
    
    # Test 2.4: Intelligent caching system
    try:
        from quantum_ctl.optimization.intelligent_caching_v2 import get_intelligent_cache
        cache = get_intelligent_cache()
        cache.put('qg_test', {'data': 'test'}, {'size': 100})
        value, hit_type = cache.get('qg_test')
        tests.append(("Intelligent caching system", value is not None))
    except Exception as e:
        tests.append(("Intelligent caching system", False, str(e)))
    
    # Print results
    for test in tests:
        if len(test) == 3:
            print_result(test[0], test[1], test[2])
        else:
            print_result(test[0], test[1])
    
    return all(t[1] for t in tests)

def run_quality_gate_3_reliability():
    """Quality Gate 3: Test system reliability and error handling."""
    print_banner("QUALITY GATE 3: RELIABILITY & ERROR HANDLING")
    
    tests = []
    
    # Test 3.1: Input validation
    try:
        from quantum_ctl.utils.comprehensive_validation import validate_system
        
        # Test valid input
        valid_state = {
            'zone_temperatures': [22.0, 21.5],
            'outside_temperature': 15.0,
            'occupancy': [0.8, 0.6]
        }
        controls = np.array([0.5, 0.3])
        config = {'prediction_horizon': 24}
        building_config = {'max_power_kw': 50}
        
        results = validate_system(valid_state, controls, config, building_config)
        valid_test = results['state'].valid and results['config'].valid
        
        # Test invalid input detection
        invalid_state = {
            'zone_temperatures': [1000.0, -100.0],  # Invalid temperatures
            'outside_temperature': 200.0,
            'occupancy': [2.0, -1.0]  # Invalid occupancy
        }
        
        invalid_results = validate_system(invalid_state, controls, config, building_config)
        invalid_test = not invalid_results['state'].valid and not invalid_results['safety'].valid
        
        tests.append(("Input validation", valid_test and invalid_test))
    except Exception as e:
        tests.append(("Input validation", False, str(e)))
    
    # Test 3.2: Recovery system
    try:
        from quantum_ctl.utils.intelligent_recovery import get_recovery_manager, FailureType
        
        recovery_manager = get_recovery_manager()
        
        # Test failure recording
        failure_event = recovery_manager.record_failure(
            FailureType.OPTIMIZATION_FAILED,
            {'test': True},
            "Test failure"
        )
        
        # Test recovery strategy generation
        strategies = recovery_manager.get_recovery_strategies(failure_event)
        
        tests.append(("Recovery system", len(strategies) > 0))
    except Exception as e:
        tests.append(("Recovery system", False, str(e)))
    
    # Test 3.3: Circuit breaker and monitoring
    try:
        from quantum_ctl.utils.enhanced_monitoring import get_health_monitor
        
        monitor = get_health_monitor()
        
        # Test metrics recording
        monitor.record_performance_metric("test_operation", 0.1, True)
        monitor.record_optimization_result(True, 0.5)
        
        # Get health report
        health_report = monitor.health_check()
        
        tests.append(("Monitoring and metrics", 'overall_health' in health_report))
    except Exception as e:
        tests.append(("Monitoring and metrics", False, str(e)))
    
    # Print results
    for test in tests:
        if len(test) == 3:
            print_result(test[0], test[1], test[2])
        else:
            print_result(test[0], test[1])
    
    return all(t[1] for t in tests)

def run_quality_gate_4_performance():
    """Quality Gate 4: Test performance and scalability components."""
    print_banner("QUALITY GATE 4: PERFORMANCE & SCALABILITY")
    
    tests = []
    
    # Test 4.1: Intelligent caching performance
    try:
        from quantum_ctl.optimization.intelligent_caching_v2 import get_intelligent_cache
        
        cache = get_intelligent_cache()
        
        # Test cache operations performance
        start_time = time.time()
        
        for i in range(100):
            cache.put(f'perf_test_{i}', {'data': f'test_{i}'}, {'size': i})
        
        for i in range(100):
            value, hit_type = cache.get(f'perf_test_{i}')
        
        operation_time = time.time() - start_time
        
        # Should complete 200 operations in reasonable time
        cache_stats = cache.get_cache_stats()
        
        tests.append(("Caching performance", operation_time < 1.0 and cache_stats['hit_rate'] > 90))
    except Exception as e:
        tests.append(("Caching performance", False, str(e)))
    
    # Test 4.2: Load balancer functionality
    try:
        from quantum_ctl.optimization.adaptive_load_balancer import get_load_balancer
        
        load_balancer = get_load_balancer()
        
        # Add worker nodes
        load_balancer.add_worker_node('test_worker1', 'http://test:8001', 100)
        load_balancer.add_worker_node('test_worker2', 'http://test:8002', 150)
        
        # Update health to make available
        load_balancer.update_node_health('test_worker1', 0.9)
        load_balancer.update_node_health('test_worker2', 0.8)
        
        # Test worker selection
        context = {'estimated_size': 50, 'priority': 'normal'}
        selected_worker = load_balancer.select_worker(context)
        
        tests.append(("Load balancer", selected_worker in ['test_worker1', 'test_worker2']))
    except Exception as e:
        tests.append(("Load balancer", False, str(e)))
    
    # Test 4.3: System resource monitoring
    try:
        from quantum_ctl.utils.enhanced_monitoring import get_health_monitor
        
        monitor = get_health_monitor()
        
        # Test metrics collection
        for i in range(10):
            monitor.record_performance_metric(f"test_op_{i%3}", 0.1 + i*0.01, True)
        
        performance_summary = monitor.get_performance_summary()
        
        tests.append(("Resource monitoring", len(performance_summary) > 0))
    except Exception as e:
        tests.append(("Resource monitoring", False, str(e)))
    
    # Print results
    for test in tests:
        if len(test) == 3:
            print_result(test[0], test[1], test[2])
        else:
            print_result(test[0], test[1])
    
    return all(t[1] for t in tests)

async def run_quality_gate_5_integration():
    """Quality Gate 5: Test complete system integration."""
    print_banner("QUALITY GATE 5: SYSTEM INTEGRATION")
    
    tests = []
    
    # Test 5.1: End-to-end optimization workflow
    try:
        from quantum_ctl import HVACController, Building
        from quantum_ctl.models.building import BuildingState, ZoneConfig
        from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
        
        # Create realistic test system
        zones = [ZoneConfig(
            zone_id="integration_zone",
            area=200.0,
            volume=600.0,
            thermal_mass=500.0,
            max_heating_power=15.0,
            max_cooling_power=12.0
        )]
        
        building = Building(building_id="integration_test", zones=zones)
        config = OptimizationConfig(
            prediction_horizon=2,  # Short for testing
            control_interval=30,
            solver="classical_fallback"
        )
        
        controller = HVACController(building, config, ControlObjectives())
        
        # Create test state
        state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([22.0]),
            outside_temperature=16.0,
            humidity=50.0,
            occupancy=np.array([0.7]),
            hvac_power=np.array([8.0]),
            control_setpoints=np.array([0.6])
        )
        
        # Test data
        weather_forecast = np.array([[16.0, 250.0, 50.0], [17.0, 260.0, 52.0], [18.0, 270.0, 54.0], [19.0, 280.0, 56.0]])
        energy_prices = np.array([0.12, 0.13, 0.14, 0.15])
        
        # Run optimization
        result = await controller.optimize(
            current_state=state,
            weather_forecast=weather_forecast,
            energy_prices=energy_prices
        )
        
        # Apply controls
        controller.apply_schedule(result)
        
        # Get system status
        status = controller.get_status()
        
        integration_test = (result is not None and 
                           len(result) > 0 and 
                           'building_id' in status)
        
        tests.append(("End-to-end workflow", integration_test))
        
    except Exception as e:
        tests.append(("End-to-end workflow", False, str(e)))
    
    # Test 5.2: Multi-component coordination
    try:
        from quantum_ctl.utils.enhanced_monitoring import get_health_monitor
        from quantum_ctl.optimization.intelligent_caching_v2 import get_intelligent_cache
        from quantum_ctl.utils.intelligent_recovery import get_recovery_manager
        
        monitor = get_health_monitor()
        cache = get_intelligent_cache()
        recovery = get_recovery_manager()
        
        # Test coordination
        monitor.record_performance_metric("integration_test", 0.5, True)
        cache.put("integration_key", {"test": "data"})
        
        # All systems should be operational
        system_status = monitor.get_system_status()
        cache_stats = cache.get_cache_stats()
        
        coordination_test = (system_status is not None and
                           cache_stats['size'] > 0)
        
        tests.append(("Multi-component coordination", coordination_test))
        
    except Exception as e:
        tests.append(("Multi-component coordination", False, str(e)))
    
    # Test 5.3: System resilience under load
    try:
        # Simulate concurrent operations
        tasks = []
        
        async def simulate_operation(op_id: int):
            from quantum_ctl.utils.enhanced_monitoring import get_health_monitor
            from quantum_ctl.optimization.intelligent_caching_v2 import get_intelligent_cache
            
            monitor = get_health_monitor()
            cache = get_intelligent_cache()
            
            # Record some activity
            monitor.record_performance_metric(f"load_test_{op_id}", 0.1, True)
            cache.put(f"load_key_{op_id}", {"data": op_id})
            
            return True
        
        # Launch concurrent operations
        for i in range(20):
            task = asyncio.create_task(simulate_operation(i))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_ops = sum(1 for r in results if r is True)
        
        tests.append(("System resilience", successful_ops >= 18))  # 90% success rate
        
    except Exception as e:
        tests.append(("System resilience", False, str(e)))
    
    # Print results
    for test in tests:
        if len(test) == 3:
            print_result(test[0], test[1], test[2])
        else:
            print_result(test[0], test[1])
    
    return all(t[1] for t in tests)

def main():
    """Run all quality gates."""
    print_banner("AUTONOMOUS SDLC QUALITY GATES")
    print("Testing Quantum HVAC Control System - All Generations")
    
    start_time = time.time()
    
    # Run all quality gates
    gate1_pass = run_quality_gate_1_imports()
    gate2_pass = run_quality_gate_2_basic_functionality()
    gate3_pass = run_quality_gate_3_reliability()
    gate4_pass = run_quality_gate_4_performance()
    gate5_pass = asyncio.run(run_quality_gate_5_integration())
    
    # Summary
    all_gates_passed = all([gate1_pass, gate2_pass, gate3_pass, gate4_pass, gate5_pass])
    total_time = time.time() - start_time
    
    print_banner("QUALITY GATES SUMMARY")
    print_result("Gate 1: System Imports", gate1_pass)
    print_result("Gate 2: Basic Functionality", gate2_pass)
    print_result("Gate 3: Reliability & Error Handling", gate3_pass)
    print_result("Gate 4: Performance & Scalability", gate4_pass)
    print_result("Gate 5: System Integration", gate5_pass)
    
    print(f"\nExecution Time: {total_time:.2f} seconds")
    
    if all_gates_passed:
        print("\n🎉 ALL QUALITY GATES PASSED!")
        print("✅ System is ready for production deployment")
        return 0
    else:
        print("\n⚠️  SOME QUALITY GATES FAILED")
        print("❌ Review failed gates before deployment")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)