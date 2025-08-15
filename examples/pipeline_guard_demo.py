"""
Comprehensive demo of the self-healing pipeline guard for quantum HVAC system.
"""

import asyncio
import time
import random
from quantum_ctl.pipeline_guard import (
    PipelineGuard,
    HealthMonitor,
    RecoveryManager,
    CircuitBreaker,
    QuantumCircuitBreaker
)
from quantum_ctl.pipeline_guard.quantum_integration import QuantumHVACPipelineGuard
from quantum_ctl.pipeline_guard.metrics_collector import MetricsCollector
from quantum_ctl.pipeline_guard.advanced_recovery import AdvancedRecoveryManager, RecoveryStrategy
from quantum_ctl.pipeline_guard.security_monitor import SecurityMonitor
from quantum_ctl.pipeline_guard.performance_optimizer import PerformanceOptimizer
from quantum_ctl.pipeline_guard.ai_predictor import AIPredictor


class MockQuantumSolver:
    """Mock quantum solver for demonstration."""
    
    def __init__(self):
        self.connection_healthy = True
        self.chain_break_rate = 0.05
        
    def test_connection(self):
        """Test quantum solver connection."""
        if not self.connection_healthy:
            raise Exception("D-Wave connection failed")
        
        # Mock result with chain break info
        class MockResult:
            def __init__(self):
                self.info = {
                    'chain_break_fraction': random.uniform(0.01, 0.15),
                    'qpu_access_time': random.uniform(5000, 20000),  # microseconds
                    'num_reads': 1000
                }
                
        return MockResult()
        
    def reset_connection(self):
        """Reset quantum solver connection."""
        self.connection_healthy = True
        print("Quantum solver connection reset")
        
    def get_metrics(self):
        """Get quantum solver metrics."""
        return {
            'chain_break_rate': self.chain_break_rate,
            'qpu_access_time': random.uniform(5000, 15000),
            'solution_quality': random.uniform(0.8, 1.0)
        }
        

class MockHVACController:
    """Mock HVAC controller for demonstration."""
    
    def __init__(self):
        self.healthy = True
        self.last_optimization = time.time()
        self.optimization_count = 0
        
    def get_current_state(self):
        """Get current HVAC state."""
        if not self.healthy:
            return None
            
        return {
            'zone_temperatures': {f'zone_{i}': random.uniform(20, 25) for i in range(5)},
            'energy_consumption': random.uniform(50, 100),
            'control_actions': {f'action_{i}': random.uniform(0, 100) for i in range(3)},
            'optimization_time': random.uniform(5, 30)
        }
        
    def can_optimize(self):
        """Check if controller can optimize."""
        return self.healthy
        
    def reset(self):
        """Reset controller."""
        self.healthy = True
        print("HVAC controller reset")
        
    def initialize(self):
        """Initialize controller."""
        self.last_optimization = time.time()
        print("HVAC controller initialized")
        
    def get_last_optimization_time(self):
        """Get last optimization time."""
        return self.last_optimization
        
    def force_optimization(self):
        """Force optimization cycle."""
        self.last_optimization = time.time()
        self.optimization_count += 1
        print(f"Forced optimization #{self.optimization_count}")
        
    def get_metrics(self):
        """Get HVAC controller metrics."""
        return {
            'optimization_count': self.optimization_count,
            'avg_temperature': 22.5,
            'energy_efficiency': random.uniform(0.7, 0.9)
        }


async def simulate_component_health_checks():
    """Simulate various component health check scenarios."""
    
    quantum_solver = MockQuantumSolver()
    hvac_controller = MockHVACController()
    
    def quantum_health_check():
        """Quantum solver health check."""
        try:
            result = quantum_solver.test_connection()
            chain_break_rate = result.info.get('chain_break_fraction', 0)
            return chain_break_rate < 0.1  # Healthy if < 10% chain breaks
        except Exception:
            return False
            
    def hvac_health_check():
        """HVAC controller health check."""
        return hvac_controller.can_optimize()
        
    def bms_health_check():
        """BMS connection health check."""
        return random.choice([True, True, True, False])  # 75% healthy
        
    def weather_api_health_check():
        """Weather API health check."""
        return random.choice([True, True, False])  # 67% healthy
        
    async def quantum_recovery():
        """Quantum solver recovery action."""
        quantum_solver.reset_connection()
        await asyncio.sleep(2)  # Simulate recovery time
        return quantum_solver.connection_healthy
        
    async def hvac_recovery():
        """HVAC controller recovery action."""
        hvac_controller.reset()
        hvac_controller.initialize()
        await asyncio.sleep(1)
        return hvac_controller.healthy
        
    async def bms_recovery():
        """BMS connection recovery action."""
        print("Attempting BMS reconnection...")
        await asyncio.sleep(3)
        return random.choice([True, False])  # 50% success rate
        
    return {
        'health_checks': {
            'quantum_solver': quantum_health_check,
            'hvac_controller': hvac_health_check,
            'bms_connector': bms_health_check,
            'weather_api': weather_api_health_check
        },
        'recovery_actions': {
            'quantum_solver': quantum_recovery,
            'hvac_controller': hvac_recovery,
            'bms_connector': bms_recovery
        },
        'components': {
            'quantum_solver': quantum_solver,
            'hvac_controller': hvac_controller
        }
    }


async def demo_basic_pipeline_guard():
    """Demonstrate basic pipeline guard functionality."""
    print("\\n=== Basic Pipeline Guard Demo ===")
    
    # Setup mock components
    components = await simulate_component_health_checks()
    
    # Create pipeline guard
    guard = PipelineGuard(check_interval=5.0)
    
    # Register components
    guard.register_component(
        name="quantum_solver",
        health_check=components['health_checks']['quantum_solver'],
        recovery_action=components['recovery_actions']['quantum_solver'],
        critical=True,
        circuit_breaker_config={
            "failure_threshold": 3,
            "recovery_timeout": 30.0
        }
    )
    
    guard.register_component(
        name="hvac_controller",
        health_check=components['health_checks']['hvac_controller'],
        recovery_action=components['recovery_actions']['hvac_controller'],
        critical=True
    )
    
    guard.register_component(
        name="bms_connector",
        health_check=components['health_checks']['bms_connector'],
        recovery_action=components['recovery_actions']['bms_connector'],
        critical=False
    )
    
    # Start monitoring
    guard.start()
    
    print("Pipeline guard started, monitoring components...")
    
    # Simulate failures and recovery
    for i in range(3):
        await asyncio.sleep(8)
        
        status = guard.get_status()
        print(f"\\nIteration {i+1}:")
        print(f"Pipeline Status: {status['pipeline_status']}")
        print(f"Healthy Components: {status['healthy_components']}/{status['total_components']}")
        
        # Simulate component failure
        if i == 1:
            print("Simulating quantum solver failure...")
            components['components']['quantum_solver'].connection_healthy = False
            
        if i == 2:
            print("Simulating HVAC controller failure...")
            components['components']['hvac_controller'].healthy = False
            
    await guard.stop()
    print("Basic pipeline guard demo completed")


async def demo_advanced_recovery():
    """Demonstrate advanced recovery strategies."""
    print("\\n=== Advanced Recovery Demo ===")
    
    recovery_manager = AdvancedRecoveryManager()
    
    # Register recovery plans
    recovery_manager.register_recovery_plan(
        component="quantum_solver",
        strategy=RecoveryStrategy.GRACEFUL_RESTART,
        steps=["shutdown_quantum", "clear_cache", "restart_quantum"],
        estimated_duration=30.0
    )
    
    recovery_manager.register_recovery_plan(
        component="quantum_solver",
        strategy=RecoveryStrategy.FALLBACK_MODE,
        steps=["switch_to_classical", "notify_admin"],
        estimated_duration=10.0
    )
    
    recovery_manager.register_recovery_plan(
        component="hvac_controller",
        strategy=RecoveryStrategy.INTELLIGENT_BACKOFF,
        steps=["pause_optimization", "reset_state", "resume_optimization"],
        estimated_duration=20.0
    )
    
    # Set component dependencies
    recovery_manager.set_component_dependencies({
        "hvac_controller": ["quantum_solver", "bms_connector"],
        "quantum_solver": [],
        "bms_connector": []
    })
    
    # Mock recovery actions
    async def restart_quantum():
        print("Restarting quantum solver...")
        await asyncio.sleep(2)
        return random.choice([True, False])
        
    async def shutdown_quantum():
        print("Shutting down quantum solver...")
        await asyncio.sleep(1)
        return True
        
    async def start_quantum():
        print("Starting quantum solver...")
        await asyncio.sleep(1)
        return True
        
    recovery_actions = {
        "restart_quantum_solver": restart_quantum,
        "shutdown_quantum_solver": shutdown_quantum,
        "start_quantum_solver": start_quantum
    }
    
    # Simulate recovery scenarios
    failure_context = {"severity": "high", "system_load": "normal"}
    
    # Test quantum solver recovery
    optimal_plan = recovery_manager.select_optimal_strategy("quantum_solver", failure_context)
    print(f"\\nSelected recovery strategy: {optimal_plan.strategy.value}")
    print(f"Estimated duration: {optimal_plan.estimated_duration}s")
    
    success = await recovery_manager.execute_recovery_plan(
        optimal_plan, failure_context, recovery_actions
    )
    print(f"Recovery result: {'Success' if success else 'Failed'}")
    
    # Get analytics
    analytics = recovery_manager.get_recovery_analytics()
    print(f"\\nRecovery Analytics:")
    print(f"Total attempts: {analytics.get('total_recovery_attempts', 0)}")
    print(f"Success rate: {analytics.get('overall_success_rate', 0):.1f}%")


async def demo_metrics_and_monitoring():
    """Demonstrate metrics collection and monitoring."""
    print("\\n=== Metrics and Monitoring Demo ===")
    
    metrics_collector = MetricsCollector()
    metrics_collector.start()
    
    # Add alert rules
    metrics_collector.add_alert_rule(
        "quantum_chain_break_fraction",
        lambda value: value > 0.15,  # Alert if chain breaks > 15%
        severity="warning"
    )
    
    metrics_collector.add_alert_rule(
        "hvac_optimization_duration",
        lambda value: value > 25,  # Alert if optimization takes > 25s
        severity="critical"
    )
    
    print("Collecting metrics for 30 seconds...")
    
    # Simulate metrics collection
    for i in range(30):
        # Quantum metrics
        chain_break_rate = random.uniform(0.05, 0.20)
        qpu_time = random.uniform(5000, 20000)
        
        metrics_collector.record_metric(
            "quantum_chain_break_fraction",
            chain_break_rate,
            labels={"solver": "quantum"},
            unit="fraction"
        )
        
        metrics_collector.record_metric(
            "quantum_qpu_access_time",
            qpu_time,
            labels={"solver": "quantum"},
            unit="microseconds"
        )
        
        # HVAC metrics
        optimization_time = random.uniform(10, 35)
        energy_consumption = random.uniform(50, 120)
        
        metrics_collector.record_histogram(
            "hvac_optimization_duration",
            optimization_time,
            labels={"controller": "main"}
        )
        
        metrics_collector.record_metric(
            "hvac_energy_consumption",
            energy_consumption,
            labels={"building": "main"},
            unit="kwh"
        )
        
        await asyncio.sleep(1)
        
    # Get metrics summary
    summary = metrics_collector.get_metric_summary("quantum_chain_break_fraction", hours=1)
    print(f"\\nQuantum Chain Break Summary:")
    print(f"Count: {summary.get('count', 0)}")
    print(f"Average: {summary.get('avg', 0):.3f}")
    print(f"Max: {summary.get('max', 0):.3f}")
    
    # Get active alerts
    active_alerts = metrics_collector.get_active_alerts()
    print(f"\\nActive Alerts: {len(active_alerts)}")
    for alert in active_alerts:
        print(f"- {alert['severity'].upper()}: {alert['message']}")
        
    await metrics_collector.stop()


async def demo_security_monitoring():
    """Demonstrate security monitoring capabilities."""
    print("\\n=== Security Monitoring Demo ===")
    
    security_monitor = SecurityMonitor()
    
    # Configure security
    security_monitor.configure_allowed_ips(["192.168.1.0/24", "10.0.0.0/8"])
    
    print("Testing security scenarios...")
    
    # Test scenarios
    test_scenarios = [
        {"ip": "192.168.1.100", "user": "admin", "op": "normal_operation", "should_pass": True},
        {"ip": "10.0.0.50", "user": "operator", "op": "read_metrics", "should_pass": True},
        {"ip": "172.16.0.1", "user": "unknown", "op": "config_change", "should_pass": False},
        {"ip": "192.168.1.100", "user": "admin", "op": "system_shutdown", "should_pass": True}
    ]
    
    for scenario in test_scenarios:
        access_allowed = security_monitor.validate_access(
            scenario["ip"], scenario["user"], scenario["op"]
        )
        
        result = "ALLOWED" if access_allowed else "BLOCKED"
        expected = "ALLOWED" if scenario["should_pass"] else "BLOCKED"
        status = "✓" if (access_allowed == scenario["should_pass"]) else "✗"
        
        print(f"{status} {scenario['ip']} -> {result} (expected: {expected})")
        
        # Simulate failed authentication for blocked access
        if not access_allowed:
            security_monitor.record_auth_failure(
                scenario["ip"], scenario["user"], "invalid_credentials"
            )
            
    # Test data integrity
    quantum_data = {"solution": [1, 0, 1, 0], "energy": -15.2}
    integrity_ok = security_monitor.validate_quantum_integrity("test_solver", quantum_data)
    print(f"\\nQuantum data integrity: {'✓ VALID' if integrity_ok else '✗ INVALID'}")
    
    # Get security summary
    summary = security_monitor.get_security_summary(hours=1)
    print(f"\\nSecurity Summary:")
    print(f"Total events: {summary['total_events']}")
    print(f"Blocked IPs: {summary['blocked_ips_count']}")
    print(f"Security status: {summary['security_status']}")


async def demo_performance_optimization():
    """Demonstrate performance optimization."""
    print("\\n=== Performance Optimization Demo ===")
    
    optimizer = PerformanceOptimizer()
    
    # Register scaling policies
    optimizer.register_scaling_policy(
        component="quantum_solver",
        scale_up_threshold=80.0,
        scale_down_threshold=30.0,
        max_instances=5,
        min_instances=1
    )
    
    # Add optimization rules
    def high_cpu_condition(metrics):
        return metrics.get("system_cpu_percent", 0) > 75
        
    async def reduce_cpu_load():
        print("Reducing CPU load...")
        await asyncio.sleep(1)
        return {"action": "cpu_throttling", "reduction": "15%"}
        
    optimizer.add_optimization_rule(
        name="high_cpu_optimization",
        condition=high_cpu_condition,
        action=reduce_cpu_load,
        priority=1
    )
    
    print("Recording performance metrics...")
    
    # Simulate performance data
    for i in range(10):
        # Simulate varying system load
        cpu_percent = 60 + (i * 3) + random.uniform(-5, 5)
        memory_percent = 45 + (i * 2) + random.uniform(-3, 3)
        
        optimizer.record_performance_metric(
            "system_cpu_percent", cpu_percent, "system"
        )
        optimizer.record_performance_metric(
            "system_memory_percent", memory_percent, "system"  
        )
        
        # Quantum solver metrics
        optimizer.record_performance_metric(
            "quantum_solver_load", 40 + (i * 5), "quantum_solver"
        )
        
        await asyncio.sleep(0.5)
        
    # Run optimization
    optimization_results = await optimizer.optimize_performance()
    print(f"\\nOptimizations applied: {len(optimization_results['optimizations_applied'])}")
    
    for opt in optimization_results['optimizations_applied']:
        print(f"- {opt['rule']}: {opt['result']}")
        
    # Get analytics
    analytics = optimizer.get_performance_analytics()
    print(f"\\nPerformance Analytics:")
    print(f"Current CPU: {analytics['current_metrics'].get('system_cpu_percent', 0):.1f}%")
    print(f"Current Memory: {analytics['current_metrics'].get('system_memory_percent', 0):.1f}%")
    
    # Get optimization suggestions
    suggestions = optimizer.suggest_optimizations()
    print(f"\\nOptimization Suggestions: {len(suggestions)}")
    for suggestion in suggestions:
        print(f"- {suggestion['type']}: {suggestion['suggestion']}")


async def demo_ai_prediction():
    """Demonstrate AI-powered failure prediction."""
    print("\\n=== AI Failure Prediction Demo ===")
    
    ai_predictor = AIPredictor()
    await ai_predictor.start_prediction_engine()
    
    print("Training AI models with simulated data...")
    
    # Generate training data
    for i in range(200):
        # Simulate metrics for different components
        
        # Quantum solver metrics
        chain_breaks = random.uniform(0.01, 0.25)
        qpu_time = random.uniform(5000, 30000)
        solution_quality = random.uniform(0.6, 1.0)
        
        ai_predictor.record_metrics("quantum_solver", {
            "chain_break_fraction": chain_breaks,
            "qpu_access_time": qpu_time,
            "solution_quality": solution_quality,
            "cpu_usage": random.uniform(20, 90)
        })
        
        # HVAC controller metrics  
        optimization_time = random.uniform(5, 45)
        energy_consumption = random.uniform(40, 150)
        temperature_variance = random.uniform(0.1, 2.0)
        
        ai_predictor.record_metrics("hvac_controller", {
            "optimization_duration": optimization_time,
            "energy_consumption": energy_consumption,
            "temperature_variance": temperature_variance,
            "control_stability": random.uniform(0.7, 1.0)
        })
        
        # Simulate occasional failures for training
        if random.random() < 0.05:  # 5% chance of failure
            component = random.choice(["quantum_solver", "hvac_controller"])
            ai_predictor.record_failure(
                component=component,
                failure_type="performance_degradation",
                metrics_before_failure={
                    "chain_break_fraction": 0.2,
                    "optimization_duration": 40
                },
                recovery_time=random.uniform(60, 300)
            )
            
    # Wait for model training
    await asyncio.sleep(3)
    
    print("Generating predictions...")
    
    # Simulate real-time prediction
    for i in range(5):
        # Record current metrics (with some concerning trends)
        ai_predictor.record_metrics("quantum_solver", {
            "chain_break_fraction": 0.15 + (i * 0.02),  # Increasing trend
            "qpu_access_time": 20000 + (i * 2000),
            "solution_quality": 0.8 - (i * 0.05),
            "cpu_usage": 70 + (i * 5)
        })
        
        ai_predictor.record_metrics("hvac_controller", {
            "optimization_duration": 25 + (i * 3),  # Increasing trend
            "energy_consumption": 100 + (i * 10),
            "temperature_variance": 0.5 + (i * 0.2),
            "control_stability": 0.9 - (i * 0.05)
        })
        
        await asyncio.sleep(1)
        
    # Get prediction summary
    summary = ai_predictor.get_prediction_summary(hours=1)
    print(f"\\nPrediction Summary:")
    print(f"Total predictions: {summary['total_predictions']}")
    print(f"Total anomalies: {summary['total_anomalies']}")
    print(f"High-risk components: {summary['high_risk_components']}")
    
    # Get active alerts
    alerts = ai_predictor.get_active_alerts()
    print(f"\\nAI Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"- {alert['type']} ({alert['severity']}): {alert['message']}")
        if alert['type'] == 'failure_prediction':
            print(f"  Time to failure: {alert.get('time_to_failure', 'unknown')} hours")


async def main():
    """Run comprehensive pipeline guard demonstration."""
    print("Quantum HVAC Self-Healing Pipeline Guard Demo")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        await demo_basic_pipeline_guard()
        await demo_advanced_recovery()
        await demo_metrics_and_monitoring()
        await demo_security_monitoring()
        await demo_performance_optimization()
        await demo_ai_prediction()
        
        print("\\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\\nThe self-healing pipeline guard provides:")
        print("✓ Automated health monitoring and recovery")
        print("✓ Advanced recovery strategies with machine learning")
        print("✓ Comprehensive metrics collection and alerting")
        print("✓ Security monitoring and threat detection")
        print("✓ Performance optimization and auto-scaling")
        print("✓ AI-powered failure prediction and prevention")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())