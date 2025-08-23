#!/usr/bin/env python3
"""
Generation 2 Validation System
Comprehensive testing and validation of robust production systems
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_ctl.resilience.production_guard_system import (
    ProductionGuardSystem, SecurityError, SystemHealthStatus
)
from quantum_ctl.breakthrough.autonomous_optimization_engine import (
    AutonomousOptimizationEngine
)

class Generation2Validator:
    """Comprehensive validation for Generation 2 robustness"""
    
    def __init__(self):
        self.guard_system = ProductionGuardSystem()
        self.autonomous_engine = AutonomousOptimizationEngine()
        self.test_results = []
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all Generation 2 validation tests"""
        
        print("🛡️ GENERATION 2 VALIDATION: MAKE IT ROBUST")
        print("=" * 60)
        
        # Start production guard
        await self.guard_system.start_guard_system()
        await asyncio.sleep(2)  # Allow systems to initialize
        
        validation_results = {
            "security_tests": await self._test_security_measures(),
            "error_handling_tests": await self._test_error_handling(),
            "health_monitoring_tests": await self._test_health_monitoring(),
            "production_resilience_tests": await self._test_production_resilience(),
            "integration_tests": await self._test_autonomous_integration()
        }
        
        # Overall assessment
        overall_score = self._calculate_overall_score(validation_results)
        
        print(f"\n🎯 GENERATION 2 VALIDATION COMPLETE")
        print(f"Overall Robustness Score: {overall_score:.1f}/100")
        
        if overall_score >= 85:
            print("✅ GENERATION 2: ROBUST - PASSED")
            print("🚀 Ready for Generation 3: Make it Scale")
        else:
            print("⚠️ GENERATION 2: NEEDS IMPROVEMENT")
            print("🔧 Additional robustness work required")
        
        return {
            "validation_results": validation_results,
            "overall_score": overall_score,
            "generation_2_status": "PASSED" if overall_score >= 85 else "NEEDS_WORK",
            "timestamp": time.time()
        }
    
    async def _test_security_measures(self) -> Dict[str, Any]:
        """Test comprehensive security measures"""
        print("\n🔒 Testing Security Measures...")
        
        security_tests = {
            "input_validation": await self._test_input_validation(),
            "rate_limiting": await self._test_rate_limiting(),
            "encryption": await self._test_encryption(),
            "anomaly_detection": await self._test_anomaly_detection()
        }
        
        passed_tests = sum(1 for result in security_tests.values() if result.get("passed", False))
        score = (passed_tests / len(security_tests)) * 100
        
        print(f"   Security Score: {score:.1f}/100 ({passed_tests}/{len(security_tests)} tests passed)")
        
        return {
            "tests": security_tests,
            "score": score,
            "passed": score >= 80
        }
    
    async def _test_input_validation(self) -> Dict[str, Any]:
        """Test input validation robustness"""
        
        # Valid request
        valid_request = {
            "temperatures": [22.0, 21.5, 23.0],
            "occupancy": [0.8, 0.6, 0.9],
            "prediction_horizon": 24
        }
        
        # Test valid input
        is_valid, alert = self.guard_system.security_validator.validate_quantum_request(
            valid_request, "test_user", "192.168.1.100"
        )
        
        if not is_valid:
            return {"passed": False, "reason": "Valid input rejected"}
        
        # Test invalid inputs
        invalid_tests = [
            # Missing required field
            {"occupancy": [0.8], "prediction_horizon": 24},
            # Invalid temperature range
            {"temperatures": [-100, 200], "occupancy": [0.5, 0.5], "prediction_horizon": 24},
            # Invalid occupancy range  
            {"temperatures": [22, 23], "occupancy": [-1, 2], "prediction_horizon": 24},
            # Invalid horizon
            {"temperatures": [22], "occupancy": [0.5], "prediction_horizon": 200}
        ]
        
        rejected_count = 0
        for invalid_request in invalid_tests:
            is_valid, _ = self.guard_system.security_validator.validate_quantum_request(
                invalid_request, "test_user", "192.168.1.100"
            )
            if not is_valid:
                rejected_count += 1
        
        success_rate = rejected_count / len(invalid_tests)
        return {
            "passed": success_rate >= 0.8,
            "success_rate": success_rate,
            "valid_rejected": rejected_count,
            "total_invalid": len(invalid_tests)
        }
    
    async def _test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting functionality"""
        
        # Simulate many requests from same user
        user_id = "rate_test_user"
        source_ip = "192.168.1.200"
        
        valid_request = {
            "temperatures": [22.0],
            "occupancy": [0.5],
            "prediction_horizon": 12
        }
        
        # Make many requests quickly
        requests_allowed = 0
        for i in range(150):  # Above rate limit
            is_valid, alert = self.guard_system.security_validator.validate_quantum_request(
                valid_request, user_id, source_ip
            )
            if is_valid:
                requests_allowed += 1
        
        # Should eventually be rate limited
        rate_limiting_working = requests_allowed < 120  # Should block some requests
        
        return {
            "passed": rate_limiting_working,
            "requests_allowed": requests_allowed,
            "rate_limiting_active": rate_limiting_working
        }
    
    async def _test_encryption(self) -> Dict[str, Any]:
        """Test data encryption functionality"""
        
        test_data = {
            "sensitive_optimization": "quantum_parameters",
            "building_data": {"zones": 5, "control_points": 20}
        }
        
        try:
            # Encrypt data
            encrypted = self.guard_system.security_validator.encrypt_quantum_data(test_data)
            
            # Decrypt data  
            decrypted = self.guard_system.security_validator.decrypt_quantum_data(encrypted)
            
            # Verify integrity
            data_integrity = (decrypted == test_data)
            
            return {
                "passed": data_integrity,
                "encryption_works": isinstance(encrypted, bytes),
                "decryption_works": decrypted == test_data,
                "data_integrity": data_integrity
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    async def _test_anomaly_detection(self) -> Dict[str, Any]:
        """Test anomaly detection capabilities"""
        
        # Normal request
        normal_request = {
            "temperatures": [22.0, 21.5, 23.0],
            "occupancy": [0.8, 0.6, 0.9],
            "prediction_horizon": 24
        }
        
        # Anomalous request (extreme values)
        anomalous_request = {
            "temperatures": [50.0, -20.0, 80.0, 15.0, 45.0] * 10,  # Extreme temperatures
            "occupancy": [0.5] * 50,
            "prediction_horizon": 168  # 1 week horizon
        }
        
        # Test normal request
        is_valid_normal, _ = self.guard_system.security_validator.validate_quantum_request(
            normal_request, "anomaly_test_user", "192.168.1.300"
        )
        
        # Test anomalous request
        is_valid_anomalous, alert = self.guard_system.security_validator.validate_quantum_request(
            anomalous_request, "anomaly_test_user", "192.168.1.300"
        )
        
        # Should accept normal, reject anomalous
        detection_working = is_valid_normal and not is_valid_anomalous
        
        return {
            "passed": detection_working,
            "normal_accepted": is_valid_normal,
            "anomaly_rejected": not is_valid_anomalous,
            "alert_generated": alert is not None if not is_valid_anomalous else False
        }
    
    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test comprehensive error handling"""
        print("\n🚨 Testing Error Handling...")
        
        error_tests = {
            "timeout_recovery": await self._test_timeout_recovery(),
            "invalid_data_handling": await self._test_invalid_data_handling(),
            "system_overload_handling": await self._test_system_overload()
        }
        
        passed_tests = sum(1 for result in error_tests.values() if result.get("passed", False))
        score = (passed_tests / len(error_tests)) * 100
        
        print(f"   Error Handling Score: {score:.1f}/100 ({passed_tests}/{len(error_tests)} tests passed)")
        
        return {
            "tests": error_tests,
            "score": score,
            "passed": score >= 70
        }
    
    async def _test_timeout_recovery(self) -> Dict[str, Any]:
        """Test timeout recovery mechanisms"""
        
        async def slow_operation(request):
            await asyncio.sleep(10)  # Simulate slow operation
            return {"result": "success"}
        
        request = {
            "temperatures": [22.0],
            "occupancy": [0.5],
            "prediction_horizon": 24
        }
        
        try:
            # This should timeout and be handled gracefully
            result = await self.guard_system.secure_quantum_operation(
                slow_operation, request, "timeout_test_user", "192.168.1.400"
            )
            return {"passed": False, "reason": "Operation should have timed out"}
            
        except asyncio.TimeoutError:
            # Expected timeout
            return {"passed": True, "timeout_handled": True}
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def _test_invalid_data_handling(self) -> Dict[str, Any]:
        """Test handling of invalid data"""
        
        async def dummy_operation(request):
            return {"result": "processed"}
        
        invalid_request = {
            "temperatures": "not_a_list",  # Invalid data type
            "occupancy": [0.5],
            "prediction_horizon": "invalid"
        }
        
        try:
            # Should be rejected by security validation
            result = await self.guard_system.secure_quantum_operation(
                dummy_operation, invalid_request, "invalid_test_user", "192.168.1.500"
            )
            return {"passed": False, "reason": "Invalid data should be rejected"}
            
        except SecurityError:
            # Expected security error
            return {"passed": True, "security_error_raised": True}
        except Exception as e:
            return {"passed": False, "unexpected_error": str(e)}
    
    async def _test_system_overload(self) -> Dict[str, Any]:
        """Test system overload handling"""
        
        # Simulate system under high load
        original_maintenance = self.guard_system.maintenance_mode
        self.guard_system.maintenance_mode = True
        
        async def dummy_operation(request):
            return {"result": "processed"}
        
        request = {
            "temperatures": [22.0],
            "occupancy": [0.5], 
            "prediction_horizon": 24
        }
        
        try:
            # Should be rejected due to maintenance mode
            result = await self.guard_system.secure_quantum_operation(
                dummy_operation, request, "overload_test_user", "192.168.1.600"
            )
            return {"passed": False, "reason": "Operation should be rejected in maintenance mode"}
            
        except RuntimeError as e:
            if "maintenance mode" in str(e):
                return {"passed": True, "maintenance_mode_respected": True}
            else:
                return {"passed": False, "wrong_error": str(e)}
        except Exception as e:
            return {"passed": False, "unexpected_error": str(e)}
        finally:
            # Restore original state
            self.guard_system.maintenance_mode = original_maintenance
    
    async def _test_health_monitoring(self) -> Dict[str, Any]:
        """Test health monitoring system"""
        print("\n💓 Testing Health Monitoring...")
        
        # Collect health metrics
        metrics = self.guard_system.health_monitor.collect_health_metrics()
        status = self.guard_system.health_monitor.assess_system_health(metrics)
        
        health_tests = {
            "metrics_collection": metrics is not None,
            "status_assessment": status in SystemHealthStatus,
            "reasonable_metrics": (
                0 <= metrics.cpu_usage <= 100 and
                0 <= metrics.memory_usage <= 100 and
                0 <= metrics.disk_usage <= 100 and
                metrics.network_latency > 0
            ) if metrics else False
        }
        
        passed_tests = sum(1 for result in health_tests.values() if result)
        score = (passed_tests / len(health_tests)) * 100
        
        print(f"   Health Monitoring Score: {score:.1f}/100 ({passed_tests}/{len(health_tests)} tests passed)")
        
        return {
            "tests": health_tests,
            "score": score,
            "passed": score >= 80,
            "current_metrics": {
                "cpu_usage": metrics.cpu_usage if metrics else None,
                "memory_usage": metrics.memory_usage if metrics else None,
                "health_status": status.value if status else None
            }
        }
    
    async def _test_production_resilience(self) -> Dict[str, Any]:
        """Test overall production resilience"""
        print("\n🏭 Testing Production Resilience...")
        
        # Get guard system status
        guard_status = self.guard_system.get_guard_status()
        
        resilience_tests = {
            "guard_system_active": guard_status.get("guard_active", False),
            "not_in_maintenance": not guard_status.get("maintenance_mode", True),
            "health_monitoring_active": guard_status.get("system_health", {}).get("status") != "unknown",
            "security_monitoring_active": isinstance(guard_status.get("security", {}), dict),
            "production_ready": guard_status.get("production_ready", False)
        }
        
        passed_tests = sum(1 for result in resilience_tests.values() if result)
        score = (passed_tests / len(resilience_tests)) * 100
        
        print(f"   Production Resilience Score: {score:.1f}/100 ({passed_tests}/{len(resilience_tests)} tests passed)")
        
        return {
            "tests": resilience_tests,
            "score": score,
            "passed": score >= 85,
            "guard_status": guard_status
        }
    
    async def _test_autonomous_integration(self) -> Dict[str, Any]:
        """Test integration with autonomous optimization engine"""
        print("\n🤖 Testing Autonomous Integration...")
        
        # Test scenario
        test_scenario = {
            "temperatures": [22.0, 21.5, 23.0],
            "occupancy": [0.8, 0.6, 0.9],
            "prediction_horizon": 24,
            "weather_forecast": {"external_temp": 20.0, "solar_radiation": 500}
        }
        
        try:
            # Run autonomous optimization
            result = await self.autonomous_engine.optimize_autonomous(test_scenario)
            
            integration_tests = {
                "autonomous_engine_works": result is not None,
                "returns_metrics": "autonomous_metrics" in result,
                "strategy_selected": "strategy_used" in result,
                "optimization_completed": "optimization_time" in result,
                "breakthrough_detection": "breakthrough_detected" in result
            }
            
            passed_tests = sum(1 for result in integration_tests.values() if result)
            score = (passed_tests / len(integration_tests)) * 100
            
            print(f"   Autonomous Integration Score: {score:.1f}/100 ({passed_tests}/{len(integration_tests)} tests passed)")
            
            return {
                "tests": integration_tests,
                "score": score,
                "passed": score >= 80,
                "sample_result": {
                    "strategy_used": result.get("strategy_used"),
                    "optimization_time": result.get("optimization_time"),
                    "breakthrough_detected": result.get("breakthrough_detected")
                }
            }
            
        except Exception as e:
            return {
                "tests": {"autonomous_engine_error": True},
                "score": 0,
                "passed": False,
                "error": str(e)
            }
    
    def _calculate_overall_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall Generation 2 robustness score"""
        
        scores = []
        weights = {
            "security_tests": 0.3,
            "error_handling_tests": 0.25, 
            "health_monitoring_tests": 0.2,
            "production_resilience_tests": 0.15,
            "integration_tests": 0.1
        }
        
        total_weight = 0
        for test_category, weight in weights.items():
            if test_category in validation_results:
                score = validation_results[test_category].get("score", 0)
                scores.append(score * weight)
                total_weight += weight
        
        return sum(scores) / total_weight if total_weight > 0 else 0

async def main():
    """Run Generation 2 validation"""
    
    validator = Generation2Validator()
    
    try:
        results = await validator.run_comprehensive_validation()
        
        # Save results
        with open("generation_2_validation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Validation results saved to: generation_2_validation_results.json")
        
        return results
        
    except Exception as e:
        print(f"❌ Generation 2 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(main())