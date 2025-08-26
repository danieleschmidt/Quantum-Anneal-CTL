"""Quality Gate Runner - Orchestrates execution of all quality gates"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from .base import GateResult
from .config import QualityGateConfig
from .gates import (
    TestCoverageGate,
    CodeQualityGate, 
    SecurityGate,
    PerformanceGate,
    DocumentationGate
)
from .metrics import QualityMetrics
from .reporter import QualityReporter

logger = logging.getLogger(__name__)


class QualityGateRunner:
    """Orchestrates execution of quality gates"""
    
    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or QualityGateConfig()
        self.gates = self._initialize_gates()
        self.reporter = QualityReporter(self.config)
        
    def _initialize_gates(self) -> List:
        """Initialize all quality gates"""
        gates = []
        
        # Core quality gates
        gates.append(TestCoverageGate(self.config))
        gates.append(CodeQualityGate(self.config))
        gates.append(SecurityGate(self.config))
        gates.append(PerformanceGate(self.config))
        gates.append(DocumentationGate(self.config))
        
        return gates
    
    async def run_all_gates(
        self, 
        context: Optional[Dict[str, Any]] = None,
        gate_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results"""
        
        if context is None:
            context = {"project_root": "."}
            
        logger.info("Starting quality gate execution")
        start_time = datetime.utcnow()
        
        # Filter gates if specified
        gates_to_run = self.gates
        if gate_filter:
            gates_to_run = [gate for gate in self.gates if gate.name in gate_filter]
        
        # Run gates in parallel or sequential based on config
        if self.config.parallel_execution:
            results = await self._run_gates_parallel(gates_to_run, context)
        else:
            results = await self._run_gates_sequential(gates_to_run, context)
        
        # Calculate overall metrics
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        overall_result = self._calculate_overall_result(results, execution_time, start_time)
        
        # Generate reports
        await self.reporter.generate_reports(overall_result)
        
        logger.info(f"Quality gates completed in {execution_time:.1f}ms")
        logger.info(f"Overall result: {'PASSED' if overall_result['passed'] else 'FAILED'}")
        
        return overall_result
    
    async def _run_gates_parallel(
        self, 
        gates: List,
        context: Dict[str, Any]
    ) -> List[GateResult]:
        """Run gates in parallel"""
        
        tasks = []
        for gate in gates:
            if gate.is_enabled():
                task = asyncio.create_task(gate.run(context))
                tasks.append(task)
        
        if not tasks:
            return []
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        gate_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Gate execution failed: {result}")
                # Create error result
                error_result = GateResult(
                    gate_name="unknown",
                    passed=False,
                    score=0.0,
                    threshold=100.0,
                    metrics={"error": str(result)},
                    messages=[f"Gate failed: {result}"],
                    execution_time_ms=0.0,
                    timestamp=datetime.utcnow()
                )
                gate_results.append(error_result)
            else:
                gate_results.append(result)
                
        return gate_results
    
    async def _run_gates_sequential(
        self,
        gates: List,
        context: Dict[str, Any]
    ) -> List[GateResult]:
        """Run gates sequentially"""
        
        results = []
        for gate in gates:
            if gate.is_enabled():
                try:
                    result = await gate.run(context)
                    results.append(result)
                    
                    # Fail fast if enabled
                    if self.config.fail_fast and not result.passed:
                        logger.warning(f"Gate {gate.name} failed - stopping execution (fail_fast enabled)")
                        break
                        
                except Exception as e:
                    logger.error(f"Gate {gate.name} failed with exception: {e}")
                    if self.config.fail_fast:
                        break
                        
        return results
    
    def _calculate_overall_result(
        self,
        results: List[GateResult],
        execution_time_ms: float,
        start_time: datetime
    ) -> Dict[str, Any]:
        """Calculate overall quality gate result"""
        
        if not results:
            return {
                "passed": False,
                "score": 0.0,
                "execution_time_ms": execution_time_ms,
                "timestamp": start_time.isoformat(),
                "gates": [],
                "summary": {
                    "total_gates": 0,
                    "passed_gates": 0,
                    "failed_gates": 0,
                    "overall_score": 0.0
                }
            }
        
        passed_gates = [r for r in results if r.passed]
        failed_gates = [r for r in results if not r.passed]
        
        # Calculate weighted overall score
        total_score = sum(r.score for r in results)
        overall_score = total_score / len(results)
        
        # Overall pass/fail logic
        overall_passed = len(failed_gates) == 0
        
        return {
            "passed": overall_passed,
            "score": overall_score,
            "execution_time_ms": execution_time_ms,
            "timestamp": start_time.isoformat(),
            "gates": [result.to_dict() for result in results],
            "summary": {
                "total_gates": len(results),
                "passed_gates": len(passed_gates),
                "failed_gates": len(failed_gates),
                "overall_score": overall_score
            },
            "metrics": QualityMetrics.from_results(results).to_dict()
        }
    
    async def run_single_gate(
        self, 
        gate_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Run a single quality gate by name"""
        
        if context is None:
            context = {"project_root": "."}
            
        for gate in self.gates:
            if gate.name == gate_name and gate.is_enabled():
                return await gate.run(context)
                
        raise ValueError(f"Gate '{gate_name}' not found or disabled")
    
    def get_available_gates(self) -> List[str]:
        """Get list of available gate names"""
        return [gate.name for gate in self.gates if gate.is_enabled()]
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration"""
        validation_result = {
            "valid": True,
            "issues": [],
            "gates": []
        }
        
        for gate in self.gates:
            gate_info = {
                "name": gate.name,
                "enabled": gate.is_enabled(),
                "config": gate.config
            }
            validation_result["gates"].append(gate_info)
            
        return validation_result