"""Base Quality Gate Implementation"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a quality gate execution"""
    gate_name: str
    passed: bool
    score: float  # 0-100
    threshold: float
    metrics: Dict[str, Any]
    messages: List[str]
    execution_time_ms: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "score": self.score,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "messages": self.messages,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat()
        }


class QualityGate(ABC):
    """Base class for all quality gates"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)
        
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> GateResult:
        """Execute the quality gate"""
        pass
    
    def is_enabled(self) -> bool:
        """Check if gate is enabled"""
        return self.enabled
    
    async def run(self, context: Dict[str, Any]) -> GateResult:
        """Run the gate with timing and error handling"""
        if not self.is_enabled():
            return GateResult(
                gate_name=self.name,
                passed=True,
                score=100.0,
                threshold=0.0,
                metrics={"status": "skipped"},
                messages=["Gate is disabled"],
                execution_time_ms=0.0,
                timestamp=datetime.utcnow()
            )
        
        start_time = asyncio.get_event_loop().time()
        try:
            result = await self.execute(context)
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            result.execution_time_ms = execution_time
            return result
            
        except Exception as e:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.error(f"Quality gate {self.name} failed with error: {e}")
            
            return GateResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                threshold=100.0,
                metrics={"error": str(e)},
                messages=[f"Gate execution failed: {e}"],
                execution_time_ms=execution_time,
                timestamp=datetime.utcnow()
            )