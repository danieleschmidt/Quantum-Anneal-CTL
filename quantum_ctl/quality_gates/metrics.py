"""Quality Metrics Collection and Analysis"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from datetime import datetime
import statistics

from .base import GateResult


@dataclass 
class QualityMetrics:
    """Comprehensive quality metrics"""
    
    # Coverage metrics
    test_coverage: float = 0.0
    docstring_coverage: float = 0.0
    
    # Quality metrics  
    code_quality_score: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    
    # Detailed metrics
    style_violations: int = 0
    security_issues: int = 0
    high_severity_security: int = 0
    complexity_issues: int = 0
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Execution metrics
    total_execution_time_ms: float = 0.0
    gates_executed: int = 0
    gates_passed: int = 0
    gates_failed: int = 0
    
    # Trends (for historical comparison)
    trends: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_results(cls, results: List[GateResult]) -> "QualityMetrics":
        """Create metrics from gate results"""
        metrics = cls()
        
        if not results:
            return metrics
            
        metrics.gates_executed = len(results)
        metrics.gates_passed = len([r for r in results if r.passed])
        metrics.gates_failed = len([r for r in results if not r.passed])
        metrics.total_execution_time_ms = sum(r.execution_time_ms for r in results)
        
        # Extract specific metrics from each gate
        for result in results:
            if result.gate_name == "test_coverage":
                metrics.test_coverage = result.metrics.get("total_coverage", 0.0)
                
            elif result.gate_name == "code_quality":
                metrics.code_quality_score = result.score
                metrics.style_violations = result.metrics.get("style_violations", 0)
                metrics.complexity_issues = result.metrics.get("complexity_issues", 0)
                
            elif result.gate_name == "security":
                metrics.security_score = result.score
                metrics.high_severity_security = result.metrics.get("high_severity_issues", 0)
                metrics.security_issues = result.metrics.get("total_issues", 0)
                
            elif result.gate_name == "performance":
                metrics.performance_score = result.score
                metrics.avg_response_time_ms = result.metrics.get("avg_response_time_ms", 0.0)
                metrics.memory_usage_mb = result.metrics.get("memory_usage_mb", 0.0)
                metrics.cpu_usage_percent = result.metrics.get("cpu_percent", 0.0)
                
            elif result.gate_name == "documentation":
                metrics.docstring_coverage = result.metrics.get("docstring_coverage", 0.0)
                
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "coverage": {
                "test_coverage": self.test_coverage,
                "docstring_coverage": self.docstring_coverage
            },
            "quality": {
                "code_quality_score": self.code_quality_score,
                "security_score": self.security_score,
                "performance_score": self.performance_score,
                "overall_score": self.calculate_overall_score()
            },
            "issues": {
                "style_violations": self.style_violations,
                "security_issues": self.security_issues,
                "high_severity_security": self.high_severity_security,
                "complexity_issues": self.complexity_issues
            },
            "performance": {
                "avg_response_time_ms": self.avg_response_time_ms,
                "memory_usage_mb": self.memory_usage_mb,
                "cpu_usage_percent": self.cpu_usage_percent
            },
            "execution": {
                "total_execution_time_ms": self.total_execution_time_ms,
                "gates_executed": self.gates_executed,
                "gates_passed": self.gates_passed,
                "gates_failed": self.gates_failed,
                "success_rate": self.calculate_success_rate()
            },
            "trends": self.trends,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def calculate_overall_score(self) -> float:
        """Calculate weighted overall quality score"""
        scores = []
        weights = []
        
        if self.test_coverage > 0:
            scores.append(self.test_coverage)
            weights.append(0.25)  # 25% weight for test coverage
            
        if self.code_quality_score > 0:
            scores.append(self.code_quality_score)
            weights.append(0.25)  # 25% weight for code quality
            
        if self.security_score > 0:
            scores.append(self.security_score)
            weights.append(0.25)  # 25% weight for security
            
        if self.performance_score > 0:
            scores.append(self.performance_score)
            weights.append(0.15)  # 15% weight for performance
            
        if self.docstring_coverage > 0:
            scores.append(self.docstring_coverage)
            weights.append(0.10)  # 10% weight for documentation
            
        if not scores:
            return 0.0
            
        # Calculate weighted average
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def calculate_success_rate(self) -> float:
        """Calculate gate success rate"""
        if self.gates_executed == 0:
            return 0.0
        return (self.gates_passed / self.gates_executed) * 100.0
    
    def get_quality_grade(self) -> str:
        """Get letter grade based on overall score"""
        score = self.calculate_overall_score()
        
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
    
    def get_improvement_suggestions(self) -> List[str]:
        """Get suggestions for improvement"""
        suggestions = []
        
        if self.test_coverage < 80:
            suggestions.append(f"Increase test coverage from {self.test_coverage:.1f}% to 80%+")
            
        if self.high_severity_security > 0:
            suggestions.append(f"Address {self.high_severity_security} high-severity security issues")
            
        if self.style_violations > 10:
            suggestions.append(f"Fix {self.style_violations} code style violations")
            
        if self.complexity_issues > 5:
            suggestions.append(f"Refactor {self.complexity_issues} complex functions")
            
        if self.avg_response_time_ms > 200:
            suggestions.append(f"Optimize response time from {self.avg_response_time_ms:.1f}ms to under 200ms")
            
        if self.docstring_coverage < 70:
            suggestions.append(f"Improve documentation coverage from {self.docstring_coverage:.1f}% to 70%+")
            
        return suggestions
    
    def compare_with_previous(self, previous_metrics: "QualityMetrics") -> Dict[str, Any]:
        """Compare with previous metrics to show trends"""
        comparison = {
            "improvements": [],
            "regressions": [],
            "deltas": {}
        }
        
        # Define comparison fields
        comparison_fields = {
            "test_coverage": "Test Coverage",
            "code_quality_score": "Code Quality",
            "security_score": "Security Score", 
            "performance_score": "Performance Score",
            "docstring_coverage": "Documentation Coverage"
        }
        
        for field, label in comparison_fields.items():
            current_value = getattr(self, field, 0.0)
            previous_value = getattr(previous_metrics, field, 0.0)
            delta = current_value - previous_value
            
            comparison["deltas"][field] = {
                "current": current_value,
                "previous": previous_value,
                "delta": delta,
                "delta_percent": (delta / previous_value * 100) if previous_value > 0 else 0
            }
            
            if delta > 1.0:  # Improvement threshold
                comparison["improvements"].append(f"{label}: +{delta:.1f} points")
            elif delta < -1.0:  # Regression threshold
                comparison["regressions"].append(f"{label}: {delta:.1f} points")
                
        return comparison