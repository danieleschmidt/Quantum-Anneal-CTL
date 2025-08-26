"""Quality Gates Configuration"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
import os


@dataclass
class QualityGateConfig:
    """Configuration for quality gates system"""
    
    # Test coverage requirements
    min_test_coverage: float = 85.0
    coverage_fail_under: float = 80.0
    
    # Performance thresholds
    max_api_response_time_ms: int = 200
    max_memory_usage_mb: int = 512
    max_cpu_usage_percent: float = 80.0
    
    # Security settings
    security_scan_enabled: bool = True
    allowed_security_issues: List[str] = field(default_factory=list)
    
    # Code quality settings
    max_complexity: int = 10
    max_line_length: int = 88
    enforce_type_hints: bool = True
    
    # Documentation requirements
    min_docstring_coverage: float = 80.0
    require_api_docs: bool = True
    
    # CI/CD integration
    fail_fast: bool = True
    parallel_execution: bool = True
    output_format: str = "json"
    
    # Reporting
    generate_html_report: bool = True
    report_output_dir: str = "quality_reports"
    
    @classmethod
    def from_env(cls) -> "QualityGateConfig":
        """Create config from environment variables"""
        return cls(
            min_test_coverage=float(os.getenv("QG_MIN_COVERAGE", "85.0")),
            max_api_response_time_ms=int(os.getenv("QG_MAX_RESPONSE_MS", "200")),
            max_memory_usage_mb=int(os.getenv("QG_MAX_MEMORY_MB", "512")),
            security_scan_enabled=os.getenv("QG_SECURITY_SCAN", "true").lower() == "true",
            fail_fast=os.getenv("QG_FAIL_FAST", "true").lower() == "true",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "min_test_coverage": self.min_test_coverage,
            "coverage_fail_under": self.coverage_fail_under,
            "max_api_response_time_ms": self.max_api_response_time_ms,
            "max_memory_usage_mb": self.max_memory_usage_mb,
            "max_cpu_usage_percent": self.max_cpu_usage_percent,
            "security_scan_enabled": self.security_scan_enabled,
            "max_complexity": self.max_complexity,
            "max_line_length": self.max_line_length,
            "enforce_type_hints": self.enforce_type_hints,
            "min_docstring_coverage": self.min_docstring_coverage,
            "require_api_docs": self.require_api_docs,
            "fail_fast": self.fail_fast,
            "parallel_execution": self.parallel_execution,
            "output_format": self.output_format,
            "generate_html_report": self.generate_html_report,
            "report_output_dir": self.report_output_dir
        }