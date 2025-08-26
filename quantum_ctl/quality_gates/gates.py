"""Core Quality Gate Implementations"""

import asyncio
import os
import subprocess
import ast
import sys
from pathlib import Path
from typing import Dict, Any, List
import json

from .base import QualityGate, GateResult
from .config import QualityGateConfig
from datetime import datetime


class TestCoverageGate(QualityGate):
    """Test coverage quality gate"""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("test_coverage", config.to_dict())
        self.min_coverage = config.min_test_coverage
        
    async def execute(self, context: Dict[str, Any]) -> GateResult:
        """Execute test coverage analysis"""
        messages = []
        
        try:
            # Run pytest with coverage
            cmd = [
                sys.executable, "-m", "pytest",
                "--cov=quantum_ctl",
                "--cov-report=json",
                "--cov-report=term-missing",
                "--tb=short",
                "-x"  # Stop at first failure
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.get("project_root", ".")
            )
            
            stdout, stderr = await proc.communicate()
            
            # Parse coverage report
            coverage_file = Path("coverage.json")
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                
                passed = total_coverage >= self.min_coverage
                messages.append(f"Test coverage: {total_coverage:.1f}% (required: {self.min_coverage}%)")
                
                if not passed:
                    missing_files = []
                    for file_path, file_data in coverage_data.get("files", {}).items():
                        file_coverage = file_data.get("summary", {}).get("percent_covered", 0.0)
                        if file_coverage < self.min_coverage:
                            missing_files.append(f"{file_path}: {file_coverage:.1f}%")
                    
                    if missing_files:
                        messages.append(f"Files below threshold: {', '.join(missing_files[:5])}")
                
                return GateResult(
                    gate_name=self.name,
                    passed=passed,
                    score=min(total_coverage, 100.0),
                    threshold=self.min_coverage,
                    metrics={
                        "total_coverage": total_coverage,
                        "files_analyzed": len(coverage_data.get("files", {})),
                        "test_result": proc.returncode == 0
                    },
                    messages=messages,
                    execution_time_ms=0.0,
                    timestamp=datetime.utcnow()
                )
            else:
                messages.append("Coverage report not found")
                return GateResult(
                    gate_name=self.name,
                    passed=False,
                    score=0.0,
                    threshold=self.min_coverage,
                    metrics={"error": "coverage_report_missing"},
                    messages=messages,
                    execution_time_ms=0.0,
                    timestamp=datetime.utcnow()
                )
                
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                threshold=self.min_coverage,
                metrics={"error": str(e)},
                messages=[f"Coverage analysis failed: {e}"],
                execution_time_ms=0.0,
                timestamp=datetime.utcnow()
            )


class CodeQualityGate(QualityGate):
    """Code quality gate using static analysis"""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("code_quality", config.to_dict())
        self.max_complexity = config.max_complexity
        
    async def execute(self, context: Dict[str, Any]) -> GateResult:
        """Execute code quality checks"""
        messages = []
        quality_score = 100.0
        
        try:
            # Run flake8 for style checks
            flake8_proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "flake8", "quantum_ctl",
                "--count", "--statistics", "--format=json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.get("project_root", ".")
            )
            
            stdout, stderr = await flake8_proc.communicate()
            
            # Parse flake8 output
            violations = 0
            if flake8_proc.returncode != 0:
                lines = stdout.decode().strip().split('\n') if stdout else []
                violations = len([line for line in lines if line.strip()])
                quality_score = max(0, 100 - (violations * 2))  # Deduct 2 points per violation
                messages.append(f"Style violations found: {violations}")
            
            # Check complexity
            complexity_issues = await self._check_complexity(context)
            if complexity_issues:
                quality_score = max(0, quality_score - (len(complexity_issues) * 5))
                messages.extend(complexity_issues[:3])  # Show first 3 issues
            
            # Check type hints
            type_hint_coverage = await self._check_type_hints(context)
            if type_hint_coverage < 80:
                quality_score = max(0, quality_score - (80 - type_hint_coverage))
                messages.append(f"Type hint coverage: {type_hint_coverage:.1f}% (recommended: 80%+)")
            
            passed = quality_score >= 75.0  # Minimum quality threshold
            
            return GateResult(
                gate_name=self.name,
                passed=passed,
                score=quality_score,
                threshold=75.0,
                metrics={
                    "style_violations": violations,
                    "complexity_issues": len(complexity_issues),
                    "type_hint_coverage": type_hint_coverage
                },
                messages=messages,
                execution_time_ms=0.0,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                threshold=75.0,
                metrics={"error": str(e)},
                messages=[f"Code quality check failed: {e}"],
                execution_time_ms=0.0,
                timestamp=datetime.utcnow()
            )
    
    async def _check_complexity(self, context: Dict[str, Any]) -> List[str]:
        """Check cyclomatic complexity"""
        issues = []
        project_root = Path(context.get("project_root", "."))
        
        for py_file in project_root.glob("quantum_ctl/**/*.py"):
            if py_file.name.startswith("test_"):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calculate_complexity(node)
                        if complexity > self.max_complexity:
                            issues.append(f"{py_file.name}:{node.name} complexity {complexity} > {self.max_complexity}")
                            
            except Exception:
                continue  # Skip files with parse errors
                
        return issues
    
    def _calculate_complexity(self, node) -> int:
        """Simple cyclomatic complexity calculation"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1
                
        return complexity
    
    async def _check_type_hints(self, context: Dict[str, Any]) -> float:
        """Check type hint coverage"""
        project_root = Path(context.get("project_root", "."))
        total_functions = 0
        typed_functions = 0
        
        for py_file in project_root.glob("quantum_ctl/**/*.py"):
            if py_file.name.startswith("test_"):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name.startswith("_"):  # Skip private functions
                            continue
                            
                        total_functions += 1
                        
                        # Check if function has type hints
                        has_return_type = node.returns is not None
                        has_arg_types = any(arg.annotation is not None for arg in node.args.args)
                        
                        if has_return_type or has_arg_types:
                            typed_functions += 1
                            
            except Exception:
                continue
                
        if total_functions == 0:
            return 100.0
            
        return (typed_functions / total_functions) * 100.0


class SecurityGate(QualityGate):
    """Security vulnerability scanning gate"""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("security", config.to_dict())
        self.enabled = config.security_scan_enabled
        
    async def execute(self, context: Dict[str, Any]) -> GateResult:
        """Execute security analysis"""
        messages = []
        
        try:
            # Check for common security issues
            security_issues = await self._scan_security_issues(context)
            
            high_severity = len([i for i in security_issues if i.get("severity") == "high"])
            medium_severity = len([i for i in security_issues if i.get("severity") == "medium"])
            
            # Security scoring
            security_score = max(0, 100 - (high_severity * 20 + medium_severity * 10))
            passed = high_severity == 0 and security_score >= 70
            
            if security_issues:
                messages.append(f"Security issues: {high_severity} high, {medium_severity} medium")
                for issue in security_issues[:3]:
                    messages.append(f"  - {issue['description']} ({issue['severity']})")
            else:
                messages.append("No security issues detected")
            
            return GateResult(
                gate_name=self.name,
                passed=passed,
                score=security_score,
                threshold=70.0,
                metrics={
                    "high_severity_issues": high_severity,
                    "medium_severity_issues": medium_severity,
                    "total_issues": len(security_issues)
                },
                messages=messages,
                execution_time_ms=0.0,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                threshold=70.0,
                metrics={"error": str(e)},
                messages=[f"Security scan failed: {e}"],
                execution_time_ms=0.0,
                timestamp=datetime.utcnow()
            )
    
    async def _scan_security_issues(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Basic security issue scanning"""
        issues = []
        project_root = Path(context.get("project_root", "."))
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password", "high"),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key", "high"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret", "high"),
            (r'token\s*=\s*["\'][^"\']{20,}["\']', "Hardcoded token", "high"),
        ]
        
        import re
        
        for py_file in project_root.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, description, severity in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        issues.append({
                            "file": str(py_file),
                            "description": description,
                            "severity": severity,
                            "pattern": pattern
                        })
                        
            except Exception:
                continue
        
        # Check for insecure imports
        insecure_imports = ["pickle", "marshal", "shelve", "eval", "exec"]
        
        for py_file in project_root.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in insecure_imports:
                                issues.append({
                                    "file": str(py_file),
                                    "description": f"Potentially insecure import: {alias.name}",
                                    "severity": "medium"
                                })
                                
            except Exception:
                continue
                
        return issues


class PerformanceGate(QualityGate):
    """Performance benchmarking gate"""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("performance", config.to_dict())
        self.max_response_time = config.max_api_response_time_ms
        
    async def execute(self, context: Dict[str, Any]) -> GateResult:
        """Execute performance benchmarks"""
        messages = []
        
        try:
            # Basic performance metrics
            performance_metrics = await self._run_performance_tests(context)
            
            avg_response_time = performance_metrics.get("avg_response_time_ms", 0)
            memory_usage = performance_metrics.get("memory_usage_mb", 0)
            
            # Performance scoring
            response_time_score = max(0, 100 - ((avg_response_time - self.max_response_time) / 10))
            memory_score = max(0, 100 - (memory_usage / 10))
            performance_score = (response_time_score + memory_score) / 2
            
            passed = avg_response_time <= self.max_response_time and performance_score >= 70
            
            messages.append(f"Average response time: {avg_response_time:.1f}ms (limit: {self.max_response_time}ms)")
            messages.append(f"Memory usage: {memory_usage:.1f}MB")
            
            return GateResult(
                gate_name=self.name,
                passed=passed,
                score=performance_score,
                threshold=70.0,
                metrics=performance_metrics,
                messages=messages,
                execution_time_ms=0.0,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                threshold=70.0,
                metrics={"error": str(e)},
                messages=[f"Performance test failed: {e}"],
                execution_time_ms=0.0,
                timestamp=datetime.utcnow()
            )
    
    async def _run_performance_tests(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Run basic performance tests"""
        import time
        import psutil
        
        # Simulate basic operations
        start_time = time.time()
        
        # Basic import test
        try:
            import quantum_ctl
            from quantum_ctl.core import controller
        except ImportError:
            pass
        
        end_time = time.time()
        
        # Get memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "avg_response_time_ms": (end_time - start_time) * 1000,
            "memory_usage_mb": memory_info.rss / (1024 * 1024),
            "cpu_percent": process.cpu_percent()
        }


class DocumentationGate(QualityGate):
    """Documentation coverage gate"""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("documentation", config.to_dict())
        self.min_docstring_coverage = config.min_docstring_coverage
        
    async def execute(self, context: Dict[str, Any]) -> GateResult:
        """Execute documentation coverage analysis"""
        messages = []
        
        try:
            docstring_coverage = await self._check_docstring_coverage(context)
            api_docs_exist = await self._check_api_documentation(context)
            
            doc_score = docstring_coverage
            if not api_docs_exist:
                doc_score = max(0, doc_score - 20)
                messages.append("API documentation missing")
            
            passed = docstring_coverage >= self.min_docstring_coverage and api_docs_exist
            
            messages.append(f"Docstring coverage: {docstring_coverage:.1f}% (required: {self.min_docstring_coverage}%)")
            
            return GateResult(
                gate_name=self.name,
                passed=passed,
                score=doc_score,
                threshold=self.min_docstring_coverage,
                metrics={
                    "docstring_coverage": docstring_coverage,
                    "api_docs_exist": api_docs_exist
                },
                messages=messages,
                execution_time_ms=0.0,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                threshold=self.min_docstring_coverage,
                metrics={"error": str(e)},
                messages=[f"Documentation check failed: {e}"],
                execution_time_ms=0.0,
                timestamp=datetime.utcnow()
            )
    
    async def _check_docstring_coverage(self, context: Dict[str, Any]) -> float:
        """Check docstring coverage"""
        project_root = Path(context.get("project_root", "."))
        total_functions = 0
        documented_functions = 0
        
        for py_file in project_root.glob("quantum_ctl/**/*.py"):
            if py_file.name.startswith("test_"):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if hasattr(node, 'name') and node.name.startswith("_"):
                            continue
                            
                        total_functions += 1
                        
                        # Check if has docstring
                        if (node.body and 
                            isinstance(node.body[0], ast.Expr) and 
                            isinstance(node.body[0].value, ast.Str)):
                            documented_functions += 1
                            
            except Exception:
                continue
                
        if total_functions == 0:
            return 100.0
            
        return (documented_functions / total_functions) * 100.0
    
    async def _check_api_documentation(self, context: Dict[str, Any]) -> bool:
        """Check if API documentation exists"""
        project_root = Path(context.get("project_root", "."))
        
        # Check for common documentation files
        doc_files = [
            "docs/api_reference.md",
            "docs/README.md",
            "README.md"
        ]
        
        for doc_file in doc_files:
            if (project_root / doc_file).exists():
                return True
                
        return False