#!/usr/bin/env python3
"""
Standalone Quality Gates Execution (No Dependencies)
Progressive Quality Gates System - Minimal Bootstrap Version
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime


def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    return True


def install_dependencies():
    """Install minimal dependencies for quality gates"""
    print("📦 Installing minimal dependencies...")
    
    minimal_deps = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0", 
        "flake8>=6.0.0",
        "psutil>=5.9.0"
    ]
    
    try:
        for dep in minimal_deps:
            print(f"   Installing {dep}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], check=True, capture_output=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def run_test_coverage():
    """Run test coverage analysis"""
    print("\n🧪 Running Test Coverage Analysis...")
    
    try:
        # Run pytest with coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "--cov=quantum_ctl",
            "--cov-report=term-missing",
            "--cov-report=json",
            "--tb=short",
            "-v"
        ], capture_output=True, text=True, timeout=300)
        
        print(f"   Exit code: {result.returncode}")
        
        # Parse coverage
        coverage = 0.0
        try:
            if Path("coverage.json").exists():
                with open("coverage.json") as f:
                    coverage_data = json.load(f)
                coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
        except Exception:
            pass
        
        passed = result.returncode == 0 and coverage >= 70.0
        
        print(f"   Coverage: {coverage:.1f}%")
        print(f"   Status: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return {
            "gate_name": "test_coverage",
            "passed": passed,
            "score": min(coverage, 100.0),
            "metrics": {"coverage": coverage},
            "messages": [f"Test coverage: {coverage:.1f}%"]
        }
        
    except Exception as e:
        print(f"   Error: {e}")
        return {
            "gate_name": "test_coverage",
            "passed": False,
            "score": 0.0,
            "metrics": {"error": str(e)},
            "messages": [f"Test coverage failed: {e}"]
        }


def run_code_quality():
    """Run code quality analysis"""
    print("\n🔍 Running Code Quality Analysis...")
    
    try:
        # Run flake8
        result = subprocess.run([
            sys.executable, "-m", "flake8", 
            "quantum_ctl",
            "--count",
            "--statistics",
            "--max-line-length=88",
            "--ignore=E203,W503"
        ], capture_output=True, text=True, timeout=120)
        
        violations = 0
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip() and line[0].isdigit():
                    violations += int(line.split()[0])
        
        quality_score = max(0, 100 - (violations * 2))
        passed = quality_score >= 75
        
        print(f"   Style violations: {violations}")
        print(f"   Quality score: {quality_score:.1f}/100")
        print(f"   Status: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return {
            "gate_name": "code_quality",
            "passed": passed,
            "score": quality_score,
            "metrics": {"violations": violations},
            "messages": [f"Style violations: {violations}"]
        }
        
    except Exception as e:
        print(f"   Error: {e}")
        return {
            "gate_name": "code_quality", 
            "passed": False,
            "score": 0.0,
            "metrics": {"error": str(e)},
            "messages": [f"Code quality failed: {e}"]
        }


def run_security_scan():
    """Run basic security analysis"""
    print("\n🔒 Running Security Analysis...")
    
    try:
        import re
        
        security_issues = []
        
        # Scan for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret")
        ]
        
        for py_file in Path("quantum_ctl").rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern, description in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        # Skip obvious test/example patterns
                        if not any(test_word in content.lower() for test_word in ['test', 'example', 'dummy']):
                            security_issues.append(f"{description} in {py_file}")
            except Exception:
                continue
        
        security_score = max(0, 100 - len(security_issues) * 20)
        passed = len(security_issues) == 0
        
        print(f"   Security issues found: {len(security_issues)}")
        print(f"   Security score: {security_score:.1f}/100")  
        print(f"   Status: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        if security_issues:
            for issue in security_issues[:3]:
                print(f"   • {issue}")
        
        return {
            "gate_name": "security",
            "passed": passed,
            "score": security_score,
            "metrics": {"issues": len(security_issues)},
            "messages": security_issues[:3] if security_issues else ["No security issues found"]
        }
        
    except Exception as e:
        print(f"   Error: {e}")
        return {
            "gate_name": "security",
            "passed": False, 
            "score": 0.0,
            "metrics": {"error": str(e)},
            "messages": [f"Security scan failed: {e}"]
        }


def run_performance_test():
    """Run basic performance analysis"""
    print("\n⚡ Running Performance Analysis...")
    
    try:
        import psutil
        
        # Monitor system resources
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss / (1024 * 1024)
        
        # Simulate some work (import quantum_ctl modules if possible)
        try:
            # Basic module import test
            import importlib
            import quantum_ctl
        except Exception:
            # Fallback performance test
            time.sleep(0.1)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)
        
        execution_time_ms = (end_time - start_time) * 1000
        memory_usage_mb = end_memory - start_memory
        
        # Performance scoring
        time_score = max(0, 100 - max(0, execution_time_ms - 100) / 10)  # Penalty after 100ms
        memory_score = max(0, 100 - max(0, memory_usage_mb - 10))  # Penalty after 10MB
        performance_score = (time_score + memory_score) / 2
        
        passed = execution_time_ms < 1000 and memory_usage_mb < 100
        
        print(f"   Execution time: {execution_time_ms:.1f}ms")
        print(f"   Memory usage: {memory_usage_mb:.1f}MB")
        print(f"   Performance score: {performance_score:.1f}/100")
        print(f"   Status: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return {
            "gate_name": "performance",
            "passed": passed,
            "score": performance_score,
            "metrics": {
                "execution_time_ms": execution_time_ms,
                "memory_usage_mb": memory_usage_mb
            },
            "messages": [f"Execution: {execution_time_ms:.1f}ms, Memory: {memory_usage_mb:.1f}MB"]
        }
        
    except Exception as e:
        print(f"   Error: {e}")
        return {
            "gate_name": "performance",
            "passed": False,
            "score": 0.0,
            "metrics": {"error": str(e)},
            "messages": [f"Performance test failed: {e}"]
        }


def run_documentation_check():
    """Run documentation analysis"""
    print("\n📚 Running Documentation Analysis...")
    
    try:
        # Count Python files and docstrings
        total_functions = 0
        documented_functions = 0
        
        import ast
        
        for py_file in Path("quantum_ctl").rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if hasattr(node, 'name') and not node.name.startswith('_'):
                            total_functions += 1
                            
                            # Check for docstring
                            if (node.body and 
                                isinstance(node.body[0], ast.Expr) and 
                                isinstance(node.body[0].value, ast.Str)):
                                documented_functions += 1
                                
            except Exception:
                continue
        
        # Check for README and docs
        docs_exist = Path("README.md").exists() or Path("docs").exists()
        
        doc_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 100
        final_score = doc_coverage * (1.1 if docs_exist else 0.9)  # Bonus/penalty for project docs
        
        passed = doc_coverage >= 60 and docs_exist
        
        print(f"   Functions documented: {documented_functions}/{total_functions}")
        print(f"   Documentation coverage: {doc_coverage:.1f}%")
        print(f"   Project docs exist: {docs_exist}")
        print(f"   Documentation score: {final_score:.1f}/100")
        print(f"   Status: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return {
            "gate_name": "documentation",
            "passed": passed,
            "score": min(final_score, 100.0),
            "metrics": {
                "doc_coverage": doc_coverage,
                "total_functions": total_functions,
                "documented_functions": documented_functions,
                "project_docs_exist": docs_exist
            },
            "messages": [f"Documentation coverage: {doc_coverage:.1f}%"]
        }
        
    except Exception as e:
        print(f"   Error: {e}")
        return {
            "gate_name": "documentation",
            "passed": False,
            "score": 0.0,
            "metrics": {"error": str(e)},
            "messages": [f"Documentation check failed: {e}"]
        }


def main():
    """Main execution"""
    print("🔬 PROGRESSIVE QUALITY GATES - STANDALONE EXECUTION")
    print("=" * 80)
    print("🚀 Bootstrap Version - Minimal Dependencies")
    print("=" * 80)
    
    # Check environment
    if not check_python_version():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Continuing without full dependencies - some gates may be limited")
    
    print(f"\n🎯 Executing quality gates at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Execute all gates
    gates = [
        run_test_coverage,
        run_code_quality, 
        run_security_scan,
        run_performance_test,
        run_documentation_check
    ]
    
    results = []
    start_time = time.time()
    
    for gate_func in gates:
        try:
            result = gate_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Gate execution failed: {e}")
            results.append({
                "gate_name": "unknown",
                "passed": False,
                "score": 0.0,
                "metrics": {"error": str(e)},
                "messages": [f"Gate failed: {e}"]
            })
    
    execution_time = time.time() - start_time
    
    # Calculate overall results
    passed_gates = [r for r in results if r.get('passed', False)]
    failed_gates = [r for r in results if not r.get('passed', True)]
    
    total_score = sum(r.get('score', 0) for r in results)
    overall_score = total_score / len(results) if results else 0
    overall_passed = len(failed_gates) == 0
    
    # Print results
    print("\n" + "="*80)
    print("📊 PROGRESSIVE QUALITY GATES RESULTS")
    print("="*80)
    
    status_icon = "✅" if overall_passed else "❌"
    status_text = "PASSED" if overall_passed else "FAILED"
    
    print(f"\n{status_icon} OVERALL STATUS: {status_text}")
    print(f"📈 Quality Score: {overall_score:.1f}/100")
    print(f"⏱️  Execution Time: {execution_time:.1f}s")
    print(f"🎯 Gates Passed: {len(passed_gates)}/{len(results)}")
    
    print(f"\n📋 INDIVIDUAL GATE RESULTS:")
    print("-" * 80)
    
    for result in results:
        gate_icon = "✅" if result.get('passed', False) else "❌"
        gate_name = result.get('gate_name', 'unknown').replace('_', ' ').title()
        gate_score = result.get('score', 0)
        
        print(f"{gate_icon} {gate_name:<20} Score: {gate_score:>6.1f}/100")
        
        messages = result.get('messages', [])
        for message in messages[:1]:  # Show first message
            print(f"   └─ {message}")
    
    # Save results
    results_data = {
        "overall_passed": overall_passed,
        "overall_score": overall_score,
        "execution_time": execution_time,
        "timestamp": datetime.now().isoformat(),
        "gates": results,
        "summary": {
            "total_gates": len(results),
            "passed_gates": len(passed_gates),
            "failed_gates": len(failed_gates)
        }
    }
    
    results_file = f"quality_gates_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    print("="*80)
    
    if overall_passed:
        print("🎉 ALL QUALITY GATES PASSED!")
        print("✅ Code meets quality standards")
        return 0
    else:
        print("⚠️  QUALITY ISSUES DETECTED")
        print("❌ Review and address failed gates")
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Quality gates execution interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)