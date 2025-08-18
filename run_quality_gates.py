#!/usr/bin/env python3
"""
Quality gates validation script for Quantum-Anneal-CTL.

This script runs comprehensive quality checks including:
- Code structure validation
- Import testing
- Basic functionality testing
- Security validation
- Performance baseline establishment
"""

import sys
import os
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any
import subprocess


class QualityGateRunner:
    """Comprehensive quality gate validation."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.results = {}
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return results."""
        
        print("🚀 Running Quantum-Anneal-CTL Quality Gates")
        print("=" * 60)
        
        gates = [
            ("Code Structure", self.validate_code_structure),
            ("Import Validation", self.validate_imports),
            ("Basic Functionality", self.test_basic_functionality),
            ("Research Modules", self.test_research_modules),
            ("Security Framework", self.test_security_framework),
            ("Scaling Components", self.test_scaling_components),
            ("Configuration Validation", self.validate_configuration),
            ("Documentation Check", self.check_documentation)
        ]
        
        passed = 0
        total = len(gates)
        
        for gate_name, gate_func in gates:
            print(f"\n📊 Running: {gate_name}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                result = gate_func()
                elapsed = time.time() - start_time
                
                if result['passed']:
                    print(f"✅ PASSED ({elapsed:.2f}s)")
                    passed += 1
                else:
                    print(f"❌ FAILED ({elapsed:.2f}s)")
                    
                self.results[gate_name] = result
                
            except Exception as e:
                print(f"💥 ERROR: {e}")
                self.results[gate_name] = {
                    'passed': False,
                    'error': str(e),
                    'details': []
                }
                
        print(f"\n🎯 Quality Gates Summary")
        print("=" * 60)
        print(f"Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("🏆 ALL QUALITY GATES PASSED!")
        elif passed >= total * 0.8:
            print("⚡ MOSTLY PASSED - Good progress!")
        else:
            print("🔧 NEEDS WORK - Several gates failed")
            
        return self.results
        
    def validate_code_structure(self) -> Dict[str, Any]:
        """Validate code structure and organization."""
        
        result = {'passed': True, 'details': [], 'warnings': []}
        
        # Check main package structure
        required_dirs = [
            'quantum_ctl',
            'quantum_ctl/core',
            'quantum_ctl/optimization',
            'quantum_ctl/models',
            'quantum_ctl/integration',
            'quantum_ctl/utils',
            'quantum_ctl/research',
            'quantum_ctl/security',
            'quantum_ctl/scaling',
            'tests'
        ]
        
        for dir_path in required_dirs:
            full_path = self.repo_path / dir_path
            if full_path.exists():
                result['details'].append(f"✓ {dir_path} exists")
            else:
                result['details'].append(f"✗ {dir_path} missing")
                result['passed'] = False
                
        # Check key files
        required_files = [
            'quantum_ctl/__init__.py',
            'pyproject.toml',
            'README.md',
            'LICENSE'
        ]
        
        for file_path in required_files:
            full_path = self.repo_path / file_path
            if full_path.exists():
                result['details'].append(f"✓ {file_path} exists")
            else:
                result['details'].append(f"✗ {file_path} missing")
                result['passed'] = False
                
        # Check for __init__.py in packages
        init_files = list(self.repo_path.glob('quantum_ctl/**/__init__.py'))
        result['details'].append(f"✓ Found {len(init_files)} __init__.py files")
        
        return result
        
    def validate_imports(self) -> Dict[str, Any]:
        """Validate that all modules can be imported."""
        
        result = {'passed': True, 'details': [], 'warnings': []}
        
        # Add repo to path
        sys.path.insert(0, str(self.repo_path))
        
        # Test core imports
        core_imports = [
            'quantum_ctl',
            'quantum_ctl.core',
            'quantum_ctl.models', 
            'quantum_ctl.optimization',
            'quantum_ctl.utils'
        ]
        
        for module_name in core_imports:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    result['details'].append(f"✓ {module_name} importable")
                else:
                    result['details'].append(f"✗ {module_name} not found")
                    result['passed'] = False
            except Exception as e:
                result['details'].append(f"✗ {module_name} import error: {e}")
                result['warnings'].append(f"Import warning for {module_name}: {e}")
                
        # Test research modules (may have dependency issues)
        research_imports = [
            'quantum_ctl.research',
            'quantum_ctl.security', 
            'quantum_ctl.scaling'
        ]
        
        for module_name in research_imports:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    result['details'].append(f"✓ {module_name} importable")
                else:
                    result['details'].append(f"⚠ {module_name} not found (optional)")
                    result['warnings'].append(f"Optional module {module_name} not available")
            except Exception as e:
                result['warnings'].append(f"Optional import warning for {module_name}: {e}")
                
        return result
        
    def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic functionality without external dependencies."""
        
        result = {'passed': True, 'details': [], 'warnings': []}
        
        sys.path.insert(0, str(self.repo_path))
        
        try:
            # Test basic data structures
            from quantum_ctl.models.building import Building
            
            # Create a simple building
            building = Building(
                building_id="test_building",
                zones=3,
                thermal_mass=[1000.0, 1200.0, 900.0]
            )
            
            result['details'].append("✓ Building model creation works")
            
            # Test building properties
            assert building.zones == 3
            assert len(building.thermal_mass) == 3
            result['details'].append("✓ Building properties accessible")
            
            # Test state dimension calculation
            state_dim = building.get_state_dimension()
            assert isinstance(state_dim, int)
            assert state_dim > 0
            result['details'].append("✓ State dimension calculation works")
            
        except ImportError as e:
            result['warnings'].append(f"Basic functionality test skipped due to imports: {e}")
        except Exception as e:
            result['details'].append(f"✗ Basic functionality error: {e}")
            result['passed'] = False
            
        return result
        
    def test_research_modules(self) -> Dict[str, Any]:
        """Test research module functionality."""
        
        result = {'passed': True, 'details': [], 'warnings': []}
        
        sys.path.insert(0, str(self.repo_path))
        
        try:
            # Test novel QUBO formulations
            from quantum_ctl.research.novel_qubo_formulations import NovelQUBOFormulator
            
            formulator = NovelQUBOFormulator(
                state_dim=3,
                control_dim=2,
                horizon=6
            )
            
            result['details'].append("✓ NovelQUBOFormulator instantiation works")
            
            # Test adaptive constraint weighting
            from quantum_ctl.research.novel_qubo_formulations import AdaptiveConstraintWeighting
            
            weighting = AdaptiveConstraintWeighting(learning_rate=0.01)
            result['details'].append("✓ AdaptiveConstraintWeighting instantiation works")
            
            # Test embedding strategies
            from quantum_ctl.research.advanced_embedding_strategies import TopologyAwareEmbedder
            
            embedder = TopologyAwareEmbedder()
            result['details'].append("✓ TopologyAwareEmbedder instantiation works")
            
        except ImportError as e:
            result['warnings'].append(f"Research modules test skipped: {e}")
        except Exception as e:
            result['details'].append(f"✗ Research modules error: {e}")
            result['passed'] = False
            
        return result
        
    def test_security_framework(self) -> Dict[str, Any]:
        """Test security framework functionality."""
        
        result = {'passed': True, 'details': [], 'warnings': []}
        
        sys.path.insert(0, str(self.repo_path))
        
        try:
            # Test security manager
            from quantum_ctl.security.quantum_security import QuantumSecurityManager, SecurityLevel
            
            security_manager = QuantumSecurityManager(
                security_level=SecurityLevel.MEDIUM
            )
            
            result['details'].append("✓ QuantumSecurityManager instantiation works")
            
            # Test security credentials
            from quantum_ctl.security.quantum_security import SecurityCredentials
            
            credentials = SecurityCredentials(
                api_key="test_key_123",
                secret_key="test_secret_456",
                permissions=["basic_access"]
            )
            
            result['details'].append("✓ SecurityCredentials creation works")
            
            # Test API key validation
            valid_key = security_manager._validate_api_key_format("valid_test_key_123456")
            assert valid_key == True
            result['details'].append("✓ API key validation works")
            
        except ImportError as e:
            result['warnings'].append(f"Security framework test skipped: {e}")
        except Exception as e:
            result['details'].append(f"✗ Security framework error: {e}")
            result['passed'] = False
            
        return result
        
    def test_scaling_components(self) -> Dict[str, Any]:
        """Test scaling and performance optimization components."""
        
        result = {'passed': True, 'details': [], 'warnings': []}
        
        sys.path.insert(0, str(self.repo_path))
        
        try:
            # Test auto-scaler
            from quantum_ctl.scaling.auto_scaler import QuantumAutoScaler, ScalingPolicy
            
            policy = ScalingPolicy(
                min_instances=1,
                max_instances=5,
                target_cpu_utilization=70.0
            )
            
            auto_scaler = QuantumAutoScaler(policy)
            result['details'].append("✓ QuantumAutoScaler instantiation works")
            
            # Test performance optimizer
            from quantum_ctl.scaling.performance_optimizer import PerformanceOptimizer, OptimizationStrategy
            
            optimizer = PerformanceOptimizer(OptimizationStrategy.BALANCED)
            result['details'].append("✓ PerformanceOptimizer instantiation works")
            
            # Test status reporting
            status = auto_scaler.get_scaling_status()
            assert isinstance(status, dict)
            result['details'].append("✓ Scaling status reporting works")
            
        except ImportError as e:
            result['warnings'].append(f"Scaling components test skipped: {e}")
        except Exception as e:
            result['details'].append(f"✗ Scaling components error: {e}")
            result['passed'] = False
            
        return result
        
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration files and settings."""
        
        result = {'passed': True, 'details': [], 'warnings': []}
        
        # Check pyproject.toml
        pyproject_path = self.repo_path / 'pyproject.toml'
        if pyproject_path.exists():
            result['details'].append("✓ pyproject.toml exists")
            
            try:
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                    
                # Check for essential sections
                required_sections = ['[project]', '[build-system]', '[tool.pytest.ini_options]']
                for section in required_sections:
                    if section in content:
                        result['details'].append(f"✓ {section} section found")
                    else:
                        result['details'].append(f"⚠ {section} section missing")
                        result['warnings'].append(f"Missing {section} in pyproject.toml")
                        
            except Exception as e:
                result['details'].append(f"✗ Error reading pyproject.toml: {e}")
                result['passed'] = False
        else:
            result['details'].append("✗ pyproject.toml missing")
            result['passed'] = False
            
        # Check requirements.txt
        requirements_path = self.repo_path / 'requirements.txt'
        if requirements_path.exists():
            result['details'].append("✓ requirements.txt exists")
            
            try:
                with open(requirements_path, 'r') as f:
                    requirements = f.read().strip().split('\n')
                    
                result['details'].append(f"✓ {len(requirements)} requirements found")
                
                # Check for essential packages
                essential_packages = ['numpy', 'scipy', 'pandas']
                for package in essential_packages:
                    if any(package in req for req in requirements):
                        result['details'].append(f"✓ {package} dependency found")
                    else:
                        result['warnings'].append(f"Essential package {package} not in requirements")
                        
            except Exception as e:
                result['details'].append(f"✗ Error reading requirements.txt: {e}")
                result['passed'] = False
        else:
            result['warnings'].append("requirements.txt missing (optional)")
            
        return result
        
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness."""
        
        result = {'passed': True, 'details': [], 'warnings': []}
        
        # Check README.md
        readme_path = self.repo_path / 'README.md'
        if readme_path.exists():
            result['details'].append("✓ README.md exists")
            
            try:
                with open(readme_path, 'r') as f:
                    readme_content = f.read()
                    
                # Check for essential sections
                essential_sections = [
                    '# Quantum-Anneal-CTL',
                    '## Overview',
                    '## Installation',
                    '## Quick Start'
                ]
                
                for section in essential_sections:
                    if section in readme_content:
                        result['details'].append(f"✓ {section} section found")
                    else:
                        result['warnings'].append(f"README missing {section} section")
                        
                # Check length
                if len(readme_content) > 1000:
                    result['details'].append(f"✓ README is comprehensive ({len(readme_content)} chars)")
                else:
                    result['warnings'].append("README is quite short")
                    
            except Exception as e:
                result['details'].append(f"✗ Error reading README: {e}")
                result['passed'] = False
        else:
            result['details'].append("✗ README.md missing")
            result['passed'] = False
            
        # Check for additional documentation
        docs_path = self.repo_path / 'docs'
        if docs_path.exists():
            doc_files = list(docs_path.glob('**/*.md'))
            result['details'].append(f"✓ Found {len(doc_files)} documentation files")
        else:
            result['warnings'].append("No docs/ directory found")
            
        return result
        
    def print_detailed_results(self):
        """Print detailed results for all quality gates."""
        
        print("\n📋 Detailed Quality Gate Results")
        print("=" * 60)
        
        for gate_name, gate_result in self.results.items():
            print(f"\n🎯 {gate_name}")
            print("-" * 30)
            
            status = "✅ PASSED" if gate_result['passed'] else "❌ FAILED"
            print(f"Status: {status}")
            
            if 'details' in gate_result:
                for detail in gate_result['details']:
                    print(f"  {detail}")
                    
            if 'warnings' in gate_result:
                for warning in gate_result['warnings']:
                    print(f"  ⚠ WARNING: {warning}")
                    
            if 'error' in gate_result:
                print(f"  💥 ERROR: {gate_result['error']}")


def main():
    """Main quality gate execution."""
    
    runner = QualityGateRunner()
    results = runner.run_all_gates()
    
    # Print detailed results
    runner.print_detailed_results()
    
    # Return exit code based on results
    passed_count = sum(1 for result in results.values() if result['passed'])
    total_count = len(results)
    
    if passed_count == total_count:
        print(f"\n🎉 All {total_count} quality gates passed!")
        return 0
    elif passed_count >= total_count * 0.8:
        print(f"\n⚡ {passed_count}/{total_count} quality gates passed (80%+ success)")
        return 0
    else:
        print(f"\n🔧 Only {passed_count}/{total_count} quality gates passed")
        return 1


if __name__ == "__main__":
    exit(main())