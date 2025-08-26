"""Quality Gates CI/CD Integrations"""

import os
import json
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .gate_runner import QualityGateRunner
from .config import QualityGateConfig

logger = logging.getLogger(__name__)


class GitHubActionsIntegration:
    """Integration with GitHub Actions"""
    
    def __init__(self, runner: QualityGateRunner):
        self.runner = runner
        self.is_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'
        
    async def run_and_report(self) -> Dict[str, Any]:
        """Run quality gates and format for GitHub Actions"""
        
        result = await self.runner.run_all_gates()
        
        if self.is_github_actions:
            await self._create_github_annotations(result)
            await self._set_github_outputs(result)
            await self._create_job_summary(result)
        
        return result
    
    async def _create_github_annotations(self, result: Dict[str, Any]):
        """Create GitHub Actions annotations for failed gates"""
        
        for gate in result.get('gates', []):
            if not gate.get('passed', True):
                gate_name = gate.get('gate_name', 'unknown')
                messages = gate.get('messages', [])
                score = gate.get('score', 0)
                
                # Create error annotation
                error_msg = f"{gate_name} failed (score: {score:.1f}/100)"
                if messages:
                    error_msg += f": {'; '.join(messages[:2])}"
                
                print(f"::error::Quality Gate Failed - {error_msg}")
                
                # Create warning for low scores
                if 50 <= score < 75:
                    print(f"::warning::Quality Gate Warning - {gate_name} score below threshold: {score:.1f}/100")
    
    async def _set_github_outputs(self, result: Dict[str, Any]):
        """Set GitHub Actions outputs"""
        
        outputs = {
            'passed': str(result.get('passed', False)).lower(),
            'score': str(result.get('score', 0)),
            'total_gates': str(result.get('summary', {}).get('total_gates', 0)),
            'passed_gates': str(result.get('summary', {}).get('passed_gates', 0)),
            'failed_gates': str(result.get('summary', {}).get('failed_gates', 0))
        }
        
        github_output = os.getenv('GITHUB_OUTPUT')
        if github_output:
            with open(github_output, 'a') as f:
                for key, value in outputs.items():
                    f.write(f"{key}={value}\n")
        
        # Also use legacy method for compatibility
        for key, value in outputs.items():
            print(f"::set-output name={key}::{value}")
    
    async def _create_job_summary(self, result: Dict[str, Any]):
        """Create GitHub Actions job summary"""
        
        github_step_summary = os.getenv('GITHUB_STEP_SUMMARY')
        if not github_step_summary:
            return
        
        # Generate markdown summary
        overall_passed = result.get('passed', False)
        overall_score = result.get('score', 0)
        summary = result.get('summary', {})
        
        status_emoji = "✅" if overall_passed else "❌"
        status_text = "PASSED" if overall_passed else "FAILED"
        
        markdown = f"""
## {status_emoji} Quality Gates Report

**Overall Status:** {status_text}  
**Score:** {overall_score:.1f}/100  
**Gates:** {summary.get('passed_gates', 0)}/{summary.get('total_gates', 0)} passed  
**Execution Time:** {result.get('execution_time_ms', 0):.1f}ms  

### Gate Results

| Gate | Status | Score | Messages |
|------|--------|-------|----------|
"""
        
        for gate in result.get('gates', []):
            gate_name = gate.get('gate_name', 'Unknown').replace('_', ' ').title()
            gate_passed = gate.get('passed', False)
            gate_score = gate.get('score', 0)
            gate_status = "✅" if gate_passed else "❌"
            gate_messages = ' • '.join(gate.get('messages', [])[:2])
            
            markdown += f"| {gate_name} | {gate_status} | {gate_score:.1f}/100 | {gate_messages} |\n"
        
        # Add metrics if available
        metrics = result.get('metrics', {})
        if metrics:
            markdown += f"\n### Detailed Metrics\n\n"
            
            coverage = metrics.get('coverage', {})
            if coverage:
                markdown += f"**Coverage:**\n"
                markdown += f"- Test Coverage: {coverage.get('test_coverage', 0):.1f}%\n"
                markdown += f"- Documentation: {coverage.get('docstring_coverage', 0):.1f}%\n\n"
            
            issues = metrics.get('issues', {})
            if issues:
                markdown += f"**Quality Issues:**\n"
                markdown += f"- Style Violations: {issues.get('style_violations', 0)}\n"
                markdown += f"- Security Issues: {issues.get('security_issues', 0)}\n"
                markdown += f"- Complexity Issues: {issues.get('complexity_issues', 0)}\n\n"
        
        with open(github_step_summary, 'a') as f:
            f.write(markdown)


class JenkinsIntegration:
    """Integration with Jenkins CI"""
    
    def __init__(self, runner: QualityGateRunner):
        self.runner = runner
        self.is_jenkins = os.getenv('JENKINS_URL') is not None
        
    async def run_and_report(self) -> Dict[str, Any]:
        """Run quality gates and format for Jenkins"""
        
        result = await self.runner.run_all_gates()
        
        if self.is_jenkins:
            await self._create_jenkins_artifacts(result)
            await self._set_build_status(result)
        
        return result
    
    async def _create_jenkins_artifacts(self, result: Dict[str, Any]):
        """Create Jenkins build artifacts"""
        
        # Create JUnit-style XML report
        await self._create_junit_report(result)
        
        # Create build badge data
        await self._create_build_badge(result)
    
    async def _create_junit_report(self, result: Dict[str, Any]):
        """Create JUnit-style XML report for Jenkins"""
        
        from xml.etree.ElementTree import Element, SubElement, tostring
        from xml.dom import minidom
        
        testsuites = Element('testsuites')
        testsuite = SubElement(testsuites, 'testsuite')
        testsuite.set('name', 'Quality Gates')
        testsuite.set('tests', str(result.get('summary', {}).get('total_gates', 0)))
        testsuite.set('failures', str(result.get('summary', {}).get('failed_gates', 0)))
        testsuite.set('time', str(result.get('execution_time_ms', 0) / 1000))
        
        for gate in result.get('gates', []):
            testcase = SubElement(testsuite, 'testcase')
            testcase.set('name', gate.get('gate_name', 'unknown'))
            testcase.set('classname', 'QualityGates')
            testcase.set('time', str(gate.get('execution_time_ms', 0) / 1000))
            
            if not gate.get('passed', True):
                failure = SubElement(testcase, 'failure')
                failure.set('message', f"Quality gate failed with score {gate.get('score', 0):.1f}/100")
                failure.text = '\n'.join(gate.get('messages', []))
        
        # Save XML report
        rough_string = tostring(testsuites, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        
        with open('quality-gates-report.xml', 'w') as f:
            f.write(reparsed.toprettyxml(indent='  '))
    
    async def _create_build_badge(self, result: Dict[str, Any]):
        """Create build badge data"""
        
        overall_passed = result.get('passed', False)
        overall_score = result.get('score', 0)
        
        badge_data = {
            'schemaVersion': 1,
            'label': 'Quality Gates',
            'message': f"{overall_score:.1f}/100" if overall_passed else 'FAILED',
            'color': 'brightgreen' if overall_passed else 'red'
        }
        
        with open('quality-gates-badge.json', 'w') as f:
            json.dump(badge_data, f, indent=2)
    
    async def _set_build_status(self, result: Dict[str, Any]):
        """Set Jenkins build status"""
        
        if not result.get('passed', False):
            # Mark build as unstable rather than failed to allow artifacts collection
            print("UNSTABLE: Quality gates failed")
            
            # Set build description
            overall_score = result.get('score', 0)
            summary = result.get('summary', {})
            description = f"Quality Score: {overall_score:.1f}/100 ({summary.get('failed_gates', 0)} gates failed)"
            
            print(f"BUILD_DESCRIPTION={description}")


class GitLabIntegration:
    """Integration with GitLab CI/CD"""
    
    def __init__(self, runner: QualityGateRunner):
        self.runner = runner
        self.is_gitlab = os.getenv('GITLAB_CI') == 'true'
        
    async def run_and_report(self) -> Dict[str, Any]:
        """Run quality gates and format for GitLab CI"""
        
        result = await self.runner.run_all_gates()
        
        if self.is_gitlab:
            await self._create_gitlab_artifacts(result)
            await self._create_test_report(result)
        
        return result
    
    async def _create_gitlab_artifacts(self, result: Dict[str, Any]):
        """Create GitLab CI artifacts"""
        
        # Create metrics report for GitLab
        metrics = result.get('metrics', {})
        gitlab_metrics = {
            'quality_score': result.get('score', 0),
            'test_coverage': metrics.get('coverage', {}).get('test_coverage', 0),
            'security_score': metrics.get('quality', {}).get('security_score', 0),
            'performance_score': metrics.get('quality', {}).get('performance_score', 0)
        }
        
        with open('quality_metrics.json', 'w') as f:
            json.dump(gitlab_metrics, f, indent=2)
    
    async def _create_test_report(self, result: Dict[str, Any]):
        """Create GitLab test report"""
        
        test_report = []
        
        for gate in result.get('gates', []):
            test_case = {
                'name': gate.get('gate_name', 'unknown'),
                'classname': 'QualityGates',
                'execution_time': gate.get('execution_time_ms', 0) / 1000,
                'status': 'passed' if gate.get('passed', True) else 'failed'
            }
            
            if not gate.get('passed', True):
                test_case['failure'] = {
                    'message': f"Quality gate failed with score {gate.get('score', 0):.1f}/100",
                    'body': '\n'.join(gate.get('messages', []))
                }
            
            test_report.append(test_case)
        
        with open('quality_gates_report.json', 'w') as f:
            json.dump(test_report, f, indent=2)


class QualityGatesIntegrator:
    """Main integrator for CI/CD systems"""
    
    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or QualityGateConfig.from_env()
        self.runner = QualityGateRunner(self.config)
        
        # Initialize integrations
        self.github_actions = GitHubActionsIntegration(self.runner)
        self.jenkins = JenkinsIntegration(self.runner)
        self.gitlab = GitLabIntegration(self.runner)
    
    async def run_for_ci(self) -> Dict[str, Any]:
        """Auto-detect CI environment and run appropriate integration"""
        
        if os.getenv('GITHUB_ACTIONS') == 'true':
            logger.info("Detected GitHub Actions environment")
            return await self.github_actions.run_and_report()
            
        elif os.getenv('JENKINS_URL') is not None:
            logger.info("Detected Jenkins environment")
            return await self.jenkins.run_and_report()
            
        elif os.getenv('GITLAB_CI') == 'true':
            logger.info("Detected GitLab CI environment")
            return await self.gitlab.run_and_report()
            
        else:
            logger.info("No specific CI environment detected, running standard gates")
            return await self.runner.run_all_gates()
    
    def create_pre_commit_hook(self, output_path: str = ".git/hooks/pre-commit"):
        """Create a pre-commit hook script"""
        
        hook_script = """#!/bin/bash
# Quality Gates Pre-commit Hook
# Generated by Quantum-Anneal-CTL

echo "🔬 Running Quality Gates..."

# Run quality gates
python -m quantum_ctl.quality_gates.cli run --fail-fast --format json

# Check exit code
if [ $? -ne 0 ]; then
    echo "❌ Quality gates failed! Commit blocked."
    echo "Run 'python -m quantum_ctl.quality_gates.cli run' for details."
    exit 1
fi

echo "✅ Quality gates passed! Proceeding with commit."
"""
        
        hook_path = Path(output_path)
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(hook_path, 'w') as f:
            f.write(hook_script)
        
        # Make executable
        hook_path.chmod(0o755)
        
        logger.info(f"Pre-commit hook created at: {hook_path}")
    
    def create_github_workflow(self, output_path: str = ".github/workflows/quality-gates.yml"):
        """Create GitHub Actions workflow"""
        
        workflow = """name: Quality Gates

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  quality-gates:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest pytest-cov flake8
    
    - name: Run Quality Gates
      run: |
        python -m quantum_ctl.quality_gates.cli run --format both
      env:
        QG_MIN_COVERAGE: "80.0"
        QG_MAX_RESPONSE_MS: "200"
        QG_SECURITY_SCAN: "true"
    
    - name: Upload Quality Reports
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: quality-reports
        path: quality_reports/
        retention-days: 30
"""
        
        workflow_path = Path(output_path)
        workflow_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(workflow_path, 'w') as f:
            f.write(workflow)
        
        logger.info(f"GitHub Actions workflow created at: {workflow_path}")


async def main():
    """CLI entry point for integrations"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'ci':
        integrator = QualityGatesIntegrator()
        result = await integrator.run_for_ci()
        
        if not result.get('passed', False):
            sys.exit(1)
    else:
        print("Usage: python -m quantum_ctl.quality_gates.integrations ci")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())