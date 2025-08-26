"""Quality Gate Reporting System"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .config import QualityGateConfig

logger = logging.getLogger(__name__)


class QualityReporter:
    """Generates quality gate reports in multiple formats"""
    
    def __init__(self, config: QualityGateConfig):
        self.config = config
        self.output_dir = Path(config.report_output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    async def generate_reports(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Generate all configured reports"""
        report_files = {}
        
        try:
            # JSON report (always generated)
            json_file = await self.generate_json_report(result)
            report_files["json"] = json_file
            
            # HTML report if enabled
            if self.config.generate_html_report:
                html_file = await self.generate_html_report(result)
                report_files["html"] = html_file
            
            # Console summary
            self.print_console_summary(result)
            
            logger.info(f"Generated reports: {list(report_files.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            
        return report_files
    
    async def generate_json_report(self, result: Dict[str, Any]) -> str:
        """Generate detailed JSON report"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"quality_gates_report_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Enhance result with metadata
        enhanced_result = {
            **result,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "generator": "quantum-anneal-ctl-quality-gates",
                "version": "1.0.0",
                "config": self.config.to_dict()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(enhanced_result, f, indent=2, default=str)
            
        logger.info(f"JSON report saved to: {filepath}")
        return str(filepath)
    
    async def generate_html_report(self, result: Dict[str, Any]) -> str:
        """Generate HTML dashboard report"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"quality_gates_report_{timestamp}.html"
        filepath = self.output_dir / filename
        
        html_content = self._create_html_report(result)
        
        with open(filepath, 'w') as f:
            f.write(html_content)
            
        logger.info(f"HTML report saved to: {filepath}")
        return str(filepath)
    
    def _create_html_report(self, result: Dict[str, Any]) -> str:
        """Create HTML report content"""
        
        # Extract key metrics
        overall_passed = result.get("passed", False)
        overall_score = result.get("score", 0.0)
        summary = result.get("summary", {})
        gates = result.get("gates", [])
        metrics = result.get("metrics", {})
        
        # Status color and icon
        status_color = "#28a745" if overall_passed else "#dc3545"
        status_icon = "✅" if overall_passed else "❌"
        status_text = "PASSED" if overall_passed else "FAILED"
        
        # Grade calculation
        grade = self._calculate_grade(overall_score)
        grade_color = self._get_grade_color(grade)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Gates Report - Quantum-Anneal-CTL</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px 8px 0 0;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .status-banner {{
            background: {status_color};
            color: white;
            padding: 20px 30px;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #666;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 1px;
        }}
        .grade {{
            color: {grade_color};
        }}
        .gates-section {{
            padding: 30px;
            border-top: 1px solid #eee;
        }}
        .gate-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            margin: 10px 0;
            border-radius: 6px;
            background: #f8f9fa;
        }}
        .gate-passed {{
            border-left: 4px solid #28a745;
        }}
        .gate-failed {{
            border-left: 4px solid #dc3545;
        }}
        .gate-name {{
            font-weight: bold;
            text-transform: capitalize;
        }}
        .gate-score {{
            font-size: 1.2em;
            font-weight: bold;
        }}
        .gate-messages {{
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }}
        .details-section {{
            padding: 30px;
            border-top: 1px solid #eee;
            background: #f8f9fa;
        }}
        .details-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        .details-card {{
            background: white;
            padding: 20px;
            border-radius: 6px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #eee;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            .details-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Quality Gates Report</h1>
            <div class="subtitle">Quantum-Anneal-CTL • Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC</div>
        </div>
        
        <div class="status-banner">
            {status_icon} Overall Status: {status_text}
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Overall Score</div>
                <div class="metric-value">{overall_score:.1f}/100</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Grade</div>
                <div class="metric-value grade">{grade}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Gates Passed</div>
                <div class="metric-value">{summary.get('passed_gates', 0)}/{summary.get('total_gates', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Execution Time</div>
                <div class="metric-value">{result.get('execution_time_ms', 0):.0f}ms</div>
            </div>
        </div>
        
        <div class="gates-section">
            <h2>Gate Results</h2>
        """
        
        # Add gate results
        for gate in gates:
            gate_class = "gate-passed" if gate.get("passed") else "gate-failed"
            gate_icon = "✅" if gate.get("passed") else "❌"
            
            html += f"""
            <div class="gate-item {gate_class}">
                <div>
                    <div class="gate-name">{gate_icon} {gate.get('gate_name', 'Unknown')}</div>
                    <div class="gate-messages">
                        {' • '.join(gate.get('messages', []))}
                    </div>
                </div>
                <div class="gate-score">{gate.get('score', 0):.1f}/100</div>
            </div>
            """
        
        # Add details section
        html += f"""
        </div>
        
        <div class="details-section">
            <h2>Detailed Metrics</h2>
            <div class="details-grid">
                <div class="details-card">
                    <h3>Coverage</h3>
                    <ul>
                        <li>Test Coverage: {metrics.get('coverage', {}).get('test_coverage', 0):.1f}%</li>
                        <li>Documentation: {metrics.get('coverage', {}).get('docstring_coverage', 0):.1f}%</li>
                    </ul>
                </div>
                <div class="details-card">
                    <h3>Quality Issues</h3>
                    <ul>
                        <li>Style Violations: {metrics.get('issues', {}).get('style_violations', 0)}</li>
                        <li>Security Issues: {metrics.get('issues', {}).get('security_issues', 0)}</li>
                        <li>Complexity Issues: {metrics.get('issues', {}).get('complexity_issues', 0)}</li>
                    </ul>
                </div>
                <div class="details-card">
                    <h3>Performance</h3>
                    <ul>
                        <li>Avg Response Time: {metrics.get('performance', {}).get('avg_response_time_ms', 0):.1f}ms</li>
                        <li>Memory Usage: {metrics.get('performance', {}).get('memory_usage_mb', 0):.1f}MB</li>
                        <li>CPU Usage: {metrics.get('performance', {}).get('cpu_usage_percent', 0):.1f}%</li>
                    </ul>
                </div>
                <div class="details-card">
                    <h3>Execution Stats</h3>
                    <ul>
                        <li>Total Gates: {metrics.get('execution', {}).get('gates_executed', 0)}</li>
                        <li>Success Rate: {metrics.get('execution', {}).get('success_rate', 0):.1f}%</li>
                        <li>Total Time: {metrics.get('execution', {}).get('total_execution_time_ms', 0):.1f}ms</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="footer">
            Generated by Quantum-Anneal-CTL Quality Gates System<br>
            <small>Report timestamp: {result.get('timestamp', 'unknown')}</small>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score"""
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
    
    def _get_grade_color(self, grade: str) -> str:
        """Get color for grade"""
        if grade.startswith('A'):
            return "#28a745"
        elif grade.startswith('B'):
            return "#17a2b8"
        elif grade.startswith('C'):
            return "#ffc107"
        elif grade.startswith('D'):
            return "#fd7e14"
        else:
            return "#dc3545"
    
    def print_console_summary(self, result: Dict[str, Any]):
        """Print summary to console"""
        overall_passed = result.get("passed", False)
        overall_score = result.get("score", 0.0)
        summary = result.get("summary", {})
        
        # Print header
        print("\n" + "="*80)
        print("🔬 QUALITY GATES REPORT - QUANTUM-ANNEAL-CTL")
        print("="*80)
        
        # Print overall status
        status_icon = "✅" if overall_passed else "❌"
        status_text = "PASSED" if overall_passed else "FAILED"
        grade = self._calculate_grade(overall_score)
        
        print(f"\n{status_icon} OVERALL STATUS: {status_text}")
        print(f"📊 Overall Score: {overall_score:.1f}/100 (Grade: {grade})")
        print(f"⏱️  Execution Time: {result.get('execution_time_ms', 0):.1f}ms")
        print(f"📈 Gates Passed: {summary.get('passed_gates', 0)}/{summary.get('total_gates', 0)}")
        
        # Print gate details
        print("\n📋 GATE DETAILS:")
        print("-" * 80)
        
        for gate in result.get("gates", []):
            gate_icon = "✅" if gate.get("passed") else "❌"
            gate_name = gate.get("gate_name", "Unknown").replace("_", " ").title()
            gate_score = gate.get("score", 0)
            
            print(f"{gate_icon} {gate_name:<25} Score: {gate_score:>5.1f}/100")
            
            # Print messages
            for message in gate.get("messages", [])[:2]:  # Show max 2 messages
                print(f"   └─ {message}")
        
        print("\n" + "="*80)
        
        # Print suggestions if failed
        if not overall_passed:
            print("💡 IMPROVEMENT SUGGESTIONS:")
            print("-" * 80)
            
            suggestions = []
            metrics = result.get("metrics", {})
            
            # Generate suggestions based on metrics
            coverage = metrics.get("coverage", {})
            if coverage.get("test_coverage", 0) < 80:
                suggestions.append(f"• Increase test coverage to 80%+ (currently {coverage.get('test_coverage', 0):.1f}%)")
            
            issues = metrics.get("issues", {})
            if issues.get("security_issues", 0) > 0:
                suggestions.append(f"• Address {issues.get('security_issues', 0)} security issues")
                
            if issues.get("style_violations", 0) > 10:
                suggestions.append(f"• Fix {issues.get('style_violations', 0)} code style violations")
            
            for suggestion in suggestions[:5]:  # Show max 5 suggestions
                print(suggestion)
                
            print("="*80)