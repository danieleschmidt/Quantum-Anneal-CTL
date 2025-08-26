"""Advanced Security Scanning for Quality Gates"""

import asyncio
import re
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Represents a security issue"""
    severity: str  # critical, high, medium, low
    category: str  # e.g., 'secrets', 'injection', 'crypto'
    description: str
    file_path: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID
    recommendation: Optional[str] = None


class AdvancedSecurityScanner:
    """Advanced security vulnerability scanner"""
    
    def __init__(self):
        self.secret_patterns = self._load_secret_patterns()
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        self.crypto_patterns = self._load_crypto_patterns()
        
    async def scan_project(self, project_root: Path) -> List[SecurityIssue]:
        """Comprehensive security scan of the project"""
        issues = []
        
        # Scan for secrets
        issues.extend(await self._scan_secrets(project_root))
        
        # Scan for code vulnerabilities
        issues.extend(await self._scan_code_vulnerabilities(project_root))
        
        # Scan for cryptographic issues
        issues.extend(await self._scan_crypto_issues(project_root))
        
        # Scan dependencies for known vulnerabilities
        issues.extend(await self._scan_dependencies(project_root))
        
        # Scan configuration files
        issues.extend(await self._scan_config_files(project_root))
        
        # Check for insecure defaults
        issues.extend(await self._scan_insecure_defaults(project_root))
        
        return sorted(issues, key=lambda x: self._severity_weight(x.severity), reverse=True)
    
    def _severity_weight(self, severity: str) -> int:
        """Get numeric weight for severity sorting"""
        weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        return weights.get(severity.lower(), 0)
    
    async def _scan_secrets(self, project_root: Path) -> List[SecurityIssue]:
        """Scan for hardcoded secrets"""
        issues = []
        
        for file_path in project_root.rglob("*.py"):
            if self._should_skip_file(file_path):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.splitlines()
                
                for line_no, line in enumerate(lines, 1):
                    for pattern_name, pattern_data in self.secret_patterns.items():
                        pattern = pattern_data['pattern']
                        severity = pattern_data['severity']
                        
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            # Additional validation to reduce false positives
                            if self._validate_secret_match(match.group(), pattern_name):
                                issues.append(SecurityIssue(
                                    severity=severity,
                                    category='secrets',
                                    description=f"Potential {pattern_name} detected",
                                    file_path=str(file_path),
                                    line_number=line_no,
                                    code_snippet=line.strip(),
                                    cwe_id='CWE-798',
                                    recommendation=f"Remove hardcoded {pattern_name} and use environment variables or secure key management"
                                ))
                                
            except Exception as e:
                logger.warning(f"Failed to scan {file_path} for secrets: {e}")
        
        return issues
    
    async def _scan_code_vulnerabilities(self, project_root: Path) -> List[SecurityIssue]:
        """Scan for code-level vulnerabilities"""
        issues = []
        
        for file_path in project_root.rglob("*.py"):
            if self._should_skip_file(file_path):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                # Analyze AST for vulnerabilities
                issues.extend(self._analyze_ast_vulnerabilities(tree, file_path, content))
                
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
        
        return issues
    
    def _analyze_ast_vulnerabilities(self, tree: ast.AST, file_path: Path, content: str) -> List[SecurityIssue]:
        """Analyze AST for security vulnerabilities"""
        issues = []
        lines = content.splitlines()
        
        class VulnerabilityVisitor(ast.NodeVisitor):
            def __init__(self, outer_self):
                self.outer_self = outer_self
                
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    if func_name in ['eval', 'exec']:
                        issues.append(SecurityIssue(
                            severity='high',
                            category='injection',
                            description=f"Use of dangerous function: {func_name}",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            code_snippet=lines[node.lineno-1].strip() if node.lineno <= len(lines) else "",
                            cwe_id='CWE-94',
                            recommendation=f"Avoid using {func_name}. Use safer alternatives like literal_eval() for eval() or avoid dynamic code execution"
                        ))
                    
                    elif func_name == 'open' and len(node.args) > 0:
                        # Check for potential path traversal
                        if isinstance(node.args[0], ast.BinOp):
                            issues.append(SecurityIssue(
                                severity='medium',
                                category='path_traversal',
                                description="Potential path traversal vulnerability in file operations",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=lines[node.lineno-1].strip() if node.lineno <= len(lines) else "",
                                cwe_id='CWE-22',
                                recommendation="Validate and sanitize file paths before use"
                            ))
                
                elif isinstance(node.func, ast.Attribute):
                    # Check for SQL-like operations
                    if (hasattr(node.func, 'attr') and 
                        node.func.attr in ['execute', 'query'] and 
                        len(node.args) > 0):
                        
                        # Check if SQL query is constructed with string concatenation
                        if isinstance(node.args[0], ast.BinOp):
                            issues.append(SecurityIssue(
                                severity='high',
                                category='injection',
                                description="Potential SQL injection vulnerability",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=lines[node.lineno-1].strip() if node.lineno <= len(lines) else "",
                                cwe_id='CWE-89',
                                recommendation="Use parameterized queries or ORM instead of string concatenation"
                            ))
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for imports of insecure modules
                insecure_modules = {
                    'pickle': ('high', 'Use json or other safe serialization formats'),
                    'marshal': ('medium', 'Use json for data serialization'),
                    'shelve': ('medium', 'Use secure database solutions'),
                    'subprocess': ('low', 'Validate all inputs when using subprocess')
                }
                
                for alias in node.names:
                    if alias.name in insecure_modules:
                        severity, recommendation = insecure_modules[alias.name]
                        issues.append(SecurityIssue(
                            severity=severity,
                            category='insecure_import',
                            description=f"Import of potentially insecure module: {alias.name}",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            code_snippet=lines[node.lineno-1].strip() if node.lineno <= len(lines) else "",
                            recommendation=recommendation
                        ))
                
                self.generic_visit(node)
                
            def visit_Str(self, node):
                # Check for hardcoded URLs with credentials
                url_pattern = r'https?://[^:]+:[^@]+@'
                if re.search(url_pattern, node.s):
                    issues.append(SecurityIssue(
                        severity='high',
                        category='secrets',
                        description="URL with embedded credentials detected",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        code_snippet=lines[node.lineno-1].strip() if node.lineno <= len(lines) else "",
                        cwe_id='CWE-798',
                        recommendation="Remove credentials from URLs and use secure authentication methods"
                    ))
                
                self.generic_visit(node)
        
        visitor = VulnerabilityVisitor(self)
        visitor.visit(tree)
        
        return issues
    
    async def _scan_crypto_issues(self, project_root: Path) -> List[SecurityIssue]:
        """Scan for cryptographic vulnerabilities"""
        issues = []
        
        for file_path in project_root.rglob("*.py"):
            if self._should_skip_file(file_path):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.splitlines()
                
                for line_no, line in enumerate(lines, 1):
                    for pattern_name, pattern_data in self.crypto_patterns.items():
                        pattern = pattern_data['pattern']
                        severity = pattern_data['severity']
                        description = pattern_data['description']
                        recommendation = pattern_data.get('recommendation', '')
                        
                        if re.search(pattern, line, re.IGNORECASE):
                            issues.append(SecurityIssue(
                                severity=severity,
                                category='cryptography',
                                description=description,
                                file_path=str(file_path),
                                line_number=line_no,
                                code_snippet=line.strip(),
                                recommendation=recommendation
                            ))
                            
            except Exception as e:
                logger.warning(f"Failed to scan {file_path} for crypto issues: {e}")
        
        return issues
    
    async def _scan_dependencies(self, project_root: Path) -> List[SecurityIssue]:
        """Scan dependencies for known vulnerabilities"""
        issues = []
        
        # Check if safety is available for dependency scanning
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, '-c', 'import safety',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            
            if proc.returncode == 0:
                # Run safety check
                safety_proc = await asyncio.create_subprocess_exec(
                    sys.executable, '-m', 'safety', 'check', '--json',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=project_root
                )
                
                stdout, stderr = await safety_proc.communicate()
                
                if stdout:
                    try:
                        vulnerabilities = json.loads(stdout.decode())
                        
                        for vuln in vulnerabilities:
                            issues.append(SecurityIssue(
                                severity='high',
                                category='dependency',
                                description=f"Vulnerable dependency: {vuln.get('package_name')} {vuln.get('installed_version')}",
                                file_path="requirements.txt",
                                recommendation=f"Update to version {vuln.get('vulnerable_spec', 'latest')}"
                            ))
                            
                    except json.JSONDecodeError:
                        pass
                        
        except Exception:
            # Safety not available, skip dependency scanning
            pass
        
        return issues
    
    async def _scan_config_files(self, project_root: Path) -> List[SecurityIssue]:
        """Scan configuration files for security issues"""
        issues = []
        
        config_patterns = [
            "*.yaml", "*.yml", "*.json", "*.ini", "*.conf", "*.cfg"
        ]
        
        for pattern in config_patterns:
            for file_path in project_root.rglob(pattern):
                if self._should_skip_file(file_path):
                    continue
                    
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = content.splitlines()
                    
                    for line_no, line in enumerate(lines, 1):
                        # Check for credentials in config
                        if re.search(r'(password|secret|key|token)\s*[:=]\s*[\'"][^\'"]{8,}[\'"]', line, re.IGNORECASE):
                            issues.append(SecurityIssue(
                                severity='high',
                                category='secrets',
                                description="Potential credentials in configuration file",
                                file_path=str(file_path),
                                line_number=line_no,
                                code_snippet=line.strip(),
                                recommendation="Use environment variables or secure key management for credentials"
                            ))
                        
                        # Check for debug mode enabled
                        if re.search(r'debug\s*[:=]\s*(true|1|yes)', line, re.IGNORECASE):
                            issues.append(SecurityIssue(
                                severity='medium',
                                category='configuration',
                                description="Debug mode enabled in configuration",
                                file_path=str(file_path),
                                line_number=line_no,
                                code_snippet=line.strip(),
                                recommendation="Disable debug mode in production"
                            ))
                            
                except Exception as e:
                    logger.warning(f"Failed to scan config file {file_path}: {e}")
        
        return issues
    
    async def _scan_insecure_defaults(self, project_root: Path) -> List[SecurityIssue]:
        """Scan for insecure default configurations"""
        issues = []
        
        # Common insecure default patterns
        insecure_defaults = {
            r'host\s*=\s*[\'"]0\.0\.0\.0[\'"]': {
                'severity': 'medium',
                'description': 'Server binding to all interfaces (0.0.0.0)',
                'recommendation': 'Bind to specific interface or use 127.0.0.1 for local access'
            },
            r'ssl_verify\s*=\s*False': {
                'severity': 'high',
                'description': 'SSL verification disabled',
                'recommendation': 'Enable SSL verification for secure communications'
            },
            r'verify\s*=\s*False': {
                'severity': 'high', 
                'description': 'Certificate verification disabled',
                'recommendation': 'Enable certificate verification'
            }
        }
        
        for file_path in project_root.rglob("*.py"):
            if self._should_skip_file(file_path):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.splitlines()
                
                for line_no, line in enumerate(lines, 1):
                    for pattern, details in insecure_defaults.items():
                        if re.search(pattern, line, re.IGNORECASE):
                            issues.append(SecurityIssue(
                                severity=details['severity'],
                                category='configuration',
                                description=details['description'],
                                file_path=str(file_path),
                                line_number=line_no,
                                code_snippet=line.strip(),
                                recommendation=details['recommendation']
                            ))
                            
            except Exception as e:
                logger.warning(f"Failed to scan {file_path} for insecure defaults: {e}")
        
        return issues
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped from scanning"""
        skip_patterns = [
            '*/test*/*',
            '*/__pycache__/*',
            '*/.*/*',
            '*/venv/*',
            '*/env/*',
            '*/node_modules/*'
        ]
        
        path_str = str(file_path)
        for pattern in skip_patterns:
            if file_path.match(pattern):
                return True
        
        return False
    
    def _validate_secret_match(self, match: str, pattern_name: str) -> bool:
        """Validate if a secret match is likely a real credential"""
        
        # Common false positive patterns
        false_positives = [
            'example', 'test', 'dummy', 'fake', 'placeholder',
            'your_key_here', 'insert_key', 'add_your_key',
            'xxxxx', '*****', '12345'
        ]
        
        match_lower = match.lower()
        
        # Check for obvious false positives
        for fp in false_positives:
            if fp in match_lower:
                return False
        
        # Additional validation based on pattern type
        if pattern_name == 'API Key':
            # API keys should have some complexity
            if len(match) < 16 or match.isdigit():
                return False
                
        elif pattern_name == 'JWT Token':
            # JWT tokens should have proper structure
            if match.count('.') != 2:
                return False
        
        return True
    
    def _load_secret_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load patterns for detecting secrets"""
        return {
            'API Key': {
                'pattern': r'api[_-]?key\s*[:=]\s*[\'"][a-zA-Z0-9_\-]{16,}[\'"]',
                'severity': 'high'
            },
            'Secret Key': {
                'pattern': r'secret[_-]?key\s*[:=]\s*[\'"][a-zA-Z0-9_\-]{16,}[\'"]',
                'severity': 'high'
            },
            'Password': {
                'pattern': r'password\s*[:=]\s*[\'"][^\'"]{8,}[\'"]',
                'severity': 'high'
            },
            'Private Key': {
                'pattern': r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
                'severity': 'critical'
            },
            'JWT Token': {
                'pattern': r'eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*',
                'severity': 'high'
            },
            'AWS Access Key': {
                'pattern': r'AKIA[0-9A-Z]{16}',
                'severity': 'critical'
            },
            'Database URL': {
                'pattern': r'postgresql://[^:]+:[^@]+@[^/]+/',
                'severity': 'high'
            }
        }
    
    def _load_vulnerability_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load patterns for code vulnerabilities"""
        return {
            'Command Injection': {
                'pattern': r'os\.system\s*\(\s*[\'"][^\'"]*(\\{|%s|\+)',
                'severity': 'high'
            },
            'Path Traversal': {
                'pattern': r'\.\./',
                'severity': 'medium'
            }
        }
    
    def _load_crypto_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load patterns for cryptographic issues"""
        return {
            'Weak Hash Algorithm': {
                'pattern': r'hashlib\.(md5|sha1)\(',
                'severity': 'medium',
                'description': 'Use of weak cryptographic hash algorithm',
                'recommendation': 'Use SHA-256 or stronger hash algorithms'
            },
            'Weak Random': {
                'pattern': r'random\.(random|randint|choice)',
                'severity': 'low',
                'description': 'Use of non-cryptographic random function',
                'recommendation': 'Use secrets module for cryptographic randomness'
            },
            'Weak SSL/TLS': {
                'pattern': r'ssl\.PROTOCOL_(SSLv|TLSv1)(?!_2)',
                'severity': 'high',
                'description': 'Use of weak SSL/TLS protocol',
                'recommendation': 'Use TLS 1.2 or higher'
            }
        }
    
    def generate_security_report(self, issues: List[SecurityIssue]) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        
        # Categorize issues
        by_severity = {}
        by_category = {}
        
        for issue in issues:
            # Group by severity
            if issue.severity not in by_severity:
                by_severity[issue.severity] = []
            by_severity[issue.severity].append(issue)
            
            # Group by category
            if issue.category not in by_category:
                by_category[issue.category] = []
            by_category[issue.category].append(issue)
        
        # Calculate scores
        total_score = 100.0
        critical_count = len(by_severity.get('critical', []))
        high_count = len(by_severity.get('high', []))
        medium_count = len(by_severity.get('medium', []))
        low_count = len(by_severity.get('low', []))
        
        # Deduct points based on severity
        total_score -= critical_count * 25  # 25 points per critical
        total_score -= high_count * 15      # 15 points per high
        total_score -= medium_count * 5     # 5 points per medium
        total_score -= low_count * 1        # 1 point per low
        
        security_score = max(0, total_score)
        
        return {
            'security_score': security_score,
            'total_issues': len(issues),
            'issues_by_severity': {
                'critical': critical_count,
                'high': high_count,
                'medium': medium_count,
                'low': low_count
            },
            'issues_by_category': {cat: len(issues) for cat, issues in by_category.items()},
            'issues': [
                {
                    'severity': issue.severity,
                    'category': issue.category,
                    'description': issue.description,
                    'file_path': issue.file_path,
                    'line_number': issue.line_number,
                    'code_snippet': issue.code_snippet,
                    'cwe_id': issue.cwe_id,
                    'recommendation': issue.recommendation
                }
                for issue in issues[:20]  # Limit to first 20 issues
            ],
            'recommendations': self._generate_recommendations(by_category),
            'scan_timestamp': datetime.utcnow().isoformat()
        }
    
    def _generate_recommendations(self, issues_by_category: Dict[str, List[SecurityIssue]]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if 'secrets' in issues_by_category:
            recommendations.append("Implement secure secret management using environment variables or dedicated services like AWS Secrets Manager")
        
        if 'injection' in issues_by_category:
            recommendations.append("Use parameterized queries and input validation to prevent injection attacks")
        
        if 'cryptography' in issues_by_category:
            recommendations.append("Update cryptographic implementations to use modern, secure algorithms")
        
        if 'dependency' in issues_by_category:
            recommendations.append("Regularly update dependencies and monitor for security advisories")
        
        recommendations.append("Implement regular security code reviews and automated security scanning in CI/CD")
        recommendations.append("Consider using static analysis tools like bandit for Python security scanning")
        
        return recommendations


class SecurityGateIntegration:
    """Integration class for security scanning in quality gates"""
    
    def __init__(self):
        self.scanner = AdvancedSecurityScanner()
    
    async def run_security_scan(self, project_root: Path) -> Dict[str, Any]:
        """Run comprehensive security scan"""
        logger.info("Starting advanced security scan...")
        
        try:
            issues = await self.scanner.scan_project(project_root)
            report = self.scanner.generate_security_report(issues)
            
            logger.info(f"Security scan completed: {len(issues)} issues found")
            logger.info(f"Security score: {report['security_score']:.1f}/100")
            
            return report
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return {
                'security_score': 0.0,
                'total_issues': 0,
                'error': str(e),
                'scan_timestamp': datetime.utcnow().isoformat()
            }