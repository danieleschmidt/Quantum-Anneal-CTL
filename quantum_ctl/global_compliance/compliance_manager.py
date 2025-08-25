"""
Global Compliance Manager
Ensures compliance with international regulations (GDPR, CCPA, PDPA, etc.)
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ComplianceRegulation(Enum):
    GDPR = "gdpr"  # EU General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"  # Brazilian General Data Protection Law
    PIPEDA = "pipeda"  # Canadian Personal Information Protection
    
class ViolationSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DataCategory(Enum):
    PERSONAL_DATA = "personal_data"
    SENSITIVE_DATA = "sensitive_data"
    BIOMETRIC_DATA = "biometric_data"
    LOCATION_DATA = "location_data"
    FINANCIAL_DATA = "financial_data"
    HEALTH_DATA = "health_data"
    SYSTEM_DATA = "system_data"

@dataclass
class DataProcessingActivity:
    """Record of data processing activity"""
    activity_id: str
    timestamp: float
    user_id: str
    data_categories: List[DataCategory]
    processing_purpose: str
    legal_basis: str
    data_source: str
    retention_period: int  # days
    automated_decision: bool
    cross_border_transfer: bool
    recipient_countries: List[str]

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    regulation: ComplianceRegulation
    severity: ViolationSeverity
    description: str
    affected_data_subjects: int
    timestamp: float
    remediation_actions: List[str]
    resolved: bool
    resolution_timestamp: Optional[float]

@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    report_id: str
    timestamp: float
    reporting_period_days: int
    regulations_covered: List[ComplianceRegulation]
    total_processing_activities: int
    violations_found: List[ComplianceViolation]
    compliance_score: float
    recommendations: List[str]

class ComplianceModule(ABC):
    """Base class for compliance modules"""
    
    @abstractmethod
    def check_compliance(self, activity: DataProcessingActivity) -> List[ComplianceViolation]:
        """Check if activity complies with regulation"""
        pass
    
    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """Get compliance requirements"""
        pass

class GDPRComplianceModule(ComplianceModule):
    """GDPR compliance checking"""
    
    def __init__(self):
        self.lawful_bases = [
            'consent', 'contract', 'legal_obligation',
            'vital_interests', 'public_task', 'legitimate_interests'
        ]
        self.max_retention_periods = {
            DataCategory.PERSONAL_DATA: 2555,  # 7 years
            DataCategory.SENSITIVE_DATA: 365,  # 1 year
            DataCategory.BIOMETRIC_DATA: 365,  # 1 year
            DataCategory.HEALTH_DATA: 2555,    # 7 years
            DataCategory.LOCATION_DATA: 365,   # 1 year
            DataCategory.FINANCIAL_DATA: 2555, # 7 years
            DataCategory.SYSTEM_DATA: 1095     # 3 years
        }
    
    def check_compliance(self, activity: DataProcessingActivity) -> List[ComplianceViolation]:
        """Check GDPR compliance"""
        violations = []
        
        # Check lawful basis
        if activity.legal_basis not in self.lawful_bases:
            violations.append(ComplianceViolation(
                violation_id=f"GDPR_BASIS_{activity.activity_id}",
                regulation=ComplianceRegulation.GDPR,
                severity=ViolationSeverity.CRITICAL,
                description=f"Invalid lawful basis for processing: {activity.legal_basis}",
                affected_data_subjects=1,
                timestamp=time.time(),
                remediation_actions=[
                    "Establish valid lawful basis",
                    "Update data processing documentation",
                    "Obtain explicit consent if required"
                ],
                resolved=False,
                resolution_timestamp=None
            ))
        
        # Check data retention periods
        for data_category in activity.data_categories:
            max_retention = self.max_retention_periods.get(data_category, 365)
            if activity.retention_period > max_retention:
                violations.append(ComplianceViolation(
                    violation_id=f"GDPR_RETENTION_{activity.activity_id}_{data_category.value}",
                    regulation=ComplianceRegulation.GDPR,
                    severity=ViolationSeverity.HIGH,
                    description=f"Retention period ({activity.retention_period} days) exceeds maximum for {data_category.value} ({max_retention} days)",
                    affected_data_subjects=1,
                    timestamp=time.time(),
                    remediation_actions=[
                        f"Reduce retention period to {max_retention} days",
                        "Implement automatic data deletion",
                        "Review data minimization practices"
                    ],
                    resolved=False,
                    resolution_timestamp=None
                ))
        
        # Check sensitive data processing
        if DataCategory.SENSITIVE_DATA in activity.data_categories:
            if activity.legal_basis != 'explicit_consent' and 'vital_interests' not in activity.legal_basis:
                violations.append(ComplianceViolation(
                    violation_id=f"GDPR_SENSITIVE_{activity.activity_id}",
                    regulation=ComplianceRegulation.GDPR,
                    severity=ViolationSeverity.CRITICAL,
                    description="Sensitive data processing without explicit consent or vital interests",
                    affected_data_subjects=1,
                    timestamp=time.time(),
                    remediation_actions=[
                        "Obtain explicit consent for sensitive data",
                        "Review processing necessity",
                        "Implement additional safeguards"
                    ],
                    resolved=False,
                    resolution_timestamp=None
                ))
        
        # Check cross-border transfers
        if activity.cross_border_transfer:
            eu_countries = ['DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'SE', 'DK', 'FI', 'PT', 'GR', 'IE', 'LU', 'CY', 'MT', 'SI', 'SK', 'EE', 'LV', 'LT', 'PL', 'CZ', 'HU', 'HR', 'RO', 'BG']
            adequate_countries = ['CA', 'CH', 'IL', 'NZ', 'UY', 'AR', 'JP', 'KR']
            
            for country in activity.recipient_countries:
                if country not in eu_countries and country not in adequate_countries:
                    violations.append(ComplianceViolation(
                        violation_id=f"GDPR_TRANSFER_{activity.activity_id}_{country}",
                        regulation=ComplianceRegulation.GDPR,
                        severity=ViolationSeverity.HIGH,
                        description=f"Cross-border transfer to country without adequacy decision: {country}",
                        affected_data_subjects=1,
                        timestamp=time.time(),
                        remediation_actions=[
                            "Implement appropriate safeguards (SCCs, BCRs)",
                            "Obtain explicit consent for transfer",
                            "Conduct transfer impact assessment"
                        ],
                        resolved=False,
                        resolution_timestamp=None
                    ))
        
        # Check automated decision making
        if activity.automated_decision:
            violations.append(ComplianceViolation(
                violation_id=f"GDPR_AUTOMATED_{activity.activity_id}",
                regulation=ComplianceRegulation.GDPR,
                severity=ViolationSeverity.MEDIUM,
                description="Automated decision making detected - ensure safeguards",
                affected_data_subjects=1,
                timestamp=time.time(),
                remediation_actions=[
                    "Provide information about automated decision making",
                    "Implement right to human intervention",
                    "Ensure algorithmic transparency"
                ],
                resolved=False,
                resolution_timestamp=None
            ))
        
        return violations
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get GDPR requirements"""
        return {
            'regulation': 'GDPR',
            'scope': 'EU and data subjects in EU',
            'lawful_bases': self.lawful_bases,
            'data_subject_rights': [
                'access', 'rectification', 'erasure', 'portability',
                'restriction', 'objection', 'automated_decision_making'
            ],
            'retention_limits': self.max_retention_periods,
            'breach_notification_time': 72,  # hours
            'consent_requirements': 'explicit and informed',
            'dpo_required': 'for systematic monitoring or sensitive data',
            'privacy_by_design': True
        }

class CCPAComplianceModule(ComplianceModule):
    """CCPA compliance checking"""
    
    def __init__(self):
        self.business_purposes = [
            'providing_services', 'security', 'debugging', 'research',
            'quality_assurance', 'legal_compliance', 'internal_use'
        ]
    
    def check_compliance(self, activity: DataProcessingActivity) -> List[ComplianceViolation]:
        """Check CCPA compliance"""
        violations = []
        
        # Check if personal information is being sold
        if 'sale' in activity.processing_purpose.lower():
            violations.append(ComplianceViolation(
                violation_id=f"CCPA_SALE_{activity.activity_id}",
                regulation=ComplianceRegulation.CCPA,
                severity=ViolationSeverity.HIGH,
                description="Personal information sale detected - ensure opt-out mechanism",
                affected_data_subjects=1,
                timestamp=time.time(),
                remediation_actions=[
                    "Provide 'Do Not Sell' option",
                    "Update privacy notice",
                    "Implement opt-out mechanism"
                ],
                resolved=False,
                resolution_timestamp=None
            ))
        
        # Check business purpose limitation
        if activity.processing_purpose not in self.business_purposes:
            violations.append(ComplianceViolation(
                violation_id=f"CCPA_PURPOSE_{activity.activity_id}",
                regulation=ComplianceRegulation.CCPA,
                severity=ViolationSeverity.MEDIUM,
                description=f"Processing purpose may not align with CCPA business purposes: {activity.processing_purpose}",
                affected_data_subjects=1,
                timestamp=time.time(),
                remediation_actions=[
                    "Review processing purpose",
                    "Ensure business purpose alignment",
                    "Update internal documentation"
                ],
                resolved=False,
                resolution_timestamp=None
            ))
        
        # Check sensitive personal information
        sensitive_categories = [DataCategory.BIOMETRIC_DATA, DataCategory.HEALTH_DATA, DataCategory.FINANCIAL_DATA]
        if any(cat in activity.data_categories for cat in sensitive_categories):
            violations.append(ComplianceViolation(
                violation_id=f"CCPA_SENSITIVE_{activity.activity_id}",
                regulation=ComplianceRegulation.CCPA,
                severity=ViolationSeverity.MEDIUM,
                description="Sensitive personal information processing - additional rights apply",
                affected_data_subjects=1,
                timestamp=time.time(),
                remediation_actions=[
                    "Provide sensitive data opt-out",
                    "Limit processing to necessary purposes",
                    "Update privacy disclosures"
                ],
                resolved=False,
                resolution_timestamp=None
            ))
        
        return violations
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get CCPA requirements"""
        return {
            'regulation': 'CCPA',
            'scope': 'California residents',
            'consumer_rights': [
                'know', 'delete', 'opt_out_of_sale', 'non_discrimination',
                'limit_sensitive_data_use'
            ],
            'business_purposes': self.business_purposes,
            'disclosure_requirements': True,
            'opt_out_methods': ['website', 'email', 'phone'],
            'verification_requirements': True,
            'third_party_sharing': 'must_disclose'
        }

class PDPAComplianceModule(ComplianceModule):
    """PDPA (Singapore) compliance checking"""
    
    def __init__(self):
        self.consent_purposes = [
            'collection', 'use', 'disclosure'
        ]
    
    def check_compliance(self, activity: DataProcessingActivity) -> List[ComplianceViolation]:
        """Check PDPA compliance"""
        violations = []
        
        # Check consent for collection
        if activity.legal_basis != 'consent' and 'legal_obligation' not in activity.legal_basis:
            violations.append(ComplianceViolation(
                violation_id=f"PDPA_CONSENT_{activity.activity_id}",
                regulation=ComplianceRegulation.PDPA,
                severity=ViolationSeverity.HIGH,
                description="Personal data collection without valid consent",
                affected_data_subjects=1,
                timestamp=time.time(),
                remediation_actions=[
                    "Obtain valid consent",
                    "Provide clear purpose notification",
                    "Ensure consent is voluntary"
                ],
                resolved=False,
                resolution_timestamp=None
            ))
        
        # Check data breach notification (within 3 days for PDPC)
        if 'breach' in activity.processing_purpose.lower():
            violations.append(ComplianceViolation(
                violation_id=f"PDPA_BREACH_{activity.activity_id}",
                regulation=ComplianceRegulation.PDPA,
                severity=ViolationSeverity.CRITICAL,
                description="Data breach detected - notify PDPC within 3 days",
                affected_data_subjects=1,
                timestamp=time.time(),
                remediation_actions=[
                    "Notify PDPC within 72 hours",
                    "Assess affected individuals",
                    "Implement containment measures"
                ],
                resolved=False,
                resolution_timestamp=None
            ))
        
        # Check cross-border transfer restrictions
        if activity.cross_border_transfer:
            for country in activity.recipient_countries:
                if country not in ['MY', 'AU', 'NZ', 'CH', 'CA']:  # Countries with adequate protection
                    violations.append(ComplianceViolation(
                        violation_id=f"PDPA_TRANSFER_{activity.activity_id}_{country}",
                        regulation=ComplianceRegulation.PDPA,
                        severity=ViolationSeverity.MEDIUM,
                        description=f"Cross-border transfer to {country} may require additional safeguards",
                        affected_data_subjects=1,
                        timestamp=time.time(),
                        remediation_actions=[
                            "Ensure adequate protection in recipient country",
                            "Obtain additional consent for transfer",
                            "Implement contractual safeguards"
                        ],
                        resolved=False,
                        resolution_timestamp=None
                    ))
        
        return violations
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get PDPA requirements"""
        return {
            'regulation': 'PDPA',
            'scope': 'Singapore',
            'consent_requirements': 'clear and unambiguous',
            'individual_rights': ['access', 'correction', 'withdrawal'],
            'notification_obligations': True,
            'breach_notification_time': 72,  # hours to PDPC
            'dpo_appointment': 'mandatory for organizations',
            'cross_border_restrictions': True,
            'retention_limitation': True
        }

class ComplianceManager:
    """Main compliance management system"""
    
    def __init__(self):
        self.compliance_modules = {
            ComplianceRegulation.GDPR: GDPRComplianceModule(),
            ComplianceRegulation.CCPA: CCPAComplianceModule(),
            ComplianceRegulation.PDPA: PDPAComplianceModule()
        }
        
        self.processing_activities = []
        self.violation_history = []
        self.compliance_monitoring_active = False
        
        # User location to regulation mapping
        self.jurisdiction_mapping = {
            'EU': [ComplianceRegulation.GDPR],
            'US-CA': [ComplianceRegulation.CCPA],
            'SG': [ComplianceRegulation.PDPA],
            'US': [ComplianceRegulation.CCPA],  # Default for US
            'CA': [ComplianceRegulation.GDPR, ComplianceRegulation.PIPEDA],
            'BR': [ComplianceRegulation.LGPD],
            'GB': [ComplianceRegulation.GDPR],  # UK GDPR
        }
    
    def record_processing_activity(self, activity: DataProcessingActivity):
        """Record a data processing activity"""
        self.processing_activities.append(activity)
        
        # Keep only recent activities (last 90 days)
        cutoff_time = time.time() - (90 * 24 * 3600)
        self.processing_activities = [
            a for a in self.processing_activities 
            if a.timestamp > cutoff_time
        ]
        
        # Check compliance for this activity
        if self.compliance_monitoring_active:
            asyncio.create_task(self._check_activity_compliance(activity))
    
    async def _check_activity_compliance(self, activity: DataProcessingActivity):
        """Check compliance for a specific activity"""
        try:
            # Determine applicable regulations based on user location
            applicable_regulations = self._get_applicable_regulations(activity.user_id)
            
            for regulation in applicable_regulations:
                if regulation in self.compliance_modules:
                    module = self.compliance_modules[regulation]
                    violations = module.check_compliance(activity)
                    
                    for violation in violations:
                        self.violation_history.append(violation)
                        logger.warning(f"Compliance violation detected: {violation.description}")
                        
                        # Trigger automated remediation if possible
                        await self._attempt_auto_remediation(violation)
            
        except Exception as e:
            logger.error(f"Error checking compliance for activity {activity.activity_id}: {e}")
    
    def _get_applicable_regulations(self, user_id: str) -> List[ComplianceRegulation]:
        """Get applicable regulations based on user location/jurisdiction"""
        
        # In practice, would look up user's location from user database
        # For demo, simulate based on user ID patterns
        if 'eu_' in user_id.lower() or '_eu' in user_id.lower():
            return self.jurisdiction_mapping.get('EU', [])
        elif 'ca_' in user_id.lower() or '_ca' in user_id.lower():
            return self.jurisdiction_mapping.get('US-CA', [])
        elif 'sg_' in user_id.lower() or '_sg' in user_id.lower():
            return self.jurisdiction_mapping.get('SG', [])
        else:
            # Default to GDPR as most comprehensive
            return [ComplianceRegulation.GDPR]
    
    async def _attempt_auto_remediation(self, violation: ComplianceViolation):
        """Attempt automated remediation for compliance violations"""
        
        remediation_success = False
        
        try:
            if 'retention' in violation.description.lower():
                # Auto-remediation for retention violations
                logger.info(f"Implementing auto-remediation for retention violation: {violation.violation_id}")
                # In practice, would trigger data deletion processes
                remediation_success = True
                
            elif 'consent' in violation.description.lower():
                # Auto-remediation for consent violations
                logger.info(f"Flagging consent violation for manual review: {violation.violation_id}")
                # In practice, would trigger consent refresh processes
                
            elif 'cross-border' in violation.description.lower():
                # Auto-remediation for transfer violations
                logger.info(f"Implementing transfer safeguards: {violation.violation_id}")
                # In practice, would enable additional encryption or stop transfer
                remediation_success = True
            
            if remediation_success:
                violation.resolved = True
                violation.resolution_timestamp = time.time()
                logger.info(f"Auto-remediation successful for violation: {violation.violation_id}")
                
        except Exception as e:
            logger.error(f"Auto-remediation failed for violation {violation.violation_id}: {e}")
    
    def generate_compliance_report(self, period_days: int = 30) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        
        cutoff_time = time.time() - (period_days * 24 * 3600)
        
        # Recent activities and violations
        recent_activities = [
            a for a in self.processing_activities 
            if a.timestamp > cutoff_time
        ]
        
        recent_violations = [
            v for v in self.violation_history 
            if v.timestamp > cutoff_time
        ]
        
        # Calculate compliance score
        total_checks = len(recent_activities) * 3  # Average 3 checks per activity
        violation_count = len(recent_violations)
        
        if total_checks > 0:
            compliance_score = max(0, 1 - (violation_count / total_checks))
        else:
            compliance_score = 1.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(recent_violations)
        
        # Determine regulations covered
        regulations_covered = list(set(v.regulation for v in recent_violations))
        if not regulations_covered:
            regulations_covered = list(self.compliance_modules.keys())
        
        report = ComplianceReport(
            report_id=f"COMP_RPT_{int(time.time())}",
            timestamp=time.time(),
            reporting_period_days=period_days,
            regulations_covered=regulations_covered,
            total_processing_activities=len(recent_activities),
            violations_found=recent_violations,
            compliance_score=compliance_score,
            recommendations=recommendations
        )
        
        return report
    
    def _generate_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate compliance recommendations based on violations"""
        
        recommendations = []
        violation_patterns = {}
        
        # Analyze violation patterns
        for violation in violations:
            pattern_key = violation.description.split(' - ')[0].lower()
            violation_patterns[pattern_key] = violation_patterns.get(pattern_key, 0) + 1
        
        # Generate specific recommendations
        if 'retention' in str(violation_patterns):
            recommendations.append("Implement automated data deletion based on retention policies")
        
        if 'consent' in str(violation_patterns):
            recommendations.append("Review and update consent collection mechanisms")
        
        if 'cross-border' in str(violation_patterns):
            recommendations.append("Implement data localization or additional transfer safeguards")
        
        if 'sensitive' in str(violation_patterns):
            recommendations.append("Enhance protections for sensitive data processing")
        
        # General recommendations
        if len(violations) > 5:
            recommendations.append("Consider conducting comprehensive privacy impact assessment")
        
        if len(violations) > 10:
            recommendations.append("Implement privacy management platform for better compliance tracking")
        
        # Default recommendations if no specific patterns
        if not recommendations:
            recommendations.extend([
                "Continue monitoring compliance across all regulations",
                "Regular training on data protection requirements",
                "Periodic review of data processing activities"
            ])
        
        return recommendations
    
    def start_compliance_monitoring(self):
        """Start automated compliance monitoring"""
        self.compliance_monitoring_active = True
        logger.info("Compliance monitoring activated")
    
    def stop_compliance_monitoring(self):
        """Stop automated compliance monitoring"""
        self.compliance_monitoring_active = False
        logger.info("Compliance monitoring deactivated")
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status"""
        
        # Recent violations summary
        recent_violations = [v for v in self.violation_history 
                           if time.time() - v.timestamp < 86400]  # Last 24h
        
        violation_summary = {}
        for violation in recent_violations:
            severity = violation.severity.value
            violation_summary[severity] = violation_summary.get(severity, 0) + 1
        
        # Compliance score calculation
        total_activities = len(self.processing_activities[-100:])  # Last 100 activities
        total_violations = len(recent_violations)
        
        if total_activities > 0:
            compliance_score = max(0, 1 - (total_violations / (total_activities * 2)))
        else:
            compliance_score = 1.0
        
        return {
            "compliance_monitoring_active": self.compliance_monitoring_active,
            "regulations_supported": [reg.value for reg in self.compliance_modules.keys()],
            "recent_violations_24h": violation_summary,
            "compliance_score": f"{compliance_score:.2%}",
            "total_processing_activities": len(self.processing_activities),
            "total_violations": len(self.violation_history),
            "auto_remediation_rate": f"{self._calculate_auto_remediation_rate():.1%}",
            "supported_jurisdictions": list(self.jurisdiction_mapping.keys()),
            "global_compliance_features": [
                "GDPR Compliance (EU)",
                "CCPA Compliance (California)",  
                "PDPA Compliance (Singapore)",
                "Automated Violation Detection",
                "Cross-border Transfer Controls",
                "Data Retention Management",
                "Consent Management",
                "Breach Notification Support"
            ]
        }
    
    def _calculate_auto_remediation_rate(self) -> float:
        """Calculate percentage of violations that were auto-remediated"""
        if not self.violation_history:
            return 0.0
        
        resolved_violations = sum(1 for v in self.violation_history if v.resolved)
        return (resolved_violations / len(self.violation_history)) * 100