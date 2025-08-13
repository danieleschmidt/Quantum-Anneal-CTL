"""
Compliance and regulatory framework for quantum HVAC systems.
"""

import logging
import json
import hashlib
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class ComplianceRegulation(Enum):
    """Supported compliance regulations."""
    GDPR = "gdpr"              # General Data Protection Regulation (EU)
    CCPA = "ccpa"              # California Consumer Privacy Act (US)
    PDPA = "pdpa"              # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"              # Lei Geral de Proteção de Dados (Brazil)
    ISO27001 = "iso27001"      # Information Security Management
    SOC2 = "soc2"              # Service Organization Control 2
    HIPAA = "hipaa"            # Health Insurance Portability and Accountability Act
    ENERGY_STAR = "energy_star" # Energy efficiency standards

@dataclass
class DataProcessingRecord:
    """Record of personal data processing activity."""
    timestamp: float
    data_type: str
    purpose: str
    legal_basis: str
    retention_period: int  # days
    user_id: Optional[str] = None
    building_id: Optional[str] = None
    processing_location: str = "local"

@dataclass
class AuditLogEntry:
    """Audit log entry for compliance tracking."""
    timestamp: float
    event_type: str
    user_id: Optional[str]
    building_id: Optional[str]
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    compliance_flags: List[str] = None

class ComplianceManager:
    """Manages compliance with various data protection and energy regulations."""
    
    def __init__(self):
        self.enabled_regulations: Set[ComplianceRegulation] = set()
        self.data_processing_records: List[DataProcessingRecord] = []
        self.audit_log: List[AuditLogEntry] = []
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.data_retention_policies: Dict[str, int] = {}
        
        self.logger = logging.getLogger(__name__)
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default compliance policies."""
        # Default data retention periods (in days)
        self.data_retention_policies = {
            "occupancy_data": 90,     # 3 months for occupancy patterns
            "energy_consumption": 365, # 1 year for energy data
            "temperature_data": 30,    # 30 days for comfort data
            "optimization_results": 180, # 6 months for performance analysis
            "user_preferences": 730,   # 2 years for user settings
            "audit_logs": 2555,       # 7 years for audit compliance
            "error_logs": 90,         # 3 months for error tracking
            "performance_metrics": 365 # 1 year for system metrics
        }
        
        # Enable common regulations by default
        self.enable_regulation(ComplianceRegulation.GDPR)
        self.enable_regulation(ComplianceRegulation.ISO27001)
        self.enable_regulation(ComplianceRegulation.ENERGY_STAR)
    
    def enable_regulation(self, regulation: ComplianceRegulation):
        """Enable compliance with a specific regulation."""
        self.enabled_regulations.add(regulation)
        self.logger.info(f"Enabled compliance for: {regulation.value}")
        
        # Log compliance enablement
        self._log_audit_event(
            event_type="compliance_enabled",
            action=f"enabled_{regulation.value}_compliance",
            details={"regulation": regulation.value}
        )
    
    def disable_regulation(self, regulation: ComplianceRegulation):
        """Disable compliance with a specific regulation."""
        if regulation in self.enabled_regulations:
            self.enabled_regulations.remove(regulation)
            self.logger.info(f"Disabled compliance for: {regulation.value}")
            
            self._log_audit_event(
                event_type="compliance_disabled",
                action=f"disabled_{regulation.value}_compliance",
                details={"regulation": regulation.value}
            )
    
    def record_data_processing(self, data_type: str, purpose: str, legal_basis: str,
                             user_id: Optional[str] = None, building_id: Optional[str] = None):
        """Record data processing activity for compliance."""
        retention_period = self.data_retention_policies.get(data_type, 365)
        
        record = DataProcessingRecord(
            timestamp=time.time(),
            data_type=data_type,
            purpose=purpose,
            legal_basis=legal_basis,
            retention_period=retention_period,
            user_id=user_id,
            building_id=building_id
        )
        
        self.data_processing_records.append(record)
        
        # Log for GDPR compliance
        if ComplianceRegulation.GDPR in self.enabled_regulations:
            self._log_audit_event(
                event_type="data_processing",
                user_id=user_id,
                building_id=building_id,
                action="process_personal_data",
                details={
                    "data_type": data_type,
                    "purpose": purpose,
                    "legal_basis": legal_basis,
                    "retention_days": retention_period
                }
            )
    
    def record_user_consent(self, user_id: str, purpose: str, granted: bool,
                          consent_text: str, building_id: Optional[str] = None):
        """Record user consent for data processing."""
        if user_id not in self.consent_records:
            self.consent_records[user_id] = {}
        
        self.consent_records[user_id][purpose] = {
            "granted": granted,
            "timestamp": time.time(),
            "consent_text": consent_text,
            "building_id": building_id,
            "checksum": hashlib.sha256(consent_text.encode()).hexdigest()
        }
        
        self._log_audit_event(
            event_type="consent_update",
            user_id=user_id,
            building_id=building_id,
            action="record_user_consent",
            details={
                "purpose": purpose,
                "granted": granted,
                "consent_checksum": self.consent_records[user_id][purpose]["checksum"]
            }
        )
        
        self.logger.info(f"Recorded consent for user {user_id}: {purpose} = {granted}")
    
    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given consent for a specific purpose."""
        if user_id in self.consent_records:
            consent_info = self.consent_records[user_id].get(purpose)
            if consent_info:
                return consent_info["granted"]
        return False
    
    def _log_audit_event(self, event_type: str, action: str, details: Dict[str, Any],
                        user_id: Optional[str] = None, building_id: Optional[str] = None,
                        ip_address: Optional[str] = None):
        """Log audit event for compliance tracking."""
        # Determine compliance flags
        compliance_flags = []
        for regulation in self.enabled_regulations:
            if self._event_affects_regulation(event_type, regulation):
                compliance_flags.append(regulation.value)
        
        entry = AuditLogEntry(
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            building_id=building_id,
            action=action,
            details=details,
            ip_address=ip_address,
            compliance_flags=compliance_flags
        )
        
        self.audit_log.append(entry)
    
    def _event_affects_regulation(self, event_type: str, regulation: ComplianceRegulation) -> bool:
        """Check if an event type affects a specific regulation."""
        gdpr_events = {
            "data_processing", "consent_update", "data_deletion", 
            "data_export", "data_access", "privacy_setting_change"
        }
        
        security_events = {
            "login", "logout", "access_denied", "permission_change",
            "security_incident", "system_access"
        }
        
        energy_events = {
            "optimization_completed", "energy_consumption", 
            "efficiency_report", "carbon_footprint"
        }
        
        if regulation == ComplianceRegulation.GDPR and event_type in gdpr_events:
            return True
        elif regulation in [ComplianceRegulation.ISO27001, ComplianceRegulation.SOC2] and event_type in security_events:
            return True
        elif regulation == ComplianceRegulation.ENERGY_STAR and event_type in energy_events:
            return True
        
        return False
    
    def cleanup_expired_data(self):
        """Clean up expired data according to retention policies."""
        current_time = time.time()
        
        # Clean up data processing records
        valid_records = []
        for record in self.data_processing_records:
            age_days = (current_time - record.timestamp) / (24 * 3600)
            if age_days < record.retention_period:
                valid_records.append(record)
            else:
                self.logger.info(f"Expired data processing record: {record.data_type}")
        
        cleaned_count = len(self.data_processing_records) - len(valid_records)
        self.data_processing_records = valid_records
        
        # Clean up audit logs (keep for 7 years by default)
        audit_retention_days = self.data_retention_policies.get("audit_logs", 2555)
        valid_audit_logs = []
        for entry in self.audit_log:
            age_days = (current_time - entry.timestamp) / (24 * 3600)
            if age_days < audit_retention_days:
                valid_audit_logs.append(entry)
        
        audit_cleaned_count = len(self.audit_log) - len(valid_audit_logs)
        self.audit_log = valid_audit_logs
        
        if cleaned_count > 0 or audit_cleaned_count > 0:
            self._log_audit_event(
                event_type="data_cleanup",
                action="automated_data_retention",
                details={
                    "processing_records_deleted": cleaned_count,
                    "audit_logs_deleted": audit_cleaned_count
                }
            )
            
            self.logger.info(f"Data cleanup: {cleaned_count} processing records, {audit_cleaned_count} audit logs")
    
    def generate_privacy_report(self, user_id: str) -> Dict[str, Any]:
        """Generate privacy report for a specific user (GDPR Article 15)."""
        if ComplianceRegulation.GDPR not in self.enabled_regulations:
            raise ValueError("GDPR compliance not enabled")
        
        # Collect user's data processing records
        user_processing = [
            asdict(record) for record in self.data_processing_records
            if record.user_id == user_id
        ]
        
        # Collect user's audit log entries
        user_audit_logs = [
            asdict(entry) for entry in self.audit_log
            if entry.user_id == user_id
        ]
        
        # Get consent records
        user_consents = self.consent_records.get(user_id, {})
        
        report = {
            "user_id": user_id,
            "report_generated": datetime.now().isoformat(),
            "data_processing_activities": user_processing,
            "consent_records": user_consents,
            "audit_trail": user_audit_logs,
            "data_retention_info": {
                data_type: f"{days} days" 
                for data_type, days in self.data_retention_policies.items()
            },
            "user_rights": {
                "right_to_access": "Fulfilled by this report",
                "right_to_rectification": "Contact system administrator",
                "right_to_erasure": "Request data deletion",
                "right_to_portability": "Data export available on request",
                "right_to_object": "Update consent preferences"
            }
        }
        
        self._log_audit_event(
            event_type="data_access",
            user_id=user_id,
            action="generate_privacy_report",
            details={"report_type": "gdpr_article_15"}
        )
        
        return report
    
    def delete_user_data(self, user_id: str, reason: str = "user_request"):
        """Delete all data for a specific user (GDPR Article 17)."""
        if ComplianceRegulation.GDPR not in self.enabled_regulations:
            raise ValueError("GDPR compliance not enabled")
        
        # Remove data processing records
        original_count = len(self.data_processing_records)
        self.data_processing_records = [
            record for record in self.data_processing_records
            if record.user_id != user_id
        ]
        processing_deleted = original_count - len(self.data_processing_records)
        
        # Remove consent records
        consent_deleted = 0
        if user_id in self.consent_records:
            consent_deleted = len(self.consent_records[user_id])
            del self.consent_records[user_id]
        
        # Anonymize audit logs (keep for compliance but remove PII)
        for entry in self.audit_log:
            if entry.user_id == user_id:
                entry.user_id = f"deleted_user_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}"
                entry.details = {"anonymized": True, "original_deleted": True}
        
        # Log the deletion
        self._log_audit_event(
            event_type="data_deletion",
            action="delete_user_data",
            details={
                "deleted_user_hash": hashlib.sha256(user_id.encode()).hexdigest()[:8],
                "reason": reason,
                "processing_records_deleted": processing_deleted,
                "consent_records_deleted": consent_deleted
            }
        )
        
        self.logger.info(f"Deleted all data for user {user_id}: {processing_deleted} processing records, {consent_deleted} consent records")
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status and statistics."""
        current_time = time.time()
        
        # Calculate data age statistics
        if self.data_processing_records:
            ages_days = [(current_time - record.timestamp) / (24 * 3600) for record in self.data_processing_records]
            avg_age = sum(ages_days) / len(ages_days)
            max_age = max(ages_days)
        else:
            avg_age = max_age = 0
        
        return {
            "enabled_regulations": [reg.value for reg in self.enabled_regulations],
            "data_processing_records": len(self.data_processing_records),
            "audit_log_entries": len(self.audit_log),
            "consent_records": len(self.consent_records),
            "data_retention_policies": self.data_retention_policies,
            "data_age_statistics": {
                "average_age_days": avg_age,
                "oldest_record_days": max_age
            },
            "compliance_summary": {
                "gdpr_enabled": ComplianceRegulation.GDPR in self.enabled_regulations,
                "security_standards": any(reg in self.enabled_regulations for reg in [ComplianceRegulation.ISO27001, ComplianceRegulation.SOC2]),
                "energy_compliance": ComplianceRegulation.ENERGY_STAR in self.enabled_regulations
            }
        }
    
    def export_compliance_data(self, filepath: str):
        """Export compliance data for auditing."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "enabled_regulations": [reg.value for reg in self.enabled_regulations],
            "data_processing_records": [asdict(record) for record in self.data_processing_records],
            "audit_log": [asdict(entry) for entry in self.audit_log],
            "consent_records": self.consent_records,
            "data_retention_policies": self.data_retention_policies,
            "compliance_status": self.get_compliance_status()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._log_audit_event(
            event_type="data_export",
            action="export_compliance_data",
            details={"export_file": filepath}
        )
        
        self.logger.info(f"Compliance data exported to: {filepath}")

# Global compliance manager instance
_compliance_manager = None

def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager instance."""
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = ComplianceManager()
    return _compliance_manager