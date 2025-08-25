"""
Breakthrough Performance Detector
Identifies and validates breakthrough performance achievements in quantum HVAC optimization
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import time
import logging
from enum import Enum
import json

logger = logging.getLogger(__name__)

class BreakthroughType(Enum):
    ENERGY_EFFICIENCY = "energy_efficiency"
    QUANTUM_ADVANTAGE = "quantum_advantage" 
    COMPUTATION_SPEED = "computation_speed"
    COMFORT_OPTIMIZATION = "comfort_optimization"
    SYSTEM_RESILIENCE = "system_resilience"
    ALGORITHMIC_INNOVATION = "algorithmic_innovation"

@dataclass
class BreakthroughCriteria:
    """Criteria for detecting breakthrough performance"""
    energy_reduction_threshold: float = 20.0  # % improvement over baseline
    quantum_speedup_threshold: float = 2.0    # speedup factor
    comfort_score_threshold: float = 95.0     # comfort satisfaction %
    stability_threshold: float = 0.9          # system stability index
    consistency_window: int = 10              # number of consistent results required
    statistical_confidence: float = 0.95     # statistical confidence level

@dataclass
class BreakthroughEvidence:
    """Evidence supporting a breakthrough claim"""
    metric_name: str
    current_value: float
    baseline_value: float
    improvement_factor: float
    confidence_level: float
    sample_size: int
    statistical_significance: bool

@dataclass
class BreakthroughRecord:
    """Complete breakthrough record"""
    breakthrough_id: str
    breakthrough_type: BreakthroughType
    timestamp: float
    evidence: List[BreakthroughEvidence]
    performance_metrics: Dict[str, float]
    experimental_conditions: Dict[str, Any]
    validation_status: str
    reproducibility_score: float

class StatisticalValidator:
    """Statistical validation for breakthrough claims"""
    
    @staticmethod
    def t_test_breakthrough(current_samples: List[float], 
                          baseline_samples: List[float],
                          confidence_level: float = 0.95) -> Tuple[bool, float]:
        """Perform t-test to validate breakthrough claim"""
        
        if len(current_samples) < 3 or len(baseline_samples) < 3:
            return False, 0.0
        
        from scipy import stats
        
        try:
            # Perform two-sample t-test
            t_statistic, p_value = stats.ttest_ind(current_samples, baseline_samples)
            
            # Check if improvement is statistically significant
            alpha = 1 - confidence_level
            is_significant = p_value < alpha and np.mean(current_samples) > np.mean(baseline_samples)
            
            return is_significant, 1 - p_value
        
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return False, 0.0
    
    @staticmethod
    def effect_size_calculation(current_samples: List[float], 
                              baseline_samples: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        
        if len(current_samples) < 2 or len(baseline_samples) < 2:
            return 0.0
        
        mean1, mean2 = np.mean(current_samples), np.mean(baseline_samples)
        std1, std2 = np.std(current_samples, ddof=1), np.std(baseline_samples, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(current_samples), len(baseline_samples)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (mean1 - mean2) / pooled_std
        return cohens_d
    
    @staticmethod
    def consistency_analysis(values: List[float], threshold: float) -> Tuple[bool, float]:
        """Analyze consistency of breakthrough performance"""
        
        if len(values) < 5:
            return False, 0.0
        
        # Check what percentage of values meet threshold
        meets_threshold = [v >= threshold for v in values]
        consistency_rate = sum(meets_threshold) / len(meets_threshold)
        
        # Calculate stability (low variance relative to mean)
        cv = np.std(values) / max(np.mean(values), 0.001)  # Coefficient of variation
        stability_score = max(0, 1 - cv)
        
        # Breakthrough is consistent if >80% meet threshold and stability is high
        is_consistent = consistency_rate > 0.8 and stability_score > 0.7
        
        return is_consistent, consistency_rate * stability_score

class BreakthroughDetector:
    """Main breakthrough detection system"""
    
    def __init__(self):
        self.criteria = BreakthroughCriteria()
        self.validator = StatisticalValidator()
        
        # Performance baselines (would be calibrated from historical data)
        self.baselines = {
            'energy_efficiency': 15.0,     # % energy reduction
            'quantum_speedup': 1.2,        # speedup factor
            'comfort_score': 80.0,         # comfort satisfaction %
            'solve_time': 0.5,             # seconds
            'stability_index': 0.7,        # stability measure
            'cost_reduction': 10.0,        # % cost reduction
        }
        
        # Historical performance data
        self.performance_history = []
        self.breakthrough_records = []
        
        # Detection state
        self.current_breakthrough_candidates = {}
        self.validation_buffer = {}
        
    def analyze_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current performance for breakthrough potential"""
        
        # Store performance data
        performance_record = {
            'timestamp': time.time(),
            'data': performance_data
        }
        self.performance_history.append(performance_record)
        
        # Limit history size
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
        
        # Analyze each metric for breakthrough potential
        breakthrough_analysis = {}
        
        # Energy efficiency analysis
        if 'energy_reduction_percent' in performance_data:
            energy_analysis = self._analyze_energy_breakthrough(
                performance_data['energy_reduction_percent']
            )
            breakthrough_analysis['energy_efficiency'] = energy_analysis
        
        # Quantum advantage analysis
        if 'quantum_advantage_factor' in performance_data:
            quantum_analysis = self._analyze_quantum_breakthrough(
                performance_data['quantum_advantage_factor']
            )
            breakthrough_analysis['quantum_advantage'] = quantum_analysis
        
        # Computation speed analysis
        if 'computation_speedup' in performance_data:
            speed_analysis = self._analyze_speed_breakthrough(
                performance_data['computation_speedup']
            )
            breakthrough_analysis['computation_speed'] = speed_analysis
        
        # Comfort optimization analysis
        if 'comfort_improvement' in performance_data:
            comfort_analysis = self._analyze_comfort_breakthrough(
                performance_data['comfort_improvement']
            )
            breakthrough_analysis['comfort_optimization'] = comfort_analysis
        
        # Overall breakthrough assessment
        overall_assessment = self._assess_overall_breakthrough(breakthrough_analysis)
        
        return {
            'breakthrough_analysis': breakthrough_analysis,
            'overall_assessment': overall_assessment,
            'detection_timestamp': time.time(),
            'analysis_confidence': self._calculate_analysis_confidence(breakthrough_analysis)
        }
    
    def _analyze_energy_breakthrough(self, current_energy_reduction: float) -> Dict[str, Any]:
        """Analyze energy efficiency for breakthrough potential"""
        
        baseline = self.baselines['energy_efficiency']
        improvement_factor = current_energy_reduction / baseline
        
        # Get recent energy performance
        recent_energy_data = [
            r['data'].get('energy_reduction_percent', baseline)
            for r in self.performance_history[-20:]
            if 'energy_reduction_percent' in r['data']
        ]
        
        # Statistical validation
        baseline_samples = [baseline] * 10  # Simulated baseline data
        is_significant, confidence = self.validator.t_test_breakthrough(
            recent_energy_data, baseline_samples, self.criteria.statistical_confidence
        )
        
        # Effect size
        effect_size = self.validator.effect_size_calculation(
            recent_energy_data, baseline_samples
        )
        
        # Consistency check
        is_consistent, consistency_score = self.validator.consistency_analysis(
            recent_energy_data, self.criteria.energy_reduction_threshold
        )
        
        breakthrough_potential = (
            current_energy_reduction >= self.criteria.energy_reduction_threshold and
            improvement_factor >= 1.3 and
            is_significant and
            is_consistent
        )
        
        return {
            'current_value': current_energy_reduction,
            'baseline_value': baseline,
            'improvement_factor': improvement_factor,
            'breakthrough_potential': breakthrough_potential,
            'statistical_significance': is_significant,
            'confidence_level': confidence,
            'effect_size': effect_size,
            'consistency_score': consistency_score,
            'threshold_met': current_energy_reduction >= self.criteria.energy_reduction_threshold,
            'evidence_strength': 'STRONG' if breakthrough_potential else 'WEAK'
        }
    
    def _analyze_quantum_breakthrough(self, quantum_advantage: float) -> Dict[str, Any]:
        """Analyze quantum advantage for breakthrough potential"""
        
        baseline = self.baselines['quantum_speedup']
        improvement_factor = quantum_advantage / baseline
        
        # Get recent quantum performance
        recent_quantum_data = [
            r['data'].get('quantum_advantage_factor', baseline)
            for r in self.performance_history[-20:]
            if 'quantum_advantage_factor' in r['data']
        ]
        
        # Statistical validation
        baseline_samples = [baseline] * 10
        is_significant, confidence = self.validator.t_test_breakthrough(
            recent_quantum_data, baseline_samples
        )
        
        # Consistency check
        is_consistent, consistency_score = self.validator.consistency_analysis(
            recent_quantum_data, self.criteria.quantum_speedup_threshold
        )
        
        breakthrough_potential = (
            quantum_advantage >= self.criteria.quantum_speedup_threshold and
            improvement_factor >= 1.5 and
            is_significant and
            is_consistent
        )
        
        return {
            'current_value': quantum_advantage,
            'baseline_value': baseline,
            'improvement_factor': improvement_factor,
            'breakthrough_potential': breakthrough_potential,
            'statistical_significance': is_significant,
            'confidence_level': confidence,
            'consistency_score': consistency_score,
            'threshold_met': quantum_advantage >= self.criteria.quantum_speedup_threshold,
            'evidence_strength': 'STRONG' if breakthrough_potential else 'MODERATE'
        }
    
    def _analyze_speed_breakthrough(self, computation_speedup: float) -> Dict[str, Any]:
        """Analyze computation speed for breakthrough potential"""
        
        baseline_time = self.baselines['solve_time']
        # Convert speedup to time reduction percentage
        time_reduction = (1 - 1/computation_speedup) * 100 if computation_speedup > 1 else 0
        
        # Get recent speed data
        recent_speed_data = [
            r['data'].get('computation_speedup', 1.0)
            for r in self.performance_history[-20:]
            if 'computation_speedup' in r['data']
        ]
        
        breakthrough_potential = (
            computation_speedup >= 2.0 and
            time_reduction >= 50 and
            len([s for s in recent_speed_data if s >= 2.0]) >= 7  # Consistent speedup
        )
        
        return {
            'current_speedup': computation_speedup,
            'time_reduction_percent': time_reduction,
            'breakthrough_potential': breakthrough_potential,
            'consistent_performance': len([s for s in recent_speed_data if s >= 2.0]),
            'evidence_strength': 'STRONG' if breakthrough_potential else 'WEAK'
        }
    
    def _analyze_comfort_breakthrough(self, comfort_score: float) -> Dict[str, Any]:
        """Analyze comfort optimization for breakthrough potential"""
        
        baseline = self.baselines['comfort_score']
        improvement = comfort_score - baseline
        
        # Get recent comfort data
        recent_comfort_data = [
            r['data'].get('comfort_improvement', baseline)
            for r in self.performance_history[-15:]
            if 'comfort_improvement' in r['data']
        ]
        
        # High comfort is breakthrough if consistently above threshold
        breakthrough_potential = (
            comfort_score >= self.criteria.comfort_score_threshold and
            improvement >= 10 and
            len([c for c in recent_comfort_data if c >= self.criteria.comfort_score_threshold]) >= 10
        )
        
        return {
            'current_score': comfort_score,
            'baseline_score': baseline,
            'improvement': improvement,
            'breakthrough_potential': breakthrough_potential,
            'consistent_high_performance': len([c for c in recent_comfort_data if c >= 90]),
            'evidence_strength': 'STRONG' if breakthrough_potential else 'MODERATE'
        }
    
    def _assess_overall_breakthrough(self, analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall breakthrough potential across all metrics"""
        
        breakthrough_indicators = []
        strong_evidence_count = 0
        
        for metric, metric_analysis in analysis.items():
            if metric_analysis.get('breakthrough_potential', False):
                breakthrough_indicators.append(metric)
            
            if metric_analysis.get('evidence_strength') == 'STRONG':
                strong_evidence_count += 1
        
        # Overall breakthrough if multiple indicators with strong evidence
        overall_breakthrough = (
            len(breakthrough_indicators) >= 2 and
            strong_evidence_count >= 1
        )
        
        # Calculate composite breakthrough score
        breakthrough_scores = []
        for metric_analysis in analysis.values():
            if 'improvement_factor' in metric_analysis:
                breakthrough_scores.append(metric_analysis['improvement_factor'])
        
        composite_score = np.mean(breakthrough_scores) if breakthrough_scores else 1.0
        
        return {
            'overall_breakthrough_detected': overall_breakthrough,
            'breakthrough_indicators': breakthrough_indicators,
            'strong_evidence_count': strong_evidence_count,
            'composite_breakthrough_score': composite_score,
            'confidence_level': 'HIGH' if overall_breakthrough and strong_evidence_count >= 2 else 'MEDIUM',
            'validation_required': overall_breakthrough,
            'next_steps': self._recommend_next_steps(overall_breakthrough, breakthrough_indicators)
        }
    
    def _calculate_analysis_confidence(self, analysis: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall confidence in breakthrough analysis"""
        
        confidence_factors = []
        
        for metric_analysis in analysis.values():
            # Statistical confidence
            if 'confidence_level' in metric_analysis:
                confidence_factors.append(metric_analysis['confidence_level'])
            
            # Consistency score
            if 'consistency_score' in metric_analysis:
                confidence_factors.append(metric_analysis['consistency_score'])
            
            # Sample size factor (more data = higher confidence)
            sample_size = len(self.performance_history)
            size_factor = min(1.0, sample_size / 50)  # Full confidence with 50+ samples
            confidence_factors.append(size_factor)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _recommend_next_steps(self, breakthrough_detected: bool, 
                            indicators: List[str]) -> List[str]:
        """Recommend next steps based on breakthrough analysis"""
        
        if not breakthrough_detected:
            return [
                "Continue monitoring performance",
                "Collect more data for statistical validation",
                "Consider parameter tuning for improvement"
            ]
        
        recommendations = []
        
        if 'energy_efficiency' in indicators:
            recommendations.extend([
                "Validate energy breakthrough with extended testing",
                "Document energy optimization methodology",
                "Prepare for deployment at scale"
            ])
        
        if 'quantum_advantage' in indicators:
            recommendations.extend([
                "Conduct quantum advantage validation study",
                "Compare with classical baselines",
                "Publish quantum breakthrough findings"
            ])
        
        if 'computation_speed' in indicators:
            recommendations.extend([
                "Benchmark speed improvements thoroughly",
                "Test scalability of speed gains",
                "Optimize for production deployment"
            ])
        
        # General breakthrough recommendations
        recommendations.extend([
            "Initiate formal breakthrough validation protocol",
            "Prepare technical documentation",
            "Plan reproducibility experiments",
            "Consider patent applications for novel methods"
        ])
        
        return recommendations
    
    def validate_breakthrough(self, breakthrough_candidate: Dict[str, Any]) -> BreakthroughRecord:
        """Formally validate a breakthrough candidate"""
        
        # Generate unique breakthrough ID
        breakthrough_id = f"BT_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        # Determine breakthrough type
        breakthrough_type = self._determine_breakthrough_type(breakthrough_candidate)
        
        # Collect evidence
        evidence = self._collect_breakthrough_evidence(breakthrough_candidate)
        
        # Validation status
        validation_status = "VALIDATED" if len(evidence) >= 2 else "REQUIRES_MORE_DATA"
        
        # Reproducibility score
        reproducibility_score = self._calculate_reproducibility_score(breakthrough_candidate)
        
        breakthrough_record = BreakthroughRecord(
            breakthrough_id=breakthrough_id,
            breakthrough_type=breakthrough_type,
            timestamp=time.time(),
            evidence=evidence,
            performance_metrics=breakthrough_candidate.get('breakthrough_analysis', {}),
            experimental_conditions=breakthrough_candidate.get('experimental_conditions', {}),
            validation_status=validation_status,
            reproducibility_score=reproducibility_score
        )
        
        # Store breakthrough record
        self.breakthrough_records.append(breakthrough_record)
        
        logger.info(f"Breakthrough {breakthrough_id} validated: {validation_status}")
        
        return breakthrough_record
    
    def _determine_breakthrough_type(self, candidate: Dict[str, Any]) -> BreakthroughType:
        """Determine the primary type of breakthrough"""
        
        analysis = candidate.get('breakthrough_analysis', {})
        
        # Find the strongest breakthrough indicator
        max_improvement = 0
        breakthrough_type = BreakthroughType.ENERGY_EFFICIENCY
        
        type_mapping = {
            'energy_efficiency': BreakthroughType.ENERGY_EFFICIENCY,
            'quantum_advantage': BreakthroughType.QUANTUM_ADVANTAGE,
            'computation_speed': BreakthroughType.COMPUTATION_SPEED,
            'comfort_optimization': BreakthroughType.COMFORT_OPTIMIZATION
        }
        
        for metric, metric_analysis in analysis.items():
            if metric in type_mapping and 'improvement_factor' in metric_analysis:
                improvement = metric_analysis['improvement_factor']
                if improvement > max_improvement:
                    max_improvement = improvement
                    breakthrough_type = type_mapping[metric]
        
        return breakthrough_type
    
    def _collect_breakthrough_evidence(self, candidate: Dict[str, Any]) -> List[BreakthroughEvidence]:
        """Collect detailed evidence for breakthrough validation"""
        
        evidence_list = []
        analysis = candidate.get('breakthrough_analysis', {})
        
        for metric, metric_analysis in analysis.items():
            if metric_analysis.get('breakthrough_potential', False):
                
                evidence = BreakthroughEvidence(
                    metric_name=metric,
                    current_value=metric_analysis.get('current_value', 0),
                    baseline_value=metric_analysis.get('baseline_value', 0),
                    improvement_factor=metric_analysis.get('improvement_factor', 1),
                    confidence_level=metric_analysis.get('confidence_level', 0.5),
                    sample_size=len(self.performance_history),
                    statistical_significance=metric_analysis.get('statistical_significance', False)
                )
                
                evidence_list.append(evidence)
        
        return evidence_list
    
    def _calculate_reproducibility_score(self, candidate: Dict[str, Any]) -> float:
        """Calculate how reproducible the breakthrough is"""
        
        # Check consistency across recent performance
        consistency_scores = []
        
        analysis = candidate.get('breakthrough_analysis', {})
        for metric_analysis in analysis.values():
            if 'consistency_score' in metric_analysis:
                consistency_scores.append(metric_analysis['consistency_score'])
        
        if not consistency_scores:
            return 0.5
        
        # Reproducibility based on consistency and sample size
        base_reproducibility = np.mean(consistency_scores)
        sample_size_factor = min(1.0, len(self.performance_history) / 30)
        
        return base_reproducibility * sample_size_factor
    
    def get_breakthrough_summary(self) -> Dict[str, Any]:
        """Get summary of all breakthrough detection activities"""
        
        total_breakthroughs = len(self.breakthrough_records)
        validated_breakthroughs = len([r for r in self.breakthrough_records 
                                     if r.validation_status == "VALIDATED"])
        
        # Breakthrough types distribution
        type_distribution = {}
        for record in self.breakthrough_records:
            bt_type = record.breakthrough_type.value
            type_distribution[bt_type] = type_distribution.get(bt_type, 0) + 1
        
        # Recent performance trends
        recent_performance = self.performance_history[-20:] if self.performance_history else []
        
        trends = {}
        if recent_performance:
            for metric in ['energy_reduction_percent', 'quantum_advantage_factor', 'comfort_improvement']:
                values = [r['data'].get(metric) for r in recent_performance if metric in r['data']]
                if values:
                    trend_slope = np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
                    trends[metric] = {
                        'current_avg': np.mean(values),
                        'trend_slope': trend_slope,
                        'improving': trend_slope > 0
                    }
        
        return {
            "breakthrough_detection_status": "ACTIVE",
            "total_performance_analyses": len(self.performance_history),
            "breakthrough_candidates_identified": total_breakthroughs,
            "validated_breakthroughs": validated_breakthroughs,
            "validation_success_rate": validated_breakthroughs / total_breakthroughs if total_breakthroughs > 0 else 0,
            "breakthrough_type_distribution": type_distribution,
            "recent_performance_trends": trends,
            "detection_criteria": {
                "energy_threshold": self.criteria.energy_reduction_threshold,
                "quantum_speedup_threshold": self.criteria.quantum_speedup_threshold,
                "comfort_threshold": self.criteria.comfort_score_threshold,
                "statistical_confidence": self.criteria.statistical_confidence
            },
            "latest_breakthrough": (
                asdict(self.breakthrough_records[-1]) if self.breakthrough_records else None
            )
        }