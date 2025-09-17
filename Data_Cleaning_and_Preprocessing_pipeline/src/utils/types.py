from dataclasses import dataclass
from typing import Dict, List, Any, Optional, TypedDict
import pandas as pd


@dataclass
class ColumnProfile:
    """Profile for individual columns to understand their characteristics"""

    name: str
    dtype: str
    unique_count: int
    null_count: int
    null_percentage: float
    sample_values: List[str]
    inferred_type: (
        str  # 'numeric', 'categorical', 'datetime', 'text', 'binary', 'identifier'
    )
    inferred_role: str  # 'feature', 'target_candidate', 'identifier', 'metadata'
    characteristics: Dict[str, Any]


@dataclass
class DatasetProfile:
    """Comprehensive profile of any uploaded dataset"""

    shape: tuple
    column_profiles: List[ColumnProfile]
    target_candidates: List[str]
    problem_type_candidates: List[
        str
    ]  # ['classification', 'regression', 'time_series', 'clustering']
    complexity_indicators: Dict[str, Any]
    duplicate_count: int = 0
    duplicate_percentage: float = 0.0


@dataclass
class DataQualityIssue:
    """data quality issue structure"""

    issue_type: str
    severity: str  # "low", "medium", "high", "critical"
    affected_columns: List[str]
    description: str
    business_impact: str
    detection_details: Dict[str, Any]
    suggested_solutions: List[Dict[str, str]]


class PreprocessingState(TypedDict):
    """State for  preprocessing workflow"""

    raw_data: pd.DataFrame
    dataset_profile: Optional[DatasetProfile]
    target_column: Optional[str]
    problem_type: Optional[
        str
    ]  # 'classification', 'regression', 'time_series', 'clustering'
    user_choices: Dict[str, Any]  # User-defined choices for preprocessing and modeling
    data_quality_issues: List[DataQualityIssue]
    preprocessing_plan: Optional[Dict[str, Any]]  # Plan for data preprocessing steps
    cleaned_data: Optional[pd.DataFrame]
    recommended_algorithms: Optional[List[str]]
    preprocessing_report: Optional[str]
    error_message: Optional[str]
    pending_questions: List[Dict[str, Any]]  # Questions to ask user for clarification
    interaction_complete: bool  # Indicates if interaction is complete
