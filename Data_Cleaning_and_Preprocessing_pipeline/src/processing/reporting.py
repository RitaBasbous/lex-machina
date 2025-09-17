import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List
from ..utils.types import DatasetProfile


class ReportGenerator:
    """Handles report generation and file output"""
    @staticmethod
    def sanitize_for_json(obj):
        """Recursively convert non-JSON-serializable types to native Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, dict):
            return {k: ReportGenerator.sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ReportGenerator.sanitize_for_json(v) for v in obj]
        else:
            return obj


    def generate_comprehensive_analysis_report(
        self,
        original_df: pd.DataFrame,
        final_df: pd.DataFrame,
        profile: DatasetProfile,
        ai_choices: Dict[str, str],
        processing_steps: List[str],
        cleaning_actions: List[str],
        problem_type: str,
        target_column: str,
    ) -> str:
        """Generate comprehensive analysis report in JSON format"""

        # Data quality metrics
        original_missing_pct = (
            original_df.isnull().sum().sum() / (original_df.shape[0] * original_df.shape[1])
        ) * 100
        final_missing_pct = (
            (final_df.isnull().sum().sum() / (final_df.shape[0] * final_df.shape[1])) * 100
            if final_df is not None
            else 100
        )

        # Column type analysis
        type_counts = {}
        for col in profile.column_profiles:
            type_counts[col.inferred_type] = type_counts.get(col.inferred_type, 0) + 1

        # Outlier detection
        outlier_columns = [
            col.name for col in profile.column_profiles
            if col.characteristics.get("has_outliers", False)
        ]

        # Build JSON structure
        report = {
            "summary": {
                "rows": original_df.shape[0],
                "columns": original_df.shape[1],
                "processed_shape": final_df.shape if final_df is not None else None,
                "problem_type": problem_type,
                "target_column": target_column,
                "data_completeness": {
                    "original": round(100 - original_missing_pct, 1),
                    "final": round(100 - final_missing_pct, 1),
                    "improvement": round((100 - final_missing_pct) - (100 - original_missing_pct), 1),
                },
                "processing_success": final_df is not None,
            },
            "composition": {
                "type_counts": type_counts,
                "type_percentages": {
                    k: round(v / len(profile.column_profiles) * 100, 1)
                    for k, v in type_counts.items()
                },
            },
            "quality_assessment": {
                "duplicate_count": profile.duplicate_count,
                "duplicate_percentage": round(profile.duplicate_percentage, 1),
                "outlier_columns": outlier_columns,
            },
            "cleaning_actions": cleaning_actions,
            "processing_steps": processing_steps,
            "algorithm_rationale": {
                "problem_type": problem_type,
                "size_category": profile.complexity_indicators["size_category"],
                "feature_complexity": profile.complexity_indicators["feature_complexity"],
            },
            "technical_specs": {
                "engine": "AI-Driven Preprocessing Pipeline",
                "ai_choices": {
                k: v.replace("_", " ").title()
                for k, v in ai_choices.items()
            },
                "feature_engineering": "Automatic type conversion",
            },
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        sanitized_report = ReportGenerator.sanitize_for_json(report)

        # Save JSON report
        os.makedirs(r"D:\projects\lex machina\outputs", exist_ok=True)
        with open(r"D:\projects\lex machina\outputs\data_analysis_report.json", "w", encoding="utf-8") as file:
            json.dump(sanitized_report, file, indent=4)

        return json.dumps(sanitized_report, indent=4)
    
    def _calculate_readiness_score(
        self, profile: DatasetProfile, final_df: pd.DataFrame
    ) -> float:
        """Calculate a data readiness score for business understanding"""
        score = 100.0

        # Deduct for missing data issues
        if final_df is not None:
            missing_pct = (
                final_df.isnull().sum().sum() / (final_df.shape[0] * final_df.shape[1])
            ) * 100
            score -= missing_pct * 2  # 2 points per % missing

        # Deduct for duplicate issues
        score -= min(profile.duplicate_percentage * 0.5, 10)  # Max 10 points deduction

        # Deduct for complexity issues
        if profile.complexity_indicators["missing_data_complexity"] == "high":
            score -= 5

        if profile.complexity_indicators["high_cardinality_features"] > 3:
            score -= 5

        # Bonus for good structure
        if len(profile.target_candidates) > 0:
            score += 5

        if profile.complexity_indicators["feature_count"] > 5:
            score += 5

        return max(score, 0)

    def generate_files(
        self,
        problem_type: str,
        df_with_ids: pd.DataFrame,
        df_for_training: pd.DataFrame,
        target_column: str,
    ) -> None:
        """Generate various output files for analysis and reporting"""

        file_name = "processed_dataset"
        # Generate final dataset with all transformations applied
        # Move the target column to the end without renaming
        cols = [col for col in df_for_training.columns if col != target_column] + [target_column]
        df_for_training = df_for_training[cols]

        # Save to CSV
        df_for_training.to_csv(rf"D:\projects\lex machina\outputs\{file_name}_{problem_type}.csv", index=False)