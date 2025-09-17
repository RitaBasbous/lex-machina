from ..utils.types import DatasetProfile, DataQualityIssue
import pandas as pd
from typing import List


class DataQualityAnalyzer:
    """Analyzes data quality issues for any type of dataset"""

    @staticmethod
    def identify_issues(
        dataset_profile: DatasetProfile, df: pd.DataFrame
    ) -> List[DataQualityIssue]:
        """Identify all data quality issues requiring user decisions"""

        issues = []

        # Outlier issues (with sample data details)
        issues.extend(DataQualityAnalyzer._analyze_outliers(dataset_profile, df))

        # Problem type selection issue (only classification and regression)
        issues.append(DataQualityAnalyzer._create_problem_type_selection_issue())

        # Missing values issues
        issues.extend(DataQualityAnalyzer._analyze_missing_values(dataset_profile, df))

        return issues

    @staticmethod
    def _analyze_outliers(
        profile: DatasetProfile, df: pd.DataFrame
    ) -> List[DataQualityIssue]:
        """Analyze outlier patterns with sample data details"""
        issues = []

        numeric_cols_with_outliers = []
        outlier_details = {}

        for col in profile.column_profiles:
            if col.inferred_type == "numeric" and col.characteristics.get(
                "has_outliers", False
            ):
                numeric_cols_with_outliers.append(col.name)

                # Get detailed outlier information
                series = df[col.name].dropna()
                if len(series) > 0:
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers_low = series[series < lower_bound]
                    outliers_high = series[series > upper_bound]

                    normal_min = series[series >= lower_bound].min()
                    normal_max = series[series <= upper_bound].max()

                    outlier_examples = []
                    if len(outliers_low) > 0:
                        outlier_examples.extend(outliers_low.head(3).tolist())
                    if len(outliers_high) > 0:
                        outlier_examples.extend(outliers_high.head(3).tolist())

                    outlier_details[col.name] = {
                        "normal_range": f"{normal_min:.2f} to {normal_max:.2f}",
                        "outlier_examples": [
                            f"{val:.2f}" for val in outlier_examples[:5]
                        ],
                        "outlier_count": len(outliers_low) + len(outliers_high),
                    }

        if numeric_cols_with_outliers:
            # Create detailed description with examples
            description_parts = []
            for col_name in numeric_cols_with_outliers:
                details = outlier_details[col_name]
                description_parts.append(
                    f"In column '{col_name}', values normally range from {details['normal_range']}, "
                    f"but there are {details['outlier_count']} outliers with values like: {', '.join(details['outlier_examples'])}"
                )

            full_description = (
                f"Outliers detected in {len(numeric_cols_with_outliers)} numeric columns:\n"
                + "\n".join(description_parts)
            )

            issues.append(
                DataQualityIssue(
                    issue_type="outliers_detected",
                    severity="medium",
                    affected_columns=numeric_cols_with_outliers,
                    description=full_description,
                    business_impact="Outliers may represent errors or important edge cases",
                    detection_details={"outlier_details": outlier_details},
                    suggested_solutions=[
                        {
                            "option": "keep_outliers",
                            "description": "Keep outliers - they might represent important patterns",
                            "pros": "No information loss, captures full business reality",
                            "cons": "May hurt model performance on typical cases",
                        },
                        {
                            "option": "cap_outliers",
                            "description": "Cap extreme values at 5th/95th percentiles",
                            "pros": "Reduces extreme influence while preserving patterns",
                            "cons": "Some information loss at the extremes",
                        },
                        {
                            "option": "remove_outliers",
                            "description": "Remove rows containing extreme outliers",
                            "pros": "Clean dataset optimized for typical patterns",
                            "cons": "Data loss, may miss important edge cases",
                        },
                    ],
                )
            )

        return issues

    @staticmethod
    def _analyze_missing_values(
        profile: DatasetProfile, df: pd.DataFrame
    ) -> List[DataQualityIssue]:
        """Analyze missing value patterns - only for 10-50% missing"""
        issues = []

        # Only process columns with 10-50% missing values
        medium_missing = [
            col.name
            for col in profile.column_profiles
            if 10 < col.null_percentage <= 50 and col.inferred_role != "identifier"
        ]

        if medium_missing:
            issues.append(
                DataQualityIssue(
                    issue_type="moderate_missing_values",
                    severity="medium",
                    affected_columns=medium_missing,
                    description=f"Columns {medium_missing} have 10-50% missing values",
                    business_impact="Moderate impact on model accuracy and reliability",
                    detection_details={
                        "missing_percentages": {
                            col: profile.column_profiles[i].null_percentage
                            for i, col in enumerate(df.columns)
                            if col in medium_missing
                        }
                    },
                    suggested_solutions=[
                        {
                            "option": "statistical_imputation",
                            "description": "Fill with statistical measures (mean/median/mode)",
                            "pros": "Simple, fast, preserves data distribution",
                            "cons": "Reduces variance, ignores relationships between variables",
                        },
                        {
                            "option": "predictive_imputation",
                            "description": "Predict missing values using other columns",
                            "pros": "More accurate, captures variable relationships",
                            "cons": "Complex, requires validation to avoid overfitting",
                        },
                    ],
                )
            )

        return issues

    @staticmethod
    def _create_problem_type_selection_issue() -> DataQualityIssue:
        """Create problem type selection issue - only classification and regression"""
        return DataQualityIssue(
            issue_type="problem_type_selection",
            severity="high",
            affected_columns=[],
            description="Please select the type of machine learning problem you want to solve",
            business_impact="Determines the type of analysis and algorithms available",
            detection_details={},
            suggested_solutions=[
                {
                    "option": "classification",
                    "description": "Predict categories/classes (e.g., spam/not spam, survived/not survived, disease/healthy)",
                    "pros": "Clear categorical predictions, well-established algorithms, easy to interpret results",
                    "cons": "Requires labeled target variable with discrete categories",
                },
                {
                    "option": "regression",
                    "description": "Predict continuous numerical values (e.g., price, age, temperature, sales amount)",
                    "pros": "Precise numerical predictions, captures relationships with continuous outcomes",
                    "cons": "Requires numerical target variable, predictions may need rounding for practical use",
                },
            ],
        )
