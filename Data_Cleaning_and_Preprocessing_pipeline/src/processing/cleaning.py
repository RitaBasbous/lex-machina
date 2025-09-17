import pandas as pd
from typing import Tuple, List
from ..utils.types import DatasetProfile


class DataCleaner:
    """Handles automatic data cleaning operations"""

    def perform_automatic_cleaning(
        self, df: pd.DataFrame, profile: DatasetProfile
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Perform automatic cleaning actions"""
        cleaning_actions = []

        print("   Performing automatic data cleaning...")

        # 1. Remove duplicate rows
        before_duplicates = len(df)
        df = df.drop_duplicates()
        after_duplicates = len(df)
        duplicates_removed = before_duplicates - after_duplicates
        if duplicates_removed > 0:
            cleaning_actions.append(
                f"Removed {duplicates_removed} duplicate rows ({(duplicates_removed / before_duplicates) * 100:.1f}%)"
            )
            print(f"     - Removed {duplicates_removed} duplicate rows")
        else:
            cleaning_actions.append("No duplicate rows found")
            print("     - No duplicate rows found")

        # 2. Remove columns with >70% missing values (more conservative)
        high_missing_cols = [
            col.name
            for col in profile.column_profiles
            if col.null_percentage > 70 and col.inferred_role != "identifier"
        ]

        if high_missing_cols:
            df = df.drop(columns=high_missing_cols)
            cleaning_actions.append(
                f"Automatically removed {len(high_missing_cols)} columns with >70% missing values: {high_missing_cols}"
            )
            print(
                f"     - Automatically removed {len(high_missing_cols)} columns with >70% missing values"
            )
        else:
            cleaning_actions.append("No columns with >70% missing values found")

        # 3. Remove rows with >80% missing data
        missing_threshold = 0.8
        before_rows = len(df)
        missing_per_row = df.isnull().sum(axis=1) / len(df.columns)
        rows_to_keep = missing_per_row <= missing_threshold
        df = df[rows_to_keep]
        after_rows = len(df)
        rows_removed = before_rows - after_rows

        if rows_removed > 0:
            cleaning_actions.append(
                f"Removed {rows_removed} rows with >80% missing data ({(rows_removed / before_rows) * 100:.1f}%)"
            )
            print(f"     - Removed {rows_removed} rows with >80% missing data")
        else:
            cleaning_actions.append("No rows with >80% missing data found")

        return df, cleaning_actions
