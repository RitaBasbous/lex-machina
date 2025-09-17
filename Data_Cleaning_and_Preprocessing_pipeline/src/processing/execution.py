import pandas as pd
from sklearn.impute import KNNImputer
from typing import Dict, Tuple, List
from ..utils.types import DatasetProfile


class PreprocessingExecutor:
    """Executes the comprehensive preprocessing operations"""

    def execute_comprehensive_preprocessing(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        ai_choices: Dict[str, str],
        target_column: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Execute preprocessing with comprehensive missing value handling"""
        processing_steps = []

        print("4. Executing comprehensive preprocessing based on AI decisions...")

        # 1. Handle data type conversions FIRST
        print("   Step 4.1: Handling data type conversions...")
        for col_profile in profile.column_profiles:
            if (
                col_profile.dtype == "object"
                and col_profile.inferred_type == "numeric"
                and col_profile.name in df.columns
            ):
                try:
                    original_dtype = df[col_profile.name].dtype
                    df[col_profile.name] = pd.to_numeric(
                        df[col_profile.name], errors="coerce"
                    )
                    processing_steps.append(
                        f"Converted {col_profile.name} from {original_dtype} to numeric"
                    )
                    print(f"     - Converted {col_profile.name} to numeric")
                except Exception as e:
                    processing_steps.append(
                        f"Failed to convert {col_profile.name} to numeric: {str(e)}"
                    )
                    print(
                        f"     - Failed to convert {col_profile.name} to numeric: {e}"
                    )

        # 2. COMPREHENSIVE missing value handling - handle ALL missing values
        print("   Step 4.2: Comprehensive missing value handling...")
        missing_strategy = "statistical_imputation_median"  # Default fallback

        # Get AI decision for missing values
        for choice_id, choice in ai_choices.items():
            if "missing_values" in choice_id:
                missing_strategy = choice
                break

        # Handle missing values for ALL columns with any missing data
        columns_with_missing = [
            col
            for col in profile.column_profiles
            if col.null_count > 0 and col.name in df.columns
        ]

        if columns_with_missing:
            print(
                f"     - Processing {len(columns_with_missing)} columns with missing values"
            )

            for col_profile in columns_with_missing:
                col = col_profile.name
                missing_count = col_profile.null_count
                missing_pct = col_profile.null_percentage

                if col_profile.inferred_type == "numeric":
                    if missing_strategy.startswith("statistical_imputation"):
                        stat_method = (
                            missing_strategy.split("_")[-1]
                            if "_" in missing_strategy
                            else "median"
                        )

                        if stat_method == "mean":
                            fill_value = df[col].mean()
                        elif stat_method == "median":
                            fill_value = df[col].median()
                        else:  # mode fallback
                            fill_value = (
                                df[col].mode().iloc[0]
                                if not df[col].mode().empty
                                else df[col].median()
                            )

                        df[col] = df[col].fillna(fill_value)
                        processing_steps.append(
                            f"Filled {missing_count} missing values ({missing_pct:.1f}%) in {col} with {stat_method} ({fill_value:.2f})"
                        )

                    elif missing_strategy == "predictive_imputation":
                        # Use KNN for numeric
                        try:
                            imputer = KNNImputer(n_neighbors=min(5, len(df) // 10))
                            df[[col]] = imputer.fit_transform(df[[col]])
                            processing_steps.append(
                                f"Applied KNN imputation to {col} ({missing_count} values, {missing_pct:.1f}%)"
                            )
                        except:
                            # Fallback to median
                            fill_value = df[col].median()
                            df[col] = df[col].fillna(fill_value)
                            processing_steps.append(
                                f"KNN failed, used median imputation for {col}: {fill_value:.2f}"
                            )

                elif col_profile.inferred_type in ["categorical", "binary"]:
                    # Always use mode for categorical
                    mode_val = (
                        df[col].mode().iloc[0]
                        if not df[col].mode().empty
                        else "Unknown"
                    )
                    df[col] = df[col].fillna(mode_val)
                    processing_steps.append(
                        f"Filled {missing_count} missing values ({missing_pct:.1f}%) in {col} with mode: '{mode_val}'"
                    )

                elif col_profile.inferred_type == "datetime":
                    # Use forward fill for datetime
                    df[col] = df[col].fillna(method="ffill")
                    # If still missing, use backward fill
                    df[col] = df[col].fillna(method="bfill")
                    processing_steps.append(
                        f"Filled {missing_count} datetime missing values in {col} using forward/backward fill"
                    )

                else:
                    # For other types, use 'Unknown' or most frequent
                    fill_value = "Unknown"
                    if col_profile.unique_count > 0:
                        try:
                            fill_value = (
                                df[col].mode().iloc[0]
                                if not df[col].mode().empty
                                else "Unknown"
                            )
                        except:
                            fill_value = "Unknown"
                    df[col] = df[col].fillna(fill_value)
                    processing_steps.append(
                        f"Filled {missing_count} missing values in {col} with: '{fill_value}'"
                    )

            # Verify no missing values remain
            remaining_missing = df.isnull().sum().sum()
            if remaining_missing > 0:
                print(
                    f"     - WARNING: {remaining_missing} missing values still remain, applying final cleanup..."
                )
                # Final cleanup - fill any remaining with appropriate defaults
                for col in df.columns:
                    if df[col].isnull().any():
                        if df[col].dtype in ["int64", "float64"]:
                            df[col] = df[col].fillna(0)
                        else:
                            df[col] = df[col].fillna("Unknown")
                processing_steps.append(
                    f"Final cleanup: filled remaining {remaining_missing} missing values"
                )
            else:
                print("     - SUCCESS: All missing values handled completely")

        # 3. Handle outliers based on AI decision
        print("   Step 4.3: Handling outliers...")
        outlier_strategy = "keep_outliers"  # Default
        for choice_id, choice in ai_choices.items():
            if "outliers" in choice_id:
                outlier_strategy = choice
                break

        if outlier_strategy == "cap_outliers":
            outliers_capped = 0
            for col_profile in profile.column_profiles:
                if (
                    col_profile.inferred_type == "numeric"
                    and col_profile.characteristics.get("has_outliers", False)
                    and col_profile.name in df.columns
                ):
                    lower_cap = df[col_profile.name].quantile(0.05)
                    upper_cap = df[col_profile.name].quantile(0.95)

                    outliers_low = (df[col_profile.name] < lower_cap).sum()
                    outliers_high = (df[col_profile.name] > upper_cap).sum()
                    total_outliers = outliers_low + outliers_high

                    df[col_profile.name] = df[col_profile.name].clip(
                        lower=lower_cap, upper=upper_cap
                    )
                    outliers_capped += total_outliers
                    processing_steps.append(
                        f"Capped {total_outliers} outliers in {col_profile.name} at 5th/95th percentiles"
                    )

            print(f"     - Capped {outliers_capped} outliers at 5th/95th percentiles")

        elif outlier_strategy == "remove_outliers":
            original_rows = len(df)
            for col_profile in profile.column_profiles:
                if (
                    col_profile.inferred_type == "numeric"
                    and col_profile.characteristics.get("has_outliers", False)
                    and col_profile.name in df.columns
                ):
                    col = col_profile.name
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            rows_removed = original_rows - len(df)
            if rows_removed > 0:
                processing_steps.append(
                    f"Removed {rows_removed} rows containing outliers"
                )
                print(f"     - Removed {rows_removed} outlier rows")

        else:  # keep_outliers
            processing_steps.append("Kept all outliers as requested by AI")
            print("     - Kept all outliers as requested by AI")

        # 4. Prepare final datasets
        print("   Step 4.4: Preparing final datasets...")
        identifier_cols = [
            col.name
            for col in profile.column_profiles
            if col.inferred_type == "identifier"
        ]

        # Create two versions: one with identifiers, one without
        df_with_identifiers = df.copy()
        df_for_training = df.drop(columns=identifier_cols, errors="ignore")

        if identifier_cols:
            processing_steps.append(
                f"Excluded identifier columns from training dataset: {identifier_cols}"
            )
            print(f"     - Excluded identifier columns: {identifier_cols}")
        else:
            processing_steps.append("No identifier columns found to exclude")

        print("   Comprehensive preprocessing completed successfully.")
        print(f"     - Training dataset shape: {df_for_training.shape}")
        print(f"     - Dataset with IDs shape: {df_with_identifiers.shape}")
        print(
            f"     - Missing values remaining: {df_for_training.isnull().sum().sum()}"
        )

        return df_for_training, df_with_identifiers, processing_steps
