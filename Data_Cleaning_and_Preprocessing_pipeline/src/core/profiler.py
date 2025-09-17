import pandas as pd
import re
from typing import Dict, List, Any
from ..utils.types import ColumnProfile, DatasetProfile


class DataProfiler:
    """Profiles any dataset using both Pandas and Polars for optimal performance"""

    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
        """Create comprehensive profile of any dataset"""

        df_pandas = df.copy()

        column_profiles = []
        target_candidates = []

        # Profile each column
        for col in df_pandas.columns:
            profile = DataProfiler._profile_column(df_pandas, col)
            column_profiles.append(profile)
            if profile.inferred_role == "target_candidate":
                target_candidates.append(col)

        # Detect duplicates
        duplicate_count = df_pandas.duplicated().sum()
        duplicate_percentage = (
            (duplicate_count / len(df_pandas)) * 100 if len(df_pandas) > 0 else 0
        )

        # Infer problem types
        problem_type_candidates = DataProfiler._infer_problem_types(
            column_profiles, df_pandas
        )

        # Complexity analysis
        complexity_indicators = DataProfiler._analyze_complexity(
            df_pandas, column_profiles
        )

        return DatasetProfile(
            shape=df_pandas.shape,
            column_profiles=column_profiles,
            target_candidates=target_candidates,
            problem_type_candidates=problem_type_candidates,
            complexity_indicators=complexity_indicators,
            duplicate_count=duplicate_count,
            duplicate_percentage=duplicate_percentage,
        )

    @staticmethod
    def _profile_column(df: pd.DataFrame, col: str) -> ColumnProfile:
        """Profile individual column to understand its nature"""

        series = df[col]
        dtype = str(series.dtype)
        unique_count = series.nunique()
        null_count = series.isnull().sum()
        null_percentage = (null_count / len(series)) * 100

        # Get sample values (non-null, converted to string)
        sample_values = []
        non_null_series = series.dropna()
        if len(non_null_series) > 0:
            sample_size = min(10, len(non_null_series))
            samples = (
                non_null_series.sample(n=sample_size, random_state=42)
                .astype(str)
                .tolist()
            )
            sample_values = samples

        # Infer column type
        inferred_type = DataProfiler._infer_column_type(series, sample_values)

        # Infer column role
        inferred_role = DataProfiler._infer_column_role(
            col, series, inferred_type, unique_count, len(df)
        )

        # Extract characteristics
        characteristics = DataProfiler._extract_column_characteristics(
            series, inferred_type, unique_count
        )

        return ColumnProfile(
            name=col,
            dtype=dtype,
            unique_count=unique_count,
            null_count=null_count,
            null_percentage=null_percentage,
            sample_values=sample_values,
            inferred_type=inferred_type,
            inferred_role=inferred_role,
            characteristics=characteristics,
        )

    @staticmethod
    def _infer_column_type(series: pd.Series, sample_values: List[str]) -> str:
        """Infer the true type of a column beyond pandas dtype"""

        # Skip if all null
        if series.isnull().all():
            return "unknown"

        non_null_series = series.dropna()

        # Check for datetime patterns
        if DataProfiler._is_datetime_column(sample_values):
            return "datetime"

        # Check for numeric data (even if stored as object)
        if DataProfiler._is_numeric_column(non_null_series):
            return "numeric"

        # Check for binary/boolean
        if unique_values := set(non_null_series.astype(str).str.lower().unique()):
            if unique_values.issubset(
                {"true", "false", "1", "0", "yes", "no", "y", "n"}
            ):
                return "binary"

        # Check for identifier patterns
        if DataProfiler._is_identifier_column(non_null_series, sample_values):
            return "identifier"

        # Check for text vs categorical
        if series.dtype == "object":
            unique_ratio = non_null_series.nunique() / len(non_null_series)
            avg_length = non_null_series.astype(str).str.len().mean()

            if unique_ratio > 0.7 and avg_length > 30:
                return "text"
            else:
                return "categorical"

        return "categorical"

    @staticmethod
    def _is_datetime_column(sample_values: List[str]) -> bool:
        """Check if column contains datetime values"""
        if not sample_values:
            return False

        datetime_patterns = [
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",  # YYYY-MM-DD or YYYY/MM/DD
            r"\d{1,2}[-/]\d{1,2}[-/]\d{4}",  # MM-DD-YYYY or MM/DD/YYYY
            r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}",  # YYYY-MM-DD HH:MM:SS
        ]

        matches = 0
        for value in sample_values[:5]:
            for pattern in datetime_patterns:
                if re.search(pattern, str(value)):
                    matches += 1
                    break

        return matches >= 3

    @staticmethod
    def _is_numeric_column(series: pd.Series) -> bool:
        """Check if column is actually numeric"""
        if pd.api.types.is_numeric_dtype(series):
            return True

        try:
            pd.to_numeric(series, errors="raise")
            return True
        except (ValueError, TypeError):
            try:
                converted = pd.to_numeric(series, errors="coerce")
                non_null_ratio = converted.notna().sum() / len(series)
                return non_null_ratio > 0.8
            except:
                return False

    @staticmethod
    def _is_identifier_column(series: pd.Series, sample_values: List[str]) -> bool:
        """Heuristic check if column is an identifier"""
        if series.empty:
            return False

        unique_ratio = series.nunique(dropna=True) / len(series)
        if unique_ratio < 0.9:
            return False

        id_patterns = [
            r"^[a-zA-Z0-9]{8,}$",
            r"^\d{5,}$",
            r"^[A-Z]{2,}\d+$",
            r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
            r"^\+?\d{7,15}$",
            r"^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$",
            r"^[\w\.-]+@[\w\.-]+\.\w+$",
            r"^[A-Z][a-z]+(\s[A-Z][a-z]+)+$",
        ]

        name_lower = series.name.lower() if series.name else ""
        id_keywords = [
            "id",
            "key",
            "uuid",
            "guid",
            "identifier",
            "code",
            "ref",
            "name",
            "fullname",
            "firstname",
            "lastname",
            "user",
            "phone",
            "mobile",
            "contact",
            "email",
            "mail",
            "ssn",
            "passport",
            "account",
        ]
        has_id_name = any(word in name_lower for word in id_keywords)

        sample_check = sample_values[: min(20, len(sample_values))]
        has_id_pattern = any(
            re.fullmatch(pattern, str(val))
            for val in sample_check
            for pattern in id_patterns
        )

        return has_id_name or has_id_pattern

    @staticmethod
    def _infer_column_role(
        col_name: str,
        series: pd.Series,
        inferred_type: str,
        unique_count: int,
        total_rows: int,
    ) -> str:
        """Infer the role of column in ML context"""

        col_lower = col_name.lower()

        if inferred_type == "identifier":
            return "identifier"

        target_keywords = [
            "target",
            "label",
            "class",
            "outcome",
            "result",
            "prediction",
            "default",
            "churn",
            "fraud",
            "risk",
            "score",
            "rating",
            "price",
            "value",
            "amount",
            "cost",
            "revenue",
            "sales",
        ]

        if any(keyword in col_lower for keyword in target_keywords):
            return "target_candidate"

        metadata_keywords = [
            "date",
            "time",
            "created",
            "updated",
            "modified",
            "name",
            "description",
            "comment",
            "note",
        ]

        if any(keyword in col_lower for keyword in metadata_keywords):
            return "metadata"

        if inferred_type == "binary" or (
            inferred_type == "categorical" and unique_count <= 5
        ):
            return "target_candidate"

        return "feature"

    @staticmethod
    def _extract_column_characteristics(
        series: pd.Series, inferred_type: str, unique_count: int
    ) -> Dict[str, Any]:
        """Extract detailed characteristics for preprocessing decisions"""

        characteristics = {
            "cardinality": "low"
            if unique_count < 10
            else "medium"
            if unique_count < 50
            else "high"
        }

        if inferred_type == "numeric":
            non_null = series.dropna()
            if len(non_null) > 0:
                characteristics.update(
                    {
                        "mean": float(non_null.mean()),
                        "std": float(non_null.std()),
                        "skewness": float(non_null.skew()) if non_null.std() > 0 else 0,
                        "has_outliers": DataProfiler._detect_outliers(non_null),
                        "distribution": "normal"
                        if abs(non_null.skew()) < 1
                        else "skewed",
                    }
                )

        elif inferred_type == "categorical":
            characteristics.update(
                {
                    "most_frequent": series.value_counts().index[0]
                    if unique_count > 0
                    else None,
                    "frequency_distribution": "balanced"
                    if series.value_counts().std() < series.value_counts().mean()
                    else "imbalanced",
                }
            )

        return characteristics

    @staticmethod
    def _detect_outliers(series: pd.Series) -> bool:
        """Simple outlier detection using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            return outliers > len(series) * 0.05
        except:
            return False

    @staticmethod
    def _infer_problem_types(
        column_profiles: List[ColumnProfile], df: pd.DataFrame
    ) -> List[str]:
        """Infer possible ML problem types based on data characteristics"""

        problem_types = []

        classification_candidates = [
            col
            for col in column_profiles
            if col.inferred_role == "target_candidate"
            and (
                col.inferred_type in ["binary", "categorical"] or col.unique_count <= 10
            )
        ]
        if classification_candidates:
            problem_types.append("classification")

        regression_candidates = [
            col
            for col in column_profiles
            if col.inferred_role == "target_candidate"
            and col.inferred_type == "numeric"
            and col.unique_count > 10
        ]
        if regression_candidates:
            problem_types.append("regression")

        datetime_cols = [
            col for col in column_profiles if col.inferred_type == "datetime"
        ]
        if datetime_cols:
            problem_types.append("time_series")

        if not problem_types:
            problem_types.append("clustering")

        return problem_types

    @staticmethod
    def _analyze_complexity(
        df: pd.DataFrame, column_profiles: List[ColumnProfile]
    ) -> Dict[str, Any]:
        """Analyze dataset complexity for algorithm selection"""

        n_rows = df.shape[0]
        n_features = len(
            [col for col in column_profiles if col.inferred_role == "feature"]
        )

        if n_rows < 10_000:
            size_category = "small"
        elif n_rows < 100_000:
            size_category = "medium"
        elif n_rows < 1_000_000:
            size_category = "large"
        else:
            size_category = "very_large"

        if n_features <= 20:
            feature_complexity = "low"
        elif n_features <= 200:
            feature_complexity = "medium"
        elif n_features <= 1000:
            feature_complexity = "high"
        else:
            feature_complexity = "very_high"

        return {
            "size_category": size_category,
            "feature_count": n_features,
            "feature_complexity": feature_complexity,
            "mixed_types": len(set(col.inferred_type for col in column_profiles)) > 3,
            "high_cardinality_features": len(
                [
                    col
                    for col in column_profiles
                    if col.characteristics.get("cardinality") == "high"
                ]
            ),
            "missing_data_complexity": "high"
            if sum(col.null_percentage for col in column_profiles)
            / len(column_profiles)
            > 20
            else "low",
        }
