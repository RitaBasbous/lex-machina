from typing import Dict, Any, Union
import pandas as pd
import polars as pl
from ..core.profiler import DataProfiler
from ..core.quality_analyzer import DataQualityAnalyzer
from .cleaning import DataCleaner
from .decisions import DecisionMaker
from .execution import PreprocessingExecutor
from .reporting import ReportGenerator
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))



class AutomatedPreprocessingPipeline:
    """Fully automated preprocessing pipeline with AI decision making"""

    def __init__(self, ai_client):
        self.ai_client = ai_client
        self.cleaner = DataCleaner()
        self.decision_maker = DecisionMaker(ai_client)
        self.executor = PreprocessingExecutor()
        self.reporter = ReportGenerator()

    def process_any_dataset(
        self, df: Union[pd.DataFrame, pl.DataFrame]
    ) -> Dict[str, Any]:
        """Process any dataset through fully automated preprocessing pipeline"""

        # Convert polars to pandas if needed
        if isinstance(df, pl.DataFrame):
            df_pandas = df.to_pandas()
        else:
            df_pandas = df.copy()

        try:
            print("Starting automated preprocessing pipeline...")

            # Step 1: Profile dataset
            print("1. Analyzing dataset structure and characteristics...")
            profile = DataProfiler.profile_dataset(df_pandas)
            print(f"   - Shape: {profile.shape}")
            print(f"   - Potential targets: {profile.target_candidates}")
            print(f"   - Problem types: {profile.problem_type_candidates}")

            # Step 2: Automatic cleaning
            df_pandas, cleaning_actions = self.cleaner.perform_automatic_cleaning(
                df_pandas, profile
            )

            # Re-profile after cleaning
            print("2. Re-profiling cleaned dataset...")
            profile = DataProfiler.profile_dataset(df_pandas)
            print(f"   - Final shape after automatic cleaning: {df_pandas.shape}")

            # Step 3: Identify issues and make AI decisions 
            print("3. Identifying data quality issues and making AI decisions...")
            issues = DataQualityAnalyzer.identify_issues(profile, df_pandas)
            user_choices = self.decision_maker.make_ai_decisions(issues, df_pandas)  
            
            # Step 4: Finalize problem setup with AI 
            problem_type, target_column = self.decision_maker.ai_finalize_problem_setup(
                profile, user_choices, df_pandas  
            )

            # Step 5: Execute preprocessing with complete missing value handling
            df_for_training, df_with_ids, processing_steps = (
                self.executor.execute_comprehensive_preprocessing(
                    df_pandas, profile, user_choices, target_column
                )
            )

            # Step 6: Generate comprehensive analysis report
            analysis_report = self.reporter.generate_comprehensive_analysis_report(
                df_pandas,
                df_for_training,
                profile,
                user_choices,
                processing_steps,
                cleaning_actions,
                problem_type,
                target_column,
            )

            self.reporter.generate_files(
                problem_type, df_with_ids, df_for_training, target_column
            )

            return {
                "success": True,
                "cleaned_data": df_for_training,
                "cleaned_data_with_ids": df_with_ids,
                "target_column": target_column,
                "problem_type": problem_type,
                "dataset_profile": profile,
                "AI_choices": user_choices,
                "analysis_report": analysis_report,
                "processing_steps": processing_steps,
                "cleaning_actions": cleaning_actions,
            }

        except Exception as e:
            return {
                "success": False,
                "cleaned_data": None,
                "cleaned_data_with_ids": None,
                "recommended_algorithms": None,
                "target_column": None,
                "problem_type": None,
                "dataset_profile": None,
                "user_choices": {},
                "analysis_report": None,
                "error": str(e),
                "processing_steps": [],
                "cleaning_actions": [],
            }
