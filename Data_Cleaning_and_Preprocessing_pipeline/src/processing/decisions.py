from typing import Dict, List, Tuple , Optional
import pandas as pd
from ..utils.types import DatasetProfile, DataQualityIssue



class DecisionMaker:
    """Handles AI-driven decision making for preprocessing with validation"""
    
    def __init__(self, ai_client):
        self.ai_client = ai_client
    
    def make_ai_decisions(self, issues: List[DataQualityIssue], df: pd.DataFrame) -> Dict[str, str]:
        """Let AI make all preprocessing decisions with validation"""
        ai_choices = {}
        
        print("   Making AI-driven decisions for data quality issues...")
        
        # Process each issue with AI
        for i, issue in enumerate(issues):
            question = {
                "id": f"{issue.issue_type}_{i}",
                "issue_type": issue.issue_type,
                "description": issue.description,
                "business_impact": issue.business_impact,
                "detection_details": issue.detection_details,
                "affected_columns": issue.affected_columns,
                "options": issue.suggested_solutions,
                "sample_data": self._get_sample_data_for_issue(df, issue)
            }
            
            ai_choice = self.ai_client.make_preprocessing_decision(question)
            ai_choices[question["id"]] = ai_choice
            print(f"     - AI Decision for {issue.issue_type}: {ai_choice}")
        
        return ai_choices
    
    def ai_finalize_problem_setup(self, profile: DatasetProfile, ai_choices: Dict[str, str], 
                                 df: pd.DataFrame) -> Tuple[str, str]:
        """AI-driven problem type and target selection with validation"""
        
        # Get problem type from AI choice with validation
        problem_type = self._get_validated_problem_type(ai_choices)
        
        # AI target column selection for supervised learning with validation
        target_column = self._get_validated_target_column(profile, problem_type, ai_choices, df)
        
        print(f"   Problem setup finalized:")
        print(f"     - Problem type: {problem_type}")
        print(f"     - Target column: {target_column}")
        
        return problem_type, target_column
    
    def _get_sample_data_for_issue(self, df: pd.DataFrame, issue: DataQualityIssue) -> str:
        """Get sample data relevant to the current issue"""
        sample_size = min(5, len(df))
        
        if issue.affected_columns:
            sample_cols = issue.affected_columns[:5]
            sample_df = df[sample_cols].head(sample_size)
        else:
            sample_df = df.head(sample_size)
        
        return f"Sample data (first {sample_size} rows):\n{sample_df.to_string()}"
    
    def _get_validated_problem_type(self, ai_choices: Dict[str, str]) -> str:
        """Extract and validate problem type from AI choices"""
        problem_type = None
        
        for choice_id, choice in ai_choices.items():
            if "problem_type_selection" in choice_id:
                problem_type = choice
                break
        
        valid_problem_types = ['classification', 'regression']
        if problem_type not in valid_problem_types:
            print(f"   WARNING: AI chose invalid problem type '{problem_type}', defaulting to classification")
            problem_type = 'classification'
        
        return problem_type
    
    def _get_validated_target_column(self, profile: DatasetProfile, problem_type: str, 
                                   ai_choices: Dict[str, str], df: pd.DataFrame) -> Optional[str]:
        """AI target column selection with sample data but unbiased options"""
        
        if problem_type not in ['classification', 'regression']:
            return None
        
        # Get ALL available columns (excluding identifiers and metadata)
        all_available_cols = [col.name for col in profile.column_profiles 
                            if col.inferred_role not in ['identifier', 'metadata']]
        
        if not all_available_cols:
            print("   No available columns found for target selection")
            return None
        
        # Create sample data preview
        sample_size = min(8, len(df))
        sample_df = df[all_available_cols].head(sample_size)
        sample_data_str = f"Dataset sample (first {sample_size} rows):\n{sample_df.to_string()}"
        
        # Create simple, unbiased options - just column names with basic info
        options = []
        
        for col_name in all_available_cols:
            col_profile = next(col for col in profile.column_profiles if col.name == col_name)
            
            # Get column-specific sample values
            col_sample = df[col_name].dropna().head(3).tolist()
            sample_values = f"Sample: {col_sample}" if col_sample else "No sample values"
            
            # Simple, unbiased description - let AI analyze the data itself
            description = f"Column: {col_name} | Type: {col_profile.inferred_type} | "
            description += f"Unique: {col_profile.unique_count} | "
            description += f"Missing: {col_profile.null_percentage:.1f}% | "
            description += sample_values
            
            # SIMPLIFIED OPTION - no pros/cons to bias the AI
            options.append({
                "option": col_name,
                "description": description
            })
        
        target_question = {
            "id": "target_column_selection",
            "issue_type": "target_column_selection", 
            "description": f"Select the MOST APPROPRIATE target column for {problem_type} prediction. "
                          f"Analyze the sample data below and choose the column that represents "
                          f"what you want to predict based on the other features.\n\n"
                          f"{sample_data_str}",
            "business_impact": "This choice determines the prediction task and model performance",
            "detection_details": {
                "problem_type": problem_type,
                "all_columns": all_available_cols,
                "dataset_sample": sample_data_str
            },
            "affected_columns": all_available_cols,
            "options": options,  # Unbiased options
            "sample_data": sample_data_str
        }
        
        # Get AI choice
        target_column = self.ai_client.make_preprocessing_decision(target_question)
        
        # Validate AI choice
        if target_column not in all_available_cols:
            print(f"   WARNING: AI chose invalid target '{target_column}', selecting first available column")
            target_column = all_available_cols[0]
        
        print(f"   AI selected target: {target_column}")
        return target_column