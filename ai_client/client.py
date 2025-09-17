import requests
import re
import json
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class DecisionMemory:
    """Stores AI decisions for chatbot retrieval"""
    preprocessing_decisions: Dict[str, Any] = field(default_factory=dict)
    algorithm_selections: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    problem_setup: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class AIClient:
    """ client that handles all AI decisions and maintains memory for chatbot"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.memory = DecisionMemory()
        
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """Generic chat completion method for AI interactions"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "AutoML-Pipeline"
        }
        
        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                error_msg = f"OpenRouter API Error {response.status_code}: {response.text}"
                raise Exception(error_msg)
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"API call failed: {e}")
            raise
    
    def make_preprocessing_decision(self, question: Dict[str, Any]) -> str:
        """Use OpenRouter AI to make preprocessing decisions"""
        
        # Build the prompt for the AI
        prompt = self._build_decision_prompt(question)
        
        try:
            response = self._call_openrouter(prompt)
            decision = self._extract_decision(response, question)
            
            # Store decision in memory
            self.memory.preprocessing_decisions[question["id"]] = {
                "decision": decision,
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
            
            return decision
        except Exception as e:
            print(f"AI decision failed: {e}")
            fallback = self._get_fallback_decision(question)
            
            # Store fallback in memory
            self.memory.preprocessing_decisions[question["id"]] = {
                "decision": fallback,
                "question": question,
                "error": str(e),
                "is_fallback": True,
                "timestamp": datetime.now().isoformat()
            }
            
            return fallback
    
    def _build_decision_prompt(self, question: Dict[str, Any]) -> str:
        """Build a clear prompt with explicit format requirements"""
        
        prompt = f"""You are a data scientist. Analyze this data and make a decision.

TASK: {question['issue_type']}
DESCRIPTION: {question['description']}

SAMPLE DATA:
{question.get('sample_data', 'No sample data provided')}

AVAILABLE OPTIONS (return the exact value only):
"""
        
        for opt in question['options']:
            prompt += f"- {opt['option']}: {opt['description']}\n"
        
        prompt += f"""
CRITICAL: You MUST return ONLY the exact value from the options above.
Do NOT add brackets, quotes, or any other text.
Do NOT return the option number or description.

Example: if you choose 'classification', return exactly: classification

VALID VALUES: {[opt['option'] for opt in question['options']]}

YOUR DECISION (exact value only): """
        
        return prompt
    
    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API with correct formatting"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "AutoML-Preprocessor"
        }
        
        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 50
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            error_msg = f"OpenRouter API Error {response.status_code}: {response.text}"
            raise Exception(error_msg)
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def _extract_decision(self, response: str, question: Dict[str, Any]) -> str:
        """Extract the decision from AI response with robust parsing"""
        
        # Clean the response
        decision = response.strip()
        
        # Remove brackets, quotes, and other common wrappers
        decision = decision.replace('[', '').replace(']', '').replace('"', '').replace("'", "").strip()
        
        # Valid options
        valid_options = [opt['option'] for opt in question['options']]
        
        # 1. Direct exact match (case insensitive)
        for opt in valid_options:
            if decision.lower() == opt.lower():
                return opt
        
        # 2. Check if response contains a valid option
        for opt in valid_options:
            if opt.lower() in decision.lower():
                return opt
        
        # 3. Try to parse choice numbers
        if any(word in decision.lower() for word in ['option', 'choice', 'select']):
            try:
                numbers = re.findall(r'\d+', decision)
                if numbers:
                    choice_num = int(numbers[0])
                    if 1 <= choice_num <= len(valid_options):
                        return valid_options[choice_num - 1]
            except (ValueError, IndexError):
                pass
        
        # 4. For problem_type_selection, handle special cases
        if question['issue_type'] == 'problem_type_selection':
            if 'classification' in decision.lower():
                return 'classification'
            elif 'regression' in decision.lower():
                return 'regression'
        
        # 5. If all else fails, use rule-based selection
        print(f"Warning: AI returned '{response}', using rule-based selection")
        return self._rule_based_fallback(question)
    
    def _rule_based_fallback(self, question: Dict[str, Any]) -> str:
        """Rule-based fallback when AI fails"""
        options = question['options']
        valid_options = [opt['option'] for opt in options]
        
        if question['issue_type'] == 'target_column_selection':
            # Score each option based on suitability
            scored_options = []
            
            for opt in options:
                score = 0
                opt_lower = opt['option'].lower()
                desc_lower = opt['description'].lower()
                
                # Positive indicators for target columns
                if any(word in opt_lower for word in ['target', 'label', 'class', 'outcome', 'result']):
                    score += 10
                if any(word in desc_lower for word in ['target', 'label', 'class', 'predict']):
                    score += 8
                if 'binary' in desc_lower:
                    score += 5
                if 'categorical' in desc_lower and 'unique: 2' in desc_lower:
                    score += 7
                
                # Negative indicators (avoid these)
                if any(word in opt_lower for word in ['id', 'key', 'index', 'number', 'code', 'ref']):
                    score -= 15
                if 'identifier' in desc_lower:
                    score -= 12
                if 'missing: 100' in desc_lower:
                    score -= 20
                
                scored_options.append((opt['option'], score))
            
            # Sort by score (highest first)
            scored_options.sort(key=lambda x: x[1], reverse=True)
            
            if scored_options and scored_options[0][1] > 0:
                return scored_options[0][0]
        
        # Fallback to first option
        return valid_options[0] if valid_options else ""
    
    def _get_fallback_decision(self, question: Dict[str, Any]) -> str:
        """Get a fallback decision if AI fails"""
        options = [opt['option'] for opt in question['options']]
        return options[0] if options else ""
    
    def select_best_algorithm(self, dataset_info: Dict[str, Any], algorithms_info: Dict[str, Any], 
                            report_content: str = None) -> str:
        """Select the best ML algorithm based on dataset characteristics and available algorithms"""
        
        # Store dataset info in memory
        self.memory.dataset_info = dataset_info
        
        # Prepare algorithm descriptions
        algo_descriptions = []
        for algo_name, algo_config in algorithms_info.items():
            algo_descriptions.append(f"- {algo_name}: {self._get_algorithm_description(algo_name, algo_config)}")
        
        # Build the prompt
        prompt = f"""
        MACHINE LEARNING ALGORITHM SELECTION TASK
        
        DATASET CHARACTERISTICS:
        - Problem Type: {dataset_info['problem_type']}
        - Samples: {dataset_info['samples']:,}
        - Features: {dataset_info['features']}
        - Numeric Features: {dataset_info['feature_types']['numeric']}
        - Categorical Features: {dataset_info['feature_types']['categorical']}
        - Target Type: {dataset_info['target_classes']}
        
        AVAILABLE ALGORITHMS:
        {chr(10).join(algo_descriptions)}
        
        """
        
        # Add report content if provided
        if report_content:
            prompt += f"""
        DATA ANALYSIS REPORT:
        {report_content[:2000]}  # Limit report length to avoid token limits
            """
        
        prompt += f"""
        SELECTION CRITERIA:
        1. Dataset size and complexity
        2. Feature types (numeric vs categorical)
        3. Problem type requirements
        4. Expected performance and interpretability
        5. Training time considerations
        6. Robustness to overfitting
        
        INSTRUCTIONS:
        - Analyze the dataset characteristics and available algorithms
        - Consider the data analysis report if provided
        - Select the SINGLE BEST algorithm for this specific task
        - Return ONLY the algorithm name (e.g., 'random_forest', 'gradient_boosting', etc.)
        - Do not include any explanations, justifications, or additional text
        
        YOUR SELECTION (algorithm name only):
        """
        
        try:
            response = self.chat_completion([
                {"role": "system", "content": "You are an expert ML engineer. Select the single best algorithm for the given dataset. Consider all factors and respond with only the algorithm name."},
                {"role": "user", "content": prompt}
            ], temperature=0.1)
            
            # Clean and validate the response
            selected_algo = self._validate_algorithm_response(response.strip(), list(algorithms_info.keys()))
            
            # Store algorithm selection in memory
            self.memory.algorithm_selections = {
                "selected_algorithm": selected_algo,
                "available_algorithms": list(algorithms_info.keys()),
                "dataset_info": dataset_info,
                "timestamp": datetime.now().isoformat()
            }
            
            return selected_algo
            
        except Exception as e:
            print(f"Algorithm selection failed: {e}")
            fallback = self._get_fallback_algorithm(dataset_info, algorithms_info)
            
            # Store fallback in memory
            self.memory.algorithm_selections = {
                "selected_algorithm": fallback,
                "available_algorithms": list(algorithms_info.keys()),
                "dataset_info": dataset_info,
                "is_fallback": True,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            return fallback
    
    def _get_algorithm_description(self, algo_name: str, algo_config: Dict[str, Any]) -> str:
        """Get descriptive information about each algorithm"""
        descriptions = {
            'random_forest': "Ensemble method, robust, handles mixed data types, provides feature importance, good for large datasets",
            'gradient_boosting': "Sequential learning, excellent performance, handles complex patterns, may overfit on small datasets",
            'logistic_regression': "Fast, interpretable, linear relationships, requires feature scaling, good for small datasets",
            'svm': "Powerful for complex patterns, memory efficient, can be slow on large datasets, sensitive to feature scaling",
            'linear_regression': "Simple, fast, interpretable, assumes linear relationships, good baseline",
            'ridge_regression': "Regularized linear model, handles multicollinearity, prevents overfitting, good for small datasets"
        }
        return descriptions.get(algo_name, f"{algo_name.replace('_', ' ').title()}: Advanced ML algorithm")
    
    def _validate_algorithm_response(self, response: str, valid_algorithms: List[str]) -> str:
        """Validate and extract the algorithm name from AI response"""
        
        # Clean the response
        cleaned = response.lower().strip().replace('"', '').replace("'", "").replace('.', '')
        
        # Check for direct matches
        for algo in valid_algorithms:
            if algo.lower() == cleaned:
                return algo
        
        # Check for partial matches
        for algo in valid_algorithms:
            if algo.lower() in cleaned or cleaned in algo.lower():
                return algo
        
        # Check for common variations
        variations = {
            'random forest': 'random_forest',
            'gradient boosting': 'gradient_boosting',
            'logistic regression': 'logistic_regression',
            'support vector machine': 'svm',
            'linear regression': 'linear_regression',
            'ridge regression': 'ridge_regression'
        }
        
        for variation, actual_algo in variations.items():
            if variation in cleaned:
                return actual_algo
        
        # If no match found, use fallback
        raise ValueError(f"Could not validate algorithm from response: {response}")
    
    def _get_fallback_algorithm(self, dataset_info: Dict[str, Any], algorithms_info: Dict[str, Any]) -> str:
        """Fallback algorithm selection logic"""
        n_samples = dataset_info['samples']
        n_features = dataset_info['features']
        
        if dataset_info['problem_type'] in ['classification', 'binary_classification']:
            if n_samples < 1000:
                return 'logistic_regression'
            elif n_features > 50:
                return 'random_forest'
            else:
                return 'gradient_boosting'
        else:  # regression
            if n_samples < 1000:
                return 'ridge_regression'
            elif n_features > 50:
                return 'random_forest'
            else:
                return 'gradient_boosting'
    
    def finalize_problem_setup(self, profile: Dict[str, Any], ai_choices: Dict[str, str], 
                              df: pd.DataFrame) -> Tuple[str, str]:
        """AI-driven problem type and target selection with validation"""
        
        # Get problem type from AI choice with validation
        problem_type = self._get_validated_problem_type(ai_choices)
        
        # AI target column selection for supervised learning with validation
        target_column = self._get_validated_target_column(profile, problem_type, ai_choices, df)
        
        # Store problem setup in memory
        self.memory.problem_setup = {
            "problem_type": problem_type,
            "target_column": target_column,
            "profile": profile,
            "timestamp": datetime.now().isoformat()
        }
        
        return problem_type, target_column
    
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
    
    def _get_validated_target_column(self, profile: Dict[str, Any], problem_type: str, 
                                   ai_choices: Dict[str, str], df: pd.DataFrame) -> Optional[str]:
        """AI target column selection with sample data but unbiased options"""
        
        if problem_type not in ['classification', 'regression']:
            return None
        
        # Get ALL available columns (excluding identifiers and metadata)
        all_available_cols = [col['name'] for col in profile['column_profiles'] 
                            if col['inferred_role'] not in ['identifier', 'metadata']]
        
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
            col_profile = next(col for col in profile['column_profiles'] if col['name'] == col_name)
            
            # Get column-specific sample values
            col_sample = df[col_name].dropna().head(3).tolist()
            sample_values = f"Sample: {col_sample}" if col_sample else "No sample values"
            
            # Simple, unbiased description - let AI analyze the data itself
            description = f"Column: {col_name} | Type: {col_profile['inferred_type']} | "
            description += f"Unique: {col_profile['unique_count']} | "
            description += f"Missing: {col_profile['null_percentage']:.1f}% | "
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
        target_column = self.make_preprocessing_decision(target_question)
        
        # Validate AI choice
        if target_column not in all_available_cols:
            print(f"   WARNING: AI chose invalid target '{target_column}', selecting first available column")
            target_column = all_available_cols[0]
        
        print(f"   AI selected target: {target_column}")
        return target_column
    
    def chatbot_query(self, query: str) -> str:
        """Chatbot interface to answer questions about AI decisions and dataset"""
        
        prompt = f"""
        You are an AI assistant for an AutoML pipeline. You have access to the decisions made during preprocessing and model selection.
        
        USER QUERY: {query}
        
        AVAILABLE INFORMATION:
        - Preprocessing Decisions: {json.dumps(self.memory.preprocessing_decisions, indent=2, default=str)}
        - Algorithm Selections: {json.dumps(self.memory.algorithm_selections, default=str)}
        - Dataset Info: {json.dumps(self.memory.dataset_info, default=str)}
        - Problem Setup: {json.dumps(self.memory.problem_setup, default=str)}
        
        INSTRUCTIONS:
        - Answer the user's question based on the available information
        - Be concise and informative
        - If you don't have information to answer the question, say so
        - Focus on explaining the reasoning behind AI decisions
        
        YOUR RESPONSE:
        """
        
        try:
            response = self.chat_completion([
                {"role": "system", "content": "You are a helpful AI assistant for an AutoML pipeline. Explain the decisions made during data preprocessing and model selection."},
                {"role": "user", "content": prompt}
            ], temperature=0.3)
            
            return response
            
        except Exception as e:
            return f"I'm sorry, I couldn't process your query due to an error: {str(e)}"