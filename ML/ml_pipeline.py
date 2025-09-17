import json
from typing import Any, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna # type: ignore
import pandas as pd
import numpy as np


class MLPipeline:
    """Unified ML Pipeline for different problem types with hyperparameter tuning"""
    
    def __init__(self, problem_type: str, ai_client=None):
        self.problem_type = problem_type.lower()
        self.ai_client = ai_client
        self.models = {}
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.training_results = {}
        
    def get_algorithms(self) -> Dict[str, Any]:
        """Get available algorithms based on problem type"""
        if self.problem_type in ['classification', 'binary_classification']:
            return {
                'random_forest': {
                    'model': RandomForestClassifier,
                    'param_space': {
                        'n_estimators': (50, 300),
                        'max_depth': (3, 20),
                        'min_samples_split': (2, 20),
                        'min_samples_leaf': (1, 10)
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier,
                    'param_space': {
                        'n_estimators': (50, 200),
                        'learning_rate': (0.01, 0.3),
                        'max_depth': (3, 10),
                        'subsample': (0.8, 1.0)
                    }
                },
                'logistic_regression': {
                    'model': LogisticRegression,
                    'param_space': {
                        'C': (0.01, 100.0),
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'saga']
                    }
                },
                'svm': {
                    'model': SVC,
                    'param_space': {
                        'C': (0.1, 100.0),
                        'kernel': ['rbf', 'linear'],
                        'gamma': ['scale', 'auto']
                    }
                }
            }
        else:  # regression
            return {
                'random_forest': {
                    'model': RandomForestRegressor,
                    'param_space': {
                        'n_estimators': (50, 300),
                        'max_depth': (3, 20),
                        'min_samples_split': (2, 20),
                        'min_samples_leaf': (1, 10)
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor,
                    'param_space': {
                        'n_estimators': (50, 200),
                        'learning_rate': (0.01, 0.3),
                        'max_depth': (3, 10),
                        'subsample': (0.8, 1.0)
                    }
                },
                'linear_regression': {
                    'model': LinearRegression,
                    'param_space': {}  # No hyperparameters to tune
                },
                'ridge_regression': {
                    'model': Ridge,
                    'param_space': {
                        'alpha': (0.01, 100.0)
                    }
                }
            }
    
    def ask_ai_for_model_selection(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Ask AI agent to select the best model BEFORE training"""
        if not self.ai_client:
            # Default fallback selection based on dataset characteristics
            return self._default_model_selection(X, y)
        
        # Prepare dataset characteristics for AI
        dataset_info = {
            'samples': len(X),
            'features': len(X.columns),
            'problem_type': self.problem_type,
            'target_classes': len(np.unique(y)) if self.problem_type in ['classification', 'binary_classification'] else 'continuous',
            'feature_types': {
                'numeric': len(X.select_dtypes(include=[np.number]).columns),
                'categorical': len(X.select_dtypes(include=['object', 'category']).columns)
            }
        }
        
        # Get available algorithms
        algorithms = self.get_algorithms()
        
        # Create algorithm descriptions for AI
        algo_descriptions = []
        for algo_name, algo_config in algorithms.items():
            if algo_name == 'random_forest':
                desc = "Random Forest: Ensemble method, handles mixed data types well, provides feature importance, robust to overfitting"
            elif algo_name == 'gradient_boosting':
                desc = "Gradient Boosting: Sequential learning, excellent performance, handles complex patterns, may overfit on small datasets"
            elif algo_name == 'logistic_regression':
                desc = "Logistic Regression: Fast, interpretable, works well with linear relationships, requires feature scaling"
            elif algo_name == 'linear_regression':
                desc = "Linear Regression: Simple, fast, interpretable, assumes linear relationships"
            elif algo_name == 'ridge_regression':
                desc = "Ridge Regression: Regularized linear model, handles multicollinearity, prevents overfitting"
            elif algo_name == 'svm':
                desc = "Support Vector Machine: Powerful for complex patterns, memory efficient, can be slow on large datasets"
            else:
                desc = f"{algo_name.replace('_', ' ').title()}: Advanced ML algorithm"
            
            algo_descriptions.append(f"- {algo_name}: {desc}")
        
        # Create AI prompt
        question = f"""
        Dataset Characteristics:
        - Problem Type: {dataset_info['problem_type']}
        - Samples: {dataset_info['samples']:,}
        - Features: {dataset_info['features']} ({dataset_info['feature_types']['numeric']} numeric, {dataset_info['feature_types']['categorical']} categorical)
        - Target Classes: {dataset_info['target_classes']}
        
        Available Algorithms:
        {chr(10).join(algo_descriptions)}
        
        Based on the dataset characteristics, which single algorithm should I train for this {self.problem_type} task?
        Consider:
        - Dataset size (small: <1000, medium: 1000-10000, large: >10000)
        - Feature complexity and types
        - Interpretability vs performance tradeoffs
        - Training time considerations
        
        Respond with ONLY the algorithm name (e.g., 'random_forest', 'gradient_boosting', 'logistic_regression', etc.).
        """
        
        try:
            ai_response = self.ai_client.chat_completion([
                {"role": "system", "content": "You are an expert ML engineer. Select the single best algorithm for the given dataset. Consider dataset size, complexity, and problem type. Respond with only the algorithm name."},
                {"role": "user", "content": question}
            ], temperature=0.1)
            
            # Clean response and find matching algorithm
            ai_choice = ai_response.strip().lower()
            for algo_name in algorithms.keys():
                if algo_name.lower() in ai_choice or ai_choice in algo_name.lower():
                    selected_algorithm = algo_name
                    break
            else:
                # Fallback to default selection
                selected_algorithm = self._default_model_selection(X, y)
            
            print(f"AI Agent selected algorithm: {selected_algorithm}")
            return selected_algorithm
            
        except Exception as e:
            print(f"AI selection failed: {e}, using default selection")
            return self._default_model_selection(X, y)
    
    def _default_model_selection(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Default model selection based on dataset characteristics"""
        n_samples = len(X)
        n_features = len(X.columns)
        
        if self.problem_type in ['classification', 'binary_classification']:
            if n_samples < 1000:
                return 'logistic_regression'  # Fast and effective for small datasets
            elif n_features > n_samples / 10:
                return 'random_forest'  # Handles high-dimensional data well
            else:
                return 'gradient_boosting'  # Generally best performance
        else:  # regression
            if n_samples < 1000:
                return 'ridge_regression'  # Regularized for small datasets
            elif n_features > n_samples / 10:
                return 'random_forest'  # Handles high-dimensional data
            else:
                return 'gradient_boosting'  # Generally best performance
    
    def save_model_metadata_before_training(self, X: pd.DataFrame, y: pd.Series, algorithm_name: str):
        """Save model metadata before training to a JSON file"""

        algo_config = self.get_algorithms()[algorithm_name]
        default_params = self._get_default_params(algorithm_name) if algo_config.get("param_space") else {}
        ALGORITHM_DESCRIPTIONS = {
    "random_forest": "An ensemble method using multiple decision trees to improve accuracy and reduce overfitting.",
    "gradient_boosting": "A boosting technique that builds models sequentially to correct errors made by previous models.",
    "logistic_regression": "A linear model for binary or multiclass classification that estimates probabilities using a logistic function.",
    "svm": "Support Vector Machine finds the optimal hyperplane to separate classes with maximum margin.",
    "linear_regression": "A simple linear model that fits a straight line to predict continuous outcomes.",
    "ridge_regression": "A regularized linear regression that penalizes large coefficients to prevent overfitting."
}
        ALGORITHM_JUSTIFICATIONS = {
    "random_forest": (
        "Chosen for its robustness and ability to handle both numerical and categorical features. "
        "It performs well on datasets with noisy or nonlinear relationships and is less prone to overfitting due to its ensemble nature."
    ),
    "gradient_boosting": (
        "Selected for its strong predictive performance on structured data. "
        "It builds models sequentially to correct errors, making it effective for capturing complex patterns and subtle interactions."
    ),
    "logistic_regression": (
        "Used for its simplicity and interpretability. "
        "It is well-suited for classification tasks with linearly separable data or when model transparency is important."
    ),
    "svm": (
        "Chosen for its effectiveness in high-dimensional spaces and its ability to create clear decision boundaries. "
        "It is particularly useful when the dataset is small to medium-sized and well-scaled."
    ),
    "linear_regression": (
        "Selected for its simplicity and efficiency in modeling linear relationships. "
        "It provides a transparent baseline for regression tasks with continuous outcomes."
    ),
    "ridge_regression": (
        "Used to improve stability and reduce overfitting in linear models. "
        "It is especially helpful when predictors are highly correlated or when regularization is needed."
    )
}
        metadata = {
        "model_name": algorithm_name,
        "description": ALGORITHM_DESCRIPTIONS.get(algorithm_name, "No description available."),
        "reason_for_selection": ALGORITHM_JUSTIFICATIONS.get(algorithm_name, "No justification available."),
        "problem_type": self.problem_type,
        "feature_columns": list(X.columns),
        "target_column": y.name if y.name else "unknown",
        "default_hyperparameters": default_params,
        
    }


        output_path = r"D:/projects/lex machina/outputs/models/model_metadata.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def train_selected_model(self, X: pd.DataFrame, y: pd.Series, algorithm_name: str, 
                       test_size: float = 0.2, n_trials: int = 50) -> Dict[str, Any]:
        """Train only the AI-selected model with hyperparameter tuning"""
        
        # Check for insufficient samples in any class
        if self.problem_type in ['classification', 'binary_classification']:
            class_counts = y.value_counts()
            if any(class_counts < 2):
                print(f"Warning: Some classes have insufficient samples: {class_counts.to_dict()}")
                print("Using simple train-test split without cross-validation for hyperparameter tuning")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Skip hyperparameter tuning and use default parameters
                algo_config = self.get_algorithms()[algorithm_name]
                
                
                # Use default parameters (no tuning)
                if algo_config['param_space']:
                    # Use reasonable defaults for the algorithm
                    default_params = self._get_default_params(algorithm_name)
                else:
                    default_params = {}
                
                self.save_model_metadata_before_training(X, y, algorithm_name)
                # Train model
                model = algo_config['model'](**default_params, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = self._evaluate_model(model, X_train, y_train)
                test_score = self._evaluate_model(model, X_test, y_test)
                
                # Store results
                results = {
                    algorithm_name: {
                        'model': model,
                        'best_params': default_params,
                        'train_metrics': train_score,
                        'test_metrics': test_score,
                        'cv_score': test_score.get('accuracy', test_score.get('r2', 0)),
                        'warning': 'Used simple train-test split due to insufficient class samples'
                    }
                }
                
                self.best_model = model
                self.best_params = default_params
                self.best_score = test_score.get('accuracy', test_score.get('r2', 0))
                self.models[algorithm_name] = model
                self.training_results = results
                
                return results
        
        # Original code for normal cases
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, 
            stratify=y if self.problem_type in ['classification', 'binary_classification'] else None
        )
        
        algorithms = self.get_algorithms()
        
        if algorithm_name not in algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not available for {self.problem_type}")
        
        algo_config = algorithms[algorithm_name]
        
        print(f"Training selected algorithm: {algorithm_name} for {self.problem_type}...")
        self.save_model_metadata_before_training(X, y, algorithm_name)
        try:
            # Optimize hyperparameters
            best_params, best_score = self._optimize_hyperparameters(
                algo_config, X_train, y_train, n_trials
            )
            
            # Train final model with best parameters
            model = algo_config['model'](**best_params, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self._evaluate_model(model, X_train, y_train)
            test_score = self._evaluate_model(model, X_test, y_test)
            
            # Store results
            results = {
                algorithm_name: {
                    'model': model,
                    'best_params': best_params,
                    'train_metrics': train_score,
                    'test_metrics': test_score,
                    'cv_score': best_score
                }
            }
            
            self.best_model = model
            self.best_params = best_params
            self.best_score = best_score
            self.models[algorithm_name] = model
            self.training_results = results
            
            return results
            
        except Exception as e:
            print(f"Error training {algorithm_name}: {str(e)}")
            return {algorithm_name: {'error': str(e)}}

    def _get_default_params(self, algorithm_name: str) -> Dict:
        """Get reasonable default parameters for algorithms"""
        defaults = {
            'random_forest': {'n_estimators': 100, 'max_depth': None},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
            'logistic_regression': {'C': 1.0, 'random_state': 42, 'max_iter': 1000},
            'svm': {'C': 1.0, 'kernel': 'rbf', 'random_state': 42},
            'linear_regression': {},
            'ridge_regression': {'alpha': 1.0, 'random_state': 42}
        }
        return defaults.get(algorithm_name, {})
    
    def _optimize_hyperparameters(self, algo_config: Dict, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int) -> Tuple[Dict, float]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = {}
            
            for param_name, param_range in algo_config['param_space'].items():
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # Handle special cases for solver compatibility in LogisticRegression
            if 'penalty' in params and 'solver' in params:
                if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
                    params['solver'] = 'liblinear'
            
            try:
                model = algo_config['model'](**params, random_state=42)
                
                # Use cross-validation for scoring
                if self.problem_type in ['classification', 'binary_classification']:
                    scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if self.problem_type.startswith('classification') else 5, scoring='accuracy')
                else:
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                return scores.mean()
                
            except Exception as e:
                return -999  # Return very low score for failed trials
        
        # Skip optimization if no parameters to tune
        if not algo_config['param_space']:
            model = algo_config['model'](random_state=42)
            if self.problem_type in ['classification', 'binary_classification']:
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            else:
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            return {}, scores.mean()
        
        # Create and run study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance based on problem type"""
        y_pred = model.predict(X)
        
        if self.problem_type in ['classification', 'binary_classification']:
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
            
            # Add AUC for binary classification
            if self.problem_type == 'binary_classification' or len(np.unique(y)) == 2:
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X)[:, 1]
                        metrics['auc'] = roc_auc_score(y, y_proba)
                    elif hasattr(model, 'decision_function'):
                        y_scores = model.decision_function(X)
                        metrics['auc'] = roc_auc_score(y, y_scores)
                except:
                    metrics['auc'] = 0.0
            
        else:  # regression
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
        
        return metrics