from datetime import datetime
from turtle import pd
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score,confusion_matrix, classification_report, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import learning_curve
import json
import os
from .feature_processor import FeatureProcessor

class ModelEvaluator:
    """Handles model evaluation, visualization, and export"""
    
    def __init__(self, problem_type: str):
        self.problem_type = problem_type.lower()
    
    def generate_evaluation_report(self, model, X_test: pd.DataFrame, y_test: pd.Series,  # type: ignore
                                 model_name: str, feature_processor: FeatureProcessor) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        evaluation_results = {
            'model_name': model_name,
            'problem_type': self.problem_type,
            'test_samples': len(X_test),
            'features_used': list(X_test.columns),
            'timestamp': datetime.now().isoformat()
        }
        
        # Get predictions
        y_pred = model.predict(X_test)
        evaluation_results['predictions_sample'] = y_pred[:10].tolist()
        
        # Calculate metrics
        if self.problem_type in ['classification', 'binary_classification']:
            # Classification metrics
            evaluation_results['accuracy'] = accuracy_score(y_test, y_pred)
            evaluation_results['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            evaluation_results['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            evaluation_results['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Classification report
            evaluation_results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            evaluation_results['confusion_matrix'] = cm.tolist()
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X_test.columns, model.feature_importances_))
                evaluation_results['feature_importance'] = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )
        
        else:  # regression
            # Regression metrics
            evaluation_results['mse'] = mean_squared_error(y_test, y_pred)
            evaluation_results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            evaluation_results['mae'] = mean_absolute_error(y_test, y_pred)
            evaluation_results['r2_score'] = r2_score(y_test, y_pred)
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X_test.columns, model.feature_importances_))
                evaluation_results['feature_importance'] = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )
        
        return evaluation_results
    
    def save_model_artifacts(self, model, feature_processor: FeatureProcessor, 
                           evaluation_results: Dict, model_name: str) -> Dict[str, str]:
        """Save all model artifacts for deployment"""
        
        artifacts = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_name}_{self.problem_type}_{timestamp}"
        
        # Save the trained model
        model_path = f"D:/projects/lex machina/outputs/models/{base_name}_model.pkl"
        os.makedirs("D:/projects/lex machina/outputs/models", exist_ok=True)
        joblib.dump(model, model_path)
        artifacts['model_path'] = model_path
        
        # Save feature processor
        processor_path = f"D:/projects/lex machina/outputs/models/{base_name}_feature_processor.pkl"
        os.makedirs("D:/projects/lex machina/outputs/models", exist_ok=True)
        joblib.dump(feature_processor, processor_path)
        artifacts['feature_processor_path'] = processor_path
        
        # Save evaluation report
        report_path = f"D:/projects/lex machina/outputs/models/model_evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        artifacts['evaluation_report_path'] = report_path
        
        # Save deployment info
        deployment_info = {
            'model_name': model_name,
            'problem_type': self.problem_type,
            'timestamp': timestamp,
            'feature_names': evaluation_results['features_used'],
            'model_path': model_path,
            'feature_processor_path': processor_path,
            'evaluation_report_path': report_path,
            'performance_metrics': {
                'accuracy': evaluation_results.get('accuracy'),
                'r2_score': evaluation_results.get('r2_score'),
                'f1_score': evaluation_results.get('f1_score')
            }
        }
        
        deployment_path = f"D:/projects/lex machina/outputs/models/deployment_info.json"
        with open(deployment_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        artifacts['deployment_info_path'] = deployment_path
        
        return artifacts