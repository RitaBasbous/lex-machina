import os
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from .feature_processor import FeatureProcessor
from .ml_pipeline import MLPipeline
from .model_evaluator import ModelEvaluator


class AutoMLPipeline:
    """Main AutoML pipeline that orchestrates the entire process"""
    
    def __init__(self, ai_client=None):
        self.ai_client = ai_client
        self.feature_processor = None
        self.ml_pipeline = None
        self.evaluator = None
        self.results = {}
    
    def run_automl(self, dataset_path: str, report_path: str = None) -> Dict[str, Any]:
        """Run complete AutoML pipeline with optional report"""
        
        print(" Starting AutoML Pipeline...")
        
        # Step 1: Load and identify dataset
        df, problem_type = self._load_and_identify_dataset(dataset_path)
        print(f" Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        print(f" Problem type detected: {problem_type}")
        
        # Step 2: Prepare features and target
        X, y = self._prepare_features_and_target(df)
        print(f" Features prepared: {X.shape[1]} features, target: {y.name}")
        
        # Step 3: Feature processing
        print(" Processing features...")
        self.feature_processor = FeatureProcessor(problem_type)
        X_processed, y_processed, processing_info = self.feature_processor.process_features(X, y)
        print(f"   - Original features: {processing_info['original_shape'][1]}")
        print(f"   - Final features: {processing_info['final_shape'][1]}")
        print(f"   - Transformations: {', '.join(processing_info['transformations_applied'])}")
        
        # Step 4: Load report if provided
        report_content = None
        if report_path and os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                print(f" Loaded analysis report from: {report_path}")
            except Exception as e:
                print(f" Warning: Could not load report file: {e}")
        
        # Step 5: AI selects model before training
        print(" AI selecting model for training...")
        self.ml_pipeline = MLPipeline(problem_type, self.ai_client)
        
        # Prepare dataset info for AI
        dataset_info = {
            'samples': len(X_processed),
            'features': len(X_processed.columns),
            'problem_type': problem_type,
            'target_classes': len(np.unique(y_processed)) if problem_type in ['classification', 'binary_classification'] else 'continuous',
            'feature_types': {
                'numeric': len(X_processed.select_dtypes(include=[np.number]).columns),
                'categorical': len(X_processed.select_dtypes(include=['object', 'category']).columns)
            }
        }
        
        # Get available algorithms
        algorithms = self.ml_pipeline.get_algorithms()
        
        # Use AI to select best algorithm
        selected_algorithm = self.ai_client.select_best_algorithm(
            dataset_info, algorithms, report_content
        )
        print(f" AI selected algorithm: {selected_algorithm}")
        
        # Step 6: Train only the selected model
        print(f" Training selected model: {selected_algorithm}...")
        training_results = self.ml_pipeline.train_selected_model(X_processed, y_processed, selected_algorithm)
        
        # Check if training was successful
        if selected_algorithm not in training_results or 'error' in training_results[selected_algorithm]:
            raise Exception(f"Training failed for selected algorithm: {selected_algorithm}")
        
        best_model_name = selected_algorithm
        print(f" Model training completed successfully!")
        
        # Step 7: Final evaluation and export
        print(" Generating evaluation and exporting artifacts...")
        self.evaluator = ModelEvaluator(problem_type)
        
        # Get test data for final evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42,
            stratify=y_processed if problem_type in ['classification', 'binary_classification'] else None
        )
        
        # Generate evaluation report
        evaluation_report = self.evaluator.generate_evaluation_report(
            self.ml_pipeline.best_model, X_test, y_test, best_model_name, self.feature_processor
        )
        
        # Save all artifacts
        artifacts = self.evaluator.save_model_artifacts(
            self.ml_pipeline.best_model, self.feature_processor, evaluation_report, best_model_name
        )
        
        # Compile final results
        final_results = {
            'success': True,
            'problem_type': problem_type,
            'dataset_shape': df.shape,
            'processed_features': X_processed.shape[1],
            'best_model': best_model_name,
            'best_model_object': self.ml_pipeline.best_model,
            'feature_processor': self.feature_processor,
            'training_results': training_results,
            'evaluation_report': evaluation_report,
            'artifacts': artifacts,
            'processing_info': processing_info,
            'ai_selected_algorithm': selected_algorithm
        }
        
        self.results = final_results
        
        # Print summary
        self._print_final_summary(final_results)
        
        return final_results
    
    def _load_and_identify_dataset(self, dataset_path: str) -> Tuple[pd.DataFrame, str]:
        """Load dataset and identify problem type from filename"""
        df = pd.read_csv(dataset_path)

        # Extract problem type from filename
        filename = os.path.basename(dataset_path).lower()
        if 'classification' in filename:
            problem_type = 'classification'
            # Check if binary classification
            target_col = df.columns[-1]
            if df[target_col].nunique() == 2:
                problem_type = 'binary_classification'
        elif 'regression' in filename:
            problem_type = 'regression'
        else:
            # Try to infer from last column
            target_col = df.columns[-1]
            target_series = df[target_col]
            unique_vals = target_series.nunique()

            if unique_vals == 2:
                problem_type = 'binary_classification'
            elif unique_vals <= 20 and target_series.dtype == 'object':
                problem_type = 'classification'
            else:
                problem_type = 'regression'

        return df, problem_type
    
    def _prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target"""
        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        
        return X, y
    
    def _print_final_summary(self, results: Dict):
        """Print comprehensive summary of AutoML results"""
        print("\n" + "="*80)
        print("AutoML Pipeline Completed Successfully!")
        print("="*80)
        print(f"Dataset: {results['dataset_shape'][0]:,} samples × {results['dataset_shape'][1]} features")
        print(f"Problem Type: {results['problem_type'].title()}")
        print(f"Features After Processing: {results['processed_features']}")
        print(f"AI Selected & Trained Model: {results['best_model'].replace('_', ' ').title()}")
        
        # Performance metrics
        eval_report = results['evaluation_report']
        if results['problem_type'] in ['classification', 'binary_classification']:
            print(f"Model Performance:")
            print(f"   • Accuracy: {eval_report.get('accuracy', 0):.3f}")
            print(f"   • F1-Score: {eval_report.get('f1_score', 0):.3f}")
            print(f"   • Precision: {eval_report.get('precision', 0):.3f}")
            print(f"   • Recall: {eval_report.get('recall', 0):.3f}")
        else:
            print(f"Model Performance:")
            print(f"   • R² Score: {eval_report.get('r2_score', 0):.3f}")
            print(f"   • RMSE: {eval_report.get('rmse', 0):.3f}")
            print(f"   • MAE: {eval_report.get('mae', 0):.3f}")
        
        print(f"\nArtifacts Saved:")
        for artifact_type, path in results['artifacts'].items():
            print(f"   • {artifact_type.replace('_', ' ').title()}: {path}")
            
        print("="*80)
    