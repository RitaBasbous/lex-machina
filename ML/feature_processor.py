import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class FeatureProcessor:
    """Handles feature selection, encoding, and normalization for different problem types"""
    
    def __init__(self, problem_type: str):
        self.problem_type = problem_type.lower()
        self.encoders = {}
        self.scalers = {}
        self.feature_selector = None
        self.selected_features = None
        self.label_encoder = None
        
    def process_features(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Complete feature processing pipeline"""
        processing_info = {
            'original_features': list(X.columns),
            'original_shape': X.shape,
            'transformations_applied': []
        }
        
        # Step 1: Handle categorical encoding
        X_encoded, encoding_info = self._encode_categorical_features(X)
        processing_info['encoding_info'] = encoding_info
        processing_info['transformations_applied'].append('categorical_encoding')
        
        # Step 2: Feature selection 
        if y is not None:
            X_selected, selection_info = self._select_features(X_encoded, y)
            processing_info['feature_selection_info'] = selection_info
            processing_info['transformations_applied'].append('feature_selection')
        else:
            X_selected = X_encoded
            processing_info['feature_selection_info'] = {'method': 'no_selection', 'features_kept': list(X_encoded.columns)}
        
        # Step 3: Normalize/scale features
        X_scaled, scaling_info = self._scale_features(X_selected)
        processing_info['scaling_info'] = scaling_info
        processing_info['transformations_applied'].append('feature_scaling')
        
        # Step 4: Process target variable if provided
        if y is not None:
            y_processed, target_info = self._process_target(y)
            processing_info['target_info'] = target_info
        else:
            y_processed = None
            processing_info['target_info'] = None
        
        processing_info['final_shape'] = X_scaled.shape
        processing_info['final_features'] = list(X_scaled.columns)
        
        return X_scaled, y_processed, processing_info
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical features using appropriate methods"""
        X_encoded = X.copy()
        encoding_info = {'methods_used': [], 'encoded_columns': []}
        
        categorical_columns = X_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            unique_values = X_encoded[col].nunique()
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.encoders[col] = {'type': 'label', 'encoder': le}
            encoding_info['methods_used'].append(f"{col}: Label Encoding : {unique_values})")
            encoding_info['encoded_columns'].append(col)
        
        return X_encoded, encoding_info
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = None) -> Tuple[pd.DataFrame, Dict]:
        """Select best features based on problem type""" 
        
        if max_features is None: max_features = min(20, max(5, len(X.columns) // 2))  # Dynamic feature selection
            # Choose selection method based on problem type
        if self.problem_type in ['classification', 'binary_classification']:
            # Check if binary classification
            if y.nunique() == 2:
                selector = SelectKBest(score_func=mutual_info_classif, k=min(max_features, len(X.columns)))
                method = 'mutual_info_classif'
            else:
                selector = SelectKBest(score_func=f_classif, k=min(max_features, len(X.columns)))
                method = 'f_classif'
        else:  # regression
            selector = SelectKBest(score_func=mutual_info_regression, k=min(max_features, len(X.columns)))
            method = 'mutual_info_regression'
        
        # Fit and transform
        X_selected = pd.DataFrame(
            selector.fit_transform(X, y),
            columns=X.columns[selector.get_support()],
            index=X.index
        )
        corr_matrix = X_selected.corr().round(4)
        corr_dict = corr_matrix.to_dict()

        # Save correlation matrix as JSON
        os.makedirs(r"D:\projects\lex machina\outputs\correlation_matrix", exist_ok=True)
        with open(r"D:\projects\lex machina\outputs\correlation_matrix\selected_feature_correlations.json", "w", encoding="utf-8") as f:
            json.dump(corr_dict, f, indent=4)

        self.feature_selector = selector
        self.selected_features = list(X_selected.columns)
        
        selection_info = {
            'method': method,
            'original_features': len(X.columns),
            'selected_features': len(X_selected.columns),
            'features_kept': self.selected_features,
            'feature_scores': dict(zip(X.columns[selector.get_support()], selector.scores_[selector.get_support()]))
        }
        
        return X_selected, selection_info
    
    def _scale_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Scale numerical features"""
        X_scaled = X.copy()
        scaling_info = {'methods_used': [], 'scaled_columns': []}
        
        # Identify numerical columns
        numerical_columns = X_scaled.select_dtypes(include=[np.number]).columns
        
        if len(numerical_columns) > 0:
            # Use StandardScaler for most cases, MinMaxScaler for algorithms sensitive to scale
            scaler = StandardScaler()
            
            X_scaled[numerical_columns] = scaler.fit_transform(X_scaled[numerical_columns])
            self.scalers['numerical'] = scaler
            
            scaling_info['methods_used'].append(f"StandardScaler applied to {len(numerical_columns)} numerical columns")
            scaling_info['scaled_columns'] = list(numerical_columns)
        
        return X_scaled, scaling_info
    
    def _process_target(self, y: pd.Series) -> Tuple[pd.Series, Dict]:
        """Process target variable based on problem type"""
        y_processed = y.copy()
        target_info = {'original_type': str(y.dtype), 'unique_values': y.nunique()}
        
        # For classification, encode string labels to numbers
        if self.problem_type in ['classification', 'binary_classification'] and y.dtype == 'object':
            le = LabelEncoder()
            y_processed = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
            self.label_encoder = le
            target_info['encoding_applied'] = True
            target_info['label_mapping'] = dict(zip(le.classes_, le.transform(le.classes_)))
        else:
            target_info['encoding_applied'] = False
        
        target_info['final_type'] = str(y_processed.dtype)
        return y_processed, target_info
    
    def transform_new_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted processors"""
        X_new = X.copy()
        
        # Apply categorical encoding
        for col, encoder_info in self.encoders.items():
            if col in X_new.columns:
                if encoder_info['type'] == 'label':
                    # Handle unseen categories
                    X_new[col] = X_new[col].astype(str)
                    mask = X_new[col].isin(encoder_info['encoder'].classes_)
                    X_new.loc[~mask, col] = encoder_info['encoder'].classes_[0]  # Default to first class
                    X_new[col] = encoder_info['encoder'].transform(X_new[col])
                    
                elif encoder_info['type'] == 'onehot':
                    dummies = pd.get_dummies(X_new[col], prefix=col, drop_first=True)
                    X_new = pd.concat([X_new.drop(col, axis=1), dummies], axis=1)
                    # Ensure all expected columns are present
                    for expected_col in encoder_info['columns']:
                        if expected_col not in X_new.columns:
                            X_new[expected_col] = 0
                    
                elif encoder_info['type'] == 'binary_label':
                    X_new[col] = X_new[col].astype(str)
                    mask = X_new[col].isin(encoder_info['encoder'].classes_)
                    X_new.loc[~mask, col] = encoder_info['encoder'].classes_[0]
                    X_new[col] = encoder_info['encoder'].transform(X_new[col])
        
        # Apply feature selection
        if self.selected_features:
            X_new = X_new[self.selected_features]
        
        # Apply scaling
        if 'numerical' in self.scalers:
            numerical_columns = X_new.select_dtypes(include=[np.number]).columns
            if len(numerical_columns) > 0:
                X_new[numerical_columns] = self.scalers['numerical'].transform(X_new[numerical_columns])
        
        return X_new
