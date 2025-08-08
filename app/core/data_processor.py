"""
Data processing module for CitizenAnalytics™ Model Selection (FastAPI version)
Handles data loading, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple, List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles all data processing operations for FastAPI backend"""
    
    def __init__(self):
        self.categorical_cols = []
        self.numeric_cols = []
        self.preprocessor = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from various file formats"""
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".json"):
                df = pd.read_json(file_path)
            elif file_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Successfully loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def get_data_preview(self, df: pd.DataFrame, n_rows: int = 5) -> Dict[str, Any]:
        """Generate data preview with insights for the frontend"""
        try:
            # Basic info
            preview_data = df.head(n_rows).fillna("").to_dict('records')
            
            # Data types
            data_types = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    data_types[col] = "numeric"
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    data_types[col] = "datetime"
                else:
                    data_types[col] = "categorical"
            
            # Missing values
            missing_values = df.isnull().sum().to_dict()
            
            # Suggest target and remove columns
            suggested_targets = self._suggest_target_columns(df)
            suggested_remove = self._suggest_remove_columns(df)
            
            return {
                "columns": df.columns.tolist(),
                "data_types": data_types,
                "missing_values": missing_values,
                "rows": len(df),
                "preview_data": preview_data,
                "suggested_target_columns": suggested_targets,
                "suggested_remove_columns": suggested_remove
            }
            
        except Exception as e:
            logger.error(f"Error generating data preview: {str(e)}")
            raise
    
    def _suggest_target_columns(self, df: pd.DataFrame) -> List[str]:
        """Suggest potential target columns based on data characteristics"""
        suggestions = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Common target column patterns
            target_patterns = [
                'target', 'label', 'class', 'outcome', 'result', 'prediction',
                'approved', 'success', 'failure', 'churned', 'converted',
                'price', 'value', 'amount', 'score', 'rating', 'revenue',
                'sales', 'profit', 'loss', 'cost'
            ]
            
            # Check if column name suggests it's a target
            if any(pattern in col_lower for pattern in target_patterns):
                suggestions.append(col)
                continue
            
            # Check binary columns (good for classification)
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                    suggestions.append(col)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _suggest_remove_columns(self, df: pd.DataFrame) -> List[str]:
        """Suggest columns that should likely be removed"""
        suggestions = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Common ID/irrelevant patterns
            remove_patterns = [
                'id', 'index', 'key', 'uuid', 'guid', 'timestamp', 'date_created',
                'created_at', 'updated_at', 'modified_at'
            ]
            
            # Check if column name suggests it should be removed
            if any(pattern in col_lower for pattern in remove_patterns):
                suggestions.append(col)
                continue
            
            # Check for high cardinality categorical columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.8:  # Very high cardinality
                    suggestions.append(col)
        
        return suggestions
    
    def validate_configuration(self, df: pd.DataFrame, target_col: str, 
                             problem_type: str, remove_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate analysis configuration and return insights"""
        remove_cols = remove_cols or []
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }
        
        # Check target column exists
        if target_col not in df.columns:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Target column '{target_col}' not found")
            return validation_result
        
        # Check if target column is in remove list
        if target_col in remove_cols:
            validation_result["valid"] = False
            validation_result["errors"].append("Target column cannot be in remove columns list")
            return validation_result
        
        # Validate target for problem type
        target_series = df[target_col]
        
        if problem_type == "classification":
            unique_vals = target_series.dropna().nunique()
            if unique_vals > 20:
                validation_result["warnings"].append(
                    f"High number of classes ({unique_vals}) for classification. Consider regression."
                )
            elif unique_vals < 2:
                validation_result["valid"] = False
                validation_result["errors"].append("Classification requires at least 2 classes")
        
        elif problem_type == "regression":
            if not pd.api.types.is_numeric_dtype(target_series):
                # Try to convert
                try:
                    pd.to_numeric(target_series, errors='raise')
                except:
                    validation_result["valid"] = False
                    validation_result["errors"].append("Regression target must be numeric")
        
        # Check remaining features after removal
        remaining_features = len(df.columns) - len(remove_cols) - 1  # -1 for target
        if remaining_features < 1:
            validation_result["valid"] = False
            validation_result["errors"].append("No features remaining after column removal")
        elif remaining_features < 3:
            validation_result["warnings"].append("Very few features remaining. Consider keeping more columns.")
        
        # Check data size
        if len(df) < 50:
            validation_result["warnings"].append("Small dataset size may lead to unreliable results")
        
        return validation_result
    
    def process_data(self, df: pd.DataFrame, target_col: str, problem_type: str, 
                    remove_cols: Optional[List[str]] = None, impute_method: str = "mean") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process the dataset for machine learning
        
        Args:
            df: Input dataframe
            target_col: Target column name
            problem_type: "classification" or "regression"
            remove_cols: List of columns to remove
            impute_method: "mean" or "iterative"
        
        Returns:
            X, y: Processed features and target
        """
        try:
            df_processed = df.copy()
            remove_cols = remove_cols or []
            
            logger.info(f"Processing data for {problem_type} with target: {target_col}")
            
            # Process target variable
            y = self._process_target(df_processed[target_col], problem_type)
            
            # Process features
            X = df_processed.drop(columns=[target_col] + remove_cols)
            X = self._process_features(X, impute_method)
            
            logger.info(f"Data processing complete: {X.shape[1]} features, {len(y)} samples")
            return X, y
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def _process_target(self, target_series: pd.Series, problem_type: str) -> pd.Series:
        """Process target variable based on problem type"""
        if problem_type == "classification":
            if target_series.dtype in ['object', 'category'] or not all(isinstance(x, (int, np.integer)) for x in target_series.dropna()):
                # Convert to integer labels
                unique_values = sorted(target_series.dropna().unique())
                label_mapping = {val: idx for idx, val in enumerate(unique_values)}
                logger.info(f"Target label mapping: {label_mapping}")
                target_series = target_series.map(label_mapping)
            
            return target_series.astype(int)
        
        else:  # regression
            if target_series.dtype == 'object':
                target_series = pd.to_numeric(target_series, errors='coerce')
                if target_series.isna().any():
                    raise ValueError("Target variable contains non-numeric values for regression")
            
            return target_series
    
    def _process_features(self, X: pd.DataFrame, impute_method: str) -> pd.DataFrame:
        """Process features: encoding and imputation"""
        # Identify categorical and numeric columns
        self.categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.categorical_cols:
            logger.info(f"Processing {len(self.categorical_cols)} categorical columns")
            # Simple one-hot encoding for categorical variables
            oh_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.preprocessor = ColumnTransformer(
                transformers=[("cat", oh_enc, self.categorical_cols)],
                remainder="passthrough"
            )
            
            X_encoded = self.preprocessor.fit_transform(X)
            
            # Create feature names
            feature_names = []
            cat_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_cols)
            feature_names.extend(cat_names)
            feature_names.extend(self.numeric_cols)
            
            X = pd.DataFrame(X_encoded, columns=feature_names)
        
        # Handle missing values
        missing_cols = X.columns[X.isnull().any()].tolist()
        if missing_cols:
            logger.info(f"Imputing missing values in {len(missing_cols)} columns using {impute_method}")
            if impute_method == "mean":
                imputer = SimpleImputer(strategy="mean")
            else:
                imputer = IterativeImputer(
                    estimator=ExtraTreesRegressor(n_estimators=10, random_state=1),
                    random_state=1,
                    max_iter=10
                )
            X[missing_cols] = imputer.fit_transform(X[missing_cols])
        
        logger.info(f"Final feature shape: {X.shape}")
        return X
    
    def check_class_imbalance(self, y: pd.Series, problem_type: str) -> Tuple[bool, Optional[str]]:
        """Check for class imbalance and determine if ADASYN should be applied"""
        if problem_type != "classification":
            return False, None
        
        n_classes = len(np.unique(y))
        
        if n_classes == 2:
            # Binary classification
            class_counts = pd.Series(y).value_counts()
            ratio = class_counts.max() / class_counts.sum()
            
            if ratio > 0.6:
                logger.info(f"Class imbalance detected: {ratio:.2%} majority class")
                return True, "binary"
            
        elif n_classes > 2:
            logger.info(f"Multiclass problem ({n_classes} classes) - ADASYN not recommended")
            return False, "multiclass"
        
        return False, None