"""
Data processing module for CitizenAnalytics™ Model Selection (FastAPI version)
Handles data loading, preprocessing, feature engineering, and categorical management
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
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
        self.categorical_configs = {}
        self.ordinal_encoders = {}
        
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
            
            # Generate categorical suggestions
            categorical_suggestions = self._suggest_categorical_types(df)
            
            return {
                "columns": df.columns.tolist(),
                "data_types": data_types,
                "missing_values": missing_values,
                "rows": len(df),
                "preview_data": preview_data,
                "suggested_target_columns": suggested_targets,
                "suggested_remove_columns": suggested_remove,
                "categorical_suggestions": categorical_suggestions
            }
            
        except Exception as e:
            logger.error(f"Error generating data preview: {str(e)}")
            raise
    
    def _suggest_categorical_types(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest ordinal vs nominal classification for categorical variables"""
        suggestions = []
        
        for col in df.columns:
            # Only process non-numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            unique_values = df[col].dropna().unique()
            unique_count = len(unique_values)
            
            # Skip columns with too many unique values (likely text/ID columns)
            if unique_count > 20:
                continue
            
            # Convert to strings for analysis
            str_values = [str(val).strip().lower() for val in unique_values]
            
            # Determine suggestion based on patterns
            suggested_type, reasoning = self._classify_categorical_variable(str_values, unique_count)
            
            suggestions.append({
                "variable_name": col,
                "suggested_type": suggested_type,
                "unique_values": [str(val) for val in unique_values],
                "unique_count": unique_count,
                "reasoning": reasoning
            })
        
        return suggestions
    
    def _classify_categorical_variable(self, str_values: List[str], unique_count: int) -> Tuple[str, str]:
        """Classify a categorical variable as ordinal or nominal"""
        
        # Binary patterns (likely ordinal)
        binary_patterns = [
            ['yes', 'no'], ['true', 'false'], ['0', '1'], ['pass', 'fail'],
            ['approved', 'rejected'], ['active', 'inactive'], ['on', 'off'],
            ['high', 'low'], ['good', 'bad'], ['positive', 'negative']
        ]
        
        # Ordinal patterns
        ordinal_patterns = [
            ['low', 'medium', 'high'], ['small', 'medium', 'large'],
            ['poor', 'fair', 'good', 'excellent'], ['bad', 'average', 'good', 'excellent'],
            ['never', 'rarely', 'sometimes', 'often', 'always'],
            ['strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree'],
            ['beginner', 'intermediate', 'advanced'], ['junior', 'senior'],
            ['first', 'second', 'third'], ['primary', 'secondary', 'tertiary']
        ]
        
        # Size/rating patterns
        size_patterns = ['xs', 's', 'm', 'l', 'xl', 'xxl']
        rating_patterns = ['a', 'b', 'c', 'd', 'f']  # Grades
        
        str_values_set = set(str_values)
        
        # Check for exact binary matches
        for pattern in binary_patterns:
            if str_values_set == set(pattern):
                return "ordinal", f"Binary classification pattern detected: {pattern}"
        
        # Check for ordinal patterns
        for pattern in ordinal_patterns:
            if str_values_set.issubset(set(pattern)) and len(str_values_set) >= 2:
                return "ordinal", f"Ordinal pattern detected: {pattern}"
        
        # Check for size patterns
        if str_values_set.issubset(set(size_patterns)):
            return "ordinal", "Size classification detected (XS, S, M, L, XL, etc.)"
        
        # Check for grade patterns
        if str_values_set.issubset(set(rating_patterns)):
            return "ordinal", "Grade/rating pattern detected (A, B, C, D, F)"
        
        # Check for numeric strings (could be ordinal)
        numeric_strings = []
        for val in str_values:
            try:
                float(val)
                numeric_strings.append(val)
            except ValueError:
                pass
        
        if len(numeric_strings) == len(str_values):
            return "ordinal", "Numeric strings detected - likely ordinal"
        
        # Check for month/day names
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                 'july', 'august', 'september', 'october', 'november', 'december']
        month_abbr = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                     'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_abbr = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        
        if str_values_set.issubset(set(months + month_abbr)):
            return "ordinal", "Month names detected - chronological order"
        
        if str_values_set.issubset(set(days + day_abbr)):
            return "ordinal", "Day names detected - chronological order"
        
        # Default heuristics
        if unique_count == 2:
            return "ordinal", "Binary variable - assuming ordinal ordering"
        elif 3 <= unique_count <= 7:
            return "ordinal", f"Small number of categories ({unique_count}) - likely ordinal"
        else:
            return "nominal", f"Many categories ({unique_count}) - likely nominal"
    
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
                'sales', 'profit', 'loss', 'cost', 'gone', 'left'
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
    
    def validate_categorical_configs(self, df: pd.DataFrame, 
                                   categorical_configs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Validate categorical configurations"""
        validated_configs = []
        warnings = []
        
        for config in categorical_configs:
            var_name = config['variable_name']
            var_type = config['variable_type']
            value_ordering = config.get('value_ordering', [])
            
            # Check if column exists
            if var_name not in df.columns:
                warnings.append(f"Column '{var_name}' not found in dataset")
                continue
            
            # Check if column is actually categorical
            if pd.api.types.is_numeric_dtype(df[var_name]):
                warnings.append(f"Column '{var_name}' is numeric, not categorical")
                continue
            
            # Get unique values
            unique_values = set(str(val) for val in df[var_name].dropna().unique())
            
            # Validate ordinal ordering
            if var_type == 'ordinal' and value_ordering:
                ordering_set = set(value_ordering)
                if not ordering_set == unique_values:
                    missing = unique_values - ordering_set
                    extra = ordering_set - unique_values
                    if missing:
                        warnings.append(f"Missing values in ordering for '{var_name}': {missing}")
                    if extra:
                        warnings.append(f"Extra values in ordering for '{var_name}': {extra}")
                    continue
            
            validated_configs.append(config)
        
        return validated_configs, warnings
    
    def validate_configuration(self, df: pd.DataFrame, target_col: str, 
                             problem_type: str, remove_cols: Optional[List[str]] = None,
                             categorical_configs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Validate analysis configuration and return insights"""
        remove_cols = remove_cols or []
        categorical_configs = categorical_configs or []
        
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
        
        # Validate categorical configurations
        if categorical_configs:
            _, cat_warnings = self.validate_categorical_configs(df, categorical_configs)
            validation_result["warnings"].extend(cat_warnings)
        
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
                    remove_cols: Optional[List[str]] = None, 
                    impute_method: str = "mean",
                    categorical_configs: Optional[List[Dict[str, Any]]] = None,
                    apply_adasyn: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process the dataset for machine learning
        
        Args:
            df: Input dataframe
            target_col: Target column name
            problem_type: "classification" or "regression"
            remove_cols: List of columns to remove
            impute_method: "mean" or "iterative"
            categorical_configs: List of categorical variable configurations
            apply_adasyn: Whether to apply ADASYN for class balancing
        
        Returns:
            X, y: Processed features and target
        """
        try:
            df_processed = df.copy()
            remove_cols = remove_cols or []
            categorical_configs = categorical_configs or []
            
            logger.info(f"Processing data for {problem_type} with target: {target_col}")
            
            # Store categorical configurations
            self.categorical_configs = {config['variable_name']: config for config in categorical_configs}
            
            # Process target variable
            y = self._process_target(df_processed[target_col], problem_type)
            
            # Process features
            X = df_processed.drop(columns=[target_col] + remove_cols)
            X = self._process_features(X, impute_method)
            
            # Apply ADASYN if requested and applicable
            if apply_adasyn and problem_type == "classification":
                X, y = self._apply_adasyn(X, y)
            
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
        """Process features: encoding and imputation with categorical management"""
        
        # Identify categorical and numeric columns
        self.categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.categorical_cols:
            logger.info(f"Processing {len(self.categorical_cols)} categorical columns with configurations")
            X = self._encode_categorical_variables(X)
        
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
    
    def _encode_categorical_variables(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables based on configurations"""
        
        ordinal_cols = []
        nominal_cols = []
        
        # Classify columns based on configurations
        for col in self.categorical_cols:
            if col in self.categorical_configs:
                config = self.categorical_configs[col]
                if config['variable_type'] == 'ordinal':
                    ordinal_cols.append(col)
                else:
                    nominal_cols.append(col)
            else:
                # Default to nominal if no configuration provided
                nominal_cols.append(col)
        
        # Process ordinal columns
        for col in ordinal_cols:
            X = self._encode_ordinal_column(X, col)
        
        # Process nominal columns with one-hot encoding
        if nominal_cols:
            X = self._encode_nominal_columns(X, nominal_cols)
        
        return X
    
    def _encode_ordinal_column(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        """Encode a single ordinal column"""
        config = self.categorical_configs[col]
        value_ordering = config.get('value_ordering', [])
        
        if value_ordering:
            # Use specified ordering
            mapping = {val: idx for idx, val in enumerate(value_ordering)}
            X[col] = X[col].map(mapping)
            self.ordinal_encoders[col] = {'type': 'custom', 'mapping': mapping}
            logger.info(f"Encoded ordinal column '{col}' with custom ordering: {value_ordering}")
        else:
            # Use automatic ordering
            label_encoder = LabelEncoder()
            X[col] = label_encoder.fit_transform(X[col].astype(str))
            self.ordinal_encoders[col] = {'type': 'auto', 'encoder': label_encoder}
            logger.info(f"Encoded ordinal column '{col}' with automatic ordering")
        
        return X
    
    def _encode_nominal_columns(self, X: pd.DataFrame, nominal_cols: List[str]) -> pd.DataFrame:
        """Encode nominal columns with one-hot encoding"""
        
        # Create one-hot encoder
        oh_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        # Apply one-hot encoding
        encoded_array = oh_enc.fit_transform(X[nominal_cols])
        
        # Create feature names
        feature_names = oh_enc.get_feature_names_out(nominal_cols)
        
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=X.index)
        
        # Remove original nominal columns and add encoded ones
        X = X.drop(columns=nominal_cols)
        X = pd.concat([X, encoded_df], axis=1)
        
        # Store encoder for future use
        self.preprocessor = oh_enc
        
        logger.info(f"One-hot encoded {len(nominal_cols)} nominal columns into {len(feature_names)} features")
        
        return X
    
    def _apply_adasyn(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply ADASYN for class balancing"""
        try:
            from imblearn.over_sampling import ADASYN
            
            logger.info("Applying ADASYN for class balancing")
            
            # Check class distribution before
            class_counts_before = pd.Series(y).value_counts().sort_index()
            logger.info(f"Class distribution before ADASYN: {dict(class_counts_before)}")
            
            # Apply ADASYN
            adasyn = ADASYN(random_state=42)
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            
            # Check class distribution after
            class_counts_after = pd.Series(y_resampled).value_counts().sort_index()
            logger.info(f"Class distribution after ADASYN: {dict(class_counts_after)}")
            
            # Convert back to DataFrame/Series
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled)
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"ADASYN failed: {str(e)}. Proceeding without resampling.")
            return X, y
    
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