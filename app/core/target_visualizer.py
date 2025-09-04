"""
Target variable visualization module for CitizenAnalytics™ Model Selection
Handles generation of target variable charts and class imbalance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from typing import Dict, Any, Optional, Tuple
import logging
from collections import Counter

logger = logging.getLogger(__name__)

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

class TargetVisualizer:
    """Handles target variable visualization and analysis"""
    
    def __init__(self):
        # Set style for consistent, professional charts
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_target_visualization(self, df: pd.DataFrame, target_col: str, 
                                    problem_type: str) -> Dict[str, Any]:
        """
        Generate appropriate visualization based on problem type
        
        Args:
            df: Input dataframe
            target_col: Target column name
            problem_type: "classification" or "regression"
        
        Returns:
            Dictionary with chart data and statistics
        """
        try:
            target_series = df[target_col].dropna()
            
            if problem_type == "classification":
                return self._generate_classification_chart(target_series, target_col)
            else:
                return self._generate_regression_chart(target_series, target_col)
                
        except Exception as e:
            logger.error(f"Error generating target visualization: {str(e)}")
            raise
    
    def _generate_classification_chart(self, target_series: pd.Series, 
                                     target_col: str) -> Dict[str, Any]:
        """Generate chart for classification target variable"""
        
        # Get value counts and statistics
        value_counts = target_series.value_counts().sort_index()
        n_classes = len(value_counts)
        total_samples = len(target_series)
        
        # Calculate class imbalance info
        class_imbalance_info = self._calculate_class_imbalance(value_counts)
        
        # Determine chart type based on number of classes
        if n_classes == 2:
            chart_base64 = self._create_binary_bar_chart(value_counts, target_col)
            chart_type = "bar_chart"
        else:
            chart_base64 = self._create_multinomial_histogram(value_counts, target_col)
            chart_type = "histogram"
        
        # Compile statistics
        statistics = {
            "total_samples": total_samples,
            "num_classes": n_classes,
            "class_counts": value_counts.to_dict(),
            "class_percentages": (value_counts / total_samples * 100).round(2).to_dict(),
            "most_frequent_class": value_counts.index[0],
            "least_frequent_class": value_counts.index[-1]
        }
        
        return {
            "chart_base64": chart_base64,
            "chart_type": chart_type,
            "statistics": statistics,
            "class_imbalance_info": class_imbalance_info
        }
    
    def _generate_regression_chart(self, target_series: pd.Series, 
                                 target_col: str) -> Dict[str, Any]:
        """Generate histogram for regression target variable"""
        
        # Create histogram
        chart_base64 = self._create_regression_histogram(target_series, target_col)
        
        # Calculate statistics
        statistics = {
            "total_samples": len(target_series),
            "mean": float(target_series.mean()),
            "median": float(target_series.median()),
            "std": float(target_series.std()),
            "min": float(target_series.min()),
            "max": float(target_series.max()),
            "q25": float(target_series.quantile(0.25)),
            "q75": float(target_series.quantile(0.75)),
            "skewness": float(target_series.skew()),
            "kurtosis": float(target_series.kurtosis())
        }
        
        return {
            "chart_base64": chart_base64,
            "chart_type": "histogram",
            "statistics": statistics,
            "class_imbalance_info": None
        }
    
    def _calculate_class_imbalance(self, value_counts: pd.Series) -> Optional[Dict[str, Any]]:
        """Calculate class imbalance metrics"""
        
        if len(value_counts) != 2:
            return None  # Only applicable for binary classification
        
        total = value_counts.sum()
        majority_count = value_counts.max()
        minority_count = value_counts.min()
        
        majority_class = value_counts.idxmax()
        minority_class = value_counts.idxmin()
        
        imbalance_ratio = majority_count / total
        is_imbalanced = imbalance_ratio > 0.6  # 60/40 threshold
        
        return {
            "class_distribution": value_counts.to_dict(),
            "imbalance_ratio": round(imbalance_ratio, 3),
            "majority_class": str(majority_class),
            "minority_class": str(minority_class),
            "is_imbalanced": is_imbalanced,
            "majority_percentage": round((majority_count / total) * 100, 1),
            "minority_percentage": round((minority_count / total) * 100, 1)
        }
    
    def _create_binary_bar_chart(self, value_counts: pd.Series, target_col: str) -> str:
        """Create side-by-side bar chart for binary classification"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        bars = ax.bar(range(len(value_counts)), value_counts.values, 
                     color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
        
        # Customize chart
        ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'Target Variable Distribution: {target_col}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set x-axis labels
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels([str(x) for x in value_counts.index], fontsize=11)
        
        # Add value labels on bars
        for bar, count in zip(bars, value_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Add percentage labels
        total = value_counts.sum()
        for i, (bar, count) in enumerate(zip(bars, value_counts.values)):
            percentage = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{percentage:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=10)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
    
    def _create_multinomial_histogram(self, value_counts: pd.Series, target_col: str) -> str:
        """Create histogram for multinomial classification"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create histogram
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        bars = ax.bar(range(len(value_counts)), value_counts.values, 
                     color=colors, alpha=0.8, edgecolor='black')
        
        # Customize chart
        ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'Target Variable Distribution: {target_col}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set x-axis labels (rotate if many classes)
        ax.set_xticks(range(len(value_counts)))
        labels = [str(x) for x in value_counts.index]
        if len(value_counts) > 8:
            ax.set_xticklabels(labels, rotation=45, ha='right')
        else:
            ax.set_xticklabels(labels)
        
        # Add value labels on bars
        for bar, count in zip(bars, value_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
    
    def _create_regression_histogram(self, target_series: pd.Series, target_col: str) -> str:
        """Create histogram for regression target variable"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram with optimal bins
        n_bins = min(50, max(10, len(target_series) // 20))
        n, bins, patches = ax.hist(target_series, bins=n_bins, alpha=0.8, 
                                  color='#2ecc71', edgecolor='black')
        
        # Add mean and median lines
        mean_val = target_series.mean()
        median_val = target_series.median()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, 
                  label=f'Median: {median_val:.2f}')
        
        # Customize chart
        ax.set_xlabel(f'{target_col} Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Target Variable Distribution: {target_col}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
    
    def get_adasyn_recommendation(self, df: pd.DataFrame, target_col: str, 
                                problem_type: str) -> Dict[str, Any]:
        """Get ADASYN recommendation based on problem type and class imbalance"""
        
        if problem_type != "classification":
            return {
                "recommended": False,
                "reason": "ADASYN is only applicable for classification problems",
                "class_imbalance_info": None
            }
        
        target_series = df[target_col].dropna()
        value_counts = target_series.value_counts()
        n_classes = len(value_counts)
        
        if n_classes != 2:
            return {
                "recommended": False,
                "reason": "ADASYN is only recommended for binary classification problems",
                "class_imbalance_info": None
            }
        
        # Calculate class imbalance
        class_imbalance_info = self._calculate_class_imbalance(value_counts)
        
        if class_imbalance_info["is_imbalanced"]:
            return {
                "recommended": True,
                "reason": f"Class imbalance detected ({class_imbalance_info['majority_percentage']:.1f}% vs {class_imbalance_info['minority_percentage']:.1f}%). ADASYN recommended to balance classes.",
                "class_imbalance_info": class_imbalance_info
            }
        else:
            return {
                "recommended": False,
                "reason": f"Classes are relatively balanced ({class_imbalance_info['majority_percentage']:.1f}% vs {class_imbalance_info['minority_percentage']:.1f}%). ADASYN not necessary.",
                "class_imbalance_info": class_imbalance_info
            }