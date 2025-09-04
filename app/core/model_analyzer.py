"""
Model analysis module for CitizenAnalytics™ Model Selection (FastAPI version)
Handles bias/variance estimation, PyCaret comparison, and Vexoo analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import base64
import os
import asyncio
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from typing import List, Tuple, Dict, Any, Optional, Callable
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """Handles model analysis, comparison, and visualization for FastAPI backend"""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pycaret_results = {}
        self.histogram_b64 = None
        
        # Metric mappings
        self.classification_metrics = {
            "accuracy": accuracy_score,
            "precision": lambda y, pred: precision_score(y, pred, zero_division=0),
            "recall": lambda y, pred: recall_score(y, pred, zero_division=0),
            "f1": lambda y, pred: f1_score(y, pred, zero_division=0)
        }
        
        self.regression_metrics = {
            "mae": mean_absolute_error,
            "rmse": lambda y, pred: mean_squared_error(y, pred, squared=False),
            "r2": r2_score
        }
        
        self.pycaret_metric_map = {
            "accuracy": "Accuracy",
            "precision": "Prec.",
            "recall": "Recall",
            "f1": "F1",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R2"
        }
    
    async def run_bias_variance_analysis(self, X: pd.DataFrame, y: pd.Series, 
                                       problem_type: str, metric: str, 
                                       iterations: int = 100,
                                       progress_callback: Optional[Callable] = None) -> List[Tuple[int, float]]:
        """
        Run bias/variance estimation across multiple random seeds (async version)
        Note: ADASYN is now handled in data preprocessing, not here
        
        Args:
            X: Features dataframe (already processed with ADASYN if needed)
            y: Target series (already processed with ADASYN if needed)
            problem_type: "classification" or "regression"
            metric: Metric to evaluate
            iterations: Number of random seeds to test
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of (seed, score) tuples
        """
        results = []
        
        try:
            # Get metric function
            if problem_type == "classification":
                metric_func = self.classification_metrics[metric.lower()]
            else:
                metric_func = self.regression_metrics[metric.lower()]
            
            # Run iterations with async breaks
            for seed in range(1, iterations + 1):
                if seed % 10 == 0:
                    logger.info(f"Progress: {seed}/{iterations}")
                    if progress_callback:
                        await progress_callback(int((seed / iterations) * 80))  # 80% for bias/variance
                    
                    # Yield control to allow other tasks
                    await asyncio.sleep(0.01)
                
                # Split data
                stratify = y if problem_type == "classification" else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, stratify=stratify, random_state=seed
                )
                
                # Train model
                if problem_type == "classification":
                    model = RandomForestClassifier(random_state=1, n_estimators=100)
                else:
                    model = RandomForestRegressor(random_state=1, n_estimators=100)
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Calculate score
                score = metric_func(y_test, predictions)
                results.append((seed, score))
            
            logger.info(f"Bias/variance analysis completed: {len(results)} iterations")
            return results
            
        except Exception as e:
            logger.error(f"Error in bias/variance analysis: {str(e)}")
            raise
    
    def generate_summary(self, results: List[Tuple[int, float]], metric: str) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        try:
            scores = [score for _, score in results]
            
            # Determine if higher is better
            higher_is_better = metric.lower() in ["accuracy", "precision", "recall", "f1", "r2"]
            
            if higher_is_better:
                best = max(results, key=lambda x: x[1])
                worst = min(results, key=lambda x: x[1])
            else:
                best = min(results, key=lambda x: x[1])
                worst = max(results, key=lambda x: x[1])
            
            # Most common score
            common_score = Counter(scores).most_common(1)[0][0]
            common_seeds = [seed for seed, score in results if score == common_score]
            
            summary = {
                "best": best,
                "worst": worst,
                "common_score": common_score,
                "common_seeds": common_seeds,
                "score_range": (min(scores), max(scores)),
                "std": np.std(scores),
                "mean": np.mean(scores),
                "higher_is_better": higher_is_better
            }
            
            logger.info(f"Generated summary: best={best[1]:.4f}, worst={worst[1]:.4f}, std={summary['std']:.4f}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise
    
    def save_visualizations(self, results: List[Tuple[int, float]], metric: str, 
                          summary: Dict[str, Any]) -> Dict[str, str]:
        """Generate and save visualization charts"""
        try:
            seeds, scores = zip(*results)
            generated_files = {}
            
            # Line plot
            plt.figure(figsize=(12, 6))
            plt.plot(seeds, scores, alpha=0.7, linewidth=1)
            plt.axhline(summary["common_score"], color='red', linestyle='--', 
                       label=f'Most Common: {summary["common_score"]:.4f}')
            plt.axvline(summary["best"][0], color='green', linestyle='--', 
                       label=f'Best Seed: {summary["best"][0]}')
            plt.axvline(summary["worst"][0], color='orange', linestyle='--', 
                       label=f'Worst Seed: {summary["worst"][0]}')
            
            plt.xlabel('Random Seed')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} across Random Seeds')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            line_plot_path = self.output_dir / 'bias_variance_line_plot.png'
            plt.savefig(line_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files['line_plot'] = str(line_plot_path)
            
            # Histogram
            plt.figure(figsize=(10, 6))
            counts, bins, patches = plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
            
            # Find tallest bin center
            tallest_bin_idx = np.argmax(counts)
            tallest_bin_center = (bins[tallest_bin_idx] + bins[tallest_bin_idx + 1]) / 2
            
            plt.axvline(summary["common_score"], color='red', linestyle='--', 
                       label=f'Most Common: {summary["common_score"]:.4f}')
            plt.axvline(summary["best"][1], color='green', linestyle='--', 
                       label=f'Best: {summary["best"][1]:.4f}')
            plt.axvline(summary["worst"][1], color='orange', linestyle='--', 
                       label=f'Worst: {summary["worst"][1]:.4f}')
            plt.axvline(tallest_bin_center, color='purple', linestyle='--', 
                       label=f'Mode: {tallest_bin_center:.4f}')
            
            plt.xlabel(metric.upper())
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {metric.upper()} Scores')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            histogram_path = self.output_dir / 'score_distribution.png'
            plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files['histogram'] = str(histogram_path)
            
            # Save histogram as base64 for Vexoo analysis
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(summary["common_score"], color='red', linestyle='--')
            plt.axvline(summary["best"][1], color='green', linestyle='--')
            plt.axvline(summary["worst"][1], color='orange', linestyle='--')
            plt.xlabel(metric.upper())
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {metric.upper()} Scores')
            plt.tight_layout()
            
            # Save as base64
            import io
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            histogram_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Save base64 to file for Vexoo analysis
            b64_path = self.output_dir / 'histogram_b64.txt'
            with open(b64_path, 'w') as f:
                f.write(histogram_b64)
            
            plt.close()
            self.histogram_b64 = histogram_b64
            generated_files['base64_histogram'] = str(b64_path)
            
            logger.info(f"Visualizations saved to {self.output_dir}")
            return generated_files
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise
    
    async def run_pycaret_comparison(self, df: pd.DataFrame, target_col: str, 
                                   problem_type: str, remove_cols: List[str], 
                                   summary: Dict[str, Any], metric: str,
                                   progress_callback: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
        """Run PyCaret model comparison for different seeds (async version)"""
        try:
            if problem_type == "classification":
                from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull as clf_pull
                setup_func = clf_setup
                compare_func = clf_compare
                pull_func = clf_pull
            else:
                from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull
                setup_func = reg_setup
                compare_func = reg_compare
                pull_func = reg_pull
            
            # Prepare data
            df_pc = df.drop(columns=remove_cols)
            sort_by = self.pycaret_metric_map.get(metric.lower(), metric.upper())
            
            seed_map = {
                "Best": summary["best"][0],
                "Worst": summary["worst"][0],
                "Most_Common": summary["common_seeds"][0]
            }
            
            self.pycaret_results = {}
            
            for i, (rank, seed) in enumerate(seed_map.items()):
                logger.info(f"Running PyCaret for {rank} seed ({seed})...")
                
                if progress_callback:
                    await progress_callback(80 + int((i / len(seed_map)) * 15))  # 80-95% for PyCaret
                
                # Setup PyCaret
                exp = setup_func(
                    data=df_pc,
                    target=target_col,
                    session_id=seed,
                    train_size=0.7,
                    verbose=False
                )
                
                # Compare models
                best_models = compare_func(
                    sort=sort_by,
                    n_select=10,
                    turbo=False,
                    verbose=False
                )
                
                results_df = pull_func()
                self.pycaret_results[rank] = results_df.copy()
                
                # Save results
                results_df.to_csv(self.output_dir / f'pycaret_{rank.lower()}_seed.csv', index=False)
                
                # Yield control
                await asyncio.sleep(0.1)
            
            logger.info(f"PyCaret results saved to {self.output_dir}")
            return self.pycaret_results
            
        except ImportError:
            logger.warning("PyCaret not installed")
            return None
        except Exception as e:
            logger.error(f"PyCaret comparison failed: {str(e)}")
            return None
    
    async def run_vexoo_analysis(self, results: List[Tuple[int, float]], 
                                summary: Dict[str, Any], pycaret_results: Optional[Dict[str, Any]], 
                                api_key: Optional[str], problem_type: str, target_col: str, 
                                metric: str, iterations: int,
                                progress_callback: Optional[Callable] = None) -> Optional[str]:
        """Run Vexoo analysis of the results (async version)"""
        try:
            import anthropic
            
            if progress_callback:
                await progress_callback(95)
            
            # Use provided API key or get from environment
            if not api_key:
                api_key = os.getenv('VEXOO_API_KEY')
                if not api_key:
                    logger.warning("No VEXOO API key found")
                    return None
            
            # Prepare context
            context = f"""
BIAS/VARIANCE ANALYSIS RESULTS:

Dataset Info:
- Problem Type: {problem_type.title()}
- Target Variable: {target_col}
- Metric Used: {metric.upper()}
- Iterations: {iterations} random seeds

Key Findings:
- Best Seed: {summary['best'][0]} → {summary['best'][1]:.4f}
- Worst Seed: {summary['worst'][0]} → {summary['worst'][1]:.4f}
- Most Common Score: {summary['common_score']:.4f}
- Score Range: {summary['score_range'][0]:.4f} to {summary['score_range'][1]:.4f}
- Standard Deviation: {summary['std']:.4f}
- Mean Score: {summary['mean']:.4f}
"""
            
            if pycaret_results:
                context += "\n\nPyCaret Model Performance:\n"
                for rank, df in pycaret_results.items():
                    context += f"\n{rank} Seed - Top 5 Models:\n"
                    context += df.head(5).to_string(index=False, float_format='%.4f')
            
            # Create Vexoo client
            client = anthropic.Anthropic(api_key=api_key)
            
            # Prepare the message content
            message_content = f"""{context}

I have also generated a histogram showing the distribution of {metric.upper()} scores across different random seeds. 

Please provide a comprehensive analysis of these bias/variance study results, focusing on:

1. Model selection recommendations based on the seed analysis
2. Bias/variance trade-offs observed in the results  
3. Interpretation of the score distribution and what it reveals about model stability
4. Actionable insights for improving model performance
5. Any patterns or anomalies in the PyCaret model comparisons (if available)

Please structure your response as a clear, actionable report for a machine learning practitioner."""
            
            # Only include image if histogram_b64 is available
            message_parts = [
                {
                    "type": "text",
                    "text": message_content
                }
            ]
            
            if hasattr(self, 'histogram_b64') and self.histogram_b64:
                message_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self.histogram_b64
                    }
                })
            
            # Create the message
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                temperature=0.3,
                system="You are an expert machine learning consultant with deep expertise in bias-variance analysis, model selection, and statistical interpretation. Provide clear, actionable insights based on experimental results.",
                messages=[
                    {
                        "role": "user",
                        "content": message_parts
                    }
                ]
            )
            
            # Extract the analysis from Vexoo's response
            analysis = message.content[0].text
            
            # Save analysis to file
            analysis_path = self.output_dir / 'vexoo_analysis.md'
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write(f"# Vexoo Analysis Report\n\n")
                f.write(f"## Context\n{context}\n\n")
                f.write(f"## Analysis\n{analysis}")
            
            logger.info(f"Vexoo analysis saved to {analysis_path}")
            
            if progress_callback:
                await progress_callback(100)
            
            return analysis
            
        except ImportError:
            logger.warning("Anthropic library not installed")
            return None
        except Exception as e:
            logger.error(f"Vexoo analysis failed: {str(e)}")
            return None
    
    def get_available_files(self) -> List[Dict[str, Any]]:
        """Get list of available files for download"""
        try:
            files = []
            
            file_patterns = {
                'bias_variance_line_plot.png': 'Line Plot',
                'score_distribution.png': 'Score Distribution',
                'vexoo_analysis.md': 'Vexoo Analysis Report',
                'pycaret_best_seed.csv': 'PyCaret Best Seed Results',
                'pycaret_worst_seed.csv': 'PyCaret Worst Seed Results',
                'pycaret_most_common_seed.csv': 'PyCaret Most Common Seed Results'
            }
            
            for filename, description in file_patterns.items():
                file_path = self.output_dir / filename
                if file_path.exists():
                    files.append({
                        'filename': filename,
                        'description': description,
                        'file_size': file_path.stat().st_size,
                        'file_type': filename.split('.')[-1].upper()
                    })
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting available files: {str(e)}")
            return []