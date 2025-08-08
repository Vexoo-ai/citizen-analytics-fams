# CitizenAnalytics™ Model Selection API

Advanced bias/variance analysis and model selection for machine learning through a powerful FastAPI backend.

## 🚀 Overview

This FastAPI application transforms your machine learning workflow by providing comprehensive bias/variance analysis, helping you select the most stable and reliable model configurations. It automatically tests your models across multiple random seeds to identify optimal performance patterns.

## ✨ Key Features

- 📊 **Bias/Variance Analysis** - Test model stability across 10-1000 random seeds
- 🤖 **AI-Powered Insights** - Claude AI analysis of your results with visualizations
- 📈 **AutoML Integration** - PyCaret model comparison across different algorithms
- 📁 **File Management** - Upload CSV/Excel files with intelligent preprocessing
- ⚡ **Async Processing** - Non-blocking background analysis with real-time progress
- 📋 **Smart Suggestions** - Auto-detect target columns and preprocessing needs
- 🎯 **Multiple Metrics** - Support for classification and regression metrics

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone and create directory:**
```bash
mkdir fastapi-ml-analyzer && cd fastapi-ml-analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set environment variables:**
```bash
# Required for Claude AI analysis
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"

# Optional: Copy and customize environment file
cp .env.example .env
```

4. **Run the server:**
```bash
cd app
python main.py
# Or: uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

5. **Access the API:**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/

## 📡 API Endpoints

### 1. **Upload Dataset** - `POST /upload`

Upload your dataset file and get instant preview with smart suggestions.

**Supported formats:** CSV, Excel (.xlsx, .xls), JSON

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_dataset.csv"
```

**Response:**
```json
{
  "file_id": "uuid-here",
  "filename": "your_dataset.csv",
  "file_size": 12345,
  "columns": ["col1", "col2", "target"],
  "rows": 1000,
  "preview": [...],
  "message": "File uploaded successfully"
}
```

### 2. **Data Preview** - `GET /preview/{file_id}`

Get detailed insights about your dataset with AI suggestions.

```bash
curl "http://localhost:8000/preview/your-file-id"
```

**Response includes:**
- Data types for each column
- Missing value counts
- Sample data preview
- **Smart suggestions** for target columns
- **Smart suggestions** for columns to remove

### 3. **Start Analysis** - `POST /analyze`

The core endpoint for bias/variance analysis with comprehensive configuration options.

#### Problem Types
- `"classification"` - For predicting categories/classes
- `"regression"` - For predicting continuous values

#### Metrics Available

**Classification:**
- `"accuracy"` - Overall prediction accuracy (default)
- `"precision"` - Precision score (positive predictive value)
- `"recall"` - Recall score (sensitivity)
- `"f1"` - F1 score (harmonic mean of precision and recall)

**Regression:**
- `"mae"` - Mean Absolute Error (default)
- `"rmse"` - Root Mean Square Error
- `"r2"` - R-squared coefficient of determination

#### Imputation Methods
- `"mean"` - Replace missing values with column mean (default)
- `"iterative"` - Advanced iterative imputation using ExtraTreesRegressor

#### Example Request Configurations

**Basic Classification Analysis:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your-file-id",
    "target_column": "approved",
    "problem_type": "classification",
    "metric": "accuracy",
    "iterations": 100,
    "remove_columns": ["customer_id"],
    "impute_method": "mean",
    "skip_pycaret": false,
    "skip_claude": false
  }'
```

**Advanced Regression Analysis:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your-file-id",
    "target_column": "house_price",
    "problem_type": "regression",
    "metric": "rmse",
    "iterations": 200,
    "remove_columns": ["property_id", "listing_date"],
    "impute_method": "iterative",
    "skip_pycaret": false,
    "skip_claude": false
  }'
```

**Quick Test Run:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your-file-id",
    "target_column": "target",
    "problem_type": "classification",
    "metric": "f1",
    "iterations": 20,
    "remove_columns": [],
    "impute_method": "mean",
    "skip_pycaret": true,
    "skip_claude": true
  }'
```

**High-Precision Analysis:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your-file-id",
    "target_column": "conversion",
    "problem_type": "classification",
    "metric": "precision",
    "iterations": 500,
    "remove_columns": ["user_id", "session_id"],
    "impute_method": "iterative",
    "skip_pycaret": false,
    "skip_claude": false
  }'
```

**Response:**
```json
{
  "job_id": "analysis-uuid",
  "status": "pending",
  "message": "Analysis started successfully",
  "estimated_duration": "300s - 800s"
}
```

### 4. **Check Progress** - `GET /status/{job_id}`

Monitor your analysis progress in real-time.

```bash
curl "http://localhost:8000/status/your-job-id"
```

**Response:**
```json
{
  "job_id": "your-job-id",
  "status": "processing",
  "progress": 65,
  "current_step": "Running PyCaret comparison",
  "message": "Analysis in progress",
  "estimated_remaining": "120s"
}
```

**Status Values:**
- `"pending"` - Analysis queued
- `"processing"` - Currently running
- `"completed"` - Analysis finished
- `"failed"` - Error occurred

### 5. **Get Results** - `GET /results/{job_id}`

Retrieve comprehensive analysis results.

```bash
curl "http://localhost:8000/results/your-job-id"
```

**Response includes:**
- **Best/worst performing seeds** with scores
- **Statistical summary** (mean, std, range)
- **All seed results** for detailed analysis
- **Model stability insights**
- **Execution time** and configuration

### 6. **Available Downloads** - `GET /downloads/{job_id}`

List all generated files available for download.

```bash
curl "http://localhost:8000/downloads/your-job-id"
```

**Generated Files:**
- 📊 `bias_variance_line_plot.png` - Performance across seeds
- 📈 `score_distribution.png` - Score distribution histogram
- 📋 `claude_analysis.md` - AI expert analysis report
- 📄 `pycaret_best_seed.csv` - Best performing models
- 📄 `pycaret_worst_seed.csv` - Worst performing models
- 📄 `pycaret_most_common_seed.csv` - Most common seed models

### 7. **Download Files** - Direct File Access

Access generated analysis files directly:

```bash
# Access via mounted static files (recommended)
curl "http://localhost:8000/files/your-job-id/claude_analysis.md" \
  --output analysis_report.md

curl "http://localhost:8000/files/your-job-id/bias_variance_line_plot.png" \
  --output performance_chart.png

curl "http://localhost:8000/files/your-job-id/pycaret_best_seed.csv" \
  --output best_models.csv
```

### 8. **Cleanup** - `DELETE /cleanup/{job_id}`

Remove job data and files to free up space.

```bash
curl -X DELETE "http://localhost:8000/cleanup/your-job-id"
```

## 📊 Use Cases & Examples

### Loan Approval Analysis
Perfect for financial institutions analyzing loan approval models:

```json
{
  "target_column": "approved",
  "problem_type": "classification",
  "metric": "precision",
  "remove_columns": ["customer_id", "application_date"]
}
```

### House Price Prediction
Real estate price modeling with high accuracy requirements:

```json
{
  "target_column": "sale_price",
  "problem_type": "regression",
  "metric": "rmse",
  "iterations": 300,
  "impute_method": "iterative"
}
```

### Customer Churn Prediction
Marketing teams optimizing retention models:

```json
{
  "target_column": "churned",
  "problem_type": "classification",
  "metric": "recall",
  "remove_columns": ["customer_id", "signup_date"]
}
```

### Medical Diagnosis
Healthcare applications requiring high precision:

```json
{
  "target_column": "diagnosis",
  "problem_type": "classification",
  "metric": "f1",
  "iterations": 500,
  "impute_method": "iterative"
}
```

## 🔧 Configuration Guide

### Environment Variables

Create a `.env` file:

```bash
# Required for Claude AI analysis
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional application settings
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000

# File storage limits
MAX_FILE_SIZE_MB=100
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs

# Security (production)
CORS_ORIGINS=https://yourdomain.com
API_KEY_HEADER=X-API-Key
```

### Performance Tuning

**For faster development/testing:**
- Use `iterations: 20-50`
- Set `skip_pycaret: true` and `skip_claude: true`

**For production analysis:**
- Use `iterations: 100-500`
- Enable all features for comprehensive insights
- Use `impute_method: "iterative"` for better data quality

**For high-stakes decisions:**
- Use `iterations: 500-1000`
- Multiple metric evaluations
- Conservative seed selection based on stability

## 🧠 Understanding the Results

### Bias/Variance Insights

The analysis reveals crucial model characteristics:

- **Low Standard Deviation** = Stable, reliable model
- **High Standard Deviation** = Unstable, sensitive to data splits
- **Best vs Worst Seed Gap** = Model sensitivity indicator

### Metric Selection Guide

**Classification:**
- **Accuracy** - Overall performance, balanced datasets
- **Precision** - Minimize false positives (e.g., fraud detection)
- **Recall** - Minimize false negatives (e.g., disease screening)
- **F1** - Balance precision and recall

**Regression:**
- **MAE** - Average absolute error, interpretable
- **RMSE** - Penalizes large errors more heavily
- **R²** - Proportion of variance explained

### Claude AI Analysis

The AI analysis provides:
- 🎯 **Model Selection Recommendations**
- 📊 **Bias/Variance Trade-off Insights**
- 📈 **Performance Distribution Analysis**
- 💡 **Actionable Improvement Suggestions**
- 🔍 **Pattern Recognition in Results**

