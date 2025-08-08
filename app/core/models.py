"""
Pydantic models for FastAPI request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class ProblemType(str, Enum):
    classification = "classification"
    regression = "regression"


class ImputeMethod(str, Enum):
    mean = "mean"
    iterative = "iterative"


class MetricType(str, Enum):
    # Classification metrics
    accuracy = "accuracy"
    precision = "precision"
    recall = "recall"
    f1 = "f1"
    # Regression metrics
    mae = "mae"
    rmse = "rmse"
    r2 = "r2"


class AnalysisStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


# Request Models
class AnalysisRequest(BaseModel):
    file_id: str = Field(..., description="ID of uploaded file")
    target_column: str = Field(..., description="Target column name")
    problem_type: ProblemType = Field(..., description="Classification or regression")
    metric: Optional[MetricType] = Field(None, description="Evaluation metric (auto-selected if not provided)")
    iterations: int = Field(100, ge=10, le=1000, description="Number of random seeds to test")
    remove_columns: List[str] = Field(default=[], description="Columns to exclude from features")
    impute_method: ImputeMethod = Field(ImputeMethod.mean, description="Missing value imputation method")
    skip_pycaret: bool = Field(False, description="Skip PyCaret model comparison")
    skip_claude: bool = Field(False, description="Skip Claude AI analysis")


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    file_size: int
    columns: List[str]
    rows: int
    preview: List[Dict[str, Any]]
    message: str


# Response Models
class AnalysisStartResponse(BaseModel):
    job_id: str
    status: AnalysisStatus
    message: str
    estimated_duration: str


class SeedResult(BaseModel):
    seed: int
    score: float


class AnalysisSummary(BaseModel):
    best_seed: int
    best_score: float
    worst_seed: int
    worst_score: float
    most_common_score: float
    score_range_min: float
    score_range_max: float
    standard_deviation: float
    mean_score: float
    higher_is_better: bool


class StatusResponse(BaseModel):
    job_id: str
    status: AnalysisStatus
    progress: Optional[int] = Field(None, description="Progress percentage (0-100)")
    current_step: Optional[str] = Field(None, description="Current processing step")
    message: str
    estimated_remaining: Optional[str] = Field(None, description="Estimated time remaining")


class AnalysisResults(BaseModel):
    job_id: str
    status: AnalysisStatus
    analysis_config: Dict[str, Any]
    summary: AnalysisSummary
    seed_results: List[SeedResult]
    pycaret_available: bool
    claude_analysis_available: bool
    files_generated: List[str]
    execution_time: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    job_id: Optional[str] = None


class DataPreview(BaseModel):
    file_id: str
    filename: str
    columns: List[str]
    data_types: Dict[str, str]
    missing_values: Dict[str, int]
    rows: int
    preview_data: List[Dict[str, Any]]
    suggested_target_columns: List[str]
    suggested_remove_columns: List[str]


class DownloadInfo(BaseModel):
    filename: str
    file_type: str
    download_url: str
    file_size: int


class AvailableDownloads(BaseModel):
    job_id: str
    available_files: List[DownloadInfo]