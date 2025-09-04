"""
FastAPI backend for CitizenAnalytics™ Model Selection
Provides REST API endpoints for bias/variance analysis with categorical management and frontend integration
"""

import os
import uuid
import time
import asyncio
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd

from core.models import (
    AnalysisRequest, AnalysisStartResponse, StatusResponse, AnalysisResults,
    FileUploadResponse, ErrorResponse, DataPreview, AvailableDownloads,
    AnalysisStatus, ProblemType, MetricType, TargetVisualization,
    CategoricalConfigRequest, CategoricalConfigResponse
)
from core.data_processor import DataProcessor
from core.model_analyzer import ModelAnalyzer
from core.target_visualizer import TargetVisualizer
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CitizenAnalytics™ Model Selection API",
    description="Advanced bias/variance analysis and model selection for machine learning with categorical management",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
FRONTEND_DIR = Path("frontend")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FRONTEND_DIR.mkdir(exist_ok=True)

# Setup templates and static files
templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
app.mount("/files", StaticFiles(directory="outputs"), name="files")

# In-memory storage for jobs and files (use Redis in production)
jobs_storage: Dict[str, Dict[str, Any]] = {}
files_storage: Dict[str, Dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the frontend application"""
    return templates.TemplateResponse("index.html", {"request": request})


async def update_job_progress(job_id: str, progress: int, step: str = None):
    """Update job progress"""
    if job_id in jobs_storage:
        jobs_storage[job_id]['progress'] = progress
        if step:
            jobs_storage[job_id]['current_step'] = step
        logger.info(f"Job {job_id}: {progress}% - {step}")


async def run_analysis_background(job_id: str, config: Dict[str, Any]):
    """Background task to run the analysis"""
    try:
        jobs_storage[job_id]['status'] = AnalysisStatus.processing
        jobs_storage[job_id]['start_time'] = time.time()
        
        # Initialize processors
        data_processor = DataProcessor()
        
        # Create job-specific output directory
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)
        model_analyzer = ModelAnalyzer(output_dir=str(job_output_dir))
        
        # Progress callback
        async def progress_callback(progress: int):
            await update_job_progress(job_id, progress)
        
        await update_job_progress(job_id, 5, "Loading data")
        
        # Load and process data
        file_path = files_storage[config['file_id']]['file_path']
        df = data_processor.load_data(file_path)
        
        await update_job_progress(job_id, 10, "Processing features")
        
        X, y = data_processor.process_data(
            df=df,
            target_col=config['target_column'],
            problem_type=config['problem_type'],
            remove_cols=config['remove_columns'],
            impute_method=config['impute_method'],
            categorical_configs=config.get('categorical_configs', []),
            apply_adasyn=config.get('apply_adasyn', False)
        )
        
        await update_job_progress(job_id, 15, "Starting bias/variance analysis")
        
        # Run bias/variance analysis
        results = await model_analyzer.run_bias_variance_analysis(
            X=X,
            y=y,
            problem_type=config['problem_type'],
            metric=config['metric'],
            iterations=config['iterations'],
            progress_callback=progress_callback
        )
        
        await update_job_progress(job_id, 80, "Generating visualizations")
        
        # Generate summary and visualizations
        summary = model_analyzer.generate_summary(results, config['metric'])
        generated_files = model_analyzer.save_visualizations(results, config['metric'], summary)
        
        # PyCaret comparison
        pycaret_results = None
        if not config['skip_pycaret']:
            await update_job_progress(job_id, 85, "Running PyCaret comparison")
            pycaret_results = await model_analyzer.run_pycaret_comparison(
                df=df,
                target_col=config['target_column'],
                problem_type=config['problem_type'],
                remove_cols=config['remove_columns'],
                summary=summary,
                metric=config['metric'],
                progress_callback=progress_callback
            )
        
        # Vexoo analysis
        vexoo_analysis = None
        if not config['skip_vexoo']:
            await update_job_progress(job_id, 95, "Running Vexoo analysis")
            vexoo_analysis = await model_analyzer.run_vexoo_analysis(
                results=results,
                summary=summary,
                pycaret_results=pycaret_results,
                api_key=None,  # Uses environment variable
                problem_type=config['problem_type'],
                target_col=config['target_column'],
                metric=config['metric'],
                iterations=config['iterations'],
                progress_callback=progress_callback
            )
        
        # Store results
        jobs_storage[job_id]['status'] = AnalysisStatus.completed
        jobs_storage[job_id]['progress'] = 100
        jobs_storage[job_id]['current_step'] = "Analysis complete"
        jobs_storage[job_id]['end_time'] = time.time()
        jobs_storage[job_id]['results'] = {
            'summary': summary,
            'seed_results': results,
            'pycaret_results': pycaret_results,
            'vexoo_analysis': vexoo_analysis,
            'generated_files': generated_files,
            'available_files': model_analyzer.get_available_files()
        }
        
        logger.info(f"Analysis completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Analysis failed for job {job_id}: {str(e)}")
        jobs_storage[job_id]['status'] = AnalysisStatus.failed
        jobs_storage[job_id]['error'] = str(e)


@app.get("/health")
async def health_check():
    """Health check endpoint for API"""
    return {
        "message": "CitizenAnalytics™ Model Selection API",
        "status": "healthy",
        "version": "2.0.0",
        "features": ["bias_variance_analysis", "categorical_management", "target_visualization", "adasyn_control"]
    }


@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload dataset file and return preview"""
    try:
        # Validate file type
        allowed_extensions = {'.csv', '.xlsx', '.xls', '.json'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique file ID and save file
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load and preview data
        data_processor = DataProcessor()
        df = data_processor.load_data(str(file_path))
        preview_data = data_processor.get_data_preview(df)
        
        # Store file info
        files_storage[file_id] = {
            'filename': file.filename,
            'file_path': str(file_path),
            'upload_time': time.time(),
            'file_size': file_path.stat().st_size,
            'columns': preview_data['columns'],
            'rows': preview_data['rows']
        }
        
        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            file_size=file_path.stat().st_size,
            columns=preview_data['columns'],
            rows=preview_data['rows'],
            preview=preview_data['preview_data'],
            message="File uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/preview/{file_id}", response_model=DataPreview)
async def get_data_preview(file_id: str):
    """Get detailed data preview with categorical suggestions"""
    try:
        if file_id not in files_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = files_storage[file_id]
        data_processor = DataProcessor()
        df = data_processor.load_data(file_info['file_path'])
        preview_data = data_processor.get_data_preview(df)
        
        return DataPreview(
            file_id=file_id,
            filename=file_info['filename'],
            columns=preview_data['columns'],
            data_types=preview_data['data_types'],
            missing_values=preview_data['missing_values'],
            rows=preview_data['rows'],
            preview_data=preview_data['preview_data'],
            suggested_target_columns=preview_data['suggested_target_columns'],
            suggested_remove_columns=preview_data['suggested_remove_columns'],
            categorical_suggestions=preview_data['categorical_suggestions']
        )
        
    except Exception as e:
        logger.error(f"Preview failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/configure-categoricals", response_model=CategoricalConfigResponse)
async def configure_categorical_variables(request: CategoricalConfigRequest):
    """Validate and store categorical variable configurations"""
    try:
        if request.file_id not in files_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = files_storage[request.file_id]
        data_processor = DataProcessor()
        df = data_processor.load_data(file_info['file_path'])
        
        # Validate configurations
        validated_configs, warnings = data_processor.validate_categorical_configs(
            df, [config.dict() for config in request.categorical_configs]
        )
        
        # Store validated configurations in file storage for later use
        files_storage[request.file_id]['categorical_configs'] = validated_configs
        
        return CategoricalConfigResponse(
            file_id=request.file_id,
            message=f"Validated {len(validated_configs)} categorical configurations",
            validated_configs=validated_configs,
            warnings=warnings
        )
        
    except Exception as e:
        logger.error(f"Categorical configuration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualize-target/{file_id}", response_model=TargetVisualization)
async def visualize_target_variable(file_id: str, target_column: str, problem_type: ProblemType):
    """Generate target variable visualization and ADASYN recommendation"""
    try:
        if file_id not in files_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = files_storage[file_id]
        data_processor = DataProcessor()
        target_visualizer = TargetVisualizer()
        
        # Load data
        df = data_processor.load_data(file_info['file_path'])
        
        # Validate target column
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        # Generate visualization
        viz_data = target_visualizer.generate_target_visualization(
            df, target_column, problem_type.value
        )
        
        # Get ADASYN recommendation
        adasyn_rec = target_visualizer.get_adasyn_recommendation(
            df, target_column, problem_type.value
        )
        
        # Store ADASYN recommendation in file storage
        files_storage[file_id]['adasyn_recommendation'] = adasyn_rec
        
        return TargetVisualization(
            file_id=file_id,
            target_column=target_column,
            problem_type=problem_type,
            chart_base64=viz_data['chart_base64'],
            chart_type=viz_data['chart_type'],
            statistics=viz_data['statistics'],
            class_imbalance_info=viz_data.get('class_imbalance_info')
        )
        
    except Exception as e:
        logger.error(f"Target visualization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/adasyn-recommendation/{file_id}")
async def get_adasyn_recommendation(file_id: str, target_column: str, problem_type: ProblemType):
    """Get ADASYN recommendation for the given target and problem type"""
    try:
        if file_id not in files_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = files_storage[file_id]
        target_visualizer = TargetVisualizer()
        data_processor = DataProcessor()
        
        # Load data
        df = data_processor.load_data(file_info['file_path'])
        
        # Validate target column
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        # Get ADASYN recommendation
        adasyn_rec = target_visualizer.get_adasyn_recommendation(
            df, target_column, problem_type.value
        )
        
        return adasyn_rec
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ADASYN recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ADASYN recommendation failed: {str(e)}")


@app.post("/analyze", response_model=AnalysisStartResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start bias/variance analysis with categorical management"""
    try:
        # Validate file exists
        if request.file_id not in files_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Load data for validation
        file_info = files_storage[request.file_id]
        data_processor = DataProcessor()
        df = data_processor.load_data(file_info['file_path'])
        
        # Convert categorical configs to dict format
        categorical_configs = [config.dict() for config in request.categorical_configs]
        
        # Validate configuration
        validation = data_processor.validate_configuration(
            df=df,
            target_col=request.target_column,
            problem_type=request.problem_type.value,
            remove_cols=request.remove_columns,
            categorical_configs=categorical_configs
        )
        
        if not validation['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid configuration: {'; '.join(validation['errors'])}"
            )
        
        # Set default metric if not provided
        if not request.metric:
            request.metric = MetricType.accuracy if request.problem_type == ProblemType.classification else MetricType.mae
        
        # Handle ADASYN decision
        apply_adasyn = request.apply_adasyn
        if apply_adasyn is None:
            # Auto-determine based on problem type and class imbalance
            target_visualizer = TargetVisualizer()
            adasyn_rec = target_visualizer.get_adasyn_recommendation(
                df, request.target_column, request.problem_type.value
            )
            apply_adasyn = adasyn_rec['recommended']
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Estimate duration
        base_time = request.iterations * 2
        if not request.skip_pycaret:
            base_time *= 1.5
        if not request.skip_vexoo:
            base_time += 30
        
        estimated_duration = f"{int(base_time)}s - {int(base_time * 1.5)}s"
        
        # Prepare analysis config
        analysis_config = request.dict()
        analysis_config['apply_adasyn'] = apply_adasyn
        analysis_config['categorical_configs'] = categorical_configs
        
        # Store job info
        jobs_storage[job_id] = {
            'status': AnalysisStatus.pending,
            'progress': 0,
            'current_step': 'Initializing',
            'config': analysis_config,
            'file_info': file_info,
            'warnings': validation.get('warnings', [])
        }
        
        # Start background analysis
        background_tasks.add_task(run_analysis_background, job_id, analysis_config)
        
        return AnalysisStartResponse(
            job_id=job_id,
            status=AnalysisStatus.pending,
            message="Analysis started successfully with categorical management",
            estimated_duration=estimated_duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis start failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_analysis_status(job_id: str):
    """Get analysis status and progress"""
    try:
        if job_id not in jobs_storage:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs_storage[job_id]
        
        # Calculate estimated remaining time
        estimated_remaining = None
        if job['status'] == AnalysisStatus.processing and 'start_time' in job:
            elapsed = time.time() - job['start_time']
            if job['progress'] > 0:
                total_estimated = elapsed * (100 / job['progress'])
                remaining = max(0, total_estimated - elapsed)
                estimated_remaining = f"{int(remaining)}s"
        
        return StatusResponse(
            job_id=job_id,
            status=job['status'],
            progress=job.get('progress'),
            current_step=job.get('current_step'),
            message=job.get('error', 'Processing...') if job['status'] == AnalysisStatus.failed else 'Analysis in progress',
            estimated_remaining=estimated_remaining
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{job_id}", response_model=AnalysisResults)
async def get_analysis_results(job_id: str):
    """Get completed analysis results"""
    try:
        if job_id not in jobs_storage:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs_storage[job_id]
        
        if job['status'] != AnalysisStatus.completed:
            raise HTTPException(
                status_code=400,
                detail=f"Analysis not completed. Current status: {job['status']}"
            )
        
        results = job['results']
        summary = results['summary']
        
        # Format seed results
        seed_results = [
            {"seed": seed, "score": score}
            for seed, score in results['seed_results']
        ]
        
        # Calculate execution time
        execution_time = job.get('end_time', 0) - job.get('start_time', 0)
        
        return AnalysisResults(
            job_id=job_id,
            status=job['status'],
            analysis_config=job['config'],
            summary={
                "best_seed": summary['best'][0],
                "best_score": summary['best'][1],
                "worst_seed": summary['worst'][0],
                "worst_score": summary['worst'][1],
                "most_common_score": summary['common_score'],
                "score_range_min": summary['score_range'][0],
                "score_range_max": summary['score_range'][1],
                "standard_deviation": summary['std'],
                "mean_score": summary['mean'],
                "higher_is_better": summary['higher_is_better']
            },
            seed_results=seed_results,
            pycaret_available=results['pycaret_results'] is not None,
            vexoo_analysis_available=results['vexoo_analysis'] is not None,
            files_generated=[file['filename'] for file in results['available_files']],
            execution_time=execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Results retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/downloads/{job_id}")
async def get_available_downloads(job_id: str):
    """Get list of available download files"""
    try:
        # First check if output directory exists
        job_output_dir = OUTPUT_DIR / job_id
        if not job_output_dir.exists():
            raise HTTPException(status_code=404, detail="Job output directory not found")
        
        # Generate file list from actual directory contents
        download_list = []
        for file_path in job_output_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                file_type = file_path.suffix.upper().replace('.', '') or 'FILE'
                download_list.append({
                    "filename": file_path.name,
                    "file_type": file_type,
                    "download_url": f"/files/{job_id}/{file_path.name}",
                    "file_size": file_path.stat().st_size
                })
        
        if not download_list:
            raise HTTPException(status_code=404, detail="No files found for this job")
        
        return {
            "job_id": job_id,
            "available_files": download_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download list failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download specific analysis file"""
    try:
        if job_id not in jobs_storage:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs_storage[job_id]
        
        if job['status'] != AnalysisStatus.completed:
            raise HTTPException(status_code=400, detail="Analysis not completed")
        
        file_path = OUTPUT_DIR / job_id / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up job files and data"""
    try:
        if job_id in jobs_storage:
            # Remove job output directory
            job_output_dir = OUTPUT_DIR / job_id
            if job_output_dir.exists():
                shutil.rmtree(job_output_dir)
            
            # Remove from storage
            del jobs_storage[job_id]
            
            return {"message": f"Job {job_id} cleaned up successfully"}
        else:
            raise HTTPException(status_code=404, detail="Job not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)