"""Project management API routes."""
import os
import uuid
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from videocuts.api.database import get_db
from videocuts.api.models import Project, Clip, ProjectStatus
from videocuts.api.schemas import ProjectCreate, ProjectResponse, ProjectListResponse, ProjectCreateResponse
from videocuts.api.settings import get_settings
from videocuts.api.download import download_video

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/projects", tags=["projects"])


async def process_video_task(project_id: str, url: str, work_dir: str, skip_download: bool = False):
    """Background task to download and process a video."""
    from videocuts.api.database import AsyncSessionLocal
    from videocuts.config import Config
    from videocuts.main import run_pipeline
    
    async with AsyncSessionLocal() as db:
        # Get project
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            logger.error(f"Project {project_id} not found")
            return
        
        try:
            video_info = {}
            if not skip_download:
                # Update status to downloading
                project.status = ProjectStatus.DOWNLOADING
                await db.commit()
                
                # Download video
                logger.info(f"Downloading video for project {project_id}")
                video_info = download_video(url, work_dir, "source")
                project.original_title = video_info["title"]
                await db.commit()
            else:
                # Reuse existing file
                logger.info(f"Skipping download for project {project_id}")
                from glob import glob
                # Find source video
                files = glob(os.path.join(work_dir, "source.*"))
                if not files:
                    raise FileNotFoundError("Source video not found for rerun")
                video_info["file_path"] = files[0]
                video_info["title"] = project.original_title
            
            # Update status to processing
            project.status = ProjectStatus.PROCESSING
            await db.commit()
            
            # Run the videocuts pipeline
            logger.info(f"Processing video for project {project_id}")
            cfg = Config()
            cfg.paths.input_video = video_info["file_path"]
            cfg.paths.project_name = work_dir
            cfg.llm.enabled = True
            
            # Limit CPU threads to half of available cores
            import multiprocessing
            try:
                cpu_count = multiprocessing.cpu_count()
                cfg.cpu_limit = max(1, cpu_count // 2)
                logger.info(f"Setting CPU limit to {cfg.cpu_limit} threads (half of {cpu_count})")
            except Exception:
                logger.warning("Could not determine CPU count, using default threading")
            
            # Run pipeline and get generated clips
            generated_clips = run_pipeline(cfg)
            
            if generated_clips:
                for clip_data in generated_clips:
                    clip = Clip(
                        id=str(uuid.uuid4()),
                        project_id=project_id,
                        clip_index=clip_data["index"],
                        title=clip_data.get("title"),
                        description=clip_data.get("description"),
                        start_time=clip_data["start"],
                        end_time=clip_data["end"],
                        file_path=clip_data["path"],
                        viral_score=clip_data.get("viral_score"),
                        transcript=clip_data.get("transcript"),
                        clip_full_transcript=clip_data.get("clip_full_transcript"),
                        hook_text=clip_data.get("hook_text"),
                        is_self_contained=True
                    )
                    db.add(clip)
            
            # Mark as completed
            project.status = ProjectStatus.COMPLETED
            await db.commit()
            logger.info(f"Project {project_id} completed successfully with {len(generated_clips or [])} clips")
            
        except Exception as e:
            logger.error(f"Project {project_id} failed: {e}")
            project.status = ProjectStatus.FAILED
            project.error_message = str(e)
            await db.commit()


@router.post("", response_model=ProjectCreateResponse, status_code=201)
async def create_project(
    request: ProjectCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Create a new video processing project."""
    settings = get_settings()
    project_id = str(uuid.uuid4())
    work_dir = os.path.join(settings.storage_path, project_id)
    os.makedirs(work_dir, exist_ok=True)
    
    project = Project(
        id=project_id,
        name=request.name,
        original_url=str(request.url),
        work_dir=work_dir,
        status=ProjectStatus.PENDING,
    )
    
    db.add(project)
    await db.commit()
    await db.refresh(project)
    
    # Queue background processing
    background_tasks.add_task(process_video_task, project_id, str(request.url), work_dir)
    
    return project


@router.post("/{project_id}/rerun", response_model=ProjectCreateResponse)
async def rerun_project(
    project_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Rerun an existing project using cached files."""
    # Get project
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    # Reset status and clear error
    project.status = ProjectStatus.PENDING
    project.error_message = None
    
    # Delete existing clips using a DELETE statement
    from sqlalchemy import delete
    await db.execute(delete(Clip).where(Clip.project_id == project_id))
    
    await db.commit()
    await db.refresh(project)
    
    # Queue background processing with skip_download=True
    background_tasks.add_task(
        process_video_task, 
        project_id, 
        project.original_url, 
        project.work_dir, 
        skip_download=True
    )
    
    return project


@router.get("", response_model=List[ProjectListResponse])
async def list_projects(
    db: AsyncSession = Depends(get_db),
    limit: int = 20,
    offset: int = 0,
):
    """List all projects."""
    result = await db.execute(
        select(Project)
        .order_by(Project.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a project by ID with its clips."""
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.clips))
        .where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return project
