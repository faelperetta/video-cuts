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
import cv2

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/projects", tags=["projects"])


async def process_video_task(project_id: str, url: str, work_dir: str, skip_download: bool = False):
    """Background task to download and process a video."""
    from videocuts.api.database import AsyncSessionLocal
    from videocuts.config import Config
    from videocuts.main import run_pipeline
    from starlette.concurrency import run_in_threadpool
    
    async with AsyncSessionLocal() as db:
        # Get project
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            logger.error(f"Project {project_id} not found")
            return
        
        try:
            logger.info(f"Starting background task for project {project_id} (skip_download={skip_download})")
            video_info = {}
            
            # Immediately set to PROCESSING if we are skipping download (rerun)
            if skip_download:
                project.status = ProjectStatus.PROCESSING
                await db.commit()
                logger.info(f"Set project {project_id} status to PROCESSING (RERUN)")

            if not skip_download:
                # Update status to downloading
                project.status = ProjectStatus.DOWNLOADING
                await db.commit()
                
                # Download video (Offload to threadpool)
                logger.info(f"Downloading video for project {project_id}")
                video_info = await run_in_threadpool(download_video, url, work_dir, "source")
                project.original_title = video_info["title"]
                await db.commit()
            else:
                # Reuse existing file
                logger.info(f"Skipping download for project {project_id}")
                from glob import glob
                # Find source video
                files = await run_in_threadpool(glob, os.path.join(work_dir, "source.*"))
                if not files:
                    raise FileNotFoundError("Source video not found for rerun")
                video_info["file_path"] = files[0]
                video_info["title"] = project.original_title
            
            # Update status to processing
            project.status = ProjectStatus.PROCESSING
            await db.commit()

            # Generate thumbnail if source exists
            try:
                source_path = video_info.get("file_path")
                if source_path and os.path.exists(source_path):
                    # Offload cv2 operations to threadpool
                    def generate_thumb():
                        cap = cv2.VideoCapture(source_path)
                        # Seek to 5 seconds or middle of video if shorter
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        duration = frame_count / fps if fps > 0 else 0
                        target_sec = min(5, duration / 2)
                        cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000)
                        ret, frame = cap.read()
                        if ret:
                            thumb_path = os.path.join(work_dir, "thumbnail.jpg")
                            cv2.imwrite(thumb_path, frame)
                        cap.release()
                        return ret
                    
                    await run_in_threadpool(generate_thumb)
                    logger.info(f"Generated thumbnail for project {project_id}")
            except Exception as te:
                logger.warning(f"Failed to generate thumbnail for project {project_id}: {te}")
            
            # Run the videocuts pipeline
            logger.info(f"Processing video for project {project_id}")
            cfg = Config()
            cfg.paths.input_video = video_info["file_path"]
            cfg.paths.project_name = work_dir
            cfg.llm.enabled = True
            
            # Limit CPU threads to half of available cores, capped between 4 and 8
            import multiprocessing
            try:
                cpu_count = multiprocessing.cpu_count()
                # Recommendation: 4-8 threads is usually ideal for system responsiveness
                cfg.cpu_limit = min(max(4, cpu_count // 2), 8)
                # Ensure we don't exceed actual cpu count on low-end systems
                cfg.cpu_limit = min(cfg.cpu_limit, cpu_count)
                logger.info(f"Setting CPU limit to {cfg.cpu_limit} threads (half of {cpu_count}, capped 4-8)")
            except Exception:
                logger.warning("Could not determine CPU count, using default threading")
            
            # Run pipeline and get generated clips (Offload to threadpool)
            generated_clips = await run_in_threadpool(run_pipeline, cfg)
            
            if generated_clips:
                for clip_data in generated_clips:
                    clip_id = str(uuid.uuid4())
                    
                    # Generate clip thumbnail (Offload to threadpool)
                    def generate_clip_thumb(c_data, w_dir):
                        clip_source = c_data["path"]
                        if os.path.exists(clip_source):
                            c_cap = cv2.VideoCapture(clip_source)
                            c_ret, c_frame = c_cap.read()
                            if c_ret:
                                thumb_filename = f"clip_{c_data['index']}_thumb.jpg"
                                clip_thumb_path = os.path.join(w_dir, thumb_filename)
                                cv2.imwrite(clip_thumb_path, c_frame)
                            c_cap.release()
                        return True

                    await run_in_threadpool(generate_clip_thumb, clip_data, work_dir)

                    clip = Clip(
                        id=clip_id,
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
            logger.error(f"Project {project_id} failed: {e}", exc_info=True)
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
    logger.info(f"Rerunning project {project_id}: setting status to PENDING")
    
    # Delete existing clips using a DELETE statement
    from sqlalchemy import delete
    await db.execute(delete(Clip).where(Clip.project_id == project_id))
    
    await db.commit()
    await db.refresh(project)
    logger.info(f"Project {project_id} committed as PENDING, starting background task")
    
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
    from sqlalchemy import func
    
    # Subquery to count clips per project
    clips_subquery = (
        select(Clip.project_id, func.count(Clip.id).label("count"))
        .group_by(Clip.project_id)
        .subquery()
    )
    
    result = await db.execute(
        select(
            Project.id,
            Project.name,
            Project.status,
            Project.original_title,
            Project.original_url,
            Project.created_at,
            func.coalesce(clips_subquery.c.count, 0).label("clips_count")
        )
        .outerjoin(clips_subquery, Project.id == clips_subquery.c.project_id)
        .order_by(Project.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    
    # Convert result rows to dictionaries that match ProjectListResponse
    projects = []
    for row in result.all():
        projects.append({
            "id": row.id,
            "name": row.name,
            "status": row.status,
            "original_title": row.original_title,
            "original_url": row.original_url,
            "created_at": row.created_at,
            "clips_count": row.clips_count
        })
    return projects


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


@router.get("/{project_id}/thumbnail")
async def get_project_thumbnail(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Serve the thumbnail for a project."""
    from fastapi.responses import FileResponse
    
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    thumb_path = os.path.join(project.work_dir, "thumbnail.jpg")
    if not os.path.exists(thumb_path):
        # Fallback to a placeholder or 404
        raise HTTPException(status_code=404, detail="Thumbnail not found")
        
    return FileResponse(thumb_path)


@router.get("/{project_id}/clips/{clip_id}/video")
async def get_clip_video(
    project_id: str,
    clip_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Serve the video file for a specific clip."""
    from fastapi.responses import FileResponse
    
    result = await db.execute(
        select(Clip)
        .where(Clip.id == clip_id)
        .where(Clip.project_id == project_id)
    )
    clip = result.scalar_one_or_none()
    
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
        
    if not os.path.exists(clip.file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
        
    return FileResponse(clip.file_path, media_type="video/mp4")


@router.get("/{project_id}/clips/{clip_id}/thumbnail")
async def get_clip_thumbnail(
    project_id: str,
    clip_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Serve the thumbnail for a specific clip."""
    from fastapi.responses import FileResponse
    
    result = await db.execute(
        select(Clip)
        .where(Clip.id == clip_id)
        .where(Clip.project_id == project_id)
    )
    clip = result.scalar_one_or_none()
    
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
        
    # Standard thumbnail path based on index
    # Note: In a production app, we would store the thumb_path in the DB.
    # For now, we follow the naming convention clip_{index}_thumb.jpg
    # Thumbnails are stored in the project work_dir
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    
    if not project:
         raise HTTPException(status_code=404, detail="Project not found")

    thumb_path = os.path.join(project.work_dir, f"clip_{clip.clip_index}_thumb.jpg")
    
    if not os.path.exists(thumb_path):
        # Fallback to project thumbnail if clip thumb is missing
        p_thumb = os.path.join(project.work_dir, "thumbnail.jpg")
        
        if p_thumb and os.path.exists(p_thumb):
            return FileResponse(p_thumb)
        raise HTTPException(status_code=404, detail="Thumbnail not found")
        
    return FileResponse(thumb_path)


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a project and all associated files."""
    import shutil
    
    # Get project
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Retrieve work_dir before deleting the project record
    work_dir = project.work_dir
    
    # Delete project (cascades to clips)
    await db.delete(project)
    await db.commit()
    
    # Delete storage files
    if work_dir and os.path.exists(work_dir):
        try:
            shutil.rmtree(work_dir)
            logger.info(f"Deleted project directory: {work_dir}")
        except Exception as e:
            logger.error(f"Failed to delete project directory {work_dir}: {e}")
            # We don't raise an error here to ensure the API call succeeds even if file deletion fails partially
            
    logger.info(f"Project {project_id} deleted")
    return None
