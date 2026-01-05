from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, HttpUrl
from videocuts.api.models import ProjectStatus


# --- Project Schemas ---

class ProjectCreate(BaseModel):
    """Request body for creating a new project."""
    name: str
    url: HttpUrl


class ClipResponse(BaseModel):
    """Response model for a single clip."""
    id: str
    clip_index: int
    title: Optional[str] = None
    description: Optional[str] = None
    start_time: float
    end_time: float
    file_path: str
    viral_score: Optional[float] = None
    transcript: Optional[str] = None
    clip_full_transcript: Optional[str] = None
    hook_text: Optional[str] = None
    is_self_contained: Optional[bool] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ProjectResponse(BaseModel):
    """Response model for a project."""
    id: str
    name: str
    status: ProjectStatus
    original_url: str
    original_title: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    clips: List[ClipResponse] = []

    class Config:
        from_attributes = True


class ProjectCreateResponse(BaseModel):
    """Response model for project creation (without clips)."""
    id: str
    name: str
    status: ProjectStatus
    original_url: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectListResponse(BaseModel):
    """Response model for listing projects."""
    id: str
    name: str
    status: ProjectStatus
    original_title: Optional[str] = None
    original_url: str
    created_at: datetime
    clips_count: int = 0

    class Config:
        from_attributes = True
