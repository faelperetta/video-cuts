import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Enum as SQLEnum, Text
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column

class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class ProjectStatus(str, Enum):
    """Project processing status."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Project(Base):
    """A video processing project."""
    __tablename__ = "projects"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[ProjectStatus] = mapped_column(SQLEnum(ProjectStatus), default=ProjectStatus.PENDING)
    original_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    original_title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    work_dir: Mapped[str] = mapped_column(String(500), nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    clips: Mapped[List["Clip"]] = relationship("Clip", back_populates="project", cascade="all, delete-orphan")


class Clip(Base):
    """A generated clip from a project."""
    __tablename__ = "clips"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    clip_index: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    viral_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    clip_full_transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    hook_text: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_self_contained: Mapped[Optional[bool]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    project: Mapped["Project"] = relationship("Project", back_populates="clips")
