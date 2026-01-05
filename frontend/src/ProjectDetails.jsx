import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import {
    ChevronLeft,
    Play,
    RotateCcw,
    Clock,
    TrendingUp,
    X
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE_URL = 'http://localhost:8000';

const VideoDialog = ({ clip, project, onClose }) => {
    if (!clip) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <motion.div
                className="modal-content"
                onClick={(e) => e.stopPropagation()}
                initial={{ scale: 0.95, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.95, opacity: 0 }}
            >
                <button className="modal-close" onClick={onClose}>
                    <X size={20} />
                </button>

                <div className="video-container">
                    <video
                        controls
                        autoPlay
                        src={`${API_BASE_URL}/v1/projects/${project.id}/clips/${clip.id}/video`}
                    />
                </div>

                <div className="info-container">
                    <h2 style={{ fontSize: '1.25rem' }}>{clip.title || 'Generated Clip'}</h2>

                    <div style={{ display: 'flex', gap: '0.3rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                            <TrendingUp size={14} color="#4ade80" />
                            <span>Score: {clip.viral_score?.toFixed(1) || 'N/A'}</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                            <Clock size={14} />
                            <span>{clip.start_time.toFixed(1)}s - {clip.end_time.toFixed(1)}s</span>
                        </div>
                    </div>

                    <div style={{ marginTop: '0.5rem' }}>
                        <h3 style={{ fontSize: '0.875rem', marginBottom: '0.375rem' }}>Description</h3>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>{clip.description}</p>
                    </div>

                    <div style={{ marginTop: '0.5rem' }}>
                        <h3 style={{ fontSize: '0.875rem', marginBottom: '0.375rem' }}>Transcript</h3>
                        <div className="transcript-area">
                            {clip.clip_full_transcript || clip.transcript}
                        </div>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};

const ClipCard = ({ clip, project, onClick }) => (
    <motion.div
        layout
        className="project-card"
        onClick={onClick}
        style={{ cursor: 'pointer' }}
    >
        <div className="card-thumbnail vertical-thumbnail">
            <img
                src={`${API_BASE_URL}/v1/projects/${project.id}/clips/${clip.id}/thumbnail`}
                alt={clip.title || 'Clip preview'}
            />
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Play size={24} style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.5))' }} />
            </div>
            <div className="status-badge" style={{ left: '1.25rem', top: '1.25rem', right: 'auto', background: 'rgba(0,0,0,0.7)' }}>
                {Math.floor(clip.end_time - clip.start_time)}s
            </div>
        </div>
        <div className="card-info" style={{ padding: '0.75rem' }}>
            <h3 className="card-title" style={{ fontSize: '0.875rem', whiteSpace: 'normal', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', height: '2.5rem' }}>
                {clip.hook_text || clip.title || `Clip #${clip.clip_index}`}
            </h3>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.25rem' }}>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                    Score: {clip.viral_score?.toFixed(1) || 'N/A'}
                </span>
            </div>
        </div>
    </motion.div>
);

const ProjectDetails = () => {
    const { id } = useParams();
    const [project, setProject] = useState(null);
    const [selectedClip, setSelectedClip] = useState(null);
    const [loading, setLoading] = useState(true);
    const [rerunning, setRerunning] = useState(false);

    const fetchProject = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/v1/projects/${id}`);
            setProject(response.data);
        } catch (err) {
            console.error('Failed to fetch project:', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchProject();
        const interval = setInterval(() => {
            if (project?.status !== 'completed' && project?.status !== 'failed') {
                fetchProject();
            }
        }, 5000);
        return () => clearInterval(interval);
    }, [id, project?.status]);

    const handleRerun = async () => {
        setRerunning(true);
        try {
            await axios.post(`${API_BASE_URL}/v1/projects/${id}/rerun`);
            fetchProject();
        } catch (err) {
            alert('Failed to rerun project');
        } finally {
            setRerunning(false);
        }
    };

    if (loading) return <div className="app-container"><p>Loading project...</p></div>;
    if (!project) return <div className="app-container"><p>Project not found.</p></div>;

    return (
        <div className="app-container">
            <div className="details-container">
                <div>
                    <Link to="/" className="back-link">
                        <ChevronLeft size={14} />
                        Back to projects
                    </Link>
                    <div className="details-header">
                        <div>
                            <h1 style={{ fontSize: '1.5rem' }}>{project.original_title || project.name}</h1>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem', fontSize: '0.8rem' }}>
                                {project.original_url}
                            </p>
                        </div>
                        <button
                            className="rerun-button"
                            onClick={handleRerun}
                            disabled={rerunning || ['pending', 'downloading', 'processing'].includes(project.status.toLowerCase())}
                        >
                            <RotateCcw size={14} className={rerunning ? 'spin' : ''} />
                            {rerunning ? 'Rerunning...' : 'Rerun Project'}
                        </button>
                    </div>
                </div>

                <section className="projects-section">
                    <h2>All clips ({project.clips?.length || 0})</h2>

                    <div className="projects-grid clips-grid" style={{ marginTop: '2rem' }}>
                        {project.clips?.map((clip) => (
                            <ClipCard
                                key={clip.id}
                                clip={clip}
                                project={project}
                                onClick={() => setSelectedClip(clip)}
                            />
                        ))}
                    </div>

                    {project.clips?.length === 0 && project.status === 'completed' && (
                        <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--text-secondary)' }}>
                            <p>No clips were found for this video.</p>
                        </div>
                    )}

                    {['pending', 'downloading', 'processing'].includes(project.status.toLowerCase()) && (
                        <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--text-secondary)' }}>
                            <p>Processing video... clips will appear here shortly.</p>
                            <div className="progress-bar" style={{ maxWidth: '400px', margin: '1rem auto' }}>
                                <motion.div
                                    className="progress-fill"
                                    initial={{ width: 0 }}
                                    animate={{ width: '100%' }}
                                    transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                                />
                            </div>
                        </div>
                    )}
                </section>
            </div>

            <AnimatePresence>
                {selectedClip && (
                    <VideoDialog
                        clip={selectedClip}
                        project={project}
                        onClose={() => setSelectedClip(null)}
                    />
                )}
            </AnimatePresence>
        </div>
    );
};

export default ProjectDetails;
