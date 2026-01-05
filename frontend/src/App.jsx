import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  Link as LinkIcon,
  Sparkles,
  Type,
  Scissors,
  Mic2,
  RefreshCcw,
  Layout,
  Clock
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import './App.css';
import ProjectDetails from './ProjectDetails';

const API_BASE_URL = 'http://localhost:8000';

const FeatureIcon = ({ icon: Icon, label }) => (
  <div className="feature-item">
    <div className="feature-icon-wrapper">
      <Icon size={20} />
    </div>
    <span style={{ fontSize: '0.75rem' }}>{label}</span>
  </div>
);

const getYoutubeThumbnail = (url) => {
  if (!url) return null;
  const regExp = /^.*(youtu\.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
  const match = url.match(regExp);
  if (match && match[2].length === 11) {
    return `https://img.youtube.com/vi/${match[2]}/hqdefault.jpg`;
  }
  return null;
};

const ProjectCard = ({ project }) => {
  const navigate = useNavigate();
  const isProcessing = ['pending', 'downloading', 'processing'].includes(project.status.toLowerCase());
  const thumbnail = getYoutubeThumbnail(project.original_url);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="project-card"
      onClick={() => ['completed', 'failed'].includes(project.status.toLowerCase()) && navigate(`/projects/${project.id}`)}
      style={{ cursor: ['completed', 'failed'].includes(project.status.toLowerCase()) ? 'pointer' : 'default' }}
    >
      <div className="card-thumbnail">
        {thumbnail ? (
          <img src={thumbnail} alt={project.original_title || project.name} onError={(e) => {
            e.target.src = 'https://images.unsplash.com/photo-1611162617474-5b21e879e113?q=80&w=1000&auto=format&fit=crop';
            e.target.style.opacity = 0.5;
          }} />
        ) : (
          <div className="placeholder-content" />
        )}
        <span className={`status-badge status-${project.status.toLowerCase()}`}>
          {project.status}
        </span>
      </div>
      <div className="card-info">
        <h3 className="card-title">{project.original_title || project.name || 'Untitled Video'}</h3>
        <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
          {new Date(project.created_at).toLocaleDateString()}
        </p>

        {isProcessing && (
          <div className="progress-bar">
            <motion.div
              className="progress-fill"
              initial={{ width: 0 }}
              animate={{ width: '100%' }}
              transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
            />
          </div>
        )}

        {project.status === 'completed' && (
          <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--accent-secondary)', fontWeight: 600 }}>
            {project.clips_count || 0} clips generated
          </p>
        )}
      </div>
    </motion.div>
  );
};

const ProjectList = () => {
  const [url, setUrl] = useState('');
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchProjects = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/v1/projects`);
      setProjects(response.data);
    } catch (err) {
      console.error('Failed to fetch projects:', err);
    }
  };

  useEffect(() => {
    fetchProjects();
    const interval = setInterval(fetchProjects, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url) return;

    setLoading(true);
    setError(null);
    try {
      await axios.post(`${API_BASE_URL}/v1/projects`, {
        name: `Project ${new Date().toISOString()}`,
        url: url
      });
      setUrl('');
      fetchProjects();
    } catch (err) {
      setError('Failed to start project. Please check the URL.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <motion.h1
          className="gradient-text"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          Video Cuts
        </motion.h1>
        <p>Turn long-form videos into viral shorts in one click. Powered by AI.</p>
      </header>

      <section className="input-section glass">
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <LinkIcon className="input-icon" size={18} />
            <input
              type="text"
              className="link-input"
              placeholder="Paste YouTube link here"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              disabled={loading}
            />
          </div>
          <button
            type="submit"
            className="submit-button"
            disabled={loading || !url}
          >
            {loading ? 'Processing...' : 'Get clips in 1 click'}
          </button>
        </form>

        {error && <p style={{ color: '#f87171', fontSize: '0.85rem' }}>{error}</p>}

        {/*
        TODO: Add features
        <div className="feature-list">
          <FeatureIcon icon={Sparkles} label="Long to shorts" />
          <FeatureIcon icon={Type} label="AI Captions" />
          <FeatureIcon icon={Scissors} label="Video editor" />
          <FeatureIcon icon={Mic2} label="Enhance speech" />
          <FeatureIcon icon={RefreshCcw} label="AI Reframe" />
          <FeatureIcon icon={Layout} label="AI B-Roll" />
        </div> */}
      </section>

      <section className="projects-section">
        <div className="projects-header">
          <h2>All projects ({projects.length})</h2>
          {/* <div style={{ display: 'flex', gap: '1rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            <span>Auto-save</span>
            <span>Auto-import</span>
          </div> */}
        </div>

        <AnimatePresence mode="popLayout">
          <div className="projects-grid">
            {projects.map((project) => (
              <ProjectCard key={project.id} project={project} />
            ))}
          </div>
        </AnimatePresence>

        {projects.length === 0 && (
          <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--text-secondary)' }}>
            <Clock size={48} strokeWidth={1} style={{ marginBottom: '1rem' }} />
            <p>No projects yet. Paste a link above to get started!</p>
          </div>
        )}
      </section>
    </div>
  );
};

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ProjectList />} />
        <Route path="/projects/:id" element={<ProjectDetails />} />
      </Routes>
    </Router>
  );
}

export default App;
