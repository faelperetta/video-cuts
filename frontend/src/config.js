/**
 * API Configuration
 * 
 * The API base URL is configurable via environment variables:
 * - Development: Set VITE_API_URL in .env.local
 * - Production: Set VITE_API_URL during build
 * 
 * Default: http://localhost:8000
 */

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
