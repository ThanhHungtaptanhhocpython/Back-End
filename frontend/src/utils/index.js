// Utility functions for video processing and data manipulation

export const formatTime = (seconds) => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
};

export const parseTime = (timeString) => {
  const parts = timeString.split(':').map(Number);
  if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  if (parts.length === 2) {
    return parts[0] * 60 + parts[1];
  }
  return parts[0] || 0;
};

export const validateVideoFile = (file) => {
  const allowedTypes = ['video/mp4', 'video/webm', 'video/ogg'];
  const maxSize = 500 * 1024 * 1024; // 500MB
  
  if (!allowedTypes.includes(file.type)) {
    throw new Error('Invalid video format. Please use MP4, WebM, or OGG.');
  }
  
  if (file.size > maxSize) {
    throw new Error('Video file is too large. Maximum size is 500MB.');
  }
  
  return true;
};

export const generateId = () => {
  return Math.random().toString(36).substr(2, 9);
};

export const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};
