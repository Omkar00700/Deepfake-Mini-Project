
/**
 * File validation utilities
 */

// File validation constants
const MAX_IMAGE_SIZE = 5 * 1024 * 1024; // 5MB
const MAX_VIDEO_SIZE = 50 * 1024 * 1024; // 50MB
const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];
const ALLOWED_VIDEO_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm'];

/**
 * Validates image files before upload
 */
export const validateImageFile = (file: File): { valid: boolean; message?: string } => {
  if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
    return { 
      valid: false, 
      message: `Invalid file type. Allowed types: ${ALLOWED_IMAGE_TYPES.join(', ')}` 
    };
  }
  
  if (file.size > MAX_IMAGE_SIZE) {
    return { 
      valid: false, 
      message: `File is too large. Maximum size: ${MAX_IMAGE_SIZE / (1024 * 1024)}MB` 
    };
  }
  
  return { valid: true };
};

/**
 * Validates video files before upload
 */
export const validateVideoFile = (file: File): { valid: boolean; message?: string } => {
  if (!ALLOWED_VIDEO_TYPES.includes(file.type)) {
    return { 
      valid: false, 
      message: `Invalid file type. Allowed types: ${ALLOWED_VIDEO_TYPES.join(', ')}` 
    };
  }
  
  if (file.size > MAX_VIDEO_SIZE) {
    return { 
      valid: false, 
      message: `File is too large. Maximum size: ${MAX_VIDEO_SIZE / (1024 * 1024)}MB` 
    };
  }
  
  return { valid: true };
};

export const getFileValidationInfo = () => {
  return {
    image: {
      maxSize: MAX_IMAGE_SIZE / (1024 * 1024),
      allowedTypes: ALLOWED_IMAGE_TYPES
    },
    video: {
      maxSize: MAX_VIDEO_SIZE / (1024 * 1024),
      allowedTypes: ALLOWED_VIDEO_TYPES
    }
  };
};
