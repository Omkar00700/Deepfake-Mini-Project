
/**
 * API configuration
 */

// Get environment variable if available
const apiUrl = import.meta.env.VITE_API_URL;

// Determine appropriate API URL based on environment
export const API_URL = (() => {
  // If explicitly configured via env var, use that
  if (apiUrl) {
    console.log(`Using API URL from environment variable: ${apiUrl}`);
    return apiUrl;
  }
  
  // Check if we're running on the same domain as the API
  const currentOrigin = window.location.origin;
  const hostname = window.location.hostname;
  
  // For local development
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    console.log('Local development detected, using http://localhost:5000');
    return 'http://localhost:5000';
  } 
  
  // For Lovable preview environments
  if (hostname.includes('lovable')) {
    console.log('Lovable preview environment detected - using mock API');
    return '/mock-api'; // We'll create a mock API implementation
  }
  
  // For all other cases (production), try to use relative API path
  console.log('Production environment detected, using relative /api path');
  return '/api';
})();

// Tell the user which API we're connecting to
console.log(`DeepDefend is configured to use API at: ${API_URL}`);

// Helper function to determine if we're using the mock API
export const isUsingMockApi = () => {
  return API_URL === '/mock-api';
};
