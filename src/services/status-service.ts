
import { API_URL, isUsingMockApi } from "./config";
import { ApiStatus } from "@/types";
import { mockCheckApiStatus } from "./mock-api";

// Check API status
export const checkApiStatus = async (): Promise<ApiStatus> => {
  // If we're using the mock API, return mock data
  if (isUsingMockApi()) {
    return await mockCheckApiStatus();
  }
  
  try {
    console.log(`Checking API status at: ${API_URL}/status`);
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
    
    const response = await fetch(`${API_URL}/status`, { 
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    // Check if the response is JSON
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      console.error(`API returned non-JSON response: ${contentType}`);
      throw new Error(`API at ${API_URL} returned HTML instead of JSON. This may indicate a routing or CORS issue.`);
    }
    
    if (!response.ok) {
      console.warn(`API returned status ${response.status}`);
      throw new Error(`API returned status ${response.status}`);
    }
    
    const data = await response.json();
    console.log("API status response:", data);
    
    return {
      status: data.status || 'offline',
      message: data.message || 'API status unknown',
      version: data.version,
      database: data.database,
      config: data.config
    };
  } catch (error) {
    console.error('API status check failed:', error);
    
    // Check if it's a connection error vs a server error
    const errorMessage = error instanceof Error ? error.message : 'Connection failed';
    const isConnectionError = errorMessage.includes('Failed to fetch') || 
                              errorMessage.includes('NetworkError') ||
                              errorMessage.includes('aborted') ||
                              errorMessage.includes('TypeError') ||
                              errorMessage.includes('HTML instead of JSON');
    
    return {
      status: 'offline',
      message: isConnectionError 
        ? `Cannot connect to API server at ${API_URL}. Make sure it's running and CORS is configured.` 
        : errorMessage
    };
  }
};
