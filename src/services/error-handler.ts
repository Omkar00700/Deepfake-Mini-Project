
import { UploadResponse } from "@/types";

/**
 * Type for consistent API error responses
 */
export type ApiErrorResponse = {
  success: boolean;
  message: string;
  error?: string;
  statusCode?: number;
  data?: any;
  contentType?: string;
};

/**
 * Handles API errors consistently
 */
export const handleApiError = (error: unknown): ApiErrorResponse => {
  console.error('API error:', error);
  
  // If error is a Response object (from fetch)
  if (error instanceof Response) {
    return {
      success: false,
      message: `API error: ${error.status} ${error.statusText}`,
      error: error.statusText,
      statusCode: error.status
    };
  }
  
  // If error is an Error object
  return {
    success: false,
    message: error instanceof Error ? error.message : 'An unknown error occurred',
    error: error instanceof Error ? error.message : 'Unknown error',
    statusCode: error instanceof Response ? error.status : 500
  };
};
