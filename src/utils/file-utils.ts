
import { saveAs } from 'file-saver';
import { toast } from 'sonner';
import { logReportGeneration } from './debug-logger';

/**
 * Save a blob as a file with the given filename
 */
export const saveFile = (blob: Blob, filename: string): boolean => {
  try {
    if (!blob) {
      throw new Error('Invalid blob: Blob is undefined or null');
    }
    
    if (blob.size === 0) {
      throw new Error('Empty blob: Cannot save a zero-size file');
    }
    
    console.log(`Saving file: ${filename}, size: ${blob.size}, type: ${blob.type}`);
    saveAs(blob, filename);
    return true;
  } catch (error) {
    console.error('Error saving file:', error);
    toast.error(`Failed to save file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    logReportGeneration('file_save_error', { 
      filename,
      error: String(error),
      blob_size: blob?.size,
      blob_type: blob?.type 
    });
    return false;
  }
};

/**
 * Create a blob from JSON data
 */
export const createJsonBlob = (data: any): Blob => {
  const jsonString = JSON.stringify(data, null, 2);
  return new Blob([jsonString], { type: 'application/json' });
};

/**
 * Create a blob from CSV data
 */
export const createCsvBlob = (csvContent: string): Blob => {
  return new Blob([csvContent], { type: 'text/csv' });
};

/**
 * Create a blob from HTML content
 */
export const createHtmlBlob = (htmlContent: string): Blob => {
  return new Blob([htmlContent], { type: 'text/html' });
};
