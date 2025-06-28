
import { DetectionResult, UploadResponse } from "@/types";
import { validateVideoFile } from "./validation";
import { API_URL, isUsingMockApi } from "./config";
import { handleApiError } from "./error-handler";
import { mockUploadVideo } from "./mock-api";
import { toast } from "sonner";

interface VideoUploadOptions {
  model?: string;
  useEnsemble?: boolean;
  debugMode?: boolean;
  token?: string;
}

export const uploadVideo = async (
  file: File,
  options: VideoUploadOptions = {}
): Promise<UploadResponse> => {
  console.log("Starting video upload process", { fileName: file.name, fileSize: file.size });
  
  // Validate file before sending to server
  const validation = validateVideoFile(file);
  if (!validation.valid) {
    console.error("Video validation failed:", validation.message);
    return {
      success: false,
      message: validation.message || 'Invalid file',
      error: validation.message,
      statusCode: 400
    };
  }

  // If we're using the mock API, return mock data
  if (isUsingMockApi()) {
    console.log("Using mock API for video upload");
    return await mockUploadVideo(file);
  }

  try {
    const formData = new FormData();
    formData.append('video', file);

    // Enable debug mode by default to collect more data about potential issues
    const debugMode = options.debugMode !== undefined ? options.debugMode : true;
    
    // Build URL with query parameters
    let url = `${API_URL}/detect-video`;
    const params = new URLSearchParams();
    
    // Add model parameter if specified
    if (options.model) {
      params.append('model', options.model);
      console.log(`Using specific model: ${options.model}`);
    }
    
    // Add ensemble parameter if specified
    if (options.useEnsemble !== undefined) {
      params.append('ensemble', options.useEnsemble ? 'true' : 'false');
      console.log(`Ensemble detection ${options.useEnsemble ? 'enabled' : 'disabled'}`);
    }
    
    // Add debug mode parameter
    if (debugMode) {
      params.append('debug', 'true');
      console.log("Debug mode enabled for detailed logging");
    }
    
    // Add params to URL if we have any
    if (params.toString()) {
      url += `?${params.toString()}`;
    }

    // Configure headers
    const headers: HeadersInit = {};
    
    // Add auth token if provided
    if (options.token) {
      headers['Authorization'] = `Bearer ${options.token}`;
    }

    console.log(`Sending request to ${url}`);
    const startTime = Date.now();
    
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      headers
    });

    const processingTime = Date.now() - startTime;
    console.log(`Received response in ${processingTime}ms with status ${response.status}`);

    const data = await response.json();
    console.log("Detection response data:", data);
    
    if (!response.ok) {
      console.error("Video detection failed:", data.message || response.statusText);
      throw new Error(data.message || `Failed to upload video (${response.status})`);
    }

    // Check if result is valid
    if (!data.result || typeof data.result.probability !== 'number') {
      console.error("Invalid detection result format:", data.result);
      throw new Error("Server returned an invalid detection result");
    }

    // Log important detection metrics
    console.log("Detection metrics:", {
      probability: data.result.probability,
      confidence: data.result.confidence,
      processingTime: data.result.processingTime,
      frameCount: data.result.frameCount
    });

    // Display debug info as toast if in debug mode
    if (debugMode) {
      const metrics = data.result.metadata?.processing_metrics;
      if (metrics) {
        const debugInfo = `Processed ${metrics.frames?.processed || 0}/${metrics.frames?.requested || 0} frames with ${metrics.faces?.total || 0} faces`;
        toast.info(debugInfo, { duration: 5000 });
      }
    }

    return {
      success: true,
      message: 'Video uploaded successfully',
      result: data.result as DetectionResult,
      statusCode: response.status
    };
  } catch (error) {
    console.error("Error during video upload:", error);
    return handleApiError(error);
  }
};
