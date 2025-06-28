
import { DetectionResult, UploadResponse } from "@/types";
import { validateImageFile } from "./validation";
import { API_URL, isUsingMockApi } from "./config";
import { handleApiError } from "./error-handler";
import { mockUploadImage } from "./mock-api";

export const uploadImage = async (file: File): Promise<UploadResponse> => {
  // Validate file before sending to server
  const validation = validateImageFile(file);
  if (!validation.valid) {
    return {
      success: false,
      message: validation.message || 'Invalid file',
      error: validation.message,
      statusCode: 400
    };
  }

  // If we're using the mock API, return mock data
  if (isUsingMockApi()) {
    return await mockUploadImage(file);
  }

  try {
    const formData = new FormData();
    formData.append('file', file);
    
    // Add additional parameters for our enhanced detection
    formData.append('model', 'indian_specialized');
    formData.append('ensemble', 'true');
    formData.append('indianEnhancement', 'true');

    const response = await fetch(`${API_URL}/api/detect`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || data.message || `Failed to upload image (${response.status})`);
    }

    // Map the API response to our frontend's expected format
    const result: DetectionResult = {
      id: data.detection_id,
      imageName: data.result.filename,
      probability: data.result.probability,
      confidence: data.result.confidence,
      processingTime: data.result.processingTime,
      timestamp: new Date().toISOString(),
      detectionType: data.result.detectionType,
      model: data.result.model,
      regions: data.result.regions.map((region: any) => ({
        box: region.box,
        probability: region.probability,
        confidence: region.confidence,
        skinTone: region.skin_tone?.indian_tone?.name || 'Unknown'
      }))
    };

    return {
      success: true,
      message: 'Image uploaded successfully',
      result: result,
      statusCode: response.status
    };
  } catch (error) {
    return handleApiError(error);
  }
};
