
/**
 * Diagnostic Service for DeepDefend
 * Provides tools for analyzing detection performance and diagnosing issues
 */

import { API_URL, isUsingMockApi } from "./config";
import { handleApiError } from "./error-handler";
import { DetectionResult, ModelInfo } from "@/types";
import { logApiInteraction, getStoredLogs } from "@/utils/debug-logger";

export interface DiagnosticData {
  result_id?: string | number;
  detection_type: 'image' | 'video';
  inference_logs: any[];
  quality_metrics: {
    overall: number;
    blur?: number;
    contrast?: number;
    noise?: number;
    [key: string]: number | undefined;
  };
  model_outputs: {
    model: string;
    raw_output: number;
    calibrated_output: number;
    confidence: number;
    weight: number;
  }[];
  ensemble_info?: {
    weights: Record<string, number>;
    normalization_factor: number;
    weighted_sum: number;
    final_prediction: number;
  };
  uncertainty_analysis?: {
    monte_carlo_samples: number[];
    variance: number;
    uncertainty_score: number;
    calibration_impact: number;
  };
  temporal_analysis?: {
    frame_consistency: number;
    frame_predictions: {
      frame: number;
      probability: number;
      confidence: number;
    }[];
    pattern_analysis?: {
      patterns_detected: string[];
      significance_score: number;
    };
  };
  performance_metrics: {
    total_processing_time: number;
    preprocessing_time?: number;
    inference_time?: number;
    postprocessing_time?: number;
    face_detection_time?: number;
  };
  system_info: {
    api_version?: string;
    model_version?: string;
    timestamp: string;
    client_info?: string;
  };
}

/**
 * Get diagnostic data for a specific detection result
 */
export async function getDiagnosticData(resultId: string | number): Promise<DiagnosticData> {
  try {
    const url = `${API_URL}/diagnostic/${resultId}`;
    
    logApiInteraction("getDiagnosticData", { resultId }, null);
    
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json'
      }
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      logApiInteraction("getDiagnosticData", { resultId }, null, errorData);
      throw new Error(errorData.message || `Failed to get diagnostic data (${response.status})`);
    }
    
    const data = await response.json();
    logApiInteraction("getDiagnosticData", { resultId }, data);
    
    return data;
  } catch (error) {
    logApiInteraction("getDiagnosticData", { resultId }, null, error);
    throw handleApiError(error);
  }
}

/**
 * Generate client-side diagnostic data from available information
 * Used as a fallback when the API endpoint is unavailable
 */
export function generateLocalDiagnosticData(result: DetectionResult): DiagnosticData {
  // Get stored logs
  const logs = getStoredLogs().filter(log => {
    // Only include logs relevant to this detection
    return log.metadata?.result_id === result.id;
  });
  
  // Extract model outputs if available in logs
  const modelOutputs = logs
    .filter(log => log.category === 'model-inference')
    .map(log => ({
      model: log.metadata?.model || 'unknown',
      raw_output: log.metadata?.prediction || 0,
      calibrated_output: log.metadata?.calibrated || log.metadata?.prediction || 0,
      confidence: log.metadata?.confidence || 0.5,
      weight: log.metadata?.weight || 1.0
    }));
  
  // Extract ensemble info from logs
  const ensembleLog = logs.find(log => log.category === 'ensemble');
  const ensembleInfo = ensembleLog ? {
    weights: ensembleLog.metadata?.weights || {},
    normalization_factor: ensembleLog.metadata?.normalization_factor || 1,
    weighted_sum: ensembleLog.metadata?.weighted_sum || 0,
    final_prediction: ensembleLog.metadata?.final_prediction || result.probability
  } : undefined;
  
  // Extract uncertainty analysis from logs
  const uncertaintyLog = logs.find(log => log.category === 'uncertainty');
  const uncertaintyAnalysis = uncertaintyLog ? {
    monte_carlo_samples: uncertaintyLog.metadata?.samples || [],
    variance: uncertaintyLog.metadata?.variance || 0,
    uncertainty_score: uncertaintyLog.metadata?.score || 0,
    calibration_impact: uncertaintyLog.metadata?.impact || 0
  } : undefined;
  
  // Extract temporal analysis for videos
  const temporalLog = logs.find(log => log.category === 'temporal');
  const temporalAnalysis = result.detectionType === 'video' ? {
    frame_consistency: temporalLog?.metadata?.consistency || result.metadata?.prediction_consistency || 0.5,
    frame_predictions: (result.regions || [])
      .filter(r => r.frame !== undefined)
      .map(r => ({
        frame: r.frame || 0,
        probability: r.probability,
        confidence: r.confidence || 0.5
      })),
    pattern_analysis: temporalLog?.metadata?.pattern_analysis
  } : undefined;
  
  // Get performance metrics from logs
  const performanceMetrics = {
    total_processing_time: result.processingTime || 0,
    preprocessing_time: logs.find(log => log.category === 'preprocessing')?.metadata?.time,
    inference_time: logs.find(log => log.category === 'model-inference')?.metadata?.time,
    postprocessing_time: logs.find(log => log.category === 'calibration')?.metadata?.time,
    face_detection_time: logs.find(log => log.category === 'face-detection')?.metadata?.time
  };
  
  return {
    result_id: result.id,
    detection_type: result.detectionType,
    inference_logs: logs,
    quality_metrics: result.metadata?.quality_scores || {
      overall: 0.5
    },
    model_outputs: modelOutputs,
    ensemble_info: ensembleInfo,
    uncertainty_analysis: uncertaintyAnalysis,
    temporal_analysis: temporalAnalysis,
    performance_metrics: performanceMetrics,
    system_info: {
      model_version: result.model,
      timestamp: result.timestamp,
      client_info: navigator.userAgent
    }
  };
}

/**
 * Submit detection feedback to improve the model
 */
export async function submitDetectionFeedback(
  resultId: string | number,
  isCorrect: boolean,
  actualLabel: 'real' | 'fake' | 'unsure',
  comment?: string
): Promise<{success: boolean, message: string}> {
  try {
    const url = `${API_URL}/feedback`;
    
    const payload = {
      result_id: resultId,
      is_correct: isCorrect,
      actual_label: actualLabel,
      comment: comment || '',
      client_info: navigator.userAgent,
      timestamp: new Date().toISOString()
    };
    
    logApiInteraction("submitDetectionFeedback", payload, null);
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      logApiInteraction("submitDetectionFeedback", payload, null, errorData);
      throw new Error(errorData.message || `Failed to submit feedback (${response.status})`);
    }
    
    const data = await response.json();
    logApiInteraction("submitDetectionFeedback", payload, data);
    
    return {
      success: true,
      message: data.message || 'Feedback submitted successfully'
    };
  } catch (error) {
    logApiInteraction("submitDetectionFeedback", { resultId, isCorrect, actualLabel }, null, error);
    
    // Return success even if API fails to avoid frustrating the user
    return {
      success: false,
      message: 'Failed to submit feedback, but your input has been recorded locally'
    };
  }
}

/**
 * Get model performance metrics
 */
export async function getModelPerformanceMetrics(): Promise<any> {
  try {
    const url = `${API_URL}/model/performance`;
    
    logApiInteraction("getModelPerformanceMetrics", {}, null);
    
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json'
      }
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      logApiInteraction("getModelPerformanceMetrics", {}, null, errorData);
      throw new Error(errorData.message || `Failed to get model metrics (${response.status})`);
    }
    
    const data = await response.json();
    logApiInteraction("getModelPerformanceMetrics", {}, data);
    
    return data;
  } catch (error) {
    logApiInteraction("getModelPerformanceMetrics", {}, null, error);
    throw handleApiError(error);
  }
}
