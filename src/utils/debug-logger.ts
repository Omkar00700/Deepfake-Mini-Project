
/**
 * Comprehensive debug logging utility for the detection pipeline
 * Enables detailed logging at each step of the inference process
 */

import { DetectionResult, DetectionRegion } from "@/types";

// Log levels
export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3
}

// Current log level - can be adjusted based on environment
let currentLogLevel = LogLevel.INFO;

// Enable more detailed logging in development
if (process.env.NODE_ENV === 'development') {
  currentLogLevel = LogLevel.DEBUG;
}

// Log categories to organize logs
export enum LogCategory {
  PREPROCESSING = 'preprocessing',
  FACE_DETECTION = 'face-detection',
  MODEL_INFERENCE = 'model-inference',
  ENSEMBLE = 'ensemble',
  CALIBRATION = 'calibration',
  TEMPORAL = 'temporal',
  QUALITY = 'quality',
  UNCERTAINTY = 'uncertainty',
  REPORT = 'report',
  API = 'api',
  GENERAL = 'general'
}

interface LogMetadata {
  [key: string]: any;
}

// Maximum number of logs to store in session storage
const MAX_STORED_LOGS = 1000;

/**
 * Log a message with optional metadata
 */
export function logMessage(
  level: LogLevel,
  category: LogCategory,
  message: string,
  metadata?: LogMetadata
) {
  if (level < currentLogLevel) return;

  const timestamp = new Date().toISOString();
  const logData = {
    timestamp,
    level: LogLevel[level],
    category,
    message,
    ...metadata
  };

  switch (level) {
    case LogLevel.DEBUG:
      console.debug(`[${category}] ${message}`, metadata || '');
      break;
    case LogLevel.INFO:
      console.info(`[${category}] ${message}`, metadata || '');
      break;
    case LogLevel.WARN:
      console.warn(`[${category}] ${message}`, metadata || '');
      break;
    case LogLevel.ERROR:
      console.error(`[${category}] ${message}`, metadata || '');
      break;
  }

  // Store logs in session storage for report generation
  storeLogForReport(logData);
}

/**
 * Store log in session storage for inclusion in reports
 */
function storeLogForReport(logData: any) {
  try {
    const logsKey = 'deepdefend_detection_logs';
    const existingLogs = JSON.parse(sessionStorage.getItem(logsKey) || '[]');
    
    // Limit log size to prevent storage issues
    if (existingLogs.length > MAX_STORED_LOGS) {
      existingLogs.shift(); // Remove oldest log
    }
    
    existingLogs.push(logData);
    sessionStorage.setItem(logsKey, JSON.stringify(existingLogs));
  } catch (error) {
    // Fail silently - logging should not break the application
    console.warn('Failed to store log in session storage', error);
  }
}

/**
 * Get all stored logs for report generation
 */
export function getStoredLogs(): any[] {
  try {
    const logsKey = 'deepdefend_detection_logs';
    return JSON.parse(sessionStorage.getItem(logsKey) || '[]');
  } catch (error) {
    console.warn('Failed to retrieve logs from session storage', error);
    return [];
  }
}

/**
 * Get logs filtered by category and/or level
 */
export function getFilteredLogs(options: {
  category?: LogCategory | LogCategory[],
  level?: LogLevel | LogLevel[],
  limit?: number
}): any[] {
  try {
    const logs = getStoredLogs();
    let filtered = logs;
    
    // Filter by category
    if (options.category) {
      const categories = Array.isArray(options.category) 
        ? options.category 
        : [options.category];
      filtered = filtered.filter(log => 
        categories.includes(log.category as LogCategory)
      );
    }
    
    // Filter by level
    if (options.level !== undefined) {
      const levels = Array.isArray(options.level) 
        ? options.level 
        : [options.level];
      filtered = filtered.filter(log => 
        levels.includes(LogLevel[log.level as keyof typeof LogLevel])
      );
    }
    
    // Apply limit
    if (options.limit && options.limit > 0) {
      filtered = filtered.slice(-options.limit);
    }
    
    return filtered;
  } catch (error) {
    console.warn('Failed to filter logs', error);
    return [];
  }
}

/**
 * Clear stored logs
 */
export function clearStoredLogs() {
  try {
    sessionStorage.removeItem('deepdefend_detection_logs');
  } catch (error) {
    console.warn('Failed to clear logs from session storage', error);
  }
}

/**
 * Create a log summary for reports
 */
export function generateLogSummary(): { 
  summary: string, 
  errorCount: number,
  warningCount: number,
  categories: Record<string, number>
} {
  try {
    const logs = getStoredLogs();
    const categories: Record<string, number> = {};
    let errorCount = 0;
    let warningCount = 0;
    
    logs.forEach(log => {
      // Count by category
      if (log.category) {
        categories[log.category] = (categories[log.category] || 0) + 1;
      }
      
      // Count errors and warnings
      if (log.level === 'ERROR') {
        errorCount++;
      } else if (log.level === 'WARN') {
        warningCount++;
      }
    });
    
    return {
      summary: `Processed ${logs.length} logs: ${errorCount} errors, ${warningCount} warnings`,
      errorCount,
      warningCount,
      categories
    };
  } catch (error) {
    console.warn('Failed to generate log summary', error);
    return {
      summary: 'Failed to generate log summary',
      errorCount: 0,
      warningCount: 0,
      categories: {}
    };
  }
}

/**
 * Log detection result with detailed metrics
 */
export function logDetectionResult(result: DetectionResult) {
  logMessage(
    LogLevel.INFO,
    LogCategory.GENERAL,
    `Detection complete: ${result.probability.toFixed(4)} (${result.confidence?.toFixed(4) || 'N/A'})`,
    {
      result_id: result.id,
      detection_type: result.detectionType,
      probability: result.probability,
      confidence: result.confidence,
      uncertainty: result.uncertainty,
      model: result.model,
      processing_time: result.processingTime,
      frame_count: result.frameCount
    }
  );

  // Log quality metrics if available
  if (result.metadata?.quality_scores) {
    logMessage(
      LogLevel.DEBUG,
      LogCategory.QUALITY,
      'Quality assessment metrics',
      { quality_scores: result.metadata.quality_scores }
    );
  }

  // Log ensemble details if available
  if (result.metadata?.ensemble_weights) {
    logMessage(
      LogLevel.DEBUG,
      LogCategory.ENSEMBLE,
      'Ensemble model weights',
      { weights: result.metadata.ensemble_weights }
    );
  }

  // Log temporal consistency for videos
  if (result.detectionType === 'video' && result.metadata?.prediction_consistency) {
    logMessage(
      LogLevel.DEBUG,
      LogCategory.TEMPORAL,
      'Temporal consistency analysis',
      { 
        consistency: result.metadata.prediction_consistency,
        frame_count: result.frameCount
      }
    );
  }

  // Log detailed region information
  if (result.regions && result.regions.length > 0) {
    logMessage(
      LogLevel.DEBUG,
      LogCategory.FACE_DETECTION,
      `Detected ${result.regions.length} regions of interest`,
      { regions: result.regions.map(simplifyRegion) }
    );
  }
}

/**
 * Simplify region data for logging (to avoid excessive data)
 */
function simplifyRegion(region: DetectionRegion) {
  return {
    probability: region.probability,
    confidence: region.confidence,
    frame: region.frame,
    dimensions: `${region.x},${region.y},${region.width},${region.height}`
  };
}

/**
 * Log preprocessing stage
 */
export function logPreprocessing(stage: string, metadata: LogMetadata) {
  logMessage(
    LogLevel.DEBUG,
    LogCategory.PREPROCESSING,
    `Preprocessing: ${stage}`,
    metadata
  );
}

/**
 * Log model inference
 */
export function logModelInference(model: string, prediction: number, metadata: LogMetadata) {
  logMessage(
    LogLevel.DEBUG,
    LogCategory.MODEL_INFERENCE,
    `Model ${model} prediction: ${prediction.toFixed(4)}`,
    metadata
  );
}

/**
 * Log ensemble aggregation
 */
export function logEnsembleAggregation(models: string[], predictions: number[], weights: number[], final: number) {
  const modelData = models.map((model, i) => ({
    model,
    prediction: predictions[i],
    weight: weights[i]
  }));

  logMessage(
    LogLevel.DEBUG,
    LogCategory.ENSEMBLE,
    `Ensemble aggregation: ${final.toFixed(4)}`,
    { models: modelData, final_prediction: final }
  );
}

/**
 * Log calibration adjustment
 */
export function logCalibration(original: number, calibrated: number, metadata: LogMetadata) {
  logMessage(
    LogLevel.DEBUG,
    LogCategory.CALIBRATION,
    `Calibration: ${original.toFixed(4)} → ${calibrated.toFixed(4)}`,
    metadata
  );
}

/**
 * Log uncertainty estimation
 */
export function logUncertainty(uncertainty: number, metadata: LogMetadata) {
  logMessage(
    LogLevel.DEBUG,
    LogCategory.UNCERTAINTY,
    `Uncertainty estimate: ${uncertainty.toFixed(4)}`,
    metadata
  );
}

/**
 * Log API request/response
 */
export function logApiInteraction(endpoint: string, request: any, response: any, error?: any) {
  if (error) {
    logMessage(
      LogLevel.ERROR,
      LogCategory.API,
      `API error: ${endpoint}`,
      { request, response, error: error.toString() }
    );
  } else {
    logMessage(
      LogLevel.DEBUG,
      LogCategory.API,
      `API call: ${endpoint}`,
      { request, response }
    );
  }
}

/**
 * Log report generation
 */
export function logReportGeneration(status: string, metadata: LogMetadata) {
  logMessage(
    LogLevel.INFO,
    LogCategory.REPORT,
    `Report generation: ${status}`,
    metadata
  );
}

// Set the current log level programmatically
export function setLogLevel(level: LogLevel) {
  currentLogLevel = level;
  logMessage(
    LogLevel.INFO,
    LogCategory.GENERAL,
    `Log level set to ${LogLevel[level]}`
  );
}
