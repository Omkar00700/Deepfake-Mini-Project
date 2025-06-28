
// Type definitions for DeepDefend detection application

// Detection Result Types
export interface DetectionResult {
  id?: number | string;
  imageName: string;
  probability: number;
  timestamp: string;
  imageUrl?: string;
  detectionType: 'image' | 'video';
  frameCount?: number;
  processingTime?: number; // Time taken to process in milliseconds
  confidence?: number; // Additional confidence metric (0-1)
  uncertainty?: number; // Uncertainty measure (0-1, lower is better)
  regions?: DetectionRegion[]; // Regions in image/frames with high probability of manipulation
  error?: string; // Any error during detection
  model?: string; // Model used for detection
  metadata?: {
    region_analysis?: string;
    comments?: string;
    region?: string;
    prediction_consistency?: number; // Consistency of predictions across regions/frames
    weighted_uncertainty?: number; // Weighted uncertainty from Monte Carlo dropout
    calibrated_probability?: number; // Probability after calibration
    calibration_applied?: boolean; // Whether calibration was applied
    calibration_method?: string; // Method used for calibration (e.g., 'platt', 'temperature')
    monte_carlo_samples?: number[]; // Raw samples from Monte Carlo dropout
    variance?: number; // Variance of MC dropout samples
    monte_carlo_variance?: number; // Alias for variance
    attention_maps?: {
      url?: string; // URL to the heatmap image
      method?: string; // Method used ('grad-cam', 'integrated-gradients', etc.)
      frame?: number; // Frame number for video
      region?: { x: number, y: number, width: number, height: number }; // Region analyzed
    };
    quality_scores?: {  // Image/video quality metrics
      overall: number;
      blur: number;
      contrast: number;
      noise: number;
      [key: string]: number;
    };
    temporal_analysis?: {
      consistency: number; // Temporal consistency score
      method: string; // Method used for temporal smoothing
      frame_predictions: { frame: number, probability: number, confidence?: number }[];
    };
    ensemble_info?: {
      models: string[]; // Model names used in ensemble
      weights: Record<string, number>; // Weights for each model
      aggregation_method: string; // How predictions were combined
    };
    [key: string]: any;
  };
  report_ready?: boolean; // Flag to indicate if the report is ready
  report_status?: {
    ready: boolean;
    missing_data?: string[];
    last_checked?: number;
    progress?: number;
  };
}

export interface DetectionRegion {
  x: number;
  y: number;
  width: number;
  height: number;
  probability: number;
  confidence?: number;
  uncertainty?: number;
  quality_score?: number;
  frame?: number; // Frame number for video detections
  metadata?: {
    processing_metrics?: {
      frames?: {
        processed: number;
        total?: number;
      };
      model?: string;
      adaptive_weights?: Record<string, number>;
      [key: string]: any;
    };
    attention_map?: any;
    [key: string]: any;
  };
}

// API Response Types
export interface UploadResponse {
  success: boolean;
  message: string;
  result?: DetectionResult;
  error?: string;
  statusCode?: number;
}

export interface ApiReportResponse {
  success: boolean;
  report_url?: string;
  message?: string;
  error?: string;
  status?: string;
  progress?: number;
  missing_data?: string[];
  data?: any;
  contentType?: string;
}

export interface ApiErrorResponse {
  success: false;
  error: string;
  message?: string;
  status?: string;
}

// Video-specific types
export interface VideoDetectionDetails {
  frameCount: number;
  processedFrames: number;
  averageProbability: number;
  detectionType: 'video';
  processingTime?: number;
  temporalConsistency?: number; // Consistency across frames
  qualityMetrics?: {
    averageQuality: number;
    blurScore: number;
    noiseLevel: number;
    [key: string]: number;
  };
}

// API and Status Types
export interface DatabaseInfo {
  connected: boolean;
  type: string;
  version?: string;
  error?: string;
}

export interface ApiStatus {
  status: 'online' | 'offline' | 'degraded';
  message: string;
  version?: string;
  database?: DatabaseInfo;
  config?: {
    using_supabase?: boolean;
    model_info?: {
      current_model: string;
      available_models: string[];
      adaptive_ensemble: boolean;
      uncertainty_enabled: boolean;
    };
  };
}

// Model Types
export interface ModelInfo {
  name: string;
  performance?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  };
  adaptive_weight?: number;
  is_current: boolean;
  inference_time?: number;
}

export interface ModelPerformance {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  auc?: number;
  uncertainty_calibration?: {
    ece: number;  // Expected Calibration Error
    correlation: number;
    predictive_auc: number;
  };
  timestamp: number;
}

// Report Types
export interface ReportStatus {
  ready: boolean;
  progress?: number;
  missingData?: string[];
  errorMessage?: string;
  lastChecked: number;
  localFallbackReady?: boolean;
  retryCount?: number;
  estimatedTimeRemaining?: number;
}

export interface ReportFormat {
  id: string;
  label: string;
  value: string;
  icon?: React.ReactNode;
}

// ReportFormatSelectorProps interface to fix the missing props error
export interface ReportFormatSelectorProps {
  formats: ReportFormat[];
  onSelect: (format: 'pdf' | 'json' | 'csv') => Promise<void> | void;
  disabled: boolean;
  isGenerating: boolean;
  selectedFormat?: string;
  onGenerate?: () => void;
}

// Utility Types
export type ProgressCallback = (progress: number) => void;

// Advanced Analytics Types
export interface ModelEnsembleDetails {
  models: {
    name: string;
    weight: number;
    raw_prediction: number;
    calibrated_prediction?: number;
    confidence: number;
  }[];
  aggregation_method: string;
  final_prediction: number;
}

export interface UncertaintyDetails {
  model_uncertainty: number;
  data_uncertainty?: number;
  total_uncertainty: number;
  confidence_interval?: [number, number];
  calibration_info?: {
    method: string;
    parameters?: Record<string, number>;
    raw_output: number;
    calibrated_output: number;
  };
}

export interface TemporalAnalysisDetails {
  temporal_consistency: number;
  smoothing_method: string;
  frame_count: number;
  analyzed_frames: number;
  frame_predictions: {
    frame: number;
    timestamp?: number;
    prediction: number;
    confidence?: number;
    uncertainty?: number;
  }[];
}

export interface InterpretabilityDetails {
  method: string;
  heatmap_url?: string;
  analyzed_regions?: {
    coordinates: { x: number, y: number, width: number, height: number };
    importance: number;
    feature_type?: string;
  }[];
  global_feature_importance?: Record<string, number>;
}
