
import { AlertTriangle, Shield, Info, Clock, CheckCircle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { DetectionResult } from "@/types";

interface DetectionSummaryProps {
  result: DetectionResult;
}

export const DetectionSummary = ({ result }: DetectionSummaryProps) => {
  const { 
    detectionType = 'image', 
    frameCount, 
    processingTime, 
    probability, 
    confidence = 0.5,
    regions = [],
    model,
    metadata = {}
  } = result;
  
  const isVideo = detectionType === 'video';
  const probabilityPercentage = Math.round(probability * 100);
  const confidencePercentage = Math.round(confidence * 100);
  
  const getResultMessage = () => {
    if (probabilityPercentage > 70) {
      return `High probability of being a deepfake ${isVideo ? 'video' : 'image'}`;
    }
    if (probabilityPercentage > 40) {
      return `Moderate probability of being a deepfake ${isVideo ? 'video' : 'image'}`;
    }
    return `Low probability of being a deepfake ${isVideo ? 'video' : 'image'}`;
  };
  
  const getPossibleReasons = () => {
    const reasons = [];
    
    if (probabilityPercentage > 70) {
      reasons.push('Facial inconsistencies detected');
      reasons.push('Unnatural lighting or shadows');
      if (isVideo) reasons.push('Temporal inconsistencies between frames');
    } else if (probabilityPercentage > 40) {
      reasons.push('Some visual artifacts detected');
      reasons.push('Slight abnormalities in facial features');
    } else {
      reasons.push('No significant manipulation patterns detected');
      reasons.push('Natural facial features identified');
    }
    
    // Add region-specific insights if available
    if (metadata.region_analysis) {
      reasons.push(metadata.region_analysis);
    }
    
    return reasons;
  };

  const formatProcessingTime = (timeMs?: number) => {
    if (!timeMs) return null;
    return timeMs < 1000 
      ? `${timeMs.toFixed(0)}ms` 
      : `${(timeMs / 1000).toFixed(2)}s`;
  };
  
  // Display weighted uncertainty if available
  const weightedUncertainty = metadata.weighted_uncertainty;
  const hasWeightedUncertainty = typeof weightedUncertainty === 'number';

  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center justify-between mb-1">
          <p className="text-sm font-medium text-muted-foreground">Deepfake Probability</p>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <Badge variant={confidencePercentage > 70 ? "default" : "secondary"}>
                  {confidencePercentage}% confidence
                </Badge>
              </TooltipTrigger>
              <TooltipContent>
                <p className="text-xs max-w-xs">
                  Confidence indicates how certain our AI is about this prediction. 
                  Higher values mean more reliable results.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <div className="flex items-center">
          <div className="w-full bg-secondary rounded-full h-4 mr-4">
            <div
              className={`h-4 rounded-full ${
                probabilityPercentage > 70
                  ? "bg-destructive"
                  : probabilityPercentage > 40
                  ? "bg-amber-500"
                  : "bg-emerald-600"
              }`}
              style={{ width: `${probabilityPercentage}%` }}
            ></div>
          </div>
          <span
            className={`font-bold text-lg ${
              probabilityPercentage > 70
                ? "text-destructive"
                : probabilityPercentage > 40
                ? "text-amber-500"
                : "text-emerald-600"
            }`}
          >
            {probabilityPercentage}%
          </span>
        </div>
      </div>

      <Alert variant={probabilityPercentage > 50 ? "destructive" : "default"} className="bg-secondary/30">
        <AlertDescription>{getResultMessage()}</AlertDescription>
      </Alert>

      {isVideo && frameCount && (
        <div className="bg-secondary/30 rounded-lg p-3 text-sm">
          <p className="font-medium mb-1">Video Analysis Details</p>
          <p className="text-muted-foreground">
            Analyzed {frameCount} frames from the uploaded video to generate this result.
          </p>
        </div>
      )}
      
      <div className="space-y-2">
        <p className="text-sm font-medium">Potential indicators:</p>
        <ul className="text-sm text-muted-foreground space-y-1">
          {getPossibleReasons().map((reason, index) => (
            <li key={index} className="flex items-start">
              <span className="mr-2 mt-0.5">
                {probabilityPercentage > 50 ? 
                  <AlertTriangle className="h-4 w-4 text-amber-500" /> : 
                  <Shield className="h-4 w-4 text-emerald-600" />
                }
              </span>
              {reason}
            </li>
          ))}
        </ul>
      </div>
      
      {hasWeightedUncertainty && (
        <div className="bg-secondary/30 rounded-lg p-3 text-sm">
          <div className="flex items-center">
            <p className="font-medium">Weighted Uncertainty: </p>
            <span className="ml-1 font-mono">
              {(weightedUncertainty * 100).toFixed(1)}%
            </span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="h-3.5 w-3.5 ml-1 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p className="text-xs max-w-xs">
                    Weighted uncertainty from Monte Carlo dropout sampling.
                    Lower values indicate more consistent predictions.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
      )}
      
      {regions && regions.length > 0 && (
        <div className="bg-secondary/30 rounded-lg p-3 text-sm">
          <p className="font-medium mb-1">Detection regions</p>
          <p className="text-muted-foreground">
            {regions.length} {regions.length === 1 ? 'face' : 'faces'} analyzed in this content
          </p>
        </div>
      )}
      
      <div className="flex flex-col space-y-2 text-sm text-muted-foreground mt-2">
        {model && (
          <div className="flex items-center">
            <CheckCircle className="h-4 w-4 mr-2" />
            <span>Model: <span className="font-medium">{model}</span></span>
          </div>
        )}
        
        {processingTime && (
          <div className="flex items-center">
            <Clock className="h-4 w-4 mr-2" />
            <span>Processing time: {formatProcessingTime(processingTime)}</span>
          </div>
        )}
        
        <div>
          <p className="mt-2 text-xs flex items-center">
            <Info className="h-3 w-3 mr-1" />
            Analyzed on {new Date(result.timestamp).toLocaleString()}
          </p>
        </div>
      </div>
    </div>
  );
};

export default DetectionSummary;
