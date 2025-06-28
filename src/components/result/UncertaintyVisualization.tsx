
import React from 'react';
import { DetectionResult } from '@/types';
import { Progress } from '@/components/ui/progress';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Info, AlertTriangle } from 'lucide-react';

interface UncertaintyVisualizationProps {
  result: DetectionResult;
}

const UncertaintyVisualization: React.FC<UncertaintyVisualizationProps> = ({ result }) => {
  // Extract uncertainty data from result
  const uncertainty = result.uncertainty || result.metadata?.weighted_uncertainty || 0;
  const confidence = result.confidence || 0.5;
  const probability = result.probability;
  
  // Determine if the prediction is borderline (close to decision threshold)
  const isBorderlinePrediction = Math.abs(probability - 0.5) < 0.15;
  
  // Calculate confidence intervals
  const lowerBound = Math.max(0, probability - uncertainty);
  const upperBound = Math.min(1, probability + uncertainty);
  
  // Check if the uncertainty range crosses the decision boundary
  const crossesThreshold = lowerBound < 0.5 && upperBound > 0.5;
  
  // Calculate width for the uncertainty bar
  const uncertaintyBarWidth = `${Math.min(100, uncertainty * 200)}%`;
  
  // Get calibrated probability if available
  const calibratedProbability = result.metadata?.calibrated_probability || probability;
  const calibrationApplied = result.metadata?.calibration_applied === true;
  
  // Get uncertainty metrics if available
  const varianceMetric = result.metadata?.variance || result.metadata?.monte_carlo_variance;
  const monteCarloSamples = result.metadata?.monte_carlo_samples || [];
  
  // Get interpretability data 
  const hasAttentionMap = !!result.metadata?.attention_maps;
  
  // Determine uncertainty level for display
  const getUncertaintyLevel = () => {
    if (uncertainty < 0.05) return { label: "Very Low", color: "text-green-600" };
    if (uncertainty < 0.1) return { label: "Low", color: "text-green-500" };
    if (uncertainty < 0.2) return { label: "Moderate", color: "text-yellow-500" };
    if (uncertainty < 0.3) return { label: "High", color: "text-orange-500" };
    return { label: "Very High", color: "text-red-500" };
  };
  
  const uncertaintyLevel = getUncertaintyLevel();

  return (
    <div className="w-full flex flex-col space-y-4">
      {/* Probability with uncertainty visualization */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">Deepfake Probability:</span>
          <span className="font-medium">
            {(probability * 100).toFixed(1)}% 
            {calibrationApplied && (
              <span className="text-xs text-muted-foreground ml-1">(calibrated)</span>
            )}
          </span>
        </div>
        
        <div className="relative h-8 bg-gray-100 rounded-md">
          {/* Decision boundary line */}
          <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-gray-400 z-10"></div>
          
          {/* Uncertainty range */}
          <div 
            className={`absolute top-0 bottom-0 bg-opacity-30 ${probability > 0.5 ? "bg-red-500" : "bg-green-500"}`}
            style={{
              left: `${lowerBound * 100}%`,
              width: `${(upperBound - lowerBound) * 100}%`
            }}
          />
          
          {/* Main probability indicator */}
          <div 
            className={`absolute top-0 bottom-0 w-1 z-20 ${probability > 0.5 ? "bg-red-500" : "bg-green-500"}`}
            style={{
              left: `calc(${probability * 100}% - 2px)`
            }}
          />
          
          {/* Scale labels */}
          <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-gray-500 px-1">
            <div>Authentic (0%)</div>
            <div>50%</div>
            <div>Deepfake (100%)</div>
          </div>
        </div>
        
        {crossesThreshold && (
          <div className="flex items-center text-yellow-600 text-xs">
            <AlertTriangle className="h-3 w-3 mr-1" />
            <span>Uncertainty range crosses the decision threshold</span>
          </div>
        )}
      </div>
      
      {/* Confidence and uncertainty metrics */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Confidence:</span>
            <span className="font-medium">{(confidence * 100).toFixed(1)}%</span>
          </div>
          <Progress value={confidence * 100} className="h-2" />
        </div>
        
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Uncertainty:</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className={`font-medium flex items-center ${uncertaintyLevel.color}`}>
                    {(uncertainty * 100).toFixed(1)}%
                    <Info className="h-3 w-3 ml-1 text-muted-foreground" />
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-xs max-w-xs">
                    Uncertainty represents the model's confidence interval. Higher values indicate less certainty in the prediction.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <Progress value={uncertainty * 100} className="h-2" />
          <div className="text-xs text-muted-foreground">
            Level: <span className={uncertaintyLevel.color}>{uncertaintyLevel.label}</span>
          </div>
        </div>
      </div>
      
      {/* Monte Carlo dropout samples visualization if available */}
      {monteCarloSamples.length > 0 && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Monte Carlo Samples:</span>
            <span className="font-medium">{monteCarloSamples.length} samples</span>
          </div>
          
          <div className="flex h-10 items-end">
            {monteCarloSamples.map((sample, i) => (
              <div 
                key={i}
                className={`w-1 mx-px ${sample > 0.5 ? 'bg-red-400' : 'bg-green-400'}`}
                style={{ height: `${Math.max(15, sample * 100)}%` }}
              />
            ))}
          </div>
          
          {varianceMetric !== undefined && (
            <div className="text-xs text-muted-foreground">
              Variance: {varianceMetric.toFixed(4)}
            </div>
          )}
        </div>
      )}
      
      {/* Confidence interval */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">Confidence Interval:</span>
          <span className="font-medium">
            {(lowerBound * 100).toFixed(1)}% – {(upperBound * 100).toFixed(1)}%
          </span>
        </div>
      </div>
      
      {/* Calibration information if available */}
      {calibrationApplied && (
        <div className="text-xs text-blue-600">
          <div className="flex items-center">
            <Info className="h-3 w-3 mr-1" />
            Calibration applied to improve prediction reliability
          </div>
          {result.metadata?.calibration_method && (
            <div className="mt-1 text-muted-foreground">
              Method: {result.metadata.calibration_method}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default UncertaintyVisualization;
