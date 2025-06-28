
import React from "react";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Info, AlertTriangle } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Progress } from "@/components/ui/progress";

interface UncertaintyVisualizerProps {
  // Main prediction probability (0-1)
  probability: number;
  // Uncertainty value (0-1)
  uncertainty?: number;
  // Monte Carlo samples if available
  samples?: number[];
  // Whether calibration was applied
  calibrated?: boolean;
  // Whether this triggered reprocessing
  triggeredReprocessing?: boolean;
  // Variance in predictions
  variance?: number;
}

/**
 * Component to visualize uncertainty in model predictions
 */
export const UncertaintyVisualizer: React.FC<UncertaintyVisualizerProps> = ({
  probability,
  uncertainty,
  samples = [],
  calibrated = false,
  triggeredReprocessing = false,
  variance
}) => {
  // If no uncertainty data is available
  if (uncertainty === undefined && samples.length === 0 && variance === undefined) {
    return (
      <Card className="w-full">
        <CardContent className="flex items-center justify-center h-40 text-center text-sm text-muted-foreground">
          <div>
            <Info className="w-10 h-10 mx-auto mb-2 opacity-50" />
            <p>No uncertainty data available for this detection</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Calculate lower and upper bounds
  const lowerBound = Math.max(0, probability - (uncertainty || 0));
  const upperBound = Math.min(1, probability + (uncertainty || 0));
  
  // Calculate confidence interval width as percentage
  const intervalWidth = (upperBound - lowerBound) * 100;
  
  // Determine if prediction crosses the decision boundary (0.5)
  const crossesBoundary = lowerBound < 0.5 && upperBound > 0.5;
  
  // Get uncertainty level description
  const getUncertaintyLevel = () => {
    if (!uncertainty) return "Unknown";
    if (uncertainty < 0.05) return "Very Low";
    if (uncertainty < 0.1) return "Low";
    if (uncertainty < 0.2) return "Moderate";
    if (uncertainty < 0.3) return "High";
    return "Very High";
  };

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center">
          Prediction Uncertainty Analysis
          {crossesBoundary && (
            <span className="ml-2 text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded-full flex items-center">
              <AlertTriangle className="h-3 w-3 mr-1" />
              Crosses threshold
            </span>
          )}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-4 w-4 ml-2 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                <p>This visualization shows the uncertainty in the model's prediction. 
                  A wider range indicates less confidence in the exact probability value. 
                  When the range crosses 50%, the prediction could be either real or fake.</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="pt-0">
        {uncertainty !== undefined && (
          <div className="mb-4">
            <div className="flex justify-between text-xs mb-1">
              <span>Confidence interval:</span>
              <span className={`font-medium ${
                crossesBoundary ? "text-yellow-600" : "text-gray-700"
              }`}>
                {(lowerBound * 100).toFixed(1)}% – {(upperBound * 100).toFixed(1)}%
              </span>
            </div>
            
            <div className="relative h-10 bg-gray-100 rounded-md">
              {/* Decision boundary line */}
              <div className="absolute top-0 bottom-0 left-1/2 w-px bg-gray-400 z-10"></div>
              
              {/* Probability range */}
              <div 
                className={`absolute top-0 bottom-0 bg-opacity-30 ${
                  probability > 0.5 ? "bg-red-500" : "bg-green-500"
                }`}
                style={{
                  left: `${lowerBound * 100}%`,
                  width: `${intervalWidth}%`
                }}
              />
              
              {/* Central prediction marker */}
              <div 
                className={`absolute top-0 bottom-0 w-1 z-20 ${
                  probability > 0.5 ? "bg-red-500" : "bg-green-500"
                }`}
                style={{
                  left: `calc(${probability * 100}% - 2px)`
                }}
              />
              
              {/* Scale labels */}
              <div className="absolute bottom-0 left-0 right-0 flex justify-between text-[10px] text-gray-500 px-1">
                <div>0%</div>
                <div>50%</div>
                <div>100%</div>
              </div>
            </div>
          </div>
        )}
        
        {samples.length > 0 && (
          <div className="mb-4">
            <div className="flex justify-between text-xs mb-1">
              <span>Monte Carlo samples:</span>
              <span className="text-gray-700 font-medium">{samples.length} samples</span>
            </div>
            
            <div className="flex items-end h-10 w-full">
              {samples.map((sample, i) => (
                <div
                  key={i}
                  className={`w-1 mx-px ${sample > 0.5 ? "bg-red-400" : "bg-green-400"}`}
                  style={{ height: `${Math.max(15, sample * 100)}%` }}
                />
              ))}
            </div>
          </div>
        )}
        
        {variance !== undefined && (
          <div className="mb-4">
            <div className="flex justify-between text-xs mb-1">
              <span>Prediction variance:</span>
              <span className="text-gray-700 font-medium">
                {variance.toFixed(4)}
              </span>
            </div>
            
            <Progress 
              value={Math.min(variance * 1000, 100)} 
              className="h-2" 
            />
          </div>
        )}
      </CardContent>
      
      <CardFooter className="flex justify-between pt-0">
        <div className="text-xs">
          <span className="text-muted-foreground">Uncertainty level: </span>
          <span className={`font-medium ${
            uncertainty && uncertainty > 0.2 ? "text-yellow-600" : "text-gray-700"
          }`}>
            {getUncertaintyLevel()}
          </span>
        </div>
        
        <div className="flex flex-col items-end text-xs space-y-1">
          {calibrated && (
            <span className="text-blue-600">Calibrated prediction</span>
          )}
          {triggeredReprocessing && (
            <span className="text-purple-600">Triggered reprocessing</span>
          )}
        </div>
      </CardFooter>
    </Card>
  );
};

export default UncertaintyVisualizer;
