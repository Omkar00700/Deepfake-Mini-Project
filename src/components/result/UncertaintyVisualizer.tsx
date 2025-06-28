
import React from 'react';
import { DetectionResult } from '@/types';
import UncertaintyVisualization from './UncertaintyVisualization';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Info } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface UncertaintyVisualizerProps {
  result: DetectionResult;
}

const UncertaintyVisualizer: React.FC<UncertaintyVisualizerProps> = ({ result }) => {
  // Check if there's uncertainty data to display
  const hasUncertaintyData = result.uncertainty !== undefined || 
    (result.metadata?.weighted_uncertainty !== undefined);
  
  if (!hasUncertaintyData) {
    return null;
  }
  
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-1">
          <Info className="h-4 w-4" />
          Uncertainty Analysis
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-3 w-3 ml-1 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="text-xs max-w-xs">
                  This visualization shows the uncertainty in the model's prediction. 
                  Higher uncertainty indicates less confidence in the exact probability value.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <UncertaintyVisualization result={result} />
      </CardContent>
    </Card>
  );
};

export default UncertaintyVisualizer;
