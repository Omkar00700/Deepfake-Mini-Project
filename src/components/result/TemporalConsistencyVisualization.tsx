
import React from 'react';
import { DetectionResult, DetectionRegion } from '@/types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Info } from 'lucide-react';
import { Tooltip as UITooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface TemporalConsistencyVisualizationProps {
  result: DetectionResult;
}

const TemporalConsistencyVisualization: React.FC<TemporalConsistencyVisualizationProps> = ({ result }) => {
  // Only for video detection results
  if (result.detectionType !== 'video' || !result.regions || result.regions.length === 0) {
    return null;
  }
  
  // Extract frame predictions from regions
  const framePredictions = result.regions
    .filter(region => region.frame !== undefined)
    .map(region => ({
      frame: region.frame || 0,
      probability: region.probability,
      confidence: region.confidence,
      uncertainty: region.uncertainty
    }))
    .sort((a, b) => a.frame - b.frame);
  
  // If we don't have frame-level data, return null
  if (framePredictions.length === 0) {
    return null;
  }
  
  // Get temporal consistency score
  const temporalConsistency = result.metadata?.prediction_consistency || 0;
  
  // Format data for the chart
  const chartData = framePredictions.map(fp => ({
    frame: fp.frame,
    probability: parseFloat((fp.probability * 100).toFixed(1)),
    confidence: fp.confidence ? parseFloat((fp.confidence * 100).toFixed(1)) : undefined,
    uncertainty: fp.uncertainty ? parseFloat((fp.uncertainty * 100).toFixed(1)) : undefined
  }));
  
  // Determine consistency level
  const getConsistencyLevel = () => {
    if (temporalConsistency > 0.7) return { label: "High", color: "text-green-500" };
    if (temporalConsistency > 0.4) return { label: "Moderate", color: "text-yellow-500" };
    return { label: "Low", color: "text-orange-500" };
  };
  
  const consistencyLevel = getConsistencyLevel();
  
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-1">
          <Info className="h-4 w-4" />
          Temporal Consistency Analysis
          <TooltipProvider>
            <UITooltip>
              <TooltipTrigger asChild>
                <Info className="h-3 w-3 ml-1 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="text-xs max-w-xs">
                  This visualization shows how consistent predictions are across video frames.
                  High variability may indicate processing issues or inconsistent manipulation.
                </p>
              </TooltipContent>
            </UITooltip>
          </TooltipProvider>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="frame" 
                label={{ value: 'Frame', position: 'insideBottomRight', offset: -5 }} 
                fontSize={11}
              />
              <YAxis 
                domain={[0, 100]} 
                label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }} 
                fontSize={11}
              />
              <Tooltip 
                formatter={(value: number, name: string) => {
                  if (name === 'probability') return [`${value}%`, 'Deepfake Probability'];
                  if (name === 'confidence') return [`${value}%`, 'Confidence'];
                  if (name === 'uncertainty') return [`±${value}%`, 'Uncertainty'];
                  return [value, name];
                }}
                labelFormatter={(frame) => `Frame ${frame}`}
              />
              <ReferenceLine y={50} stroke="rgba(0,0,0,0.3)" strokeDasharray="4 4" />
              <Line 
                type="monotone" 
                dataKey="probability" 
                stroke="rgba(79, 70, 229, 0.8)" 
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
              />
              {chartData[0].confidence !== undefined && (
                <Line 
                  type="monotone" 
                  dataKey="confidence" 
                  stroke="rgba(16, 185, 129, 0.8)" 
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  dot={{ r: 2 }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        <div className="flex justify-between mt-4 text-sm">
          <div className="flex items-center">
            <span className="block w-3 h-3 bg-indigo-500 rounded-full mr-1"></span>
            <span>Probability</span>
            {chartData[0].confidence !== undefined && (
              <span className="ml-4 flex items-center">
                <span className="block w-3 h-3 bg-emerald-500 rounded-full mr-1"></span>
                <span>Confidence</span>
              </span>
            )}
          </div>
          <div className="text-sm">
            <span className="text-muted-foreground">Temporal Consistency: </span>
            <span className={consistencyLevel.color}>{(temporalConsistency * 100).toFixed()}% ({consistencyLevel.label})</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default TemporalConsistencyVisualization;
