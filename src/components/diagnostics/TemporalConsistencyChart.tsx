
import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Info } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine } from "recharts";

interface FramePrediction {
  frame: number;
  probability: number;
  confidence?: number;
  uncertainty?: number;
}

interface TemporalConsistencyChartProps {
  framePredictions: FramePrediction[];
  temporalConsistency?: number;
  smoothedPrediction?: number;
  showUncertainty?: boolean;
}

/**
 * Component to visualize temporal consistency across video frames
 */
export const TemporalConsistencyChart: React.FC<TemporalConsistencyChartProps> = ({
  framePredictions,
  temporalConsistency,
  smoothedPrediction,
  showUncertainty = true
}) => {
  if (!framePredictions || framePredictions.length === 0) {
    return (
      <Card className="w-full">
        <CardContent className="flex items-center justify-center h-40 text-center text-sm text-muted-foreground">
          <div>
            <Info className="w-10 h-10 mx-auto mb-2 opacity-50" />
            <p>No temporal data available for this detection</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Process data for the chart
  const chartData = framePredictions.map(fp => ({
    frame: fp.frame,
    probability: parseFloat((fp.probability * 100).toFixed(1)),
    confidence: fp.confidence ? parseFloat((fp.confidence * 100).toFixed(1)) : undefined,
    uncertainty: fp.uncertainty ? parseFloat((fp.uncertainty * 100).toFixed(1)) : undefined,
    upperBound: fp.uncertainty ? parseFloat(((fp.probability + fp.uncertainty) * 100).toFixed(1)) : undefined,
    lowerBound: fp.uncertainty ? parseFloat(((fp.probability - fp.uncertainty) * 100).toFixed(1)) : undefined,
  }));

  // Sort by frame number
  chartData.sort((a, b) => a.frame - b.frame);

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center">
          Temporal Consistency Analysis
          {temporalConsistency !== undefined && (
            <span className={`ml-2 text-xs px-2 py-0.5 rounded-full ${
              temporalConsistency > 0.7 
                ? "bg-emerald-100 text-emerald-700" 
                : temporalConsistency > 0.4 
                  ? "bg-yellow-100 text-yellow-700" 
                  : "bg-red-100 text-red-700"
            }`}>
              {temporalConsistency > 0.7 
                ? "High consistency" 
                : temporalConsistency > 0.4 
                  ? "Moderate consistency" 
                  : "Low consistency"}
            </span>
          )}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-4 w-4 ml-2 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                <p>This chart shows predictions across video frames and how consistent they are over time. High variability may indicate processing issues or complex manipulation.</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="pt-0">
        <div className="w-full h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 10, right: 10, left: 0, bottom: 15 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="frame" 
                label={{ 
                  value: 'Frame', 
                  position: 'insideBottomRight', 
                  offset: -5 
                }}
                fontSize={11}
              />
              <YAxis 
                domain={[0, 100]} 
                label={{ 
                  value: 'Probability (%)', 
                  angle: -90, 
                  position: 'insideLeft',
                  style: { textAnchor: 'middle' }
                }} 
                fontSize={11}
              />
              <RechartsTooltip 
                formatter={(value: number, name: string) => {
                  if (name === 'probability') return [`${value}%`, 'Deepfake Probability'];
                  if (name === 'confidence') return [`${value}%`, 'Confidence'];
                  if (name === 'uncertainty') return [`±${value}%`, 'Uncertainty'];
                  return [value, name];
                }}
                labelFormatter={(frame) => `Frame ${frame}`}
              />
              
              {/* Decision threshold line */}
              <ReferenceLine y={50} stroke="rgba(0,0,0,0.3)" strokeDasharray="4 4" label={{ 
                value: 'Threshold', 
                position: 'right',
                fontSize: 10,
                fill: 'rgba(0,0,0,0.5)'
              }} />
              
              {/* Smoothed prediction line */}
              {smoothedPrediction !== undefined && (
                <ReferenceLine y={smoothedPrediction * 100} stroke="rgba(0,128,0,0.5)" strokeDasharray="3 3" label={{ 
                  value: 'Smoothed', 
                  position: 'left',
                  fontSize: 10,
                  fill: 'rgba(0,128,0,0.7)'
                }} />
              )}
              
              {/* Uncertainty bounds */}
              {showUncertainty && chartData[0].uncertainty !== undefined && (
                <>
                  <Line 
                    type="monotone" 
                    dataKey="upperBound" 
                    stroke="rgba(79, 70, 229, 0.2)" 
                    fill="rgba(79, 70, 229, 0.1)" 
                    strokeWidth={0}
                    dot={false}
                    activeDot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="lowerBound" 
                    stroke="rgba(79, 70, 229, 0.2)" 
                    fill="rgba(79, 70, 229, 0.1)" 
                    strokeWidth={0}
                    dot={false}
                    activeDot={false}
                  />
                </>
              )}
              
              {/* Main probability line */}
              <Line 
                type="monotone" 
                dataKey="probability" 
                stroke="rgba(79, 70, 229, 0.8)" 
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
              />
              
              {/* Confidence line */}
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
        
        <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            <div className="flex items-center">
              <span className="block w-3 h-3 bg-indigo-500 rounded-full mr-1"></span>
              <span>Probability</span>
            </div>
            {chartData[0].confidence !== undefined && (
              <div className="flex items-center">
                <span className="block w-3 h-3 bg-emerald-500 rounded-full mr-1"></span>
                <span>Confidence</span>
              </div>
            )}
            {showUncertainty && chartData[0].uncertainty !== undefined && (
              <div className="flex items-center">
                <span className="block w-3 h-3 bg-indigo-200 rounded-full mr-1"></span>
                <span>Uncertainty</span>
              </div>
            )}
          </div>
          <div>
            {temporalConsistency !== undefined && (
              <span>Consistency score: {(temporalConsistency * 100).toFixed(1)}%</span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default TemporalConsistencyChart;
