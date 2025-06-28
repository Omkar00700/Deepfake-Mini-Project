
import React, { useState } from 'react';
import { DetectionResult } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Info, ZoomIn, ZoomOut, Layers } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface InterpretabilityVisualizationProps {
  result: DetectionResult;
}

const InterpretabilityVisualization: React.FC<InterpretabilityVisualizationProps> = ({ result }) => {
  const [opacity, setOpacity] = useState(70);
  const [zoom, setZoom] = useState(100);
  
  // Check if we have attention maps/heatmaps to display
  const hasHeatmaps = result.metadata?.attention_maps !== undefined;
  
  // Get original image URL
  const originalImageUrl = result.imageUrl;
  
  // Get heatmap URL or base64 from the metadata
  const heatmapUrl = result.metadata?.attention_maps?.url || '';
  
  // If no heatmap data is available, don't render the component
  if (!hasHeatmaps || !heatmapUrl) {
    return null;
  }
  
  // Get method used for heatmap generation
  const heatmapMethod = result.metadata?.attention_maps?.method || 'Grad-CAM';
  
  // Get frame number for video
  const frameNumber = result.metadata?.attention_maps?.frame;
  
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-1">
          <Info className="h-4 w-4" />
          Model Attention Visualization
          {frameNumber !== undefined && (
            <span className="ml-2 text-xs bg-secondary px-2 py-0.5 rounded-full">
              Frame {frameNumber}
            </span>
          )}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-3 w-3 ml-1 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="text-xs max-w-xs">
                  This visualization shows the regions that most influenced the model's decision.
                  Brighter areas had more impact on the detection result.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </CardTitle>
      </CardHeader>
      <CardContent className="relative">
        <Tabs defaultValue="comparison" className="w-full">
          <TabsList className="w-full grid grid-cols-3">
            <TabsTrigger value="original">Original</TabsTrigger>
            <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
            <TabsTrigger value="comparison">Overlay</TabsTrigger>
          </TabsList>
          
          <TabsContent value="original" className="relative mt-2">
            <div className="relative w-full h-64 bg-black/5 rounded-md overflow-hidden flex items-center justify-center">
              {originalImageUrl ? (
                <img 
                  src={originalImageUrl} 
                  alt="Original detection image" 
                  className="object-contain max-h-full max-w-full"
                  style={{ transform: `scale(${zoom/100})` }}
                />
              ) : (
                <div className="text-sm text-muted-foreground">Original image not available</div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="heatmap" className="relative mt-2">
            <div className="relative w-full h-64 bg-black/5 rounded-md overflow-hidden flex items-center justify-center">
              {heatmapUrl ? (
                <img 
                  src={heatmapUrl} 
                  alt="Model attention heatmap" 
                  className="object-contain max-h-full max-w-full"
                  style={{ transform: `scale(${zoom/100})` }}
                />
              ) : (
                <div className="text-sm text-muted-foreground">Heatmap not available</div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="comparison" className="relative mt-2">
            <div className="relative w-full h-64 bg-black/5 rounded-md overflow-hidden flex items-center justify-center">
              {originalImageUrl && heatmapUrl ? (
                <>
                  <img 
                    src={originalImageUrl} 
                    alt="Original detection image" 
                    className="absolute object-contain max-h-full max-w-full"
                    style={{ transform: `scale(${zoom/100})` }}
                  />
                  <img 
                    src={heatmapUrl} 
                    alt="Model attention heatmap" 
                    className="absolute object-contain max-h-full max-w-full"
                    style={{ 
                      transform: `scale(${zoom/100})`,
                      opacity: opacity / 100 
                    }}
                  />
                </>
              ) : (
                <div className="text-sm text-muted-foreground">Comparison not available</div>
              )}
            </div>
            
            <div className="mt-2 px-1">
              <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                <span>Overlay Opacity</span>
                <span>{opacity}%</span>
              </div>
              <Slider 
                value={[opacity]} 
                min={0} 
                max={100} 
                step={5}
                onValueChange={(values) => setOpacity(values[0])} 
              />
            </div>
          </TabsContent>
        </Tabs>
        
        <div className="absolute top-1 right-1 flex space-x-1">
          <Button 
            variant="outline" 
            size="icon" 
            className="h-6 w-6" 
            onClick={() => setZoom(Math.max(zoom - 10, 50))}
          >
            <ZoomOut className="h-3 w-3" />
          </Button>
          <Button 
            variant="outline" 
            size="icon" 
            className="h-6 w-6" 
            onClick={() => setZoom(Math.min(zoom + 10, 200))}
          >
            <ZoomIn className="h-3 w-3" />
          </Button>
          <Button 
            variant="outline" 
            size="icon" 
            className="h-6 w-6" 
            onClick={() => setZoom(100)}
          >
            <Layers className="h-3 w-3" />
          </Button>
        </div>
        
        <div className="mt-4 text-xs text-muted-foreground">
          <div className="flex justify-between">
            <span>Visualization method: {heatmapMethod}</span>
            <span>Probability: {(result.probability * 100).toFixed(1)}%</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default InterpretabilityVisualization;
