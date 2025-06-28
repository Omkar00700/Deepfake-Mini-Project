
import React, { useState } from "react";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Info, ZoomIn, ZoomOut, Layers } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface HeatmapProps {
  // Base64 encoded heatmap image
  heatmapSrc?: string;
  // Original image for comparison
  originalSrc?: string;
  // Frame number (for video)
  frameNumber?: number;
  // Confidence score associated with this detection
  confidence?: number;
  // Prediction probability
  probability?: number;
  // Whether this is from a calibrated model
  isCalibratedModel?: boolean;
  // Region metadata
  regionMetadata?: Record<string, any>;
}

/**
 * Component to visualize model attention heatmaps and interpretability data
 */
export const HeatmapVisualizer: React.FC<HeatmapProps> = ({
  heatmapSrc,
  originalSrc,
  frameNumber,
  confidence = 0,
  probability = 0,
  isCalibratedModel = false,
  regionMetadata = {}
}) => {
  const [opacity, setOpacity] = useState(70);
  const [zoom, setZoom] = useState(100);
  
  // If no heatmap is available
  if (!heatmapSrc && !originalSrc) {
    return (
      <Card className="w-full">
        <CardContent className="flex items-center justify-center h-40 text-center text-sm text-muted-foreground">
          <div>
            <Info className="w-10 h-10 mx-auto mb-2 opacity-50" />
            <p>No visualization data available for this detection</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center">
          Model Attention Visualization
          {frameNumber !== undefined && (
            <span className="ml-2 text-xs bg-secondary px-2 py-0.5 rounded-full">
              Frame {frameNumber}
            </span>
          )}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-4 w-4 ml-2 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                <p>This heatmap shows which areas of the image influenced the model's decision most strongly.</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="relative pt-0">
        <Tabs defaultValue="comparison" className="w-full">
          <TabsList className="w-full grid grid-cols-3">
            <TabsTrigger value="original">Original</TabsTrigger>
            <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
            <TabsTrigger value="comparison">Overlay</TabsTrigger>
          </TabsList>
          
          <TabsContent value="original" className="relative mt-2">
            <div className="relative w-full h-64 bg-black/5 rounded-md overflow-hidden flex items-center justify-center">
              {originalSrc ? (
                <img 
                  src={originalSrc} 
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
              {heatmapSrc ? (
                <img 
                  src={heatmapSrc} 
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
              {originalSrc && heatmapSrc ? (
                <>
                  <img 
                    src={originalSrc} 
                    alt="Original detection image" 
                    className="absolute object-contain max-h-full max-w-full"
                    style={{ transform: `scale(${zoom/100})` }}
                  />
                  <img 
                    src={heatmapSrc} 
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
      </CardContent>
      
      <CardFooter className="flex flex-col items-start pt-0">
        <div className="w-full grid grid-cols-2 gap-2 text-xs">
          <div className="flex flex-col space-y-1">
            <span className="text-muted-foreground">Prediction:</span>
            <span className={probability > 0.5 ? "text-destructive font-medium" : "text-emerald-600 font-medium"}>
              {probability > 0.5 ? "Manipulated" : "Authentic"} ({(probability * 100).toFixed(1)}%)
            </span>
          </div>
          <div className="flex flex-col space-y-1">
            <span className="text-muted-foreground">Confidence:</span>
            <span className="font-medium">
              {(confidence * 100).toFixed(1)}%
              {isCalibratedModel && <span className="text-xs text-muted-foreground ml-1">(calibrated)</span>}
            </span>
          </div>
        </div>
      </CardFooter>
    </Card>
  );
};

export default HeatmapVisualizer;
