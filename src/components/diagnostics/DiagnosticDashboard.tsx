
import React, { useState, useEffect } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { DetectionResult } from "@/types";
import { getDiagnosticData, generateLocalDiagnosticData } from "@/services/diagnostic-service";
import HeatmapVisualizer from "./HeatmapVisualizer";
import TemporalConsistencyChart from "./TemporalConsistencyChart";
import UncertaintyVisualizer from "./UncertaintyVisualizer";
import { Info, FileDown, Layers, Clock, Cpu, BarChart, AlertTriangle, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { logReportGeneration } from "@/utils/debug-logger";

interface DiagnosticDashboardProps {
  result: DetectionResult;
  onDiagnosticDataLoaded?: (data: any) => void;
}

/**
 * Comprehensive dashboard for detection diagnostic information
 * Displays model interpretability, uncertainty metrics, and temporal analysis
 */
export const DiagnosticDashboard: React.FC<DiagnosticDashboardProps> = ({ 
  result, 
  onDiagnosticDataLoaded 
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [diagnosticData, setDiagnosticData] = useState<any>(null);
  const [currentTab, setCurrentTab] = useState("overview");
  const [error, setError] = useState<string | null>(null);
  
  // Load diagnostic data for this detection result
  useEffect(() => {
    if (!result?.id) return;
    
    const loadDiagnosticData = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        // Try to get diagnostic data from API
        const data = await getDiagnosticData(result.id);
        setDiagnosticData(data);
        
        if (onDiagnosticDataLoaded) {
          onDiagnosticDataLoaded(data);
        }
        
        logReportGeneration("diagnostic data loaded", { result_id: result.id });
      } catch (err) {
        console.error("Failed to load diagnostic data:", err);
        
        // Generate fallback local diagnostic data
        const fallbackData = generateLocalDiagnosticData(result);
        setDiagnosticData(fallbackData);
        
        if (onDiagnosticDataLoaded) {
          onDiagnosticDataLoaded(fallbackData);
        }
        
        setError("Couldn't load detailed diagnostic data from server. Showing limited local data.");
        logReportGeneration("using local diagnostic fallback", { result_id: result.id, error: String(err) });
      } finally {
        setIsLoading(false);
      }
    };
    
    loadDiagnosticData();
  }, [result?.id, onDiagnosticDataLoaded]);
  
  // Extract frame predictions if available
  const framePredictions = diagnosticData?.temporal_analysis?.frame_predictions || [];
  
  // Extract uncertainty data
  const uncertaintyData = diagnosticData?.uncertainty_analysis || {
    monte_carlo_samples: [],
    variance: 0,
    uncertainty_score: 0
  };
  
  // Extract model outputs
  const modelOutputs = diagnosticData?.model_outputs || [];
  
  // Determine if we have visualization data
  const hasHeatmaps = !!diagnosticData?.visualization?.heatmaps?.length;
  const hasTemporalData = framePredictions.length > 0;
  const hasUncertaintyData = !!uncertaintyData;
  
  if (isLoading) {
    return (
      <Card className="w-full">
        <CardContent className="flex flex-col items-center justify-center py-10">
          <Loader2 className="h-10 w-10 animate-spin text-muted-foreground mb-4" />
          <p className="text-muted-foreground">Loading diagnostic data...</p>
        </CardContent>
      </Card>
    );
  }
  
  if (error) {
    toast.error(error, { duration: 5000 });
  }
  
  return (
    <div className="w-full space-y-6">
      {error && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3 flex items-start">
          <AlertTriangle className="h-5 w-5 text-yellow-500 mt-0.5 mr-2 flex-shrink-0" />
          <div className="text-sm text-yellow-700">{error}</div>
        </div>
      )}
      
      <Tabs defaultValue={currentTab} onValueChange={setCurrentTab}>
        <TabsList className="grid grid-cols-4 mb-4">
          <TabsTrigger value="overview">
            <Info className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="interpretability" disabled={!hasHeatmaps}>
            <Layers className="h-4 w-4 mr-2" />
            Interpretability
          </TabsTrigger>
          <TabsTrigger value="temporal" disabled={!hasTemporalData}>
            <Clock className="h-4 w-4 mr-2" />
            Temporal Analysis
          </TabsTrigger>
          <TabsTrigger value="models">
            <Cpu className="h-4 w-4 mr-2" />
            Model Details
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Detection Summary</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <dl className="grid grid-cols-2 gap-2 text-sm">
                  <dt className="text-muted-foreground">Detection Type:</dt>
                  <dd className="font-medium">{result.detectionType}</dd>
                  
                  <dt className="text-muted-foreground">Probability:</dt>
                  <dd className={`font-medium ${result.probability > 0.5 ? "text-destructive" : "text-emerald-600"}`}>
                    {(result.probability * 100).toFixed(1)}%
                  </dd>
                  
                  <dt className="text-muted-foreground">Confidence:</dt>
                  <dd className="font-medium">{(result.confidence || 0) * 100}%</dd>
                  
                  <dt className="text-muted-foreground">Processing Time:</dt>
                  <dd className="font-medium">{result.processingTime} ms</dd>
                  
                  {result.detectionType === 'video' && (
                    <>
                      <dt className="text-muted-foreground">Frames Analyzed:</dt>
                      <dd className="font-medium">{result.frameCount || 'Unknown'}</dd>
                    </>
                  )}
                  
                  <dt className="text-muted-foreground">Model:</dt>
                  <dd className="font-medium">{result.model || 'Default'}</dd>
                </dl>
              </CardContent>
            </Card>
            
            <UncertaintyVisualizer 
              probability={result.probability} 
              uncertainty={uncertaintyData.uncertainty_score}
              variance={uncertaintyData.variance}
              samples={uncertaintyData.monte_carlo_samples}
              calibrated={modelOutputs.some(m => m.calibrated_output !== undefined)}
              triggeredReprocessing={diagnosticData?.processing_metrics?.reprocessing_triggered}
            />
          </div>
          
          {hasTemporalData && (
            <div className="mt-4">
              <TemporalConsistencyChart 
                framePredictions={framePredictions}
                temporalConsistency={diagnosticData?.temporal_analysis?.frame_consistency}
                smoothedPrediction={result.probability}
              />
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="interpretability">
          {hasHeatmaps ? (
            <div className="space-y-4">
              {diagnosticData.visualization.heatmaps.map((heatmap: any, index: number) => (
                <HeatmapVisualizer 
                  key={index}
                  heatmapSrc={heatmap.heatmap_url}
                  originalSrc={heatmap.original_url}
                  frameNumber={heatmap.frame_number}
                  probability={heatmap.probability}
                  confidence={heatmap.confidence}
                  isCalibratedModel={heatmap.is_calibrated}
                  regionMetadata={heatmap.metadata}
                />
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center py-10">
                <div className="text-center">
                  <Info className="h-10 w-10 text-muted-foreground mb-4 mx-auto" />
                  <p className="text-muted-foreground">No interpretability data available for this detection.</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
        
        <TabsContent value="temporal">
          {hasTemporalData ? (
            <div className="space-y-4">
              <TemporalConsistencyChart 
                framePredictions={framePredictions}
                temporalConsistency={diagnosticData?.temporal_analysis?.frame_consistency}
                smoothedPrediction={result.probability}
                showUncertainty={true}
              />
            </div>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center py-10">
                <div className="text-center">
                  <Clock className="h-10 w-10 text-muted-foreground mb-4 mx-auto" />
                  <p className="text-muted-foreground">No temporal data available for this detection.</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
        
        <TabsContent value="models">
          {modelOutputs.length > 0 ? (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Model Outputs</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr>
                        <th className="text-left py-2">Model</th>
                        <th className="text-left py-2">Output</th>
                        <th className="text-left py-2">Confidence</th>
                        <th className="text-left py-2">Weight</th>
                      </tr>
                    </thead>
                    <tbody>
                      {modelOutputs.map((model: any, index: number) => (
                        <tr key={index}>
                          <td className="py-2">{model.model}</td>
                          <td className="py-2">{(model.raw_output * 100).toFixed(1)}%</td>
                          <td className="py-2">{(model.confidence * 100).toFixed(1)}%</td>
                          <td className="py-2">{model.weight?.toFixed(2) || 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center py-10">
                <div className="text-center">
                  <Cpu className="h-10 w-10 text-muted-foreground mb-4 mx-auto" />
                  <p className="text-muted-foreground">No model details available for this detection.</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default DiagnosticDashboard;
