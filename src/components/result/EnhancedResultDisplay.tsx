
import React, { useState, useEffect } from 'react';
import { DetectionResult } from '@/types';
import ConfidenceMeter from './ConfidenceMeter';
import UncertaintyVisualizer from './UncertaintyVisualizer';
import TemporalConsistencyVisualization from './TemporalConsistencyVisualization';
import InterpretabilityVisualization from './InterpretabilityVisualization';
import AdvancedReportDownloadHandler from '../report/AdvancedReportDownloadHandler';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { FileDown, BarChart3, LineChart, Info, FileText } from 'lucide-react';
import { useReportStatus } from '@/hooks/useReportStatus';
import { toast } from 'sonner';

interface EnhancedResultDisplayProps {
  result: DetectionResult;
}

const EnhancedResultDisplay: React.FC<EnhancedResultDisplayProps> = ({ result }) => {
  const [activeTab, setActiveTab] = useState('interpretability');
  const [isDownloading, setIsDownloading] = useState(false);
  
  // Use the report status hook to track report readiness
  const { isReportReady, status, isPolling } = useReportStatus(result, {
    pollingInterval: 5000,
    autoStart: true
  });
  
  // When changing to report tab, show loading state if report isn't ready
  useEffect(() => {
    if (activeTab === 'report' && !isReportReady && !status.localFallbackReady) {
      if (!isPolling) {
        toast.info('Checking report availability...', {
          duration: 3000,
          icon: <Info className="h-4 w-4" />
        });
      }
    }
  }, [activeTab, isReportReady, status.localFallbackReady, isPolling]);
  
  if (!result) return null;
  
  // Determine if it's a video analysis
  const isVideoAnalysis = result.detectionType === 'video';
  
  // Check if we have interpretability data
  const hasInterpretabilityData = result.metadata?.attention_maps !== undefined;
  
  // Check if we have uncertainty data
  const hasUncertaintyData = result.uncertainty !== undefined || 
    result.metadata?.weighted_uncertainty !== undefined;
  
  // Function to handle downloading status changes
  const handleDownloadStatusChange = (downloading: boolean) => {
    setIsDownloading(downloading);
    
    // If download started and we're on another tab, switch to report tab
    if (downloading && activeTab !== 'report') {
      setActiveTab('report');
    }
  };
  
  return (
    <div className="w-full space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Main detection result with confidence meter */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-1">
              <BarChart3 className="h-4 w-4" />
              Detection Result
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ConfidenceMeter 
              probability={result.probability} 
              confidence={result.confidence}
              uncertainty={result.uncertainty}
            />
          </CardContent>
        </Card>
        
        {/* Uncertainty visualization */}
        {hasUncertaintyData && (
          <UncertaintyVisualizer result={result} />
        )}
      </div>
      
      <Tabs 
        value={activeTab} 
        onValueChange={setActiveTab} 
        className="w-full"
      >
        <TabsList>
          <TabsTrigger value="interpretability" disabled={!hasInterpretabilityData}>
            <Info className="h-4 w-4 mr-1" />
            Model Interpretability
          </TabsTrigger>
          <TabsTrigger value="temporal" disabled={!isVideoAnalysis}>
            <LineChart className="h-4 w-4 mr-1" />
            Temporal Analysis
          </TabsTrigger>
          <TabsTrigger value="report">
            <FileText className="h-4 w-4 mr-1" />
            Download Report
            {isDownloading && <span className="ml-1 animate-pulse">...</span>}
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="interpretability" className="mt-4">
          {hasInterpretabilityData ? (
            <InterpretabilityVisualization result={result} />
          ) : (
            <Card>
              <CardContent className="py-8 text-center">
                <Info className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
                <p className="text-muted-foreground">
                  No interpretability data available for this detection.
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
        
        <TabsContent value="temporal" className="mt-4">
          {isVideoAnalysis ? (
            <TemporalConsistencyVisualization result={result} />
          ) : (
            <Card>
              <CardContent className="py-8 text-center">
                <LineChart className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
                <p className="text-muted-foreground">
                  Temporal analysis is only available for video detections.
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
        
        <TabsContent value="report" className="mt-4">
          <AdvancedReportDownloadHandler 
            result={result} 
            onDownloadStatusChange={handleDownloadStatusChange}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default EnhancedResultDisplay;
