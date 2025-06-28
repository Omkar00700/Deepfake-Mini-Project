
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { DetectionResult, ReportFormat } from "@/types";
import { useReportStatus } from "@/hooks/useReportStatus";
import { toast } from "sonner";
import { FileDown, Check, X, AlertTriangle, FileText, FileJson, FileSpreadsheet, Loader2 } from "lucide-react";
import { 
  generateDetailedReport, 
  downloadApiReport
} from "@/services/report-service";
import { logReportGeneration } from "@/utils/debug-logger";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface AdvancedReportDownloadHandlerProps {
  result: DetectionResult;
  onDownloadStatusChange?: (isDownloading: boolean) => void;
}

const AdvancedReportDownloadHandler: React.FC<AdvancedReportDownloadHandlerProps> = ({ 
  result,
  onDownloadStatusChange
}) => {
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState<'pdf' | 'json' | 'csv'>('pdf');
  const [downloadAttempts, setDownloadAttempts] = useState(0);
  const MAX_DOWNLOAD_ATTEMPTS = 3;
  
  // Use the report status hook with more frequent polling
  const {
    status,
    isPolling,
    retryAttempts,
    maxRetries,
    isReportReady,
    checkStatus,
    hasDiagnosticData,
    startPolling,
    stopPolling
  } = useReportStatus(result, {
    pollingInterval: 3000,  // Poll more frequently
    maxRetries: 5,          // More retries
    autoStart: true
  });
  
  // Available report formats
  const reportFormats: ReportFormat[] = [
    {
      id: 'pdf',
      label: 'PDF Report',
      value: 'pdf',
      icon: <FileText className="mr-2 h-4 w-4" />
    },
    {
      id: 'json',
      label: 'JSON Data',
      value: 'json',
      icon: <FileJson className="mr-2 h-4 w-4" />
    },
    {
      id: 'csv',
      label: 'CSV Format',
      value: 'csv',
      icon: <FileSpreadsheet className="mr-2 h-4 w-4" />
    }
  ];
  
  // Check for report readiness when result changes
  useEffect(() => {
    if (result?.id) {
      console.log("Starting report status polling for result ID:", result.id);
      startPolling();
      
      // Log that we're checking for report readiness
      logReportGeneration("report status polling started", { 
        result_id: result.id 
      });
    }
    
    return () => {
      console.log("Stopping report status polling");
      stopPolling();
    };
  }, [result?.id, startPolling, stopPolling]);
  
  // Handle download report in the selected format
  const handleDownloadReport = async () => {
    console.log("Download button clicked for format:", selectedFormat);
    
    if (!result) {
      toast.error("No detection result available", {
        icon: <X className="h-4 w-4 text-red-500" />
      });
      return;
    }
    
    // Verify report readiness, except when using local fallback
    if (!isReportReady && 
        !status.localFallbackReady && 
        retryAttempts < maxRetries) {
      
      console.log("Performing final report readiness check before download");
      // Make one final check in case status is stale
      const isReady = await checkStatus();
      console.log("Final readiness check result:", isReady);
      
      if (!isReady) {
        toast.error(
          `Report data is not ready yet. ${status.missingData?.join(', ') || 'Processing incomplete'}`, 
          {
            icon: <AlertTriangle className="h-4 w-4 text-yellow-500" />,
            duration: 5000
          }
        );
        return;
      }
    }
    
    // Set generating state and notify parent component
    setIsGeneratingReport(true);
    if (onDownloadStatusChange) onDownloadStatusChange(true);
    
    console.log("Starting report download process", { 
      format: selectedFormat, 
      result_id: result.id,
      using_fallback: status.localFallbackReady && !status.ready
    });
    
    logReportGeneration("download initiated", { 
      format: selectedFormat, 
      result_id: result.id,
      using_fallback: status.localFallbackReady && !status.ready
    });
    
    try {
      if (result.id && !status.localFallbackReady) {
        // Try API download for reports with an ID and when server report should be available
        setDownloadAttempts(prev => prev + 1);
        let attempts = 0;
        const maxAttempts = MAX_DOWNLOAD_ATTEMPTS;
        let success = false;
        
        while (attempts < maxAttempts && !success) {
          try {
            console.log(`API report download attempt ${attempts + 1} for ${selectedFormat}`);
            
            toast.loading(`Downloading ${selectedFormat.toUpperCase()} report...`, {
              id: "report-download",
              duration: 10000
            });
            
            success = await downloadApiReport(result.id, selectedFormat);
            console.log("Download API call result:", success);
            
            if (success) break;
          } catch (err) {
            console.error(`API report download attempt ${attempts + 1} failed:`, err);
            attempts++;
            
            if (attempts < maxAttempts) {
              await new Promise(resolve => setTimeout(resolve, 1000 * attempts));
            }
          }
        }
        
        if (success) {
          toast.success(`${selectedFormat.toUpperCase()} report downloaded successfully`, {
            id: "report-download",
            icon: <Check className="h-4 w-4 text-green-500" />
          });
          setIsGeneratingReport(false);
          if (onDownloadStatusChange) onDownloadStatusChange(false);
          return;
        } else if (attempts >= maxAttempts) {
          console.log("API download failed after max attempts, falling back to local generation");
          toast.warning(`API report download failed. Trying local generation...`, {
            id: "report-download",
            icon: <AlertTriangle className="h-4 w-4 text-yellow-500" />
          });
        }
      }
      
      // Fallback to local report generation
      console.log("Generating report locally for format:", selectedFormat);
      
      toast.loading(`Generating ${selectedFormat.toUpperCase()} report locally...`, {
        id: "report-download",
        duration: 10000
      });
      
      await generateDetailedReport(result, selectedFormat);
      
      toast.success(`${selectedFormat.toUpperCase()} report generated successfully`, {
        id: "report-download",
        icon: <Check className="h-4 w-4 text-green-500" />
      });
    } catch (error) {
      console.error("Error generating report:", error);
      
      toast.error(`Failed to generate ${selectedFormat.toUpperCase()} report: ${error instanceof Error ? error.message : 'Unknown error'}`, {
        id: "report-download",
        icon: <X className="h-4 w-4 text-red-500" />
      });
    } finally {
      setIsGeneratingReport(false);
      if (onDownloadStatusChange) onDownloadStatusChange(false);
    }
  };

  // Get tooltip message based on current state
  const getTooltipMessage = () => {
    if (isGeneratingReport) {
      return "Report is being generated...";
    }
    
    if (!isReportReady && !status.localFallbackReady) {
      if (status.missingData && status.missingData.length > 0) {
        return `Report not ready: ${status.missingData[0]}`;
      }
      return "Report data is still being prepared";
    }
    
    if (downloadAttempts >= MAX_DOWNLOAD_ATTEMPTS) {
      return "Multiple download attempts failed. Using local generation as fallback.";
    }
    
    return `Download a detailed ${selectedFormat.toUpperCase()} report with complete analysis`;
  };
  
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-1">
          <FileDown className="h-4 w-4" />
          Detailed Analysis Report
        </CardTitle>
        <CardDescription className="text-xs">
          Download a comprehensive report with complete detection metrics and visualizations
        </CardDescription>
      </CardHeader>
      
      <CardContent className="pb-2">
        <Tabs 
          defaultValue="pdf" 
          className="w-full"
          onValueChange={(value) => setSelectedFormat(value as 'pdf' | 'json' | 'csv')}
        >
          <TabsList className="w-full grid grid-cols-3">
            <TabsTrigger value="pdf">PDF Report</TabsTrigger>
            <TabsTrigger value="json">JSON Data</TabsTrigger>
            <TabsTrigger value="csv">CSV Format</TabsTrigger>
          </TabsList>
        </Tabs>
        
        {/* Status indicators */}
        {!isReportReady && status.progress !== undefined && status.progress > 0 && (
          <div className="mt-4">
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
              <span>Preparing report data...</span>
              <span>{status.progress}%</span>
            </div>
            <Progress value={status.progress} className="h-1.5" />
          </div>
        )}
        
        {/* Error message */}
        {!isReportReady && status.missingData && status.missingData.length > 0 && (
          <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs text-yellow-800">
            <div className="flex items-start">
              <AlertTriangle className="h-3 w-3 mt-0.5 mr-1 flex-shrink-0" />
              <div>
                <p className="font-medium">Waiting for report data:</p>
                <ul className="list-disc list-inside mt-1 space-y-1">
                  {status.missingData.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
        
        {/* Retry attempts */}
        {retryAttempts > 0 && (
          <div className="mt-2 text-xs text-muted-foreground">
            <div className="flex items-center">
              <Loader2 className="h-3 w-3 mr-1 animate-spin" />
              <span>Retry attempt {retryAttempts}/{maxRetries}...</span>
            </div>
          </div>
        )}
      </CardContent>
      
      <CardFooter>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="w-full">
                <Button 
                  className="w-full"
                  disabled={isGeneratingReport || (!isReportReady && !status.localFallbackReady) || isPolling}
                  onClick={handleDownloadReport}
                >
                  {isGeneratingReport ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating Report...
                    </>
                  ) : (
                    <>
                      <FileDown className="mr-2 h-4 w-4" />
                      Download {selectedFormat.toUpperCase()} Report
                    </>
                  )}
                </Button>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p className="text-xs max-w-xs">
                {getTooltipMessage()}
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </CardFooter>
    </Card>
  );
};

export default AdvancedReportDownloadHandler;
