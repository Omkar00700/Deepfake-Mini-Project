
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { DetectionResult, ReportFormat } from "@/types";
import { useReportStatus } from "@/hooks/useReportStatus";
import { toast } from "sonner";
import { FileDown, Check, X, AlertTriangle } from "lucide-react";
import { 
  generateDetailedReport, 
  downloadApiReport
} from "@/services/report-service";
import { logReportGeneration } from "@/utils/debug-logger";
import ReportStatusIndicator from "./ReportStatusIndicator";
import ReportFormatSelector from "./ReportFormatSelector";
import ReportErrorMessage from "./ReportErrorMessage";
import DiagnosticDashboard from "../diagnostics/DiagnosticDashboard";
import { Progress } from "@/components/ui/progress";

interface ReportDownloadHandlerProps {
  result: DetectionResult;
  includeVisualizations?: boolean;
  onDownloadStatusChange?: (isDownloading: boolean) => void;
}

/**
 * Component to handle report downloads with enhanced status checking and diagnostics
 */
export const ReportDownloadHandler: React.FC<ReportDownloadHandlerProps> = ({ 
  result,
  includeVisualizations = true,
  onDownloadStatusChange
}) => {
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [downloadFormat, setDownloadFormat] = useState<'pdf' | 'json' | 'csv'>('pdf');
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [diagnosticData, setDiagnosticData] = useState<any>(null);
  const [downloadProgress, setDownloadProgress] = useState(0);
  
  const {
    status,
    isPolling,
    retryAttempts,
    maxRetries,
    checkStatus,
    hasDiagnosticData
  } = useReportStatus(result, {
    pollingInterval: 5000,
    maxRetries: 3,
    autoStart: true
  });

  const isReportReady = status.ready || status.localFallbackReady;
  
  const reportFormats: ReportFormat[] = [
    {
      id: 'pdf',
      label: 'PDF Report',
      value: 'pdf',
      icon: <FileDown className="mr-2 h-4 w-4" />
    },
    {
      id: 'json',
      label: 'JSON Report',
      value: 'json',
      icon: <FileDown className="mr-2 h-4 w-4" />
    },
    {
      id: 'csv',
      label: 'CSV Report',
      value: 'csv',
      icon: <FileDown className="mr-2 h-4 w-4" />
    }
  ];
  
  const handleDownloadReport = async (format: 'pdf' | 'json' | 'csv') => {
    if (!result) {
      toast.error("No detection result available", {
        icon: <X className="h-4 w-4 text-red-500" />
      });
      return;
    }
    
    setDownloadFormat(format);
    
    // Only check status if we're not using local fallback and haven't exceeded retry attempts
    if (!isReportReady && 
        !status.localFallbackReady && 
        retryAttempts < maxRetries && 
        !status.missingData?.some(msg => msg.toLowerCase().includes("error"))) {
      
      setDownloadProgress(10);
      const isReady = await checkStatus();
      setDownloadProgress(25);
      
      if (!isReady) {
        toast.error(
          `Report data is not ready yet. ${status.missingData?.join(', ') || 'Processing incomplete'}`, 
          {
            icon: <AlertTriangle className="h-4 w-4 text-yellow-500" />,
            duration: 5000
          }
        );
        setDownloadProgress(0);
        return;
      }
    }
    
    setIsGeneratingReport(true);
    setDownloadProgress(30);
    if (onDownloadStatusChange) onDownloadStatusChange(true);
    
    logReportGeneration("download initiated", { 
      format, 
      result_id: result.id,
      using_fallback: status.localFallbackReady && !status.ready
    });
    
    try {
      if (result.id && !status.localFallbackReady) {
        let attempts = 0;
        const maxAttempts = 3;
        let success = false;
        
        while (attempts < maxAttempts && !success) {
          try {
            console.log(`API report download attempt ${attempts + 1} for ${format}`);
            logReportGeneration("download attempt", { attempt: attempts + 1, format });
            
            setDownloadProgress(40 + attempts * 10);
            success = await downloadApiReport(result.id, format);
            if (success) {
              setDownloadProgress(100);
              break;
            }
          } catch (err) {
            console.error(`API report download attempt ${attempts + 1} failed:`, err);
            logReportGeneration("download error", { attempt: attempts + 1, error: String(err) });
            attempts++;
            
            if (attempts < maxAttempts) {
              await new Promise(resolve => setTimeout(resolve, 1000 * attempts));
            }
          }
        }
        
        if (success) {
          toast.success(`${format.toUpperCase()} report downloaded successfully`, {
            icon: <Check className="h-4 w-4 text-green-500" />
          });
          logReportGeneration("download success", { format });
          setIsGeneratingReport(false);
          if (onDownloadStatusChange) onDownloadStatusChange(false);
          return;
        } else if (attempts >= maxAttempts) {
          toast.warning(`API report download failed after ${maxAttempts} attempts. Trying local generation...`, {
            icon: <AlertTriangle className="h-4 w-4 text-yellow-500" />,
            duration: 5000
          });
          logReportGeneration("fallback to local", { after_attempts: attempts });
        }
      }
      
      console.log("Generating report locally");
      logReportGeneration("local generation", { format });
      setDownloadProgress(70);
      
      await generateDetailedReport(result, format, diagnosticData || undefined);
      setDownloadProgress(100);
      toast.success(`${format.toUpperCase()} report generated locally`, {
        icon: <Check className="h-4 w-4 text-green-500" />
      });
    } catch (error) {
      console.error("Error generating report:", error);
      logReportGeneration("generation error", { error: String(error) });
      
      toast.error(`Failed to generate ${format.toUpperCase()} report: ${error instanceof Error ? error.message : 'Unknown error'}`, {
        icon: <X className="h-4 w-4 text-red-500" />,
        duration: 7000
      });
    } finally {
      setIsGeneratingReport(false);
      if (onDownloadStatusChange) onDownloadStatusChange(false);
      // Reset progress after a delay
      setTimeout(() => setDownloadProgress(0), 1000);
    }
  };
  
  const handleGenerateReport = () => {
    handleDownloadReport(downloadFormat);
  };
  
  const isButtonDisabled = () => {
    if (isGeneratingReport) return true;
    if (isPolling) return true;
    
    if (!status.ready && status.localFallbackReady) {
      return false;
    }
    
    return !isReportReady;
  };
  
  const handleDiagnosticDataLoaded = (data: any) => {
    setDiagnosticData(data);
  };
  
  const toggleDiagnostics = () => {
    setShowDiagnostics(!showDiagnostics);
  };
  
  return (
    <div className="w-full space-y-2">
      <ReportFormatSelector 
        formats={reportFormats}
        onSelect={setDownloadFormat}
        disabled={isButtonDisabled()}
        isGenerating={isGeneratingReport}
        selectedFormat={downloadFormat}
        onGenerate={handleGenerateReport}
      />
      
      {downloadProgress > 0 && (
        <div className="w-full py-1">
          <Progress value={downloadProgress} className="h-1.5" />
          <p className="text-xs text-muted-foreground text-right mt-1">
            {downloadProgress === 100 ? 'Complete' : 'Processing...'}
          </p>
        </div>
      )}
      
      <ReportStatusIndicator 
        status={status} 
        retryAttempts={retryAttempts}
        maxRetries={maxRetries}
      />
      
      {!status.ready && !isReportReady && (
        <ReportErrorMessage 
          status={status}
          retryAttempts={retryAttempts}
          maxRetries={maxRetries}
        />
      )}
      
      {includeVisualizations && (result.detectionType === 'video' || hasDiagnosticData) && (
        <div className="mt-4">
          <Button 
            variant="ghost" 
            className="w-full text-sm" 
            size="sm"
            onClick={toggleDiagnostics}
          >
            {showDiagnostics ? "Hide Diagnostics" : "Show Advanced Diagnostics"}
          </Button>
        </div>
      )}
      
      {includeVisualizations && showDiagnostics && (
        <div className="mt-4">
          <DiagnosticDashboard 
            result={result}
            onDiagnosticDataLoaded={handleDiagnosticDataLoaded}
          />
        </div>
      )}
    </div>
  );
};

export default ReportDownloadHandler;
