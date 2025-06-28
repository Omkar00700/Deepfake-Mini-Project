
import { useState } from "react";
import { DetectionResult } from "@/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { useReportStatus } from "@/hooks/useReportStatus";
import { useReportData } from "@/hooks/useReportData";
import ReportFormatSelector from "./ReportFormatSelector";
import ReportStatusInfo from "./ReportStatusInfo";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { FileDown, Info } from "lucide-react";

interface EnhancedReportDownloadButtonProps {
  result: DetectionResult;
  initialReady?: boolean;
  compact?: boolean;
}

/**
 * Enhanced report download button with format selection and status display
 */
const EnhancedReportDownloadButton = ({
  result,
  initialReady = false,
  compact = false
}: EnhancedReportDownloadButtonProps) => {
  const [diagnosticData, setDiagnosticData] = useState<any>(null);
  
  // Use our custom hooks for report status and data
  const {
    status,
    isPolling,
    retryAttempts,
    maxRetries,
    isReportReady,
    checkStatus,
    hasDiagnosticData
  } = useReportStatus(result, {
    initialReady,
    autoStart: true,
    pollingInterval: 4000
  });
  
  const {
    isGenerating,
    selectedFormat,
    reportFormats,
    selectFormat,
    generateReport
  } = useReportData(result);

  // Handle diagnostic data loaded
  const handleDiagnosticDataLoaded = (data: any) => {
    setDiagnosticData(data);
  };
  
  // Get tooltip message based on current state
  const getTooltipMessage = () => {
    if (isGenerating) {
      return "Report is being generated...";
    }
    
    if (isPolling) {
      return "Checking report status...";
    }
    
    if (!isReportReady && !status.localFallbackReady) {
      if (status.missingData && status.missingData.length > 0) {
        return `Report not ready: ${status.missingData[0]}`;
      }
      return "Report data is still being prepared";
    }
    
    return `Download a detailed ${selectedFormat.toUpperCase()} report with complete analysis`;
  };

  // Handle download action
  const handleDownload = async () => {
    // Perform one last check before download
    if (!isReportReady && !status.localFallbackReady) {
      const isReady = await checkStatus();
      if (!isReady) {
        // Show a toast message if report is not ready
        toast.error("Report is not ready yet. Please try again in a few moments.");
        return;
      }
    }
    
    try {
      // Show loading toast
      toast.loading(`Preparing ${selectedFormat.toUpperCase()} report...`, {
        id: "report-download",
        duration: 10000
      });
      
      // Generate the report
      const success = await generateReport(diagnosticData);
      
      if (success) {
        toast.success(`${selectedFormat.toUpperCase()} report downloaded successfully`, {
          id: "report-download"
        });
      } else {
        toast.error(`Failed to download report. Please try again.`, {
          id: "report-download"
        });
      }
    } catch (error) {
      console.error("Error in handleDownload:", error);
      toast.error(`Download failed: ${error instanceof Error ? error.message : 'Unknown error'}`, {
        id: "report-download"
      });
    }
  };
  
  // Use compact version for small screens or when requested
  if (compact) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="w-full">
              <ReportFormatSelector 
                formats={reportFormats}
                selectedFormat={selectedFormat}
                isGenerating={isGenerating}
                disabled={!isReportReady && !status.localFallbackReady && !hasDiagnosticData}
                onSelect={selectFormat}
                onGenerate={handleDownload}
              />
            </div>
          </TooltipTrigger>
          <TooltipContent>
            <p className="text-xs max-w-xs">
              {getTooltipMessage()}
            </p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }
  
  // Full version with card
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-1">
          <FileDown className="h-4 w-4" />
          Detection Report
        </CardTitle>
        <CardDescription className="text-xs">
          Download a comprehensive report with complete detection metrics
        </CardDescription>
      </CardHeader>
      
      <CardContent className="pb-2">
        <ReportFormatSelector 
          formats={reportFormats}
          selectedFormat={selectedFormat}
          isGenerating={isGenerating}
          disabled={!isReportReady && !status.localFallbackReady && !hasDiagnosticData}
          onSelect={selectFormat}
          onGenerate={handleDownload}
        />
        
        <ReportStatusInfo 
          status={status}
          retryAttempts={retryAttempts}
          maxRetries={maxRetries}
          isPolling={isPolling}
        />
      </CardContent>
      
      <CardFooter className="pt-0">
        <div className="w-full text-xs text-muted-foreground flex items-center">
          <Info className="h-3 w-3 mr-1" />
          {hasDiagnosticData ? 
            "Diagnostic data available for enhanced reporting" : 
            "Basic report available"}
        </div>
      </CardFooter>
    </Card>
  );
};

export default EnhancedReportDownloadButton;
