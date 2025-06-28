
import { useState, useEffect, useRef } from "react";
import { DetectionResult, ReportStatus } from "@/types";
import { checkReportStatus } from "@/services/report-service";
import { getDiagnosticData } from "@/services/diagnostic-service";
import { logReportGeneration } from "@/utils/debug-logger";
import { toast } from "sonner";
import { Info, Loader2, Check } from "lucide-react";

interface UseReportStatusOptions {
  initialReady?: boolean;
  pollingInterval?: number;
  maxRetries?: number;
  autoStart?: boolean;
}

/**
 * Custom hook that manages report status and readiness
 * Combines polling with additional readiness logic
 */
export function useReportStatus(
  result: DetectionResult | null,
  options: UseReportStatusOptions = {}
) {
  const {
    initialReady = false,
    pollingInterval = 5000,
    maxRetries = 3,
    autoStart = true
  } = options;
  
  // Status state
  const [status, setStatus] = useState<ReportStatus>({
    ready: initialReady,
    progress: initialReady ? 100 : 0,
    missingData: [],
    lastChecked: Date.now()
  });
  
  // Polling and retry state
  const [isPolling, setIsPolling] = useState(false);
  const [retryAttempts, setRetryAttempts] = useState(0);
  const pollingIntervalRef = useRef<number | null>(null);
  
  // Additional state to track preparatory diagnostics
  const [isDiagnosticsLoading, setIsDiagnosticsLoading] = useState(false);
  const [hasDiagnosticData, setHasDiagnosticData] = useState(false);
  const [hasTriedStatusCheck, setHasTriedStatusCheck] = useState(false);
  
  // Function to start polling for report status
  const startPolling = () => {
    console.log("Starting report status polling");
    if (pollingIntervalRef.current !== null) {
      window.clearInterval(pollingIntervalRef.current);
    }
    
    checkCurrentStatus();
    pollingIntervalRef.current = window.setInterval(
      checkCurrentStatus, 
      pollingInterval
    );
  };
  
  // Function to stop polling
  const stopPolling = () => {
    console.log("Stopping report status polling");
    if (pollingIntervalRef.current !== null) {
      window.clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };
  
  // Function to check the current report status
  const checkCurrentStatus = async () => {
    if (!result?.id) {
      console.log("No result ID available for status check");
      return;
    }
    
    if (status.ready) {
      console.log("Report already marked as ready, skipping status check");
      stopPolling();
      return;
    }
    
    setIsPolling(true);
    setHasTriedStatusCheck(true);
    
    try {
      console.log("Checking report status for ID:", result.id);
      const apiStatus = await checkReportStatus(result.id);
      
      console.log("Received report status:", apiStatus);
      
      const newStatus: ReportStatus = {
        ready: apiStatus.ready,
        progress: apiStatus.progress,
        missingData: apiStatus.missingData,
        lastChecked: Date.now()
      };
      
      setStatus(newStatus);
      
      // Update result with latest status if it changed
      if (result.report_status?.ready !== newStatus.ready || 
          result.report_status?.progress !== newStatus.progress) {
        console.log("Updating result object with new report status");
        result.report_status = {
          ready: newStatus.ready,
          missing_data: newStatus.missingData,
          progress: newStatus.progress,
          last_checked: newStatus.lastChecked
        };
        result.report_ready = newStatus.ready;
      }
      
      // Log report status
      logReportGeneration("status check", {
        result_id: result.id,
        status: newStatus.ready ? "ready" : "not ready",
        progress: newStatus.progress,
        missing_data: newStatus.missingData
      });
      
      // Show toast for report ready
      if (newStatus.ready && !status.ready) {
        console.log("Report is now ready, stopping polling");
        stopPolling();
        toast.success("Report is ready for download", {
          icon: <Check className="h-4 w-4 text-green-500" />
        });
      } 
      // Show toast for progress update (but not too frequently)
      else if (
        newStatus.progress && 
        newStatus.progress > 0 && 
        (!status.progress || Math.abs(newStatus.progress - status.progress) > 20)
      ) {
        toast.info(`Report generation in progress: ${newStatus.progress}% complete`, {
          icon: <Loader2 className="h-4 w-4 animate-spin" />,
          duration: 2000,
        });
      }
      
      // If report is fully ready, we can stop polling
      if (newStatus.ready) {
        stopPolling();
      }
      
      return newStatus.ready;
    } catch (error) {
      console.error("Failed to check report status:", error);
      logReportGeneration("status check error", { error: String(error) });
      
      setStatus(prev => ({
        ...prev,
        errorMessage: error instanceof Error ? error.message : String(error),
        lastChecked: Date.now()
      }));
      
      return false;
    } finally {
      setIsPolling(false);
    }
  };
  
  // Check for diagnostics data availability
  useEffect(() => {
    if (!result?.id || status.ready || hasDiagnosticData || isDiagnosticsLoading) {
      return;
    }
    
    const loadDiagnostics = async () => {
      setIsDiagnosticsLoading(true);
      console.log("Attempting to load diagnostic data for ID:", result.id);
      
      try {
        const data = await getDiagnosticData(result.id);
        const hasData = !!data && (
          (data.temporal_analysis && data.temporal_analysis.frame_predictions?.length > 0) ||
          (data.model_outputs?.length > 0) ||
          // Fix: Check for uncertainty analysis in the correct property
          data.uncertainty_analysis !== undefined
        );
        
        console.log("Diagnostic data loaded, has valid data:", hasData);
        setHasDiagnosticData(hasData);
        
        logReportGeneration("diagnostic data loaded", { 
          result_id: result.id,
          has_data: hasData
        });
      } catch (err) {
        console.log("Diagnostic data not available yet:", err);
        setHasDiagnosticData(false);
      } finally {
        setIsDiagnosticsLoading(false);
      }
    };
    
    loadDiagnostics();
  }, [result?.id, status.ready, hasDiagnosticData, isDiagnosticsLoading]);
  
  // Auto-retry logic
  useEffect(() => {
    // Only retry if we've already tried at least once and got an error
    if (
      hasTriedStatusCheck && 
      !status.ready && 
      status.missingData?.some(msg => msg.toLowerCase().includes("error")) && 
      retryAttempts < maxRetries
    ) {
      console.log(`Setting up auto-retry for attempt ${retryAttempts + 1}/${maxRetries}`);
      
      const retryTimer = setTimeout(() => {
        console.log(`Auto-retrying report status check (attempt ${retryAttempts + 1}/${maxRetries})`);
        logReportGeneration("auto-retry", { attempt: retryAttempts + 1, max_attempts: maxRetries });
        setRetryAttempts(prev => prev + 1);
        
        if (result?.id) {
          checkCurrentStatus().then(isReady => {
            if (isReady) {
              console.log("Auto-retry successful, report is now ready");
              toast.success("Report generation successful after retry", {
                icon: <Check className="h-4 w-4 text-green-500" />
              });
              stopPolling();
            }
          });
        }
      }, 3000 * (retryAttempts + 1));
      
      return () => {
        console.log("Clearing retry timer");
        clearTimeout(retryTimer);
      };
    }
  }, [status, retryAttempts, result?.id, maxRetries, hasTriedStatusCheck]);
  
  // Start/stop polling based on result ID and autoStart option
  useEffect(() => {
    if (result?.id && autoStart && !status.ready) {
      console.log("Auto-starting polling for ID:", result.id);
      startPolling();
    } else if (!result?.id || status.ready) {
      console.log("Stopping polling due to missing ID or report ready");
      stopPolling();
    }
    
    return () => {
      stopPolling();
    };
  }, [result?.id, status.ready, autoStart]);
  
  // Reset function to clear status and retry attempts
  const resetStatus = () => {
    console.log("Resetting report status");
    stopPolling();
    setStatus({
      ready: false,
      progress: 0,
      missingData: [],
      lastChecked: Date.now()
    });
    setRetryAttempts(0);
    setHasTriedStatusCheck(false);
    setHasDiagnosticData(false);
  };
  
  // Get combined status (API polling status + our internal checks)
  const getCombinedStatus = (): ReportStatus => {
    // If API says it's ready, we're ready
    if (status.ready) {
      return status;
    }
    
    // If we have explicit readiness flag from result object
    if (
      result?.report_ready === true || 
      result?.report_status?.ready === true
    ) {
      return {
        ...status,
        ready: true,
        progress: 100
      };
    }
    
    // If we've exceeded retries and have diagnostic data, we can generate locally
    if (retryAttempts >= maxRetries && hasDiagnosticData) {
      return {
        ...status,
        localFallbackReady: true
      };
    }
    
    // Otherwise return the current status
    return status;
  };
  
  // Check if all required data for report is available
  const checkReadiness = () => {
    if (!result) return false;
    
    // Check if API explicitly says report is ready
    if (status.ready) return true;
    
    // Check if we have the report_ready flag
    if (result.report_ready === true) return true;
    
    // Check for detailed report status
    if (result.report_status?.ready === true) return true;
    
    // Check if we can do local fallback
    if (hasDiagnosticData && retryAttempts >= maxRetries) return true;
    
    return false;
  };
  
  // Manual check function
  const manualCheck = async () => {
    console.log("Manual report status check requested");
    if (!result?.id) return false;
    
    setIsPolling(true);
    try {
      const isReady = await checkCurrentStatus();
      console.log("Manual status check result:", isReady);
      return isReady;
    } catch (error) {
      console.error("Manual report status check failed:", error);
      return false;
    } finally {
      setIsPolling(false);
    }
  };
  
  // Combine all our status data
  const combinedStatus = getCombinedStatus();
  
  return {
    status: combinedStatus,
    isPolling: isPolling || isDiagnosticsLoading,
    retryAttempts,
    maxRetries,
    isReportReady: checkReadiness(),
    checkStatus: manualCheck,
    startPolling,
    stopPolling,
    resetStatus,
    hasDiagnosticData
  };
}

export default useReportStatus;
