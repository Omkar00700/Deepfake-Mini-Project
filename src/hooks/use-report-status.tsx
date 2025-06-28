
import { useState, useEffect, useRef } from "react";
import { DetectionResult, ReportStatus } from "@/types";
import { checkReportStatus } from "@/services/report-service";
import { logReportGeneration } from "@/utils/debug-logger";
import { toast } from "sonner";
import { Info, Loader2, Check } from "lucide-react";

interface UseReportStatusOptions {
  initialReady?: boolean;
  pollingInterval?: number;
  maxRetries?: number;
}

/**
 * Custom hook that handles polling the API for report status
 * Manages report status state, polling, and auto-retries
 */
export function useReportStatus(
  result: DetectionResult | null,
  options: UseReportStatusOptions = {}
) {
  const {
    initialReady = false,
    pollingInterval = 5000,
    maxRetries = 3
  } = options;
  
  const [status, setStatus] = useState<ReportStatus>({
    ready: initialReady,
    progress: initialReady ? 100 : 0,
    missingData: [],
    lastChecked: Date.now()
  });
  
  const [isPolling, setIsPolling] = useState(false);
  const [retryAttempts, setRetryAttempts] = useState(0);
  const pollingIntervalRef = useRef<number | null>(null);
  
  // Function to start polling for report status
  const startPolling = () => {
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
    if (pollingIntervalRef.current !== null) {
      window.clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };
  
  // Function to check the current report status
  const checkCurrentStatus = async () => {
    if (!result?.id || status.ready) return;
    
    setIsPolling(true);
    try {
      const apiStatus = await checkReportStatus(result.id);
      
      const newStatus: ReportStatus = {
        ready: apiStatus.ready,
        progress: apiStatus.progress,
        missingData: apiStatus.missingData,
        lastChecked: Date.now()
      };
      
      setStatus(newStatus);
      
      // Update result with latest status
      if (result.report_status !== newStatus) {
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
        stopPolling();
        toast.success("Report is ready for download", {
          icon: <Check className="h-4 w-4 text-green-500" />
        });
      } 
      // Show toast for progress update (but not too frequently)
      else if (
        newStatus.progress && 
        newStatus.progress > 0 && 
        (!status.progress || Math.abs(newStatus.progress - status.progress) > 10)
      ) {
        toast.info(`Report generation in progress: ${newStatus.progress}% complete`, {
          icon: <Loader2 className="h-4 w-4 animate-spin" />,
          duration: 2000,
        });
      }
    } catch (error) {
      console.error("Failed to check report status:", error);
      logReportGeneration("status check error", { error: String(error) });
      
      setStatus(prev => ({
        ...prev,
        errorMessage: error instanceof Error ? error.message : String(error),
        lastChecked: Date.now()
      }));
    } finally {
      setIsPolling(false);
    }
  };
  
  // Reset function to clear status and retry attempts
  const resetStatus = () => {
    setStatus({
      ready: false,
      progress: 0,
      missingData: [],
      lastChecked: Date.now()
    });
    setRetryAttempts(0);
  };
  
  // Auto-retry logic
  useEffect(() => {
    if (
      status.missingData?.some(msg => msg.toLowerCase().includes("error")) && 
      retryAttempts < maxRetries
    ) {
      const retryTimer = setTimeout(() => {
        console.log(`Auto-retrying report status check (attempt ${retryAttempts + 1}/${maxRetries})`);
        logReportGeneration("auto-retry", { attempt: retryAttempts + 1, max_attempts: maxRetries });
        setRetryAttempts(prev => prev + 1);
        
        if (result?.id) {
          checkReportStatus(result.id)
            .then(apiStatus => {
              setStatus({
                ready: apiStatus.ready,
                progress: apiStatus.progress,
                missingData: apiStatus.missingData,
                lastChecked: Date.now()
              });
              
              if (apiStatus.ready) {
                toast.success("Report generation successful after retry", {
                  icon: <Check className="h-4 w-4 text-green-500" />
                });
              }
            })
            .catch(err => {
              console.error("Auto-retry failed:", err);
              logReportGeneration("auto-retry failed", { error: String(err) });
            });
        }
      }, 3000 * (retryAttempts + 1));
      
      return () => clearTimeout(retryTimer);
    }
  }, [status.missingData, retryAttempts, result?.id, maxRetries]);
  
  // Start/stop polling based on result
  useEffect(() => {
    if (result?.id && !status.ready) {
      startPolling();
    } else if (!result?.id || status.ready) {
      stopPolling();
    }
    
    return () => {
      stopPolling();
    };
  }, [result?.id, status.ready]);
  
  // Manual check function
  const manualCheck = async () => {
    if (!result?.id) return;
    
    setIsPolling(true);
    try {
      const apiStatus = await checkReportStatus(result.id);
      
      setStatus({
        ready: apiStatus.ready,
        progress: apiStatus.progress,
        missingData: apiStatus.missingData,
        lastChecked: Date.now()
      });
      
      return apiStatus.ready;
    } catch (error) {
      console.error("Manual report status check failed:", error);
      return false;
    } finally {
      setIsPolling(false);
    }
  };
  
  return {
    status,
    isPolling,
    retryAttempts,
    maxRetries,
    checkStatus: manualCheck,
    resetStatus
  };
}
