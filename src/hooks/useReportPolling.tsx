
import { useState, useEffect, useRef } from "react";
import { DetectionResult, ReportStatus } from "@/types";
import { checkReportStatus } from "@/services/report-service";
import { logReportGeneration } from "@/utils/debug-logger";
import { toast } from "sonner";

interface UseReportPollingOptions {
  pollingInterval?: number;
  maxRetries?: number;
  autoStart?: boolean;
}

/**
 * Custom hook that handles polling the API for report status
 * Only focuses on the polling mechanism and status updates
 */
export function useReportPolling(
  result: DetectionResult | null,
  options: UseReportPollingOptions = {}
) {
  const {
    pollingInterval = 5000,
    maxRetries = 3,
    autoStart = true
  } = options;
  
  const [status, setStatus] = useState<ReportStatus>({
    ready: false,
    progress: 0,
    missingData: [],
    lastChecked: Date.now()
  });
  
  const [isPolling, setIsPolling] = useState(false);
  const [retryAttempts, setRetryAttempts] = useState(0);
  const pollingIntervalRef = useRef<number | null>(null);
  
  // Function to start polling for report status
  const startPolling = () => {
    if (!result?.id) return;
    
    // Clear any existing interval
    if (pollingIntervalRef.current !== null) {
      window.clearInterval(pollingIntervalRef.current);
    }
    
    // Do an immediate check
    checkCurrentStatus();
    
    // Start polling at regular intervals
    pollingIntervalRef.current = window.setInterval(
      checkCurrentStatus, 
      pollingInterval
    );
    
    logReportGeneration("polling started", {
      result_id: result.id,
      interval: pollingInterval
    });
  };
  
  // Function to stop polling
  const stopPolling = () => {
    if (pollingIntervalRef.current !== null) {
      window.clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
      setIsPolling(false);
      
      if (result?.id) {
        logReportGeneration("polling stopped", { result_id: result.id });
      }
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
        progress: apiStatus.progress || 0,
        missingData: apiStatus.missingData || [],
        lastChecked: Date.now()
      };
      
      setStatus(newStatus);
      
      logReportGeneration("status check", {
        result_id: result.id,
        status: newStatus.ready ? "ready" : "not ready",
        progress: newStatus.progress,
        missing_data: newStatus.missingData
      });
      
      // If report is ready, stop polling
      if (newStatus.ready) {
        stopPolling();
      }
    } catch (error) {
      console.error("Failed to check report status:", error);
      logReportGeneration("status check error", { 
        result_id: result?.id,
        error: String(error) 
      });
      
      setStatus(prev => ({
        ...prev,
        errorMessage: error instanceof Error ? error.message : String(error),
        lastChecked: Date.now()
      }));
    } finally {
      setIsPolling(false);
    }
  };
  
  // Auto-retry logic for errors
  useEffect(() => {
    // Only attempt retries if there are error messages and we haven't exceeded max retries
    if (
      status.missingData?.some(msg => msg.toLowerCase().includes("error")) && 
      retryAttempts < maxRetries &&
      result?.id
    ) {
      const retryTimer = setTimeout(() => {
        console.log(`Auto-retrying report status check (attempt ${retryAttempts + 1}/${maxRetries})`);
        logReportGeneration("auto-retry", { 
          attempt: retryAttempts + 1, 
          max_attempts: maxRetries,
          result_id: result.id
        });
        
        setRetryAttempts(prev => prev + 1);
        checkCurrentStatus();
      }, 3000 * (retryAttempts + 1)); // Increasing backoff
      
      return () => clearTimeout(retryTimer);
    }
  }, [status.missingData, retryAttempts, maxRetries, result?.id]);
  
  // Start/stop polling based on result
  useEffect(() => {
    if (result?.id && !status.ready && autoStart) {
      startPolling();
    } else if (!result?.id || status.ready) {
      stopPolling();
    }
    
    return () => {
      stopPolling();
    };
  }, [result?.id, status.ready, autoStart]);
  
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
  
  // Manual check function
  const manualCheck = async () => {
    if (!result?.id) return false;
    
    setIsPolling(true);
    try {
      const apiStatus = await checkReportStatus(result.id);
      
      setStatus({
        ready: apiStatus.ready,
        progress: apiStatus.progress || 0,
        missingData: apiStatus.missingData || [],
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
    startPolling,
    stopPolling,
    resetStatus
  };
}

export default useReportPolling;
