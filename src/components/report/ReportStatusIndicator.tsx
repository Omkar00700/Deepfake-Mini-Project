
import { ReportStatus } from "@/types";
import { Progress } from "@/components/ui/progress";
import { Loader2, AlertTriangle, Info } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface ReportStatusIndicatorProps {
  status: ReportStatus;
  retryAttempts: number;
  maxRetries: number;
}

/**
 * Component that displays the current status of report generation
 * Shows progress bar, loading indicators or error states
 */
export const ReportStatusIndicator = ({ 
  status, 
  retryAttempts,
  maxRetries 
}: ReportStatusIndicatorProps) => {
  // No status to show if report is ready
  if (status.ready) {
    return null;
  }
  
  // Show retry status if we're retrying
  if (retryAttempts > 0) {
    return (
      <div className="mt-2 text-xs">
        <div className="flex items-center text-yellow-600">
          <AlertTriangle className="h-3 w-3 mr-1" />
          <span>Retry attempt {retryAttempts}/{maxRetries}</span>
        </div>
      </div>
    );
  }
  
  // Show progress if we have it
  if (status.progress !== undefined && status.progress > 0 && status.progress < 100) {
    return (
      <div className="mt-2">
        <Progress value={status.progress} className="h-1" />
        <p className="text-xs text-gray-500 text-center mt-1">
          {status.progress}% complete
        </p>
      </div>
    );
  }
  
  // Show loading indicator by default
  return (
    <div className="mt-2 flex items-center justify-center gap-2">
      <Loader2 className="h-3 w-3 animate-spin" />
      <span className="text-xs text-gray-500">Checking report status...</span>
    </div>
  );
};

export default ReportStatusIndicator;
