
import { ReportStatus } from "@/types";
import { AlertTriangle } from "lucide-react";

interface ReportErrorMessageProps {
  status: ReportStatus;
  retryAttempts: number;
  maxRetries: number;
}

/**
 * Component to display error messages related to report generation
 */
export const ReportErrorMessage = ({ 
  status, 
  retryAttempts,
  maxRetries
}: ReportErrorMessageProps) => {
  const hasErrors = status.missingData?.some(msg => msg.toLowerCase().includes("error"));
  const hasExceededRetries = retryAttempts >= maxRetries;
  
  // No errors or not in error state
  if (!hasErrors || status.ready) {
    return null;
  }
  
  // Display a message about using local fallback
  if (hasExceededRetries) {
    return (
      <div className="mt-2 text-xs flex items-start text-yellow-600">
        <AlertTriangle className="h-3 w-3 mt-0.5 mr-1 flex-shrink-0" />
        <span>
          Server report generation failed. Click download to generate a local report.
        </span>
      </div>
    );
  }
  
  // Regular error message
  return (
    <div className="mt-2 text-xs flex items-start text-red-600">
      <AlertTriangle className="h-3 w-3 mt-0.5 mr-1 flex-shrink-0" />
      <span>
        {status.missingData?.filter(msg => msg.toLowerCase().includes("error")).join('. ')}
      </span>
    </div>
  );
};

export default ReportErrorMessage;
