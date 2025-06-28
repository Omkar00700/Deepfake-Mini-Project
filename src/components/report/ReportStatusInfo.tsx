
import { ReportStatus } from "@/types";
import { Progress } from "@/components/ui/progress";
import { AlertTriangle, Loader2, CheckCircle } from "lucide-react";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";

interface ReportStatusInfoProps {
  status: ReportStatus;
  retryAttempts: number;
  maxRetries: number;
  isPolling: boolean;
}

const ReportStatusInfo = ({
  status,
  retryAttempts,
  maxRetries,
  isPolling
}: ReportStatusInfoProps) => {
  // Show nothing if report is ready
  if (status.ready) {
    return (
      <div className="flex items-center text-green-600 text-sm">
        <CheckCircle className="h-4 w-4 mr-2" />
        Report is ready for download
      </div>
    );
  }

  // Polling status and progress
  if (isPolling) {
    return (
      <div>
        {status.progress !== undefined && status.progress > 0 && (
          <div className="mt-2">
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
              <span className="flex items-center">
                <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                Preparing report data...
              </span>
              <span>{status.progress}%</span>
            </div>
            <Progress value={status.progress} className="h-1.5" />
          </div>
        )}
      </div>
    );
  }

  // Show retry attempts
  if (retryAttempts > 0) {
    return (
      <div className="mt-2 text-xs text-muted-foreground">
        <div className="flex items-center">
          <Loader2 className="h-3 w-3 mr-1 animate-spin" />
          <span>Retry attempt {retryAttempts}/{maxRetries}...</span>
        </div>
      </div>
    );
  }

  // Show errors or missing data
  if (status.missingData && status.missingData.length > 0) {
    return (
      <Alert variant="warning" className="mt-2">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Waiting for report data</AlertTitle>
        <AlertDescription className="text-xs">
          {status.missingData[0]}
        </AlertDescription>
      </Alert>
    );
  }

  return null;
};

export default ReportStatusInfo;
