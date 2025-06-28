
import { DetectionResult } from "@/types";
import { logReportGeneration } from "@/utils/debug-logger";
import EnhancedReportDownloadButton from "../report/EnhancedReportDownloadButton";

interface ReportDownloadButtonProps {
  result: DetectionResult;
  isReportReady?: boolean;
}

/**
 * Enhanced component for downloading detection result reports
 * Uses the new EnhancedReportDownloadButton component
 */
export const ReportDownloadButton = ({ 
  result, 
  isReportReady: initialReportReady = false
}: ReportDownloadButtonProps) => {
  // Log when the component is rendered
  console.log("Rendering ReportDownloadButton for result:", result?.id);
  logReportGeneration("report button rendered", { 
    result_id: result.id,
    initial_ready: initialReportReady
  });
  
  return (
    <EnhancedReportDownloadButton
      result={result}
      initialReady={initialReportReady}
      compact={true}
    />
  );
};

export default ReportDownloadButton;
