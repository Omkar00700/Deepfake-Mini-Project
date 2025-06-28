
import { useState } from "react";
import { DetectionResult, ReportFormat } from "@/types";
import { toast } from "sonner";
import { generateDetailedReport, downloadApiReport } from "@/services/report-service";
import { logReportGeneration } from "@/utils/debug-logger";

/**
 * Custom hook that manages report data generation and downloading
 */
export function useReportData(result: DetectionResult | null) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState<'pdf' | 'json' | 'csv'>('pdf');
  const [downloadAttempts, setDownloadAttempts] = useState(0);
  const MAX_DOWNLOAD_ATTEMPTS = 3;

  // Available report formats
  const reportFormats: ReportFormat[] = [
    { id: 'pdf', label: 'PDF Report', value: 'pdf' },
    { id: 'json', label: 'JSON Data', value: 'json' },
    { id: 'csv', label: 'CSV Format', value: 'csv' }
  ];

  // Handle format selection
  const selectFormat = (format: 'pdf' | 'json' | 'csv') => {
    setSelectedFormat(format);
  };

  // Generate or download report
  const generateReport = async (
    diagnosticData?: any, 
    onDownloadStatusChange?: (isDownloading: boolean) => void
  ) => {
    if (!result) {
      toast.error("No detection result available");
      return false;
    }

    setIsGenerating(true);
    if (onDownloadStatusChange) onDownloadStatusChange(true);

    try {
      logReportGeneration("download initiated", { 
        format: selectedFormat, 
        result_id: result.id,
      });

      // Try API download if we have a valid ID
      if (result.id) {
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
            id: "report-download"
          });
          return true;
        } else if (attempts >= maxAttempts) {
          toast.warning(`API report download failed. Trying local generation...`, {
            id: "report-download"
          });
        }
      }
      
      // Fall back to local report generation
      toast.loading(`Generating ${selectedFormat.toUpperCase()} report locally...`, {
        id: "report-download",
        duration: 10000
      });
      
      await generateDetailedReport(result, selectedFormat, diagnosticData);
      
      toast.success(`${selectedFormat.toUpperCase()} report generated successfully`, {
        id: "report-download"
      });
      return true;
    } catch (error) {
      console.error("Error generating report:", error);
      
      toast.error(`Failed to generate ${selectedFormat.toUpperCase()} report: ${error instanceof Error ? error.message : 'Unknown error'}`, {
        id: "report-download"
      });
      return false;
    } finally {
      setIsGenerating(false);
      if (onDownloadStatusChange) onDownloadStatusChange(false);
    }
  };

  return {
    isGenerating,
    selectedFormat,
    reportFormats,
    downloadAttempts,
    MAX_DOWNLOAD_ATTEMPTS,
    selectFormat,
    generateReport
  };
}

export default useReportData;
