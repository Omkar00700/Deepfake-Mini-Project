
import { DetectionResult } from "@/types";
import { API_URL, isUsingMockApi } from "./config";
import { logReportGeneration } from "@/utils/debug-logger";
import { toast } from "sonner";
import { saveFile, createJsonBlob, createCsvBlob, createHtmlBlob } from "@/utils/file-utils";

/**
 * Check if a report is ready for the specified detection result
 */
export async function checkReportStatus(detectionId: string | number) {
  try {
    console.log("Checking report status for ID:", detectionId);
    
    if (isUsingMockApi()) {
      // For testing, randomly decide if report is ready
      const mockReady = Math.random() > 0.3;
      const mockProgress = mockReady ? 100 : Math.floor(Math.random() * 80) + 10;
      
      console.log("Using mock API for report status:", {
        ready: mockReady,
        progress: mockProgress
      });
      
      return {
        ready: mockReady,
        progress: mockProgress,
        missingData: mockReady ? [] : ['Mock API: Report still generating']
      };
    }
    
    const url = `${API_URL}/report-status/${detectionId}`;
    console.log("Fetching report status from:", url);
    
    const response = await fetch(url);
    
    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = errorData.error || `Failed to check report status: ${response.status}`;
      console.error("Report status error:", errorMessage);
      throw new Error(errorMessage);
    }
    
    const data = await response.json();
    console.log("Report status response:", data);
    
    return {
      ready: data.ready === true,
      progress: data.progress,
      missingData: data.missing_data || []
    };
  } catch (error) {
    console.error("Error checking report status:", error);
    logReportGeneration("status check error", {
      detection_id: detectionId,
      error: String(error)
    });
    
    // For better UX, we'll return a structure that indicates the report is not ready
    return {
      ready: false,
      progress: 0,
      missingData: [
        error instanceof Error ? error.message : 'Error communicating with report service'
      ]
    };
  }
}

/**
 * Download a report from the API with retry logic
 */
export async function downloadApiReport(detectionId: string | number, format: 'pdf' | 'json' | 'csv' | 'html' = 'pdf'): Promise<boolean> {
  try {
    // Use our new API endpoint for reports
    const url = `${API_URL}/api/report/${detectionId}?format=${format}`;
    console.log("Attempting to download report from:", url);
    
    logReportGeneration("download requested", {
      detection_id: detectionId,
      format,
      url
    });
    
    const response = await fetch(url);
    
    if (!response.ok) {
      // Try to get error message from JSON response
      const errorText = await response.text();
      let errorMessage = `Failed to download report: ${response.status}`;
      console.error("Download failed with status:", response.status);
      console.log("Error response body:", errorText);
      
      try {
        const errorData = JSON.parse(errorText);
        errorMessage = errorData.error || errorData.message || errorMessage;
      } catch (e) {
        // If parsing fails, use the raw text
        if (errorText && errorText.length < 100) {
          errorMessage = errorText;
        }
      }
      
      throw new Error(errorMessage);
    }
    
    // Check content type to determine how to handle the response
    const contentType = response.headers.get('content-type');
    console.log("Response content-type:", contentType);
    
    if (contentType && contentType.includes('application/json')) {
      // This is a JSON response, not a file download
      const data = await response.json();
      console.log("JSON response for report download:", data);
      
      if (!data.success) {
        throw new Error(data.error || data.message || 'Failed to generate report');
      }
      
      // Check if we have a report URL to download
      if (data.report_url) {
        console.log("Following report URL to download file:", data.report_url);
        // Follow the URL to download the actual file
        const fileResponse = await fetch(data.report_url);
        
        if (!fileResponse.ok) {
          throw new Error(`Failed to download report file: ${fileResponse.status}`);
        }
        
        const blob = await fileResponse.blob();
        console.log("Report blob received, size:", blob.size, "type:", blob.type);
        
        if (blob.size === 0) {
          throw new Error('Received empty blob from server');
        }
        
        const success = saveFile(blob, `deepdefend-report-${detectionId}.${format}`);
        
        if (success) {
          logReportGeneration("download success", {
            detection_id: detectionId,
            format,
            from_url: true
          });
          return true;
        } else {
          throw new Error('Failed to save the report file');
        }
      }
      
      // If we have raw data, convert it to a Blob and download
      if (data.data) {
        console.log("Using raw data from response to create blob");
        
        const content = typeof data.data === 'string' 
          ? data.data 
          : JSON.stringify(data.data, null, 2);
        
        const blob = new Blob([content], {
          type: data.contentType || 'application/octet-stream'
        });
        
        console.log("Created blob from data, size:", blob.size);
        
        if (blob.size === 0) {
          throw new Error('Generated empty blob from API data');
        }
        
        const success = saveFile(blob, `deepdefend-report-${detectionId}.${format}`);
        
        if (success) {
          logReportGeneration("download success", {
            detection_id: detectionId,
            format,
            from_data: true
          });
          return true;
        } else {
          throw new Error('Failed to save the report file');
        }
      }
      
      throw new Error('Report API returned success but no downloadable content');
    } else {
      // This is a direct file download
      console.log("Direct file download detected");
      
      const blob = await response.blob();
      console.log("Report blob received, size:", blob.size, "type:", blob.type);
      
      if (blob.size === 0) {
        throw new Error('Received empty blob from server');
      }
      
      const success = saveFile(blob, `deepdefend-report-${detectionId}.${format}`);
      
      if (success) {
        logReportGeneration("download success", {
          detection_id: detectionId,
          format,
          direct_download: true
        });
        return true;
      } else {
        throw new Error('Failed to save the report file');
      }
    }
  } catch (error) {
    console.error("Error downloading report:", error);
    logReportGeneration("download error", {
      detection_id: detectionId,
      format,
      error: String(error)
    });
    
    throw error;
  }
}

/**
 * Generate a detailed report from the detection result
 * This is a client-side fallback when API report generation fails
 */
export async function generateDetailedReport(
  result: DetectionResult, 
  format: 'pdf' | 'json' | 'csv' = 'pdf', 
  diagnosticData?: any
) {
  try {
    console.log("Starting local report generation, format:", format);
    logReportGeneration("local generation started", {
      result_id: result.id,
      format,
      has_diagnostic_data: !!diagnosticData
    });
    
    if (format === 'pdf') {
      // Generate PDF with the detection result and any diagnostic data
      console.log("Generating PDF report");
      const blob = await createPDFReport(result, diagnosticData);
      console.log("PDF blob created, size:", blob.size, "type:", blob.type);
      
      if (blob.size === 0) {
        throw new Error('Generated empty PDF blob');
      }
      
      saveFile(blob, `deepdefend-report-${result.id || 'local'}.pdf`);
    } else if (format === 'json') {
      // Create a comprehensive JSON report
      console.log("Generating JSON report");
      const reportData = {
        timestamp: new Date().toISOString(),
        detection_result: result,
        diagnostic_data: diagnosticData || null,
        report_type: 'client_generated',
        schema_version: '1.0'
      };
      
      const blob = createJsonBlob(reportData);
      console.log("JSON blob created, size:", blob.size);
      
      saveFile(blob, `deepdefend-report-${result.id || 'local'}.json`);
    } else if (format === 'csv') {
      // Create a simple CSV with key metrics
      console.log("Generating CSV report");
      
      let csvContent = "Timestamp,Detection Type,Image Name,Probability,Confidence,Processing Time\n";
      csvContent += `${result.timestamp},${result.detectionType},${result.imageName},`;
      csvContent += `${result.probability},${result.confidence || 'N/A'},${result.processingTime || 'N/A'}\n`;
      
      // Add frame data for videos if available
      if (result.detectionType === 'video' && diagnosticData?.temporal_analysis?.frame_predictions) {
        csvContent += "\nFrame Analysis\n";
        csvContent += "Frame Number,Probability,Confidence\n";
        
        diagnosticData.temporal_analysis.frame_predictions.forEach((frame: any) => {
          csvContent += `${frame.frame},${frame.probability},${frame.confidence || 'N/A'}\n`;
        });
      }
      
      const blob = createCsvBlob(csvContent);
      console.log("CSV blob created, size:", blob.size);
      
      saveFile(blob, `deepdefend-report-${result.id || 'local'}.csv`);
    }
    
    logReportGeneration("local generation success", {
      result_id: result.id,
      format
    });
    
    console.log("Local report generation completed successfully");
    return true;
  } catch (error) {
    console.error("Error generating local report:", error);
    logReportGeneration("local generation error", {
      result_id: result?.id,
      format,
      error: String(error)
    });
    
    toast.error(`Error generating ${format.toUpperCase()} report: ${error instanceof Error ? error.message : 'Unknown error'}`);
    
    throw error;
  }
}

/**
 * Generate a PDF file (placeholder implementation)
 * In a real app, this would use a proper PDF generation library
 */
export async function createPDFReport(result: DetectionResult, diagnosticData?: any): Promise<Blob> {
  // This is a simplified version - in a real app you'd use a PDF library
  // For this example, we'll create an HTML string and convert it to a Blob
  const title = result.probability > 0.5 ? "Deepfake Detected" : "Authentic Media Detected";
  const probability = (result.probability * 100).toFixed(1);
  const confidence = result.confidence ? (result.confidence * 100).toFixed(1) : 'N/A';
  
  let html = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>DeepDefend Analysis Report</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: ${result.probability > 0.5 ? '#e53935' : '#43a047'}; }
        .section { margin-bottom: 20px; }
        .label { font-weight: bold; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
      </style>
    </head>
    <body>
      <h1>DeepDefend Analysis Report: ${title}</h1>
      
      <div class="section">
        <h2>Detection Summary</h2>
        <table>
          <tr>
            <th>Field</th>
            <th>Value</th>
          </tr>
          <tr>
            <td>Detection Type</td>
            <td>${result.detectionType}</td>
          </tr>
          <tr>
            <td>Media Name</td>
            <td>${result.imageName}</td>
          </tr>
          <tr>
            <td>Timestamp</td>
            <td>${result.timestamp}</td>
          </tr>
          <tr>
            <td>Deepfake Probability</td>
            <td>${probability}%</td>
          </tr>
          <tr>
            <td>Confidence</td>
            <td>${confidence}%</td>
          </tr>
          <tr>
            <td>Processing Time</td>
            <td>${result.processingTime || 'N/A'} ms</td>
          </tr>
          <tr>
            <td>Model</td>
            <td>${result.model || 'Standard detection model'}</td>
          </tr>
        </table>
      </div>
  `;
  
  // Add video-specific information
  if (result.detectionType === 'video') {
    html += `
      <div class="section">
        <h2>Video Analysis</h2>
        <table>
          <tr>
            <th>Metric</th>
            <th>Value</th>
          </tr>
          <tr>
            <td>Frames Analyzed</td>
            <td>${result.frameCount || 'N/A'}</td>
          </tr>
    `;
    
    if (diagnosticData?.temporal_analysis) {
      html += `
          <tr>
            <td>Temporal Consistency</td>
            <td>${(diagnosticData.temporal_analysis.frame_consistency * 100).toFixed(1)}%</td>
          </tr>
      `;
    }
    
    html += `
        </table>
      </div>
    `;
    
    // Add frame predictions if available
    if (diagnosticData?.temporal_analysis?.frame_predictions?.length) {
      html += `
        <div class="section">
          <h2>Frame-by-Frame Analysis</h2>
          <table>
            <tr>
              <th>Frame</th>
              <th>Probability</th>
              <th>Confidence</th>
            </tr>
      `;
      
      diagnosticData.temporal_analysis.frame_predictions.forEach((frame: any) => {
        html += `
          <tr>
            <td>${frame.frame}</td>
            <td>${(frame.probability * 100).toFixed(1)}%</td>
            <td>${frame.confidence ? (frame.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
          </tr>
        `;
      });
      
      html += `
          </table>
        </div>
      `;
    }
  }
  
  // Add model details if available
  if (diagnosticData?.model_outputs?.length) {
    html += `
      <div class="section">
        <h2>Model Details</h2>
        <table>
          <tr>
            <th>Model</th>
            <th>Raw Output</th>
            <th>Calibrated Output</th>
            <th>Confidence</th>
            <th>Weight</th>
          </tr>
    `;
    
    diagnosticData.model_outputs.forEach((model: any) => {
      html += `
        <tr>
          <td>${model.model}</td>
          <td>${(model.raw_output * 100).toFixed(1)}%</td>
          <td>${model.calibrated_output ? (model.calibrated_output * 100).toFixed(1) + '%' : 'N/A'}</td>
          <td>${(model.confidence * 100).toFixed(1)}%</td>
          <td>${model.weight.toFixed(2)}</td>
        </tr>
      `;
    });
    
    html += `
        </table>
      </div>
    `;
  }
  
  // Add technical details
  html += `
      <div class="section">
        <h2>Technical Information</h2>
        <p>This report was generated client-side by DeepDefend.</p>
        <p>Report ID: ${result.id || 'local'}-${Date.now()}</p>
        <p>Generation time: ${new Date().toISOString()}</p>
      </div>
    </body>
    </html>
  `;
  
  // Create a proper HTML blob
  return createHtmlBlob(html);
}
