import { useRef, useEffect, useState } from "react";
import { DetectionResult } from "@/types";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import ConfidenceMeter from "./result/ConfidenceMeter";
import { motion } from "framer-motion";
import FeedbackForm from "./result/FeedbackForm";
import { Info } from "lucide-react";
import DetectionSummary from "./result/DetectionSummary";
import UncertaintyVisualizer from "./result/UncertaintyVisualizer";
import ReportDownloadButton from "./result/ReportDownloadButton";
import { logReportGeneration } from "@/utils/debug-logger";

interface ResultDisplayProps {
  result: DetectionResult | null;
  onReset: () => void;
}

const ResultDisplay = ({ result, onReset }: ResultDisplayProps) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [isReportReady, setIsReportReady] = useState(false);

  // Check if report is ready when result changes
  useEffect(() => {
    if (result) {
      // Check report readiness based on result data
      const hasRequiredData = 
        result.probability !== undefined && 
        result.id !== undefined &&
        result.processingTime !== undefined;
      
      const isExplicitlyReady = 
        result.report_ready === true || 
        (result.report_status?.ready === true);
      
      // If we have explicit readiness flag, use that
      if (isExplicitlyReady) {
        setIsReportReady(true);
        logReportGeneration("report ready from flag", { id: result.id });
        return;
      }
      
      // Otherwise check for required data
      const hasSufficientFrames = 
        result.detectionType !== 'video' || 
        (result.frameCount !== undefined && result.frameCount >= 5);
      
      const hasProcessedRegions = 
        result.regions?.length && 
        result.regions.some(r => r.metadata?.processing_metrics);
      
      // Determine if we have enough data for report
      const readyForReport = 
        hasRequiredData && hasSufficientFrames && hasProcessedRegions;
      
      setIsReportReady(readyForReport);
      
      if (readyForReport) {
        logReportGeneration("report ready from data check", { 
          id: result.id,
          has_required_data: hasRequiredData,
          has_sufficient_frames: hasSufficientFrames,
          has_processed_regions: hasProcessedRegions
        });
      }
    } else {
      setIsReportReady(false);
    }
  }, [result]);

  useEffect(() => {
    if (result && cardRef.current) {
      cardRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [result]);

  if (!result) return null;

  const isDeepfake = result.probability > 0.5;

  return (
    <motion.div
      ref={cardRef}
      className="w-full max-w-2xl mx-auto mt-10"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="relative overflow-hidden backdrop-blur-sm border glass-panel">
        <div
          className={`absolute inset-0 opacity-5 ${
            isDeepfake ? "bg-destructive" : "bg-emerald-500"
          }`}
        ></div>
        <CardContent className="pt-6 pb-8 px-8">
          <div className="flex items-center mb-4">
            <h3 className="text-xl font-semibold">
              Analysis Results
            </h3>
            <div className="ml-auto flex items-center gap-2">
              {feedbackSubmitted ? (
                <span className="text-sm text-muted-foreground">
                  Thank you for your feedback!
                </span>
              ) : (
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={() => setShowFeedback(!showFeedback)}
                  className="text-sm"
                >
                  <Info className="mr-1 h-4 w-4" />
                  {showFeedback ? "Hide Feedback" : "Provide Feedback"}
                </Button>
              )}
            </div>
          </div>

          <div className="flex flex-col md:flex-row gap-6">
            <div className="flex-1">
              <DetectionSummary result={result} />
              
              <div className="mt-6">
                <UncertaintyVisualizer result={result} />
              </div>
              
              {showFeedback && !feedbackSubmitted && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-4 pt-4 border-t"
                >
                  <FeedbackForm 
                    detectionId={result.id ? result.id.toString() : "unknown"} 
                    predictedLabel={isDeepfake ? "deepfake" : "real"}
                    confidence={result.confidence || 0}
                    onSubmitSuccess={() => setFeedbackSubmitted(true)}
                  />
                </motion.div>
              )}
            </div>

            <div className="flex flex-col items-center gap-4 min-w-[160px]">
              <ConfidenceMeter 
                probability={result.probability} 
                confidence={result.confidence} 
                uncertainty={result.uncertainty}
              />
              
              <div className="w-full space-y-2">
                <ReportDownloadButton 
                  result={result}
                  isReportReady={isReportReady}
                />
                
                <Button
                  variant="outline"
                  onClick={onReset}
                  className="w-full"
                >
                  Analyze Another
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default ResultDisplay;
