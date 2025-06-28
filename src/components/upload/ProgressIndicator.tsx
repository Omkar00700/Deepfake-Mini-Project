
import { Progress } from "@/components/ui/progress";
import { Loader2, CheckCircle } from "lucide-react";
import { motion } from "framer-motion";

interface ProgressIndicatorProps {
  isProcessing: boolean;
  progress: number;
  type: "image" | "video";
}

const ProgressIndicator = ({ isProcessing, progress, type }: ProgressIndicatorProps) => {
  if (!isProcessing) return null;
  
  const getProgressMessage = () => {
    if (progress < 10) return "Initializing...";
    if (progress < 30) return type === "video" ? "Extracting frames..." : "Preprocessing image...";
    if (progress < 50) return "Detecting faces...";
    if (progress < 70) return "Analyzing content...";
    if (progress < 90) return "Running deepfake detection...";
    if (progress < 100) return "Finalizing results...";
    return "Analysis complete!";
  };
  
  return (
    <motion.div 
      className="mt-6 p-4 border rounded-lg bg-background shadow-sm"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center mb-2">
        {progress < 100 ? (
          <Loader2 className="h-4 w-4 mr-2 animate-spin text-primary" />
        ) : (
          <CheckCircle className="h-4 w-4 mr-2 text-primary" />
        )}
        <p className="text-sm font-medium">
          {getProgressMessage()}
        </p>
        <span className="ml-auto text-xs font-medium">{progress}%</span>
      </div>
      
      <Progress value={progress} className="h-2" />
      
      <p className="text-xs mt-2 text-muted-foreground">
        {type === "video" 
          ? "Video analysis may take longer due to frame-by-frame processing"
          : "Image analysis typically completes in a few seconds"}
      </p>
    </motion.div>
  );
};

export default ProgressIndicator;
