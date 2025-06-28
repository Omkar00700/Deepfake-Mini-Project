
import { ReactNode } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FilePreviewProps {
  preview: string;
  fileName: string;
  isProcessing: boolean;
  onClear: () => void;
  onAnalyze: () => void;
  isVideo: boolean;
  children?: ReactNode;
}

const FilePreview = ({
  preview,
  fileName,
  isProcessing,
  onClear,
  onAnalyze,
  isVideo,
  children
}: FilePreviewProps) => {
  return (
    <div className="relative">
      <Button
        variant="outline"
        size="icon"
        className="absolute top-0 right-0 z-10 rounded-full bg-background/80 backdrop-blur-sm hover:bg-destructive hover:text-destructive-foreground"
        onClick={onClear}
        disabled={isProcessing}
      >
        <X className="h-4 w-4" />
      </Button>
      <div className="flex flex-col items-center">
        <div className="relative w-full max-w-xs mx-auto rounded-lg overflow-hidden shadow-lg mb-4">
          {isVideo ? (
            <video
              src={preview}
              controls
              className="w-full h-auto"
            />
          ) : (
            <img
              src={preview}
              alt="Preview"
              className="w-full h-auto object-contain"
            />
          )}
        </div>
        <p className="text-sm text-muted-foreground mb-4">
          {fileName}
        </p>
        {isVideo && (
          <p className="text-xs text-amber-500 mb-4">
            Video analysis may take longer due to frame processing
          </p>
        )}
        {children}
        <Button
          onClick={onAnalyze}
          disabled={isProcessing}
          className="relative overflow-hidden group"
        >
          <span className="relative z-10">
            {isProcessing ? "Processing..." : isVideo ? "Analyze Video" : "Analyze Image"}
          </span>
          <span className="absolute inset-0 bg-gradient-to-r from-primary via-accent to-primary bg-[length:200%_100%] group-hover:animate-gradient" />
        </Button>
      </div>
    </div>
  );
};

export default FilePreview;
