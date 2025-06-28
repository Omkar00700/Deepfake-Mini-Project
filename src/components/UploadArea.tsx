import { useState } from "react";
import { uploadImage, uploadVideo } from "@/services/api";
import { DetectionResult } from "@/types";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ImageIcon, Video } from "lucide-react";
import { toast } from "sonner";
import ImageUploader from "./upload/ImageUploader";
import VideoUploader from "./upload/VideoUploader";
import ProgressIndicator from "./upload/ProgressIndicator";
import ProcessingSteps from "./upload/ProcessingSteps";
import FileValidationError from "./upload/FileValidationError";

interface UploadAreaProps {
  onUploadComplete: (result: DetectionResult) => void;
  isProcessing: boolean;
  setIsProcessing: (value: boolean) => void;
}

const UploadArea = ({ onUploadComplete, isProcessing, setIsProcessing }: UploadAreaProps) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("image");

  const clearSelection = () => {
    setSelectedFile(null);
    setProgress(0);
  };

  const updateProgress = (newProgress: number) => {
    setProgress(newProgress);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);
    const isVideo = activeTab === "video";
    
    try {
      let response;
      if (isVideo) {
        response = await uploadVideo(selectedFile);
      } else {
        // For images, simulate a smoother progress since we don't get real-time updates
        const interval = setInterval(() => {
          setProgress((prev) => {
            if (prev >= 90) {
              clearInterval(interval);
              return 90;
            }
            // Increase by random amount between 5-15% for more natural feel
            return Math.min(90, prev + Math.floor(Math.random() * 10) + 5);
          });
        }, 300);
        
        response = await uploadImage(selectedFile);
        clearInterval(interval);
      }
      
      setProgress(100);
      
      if (response.success && response.result) {
        setTimeout(() => {
          toast.success(`${isVideo ? 'Video' : 'Image'} analyzed successfully`);
          onUploadComplete(response.result);
        }, 500);
      } else {
        throw new Error(response.error || `Failed to analyze ${isVideo ? 'video' : 'image'}`);
      }
    } catch (error) {
      setProgress(0);
      setError(error instanceof Error ? error.message : "An unknown error occurred");
      toast.error(`Failed to analyze ${isVideo ? 'video' : 'image'}`);
    } finally {
      setTimeout(() => {
        setIsProcessing(false);
      }, 500);
    }
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setError(null);
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <Tabs 
        defaultValue="image" 
        value={activeTab} 
        onValueChange={setActiveTab}
        className="mb-6"
      >
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="image" disabled={isProcessing}>
            <ImageIcon className="h-4 w-4 mr-2" />
            Image Detection
          </TabsTrigger>
          <TabsTrigger value="video" disabled={isProcessing}>
            <Video className="h-4 w-4 mr-2" />
            Video Detection
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="image" className="mt-4">
          <ImageUploader
            onFileSelect={handleFileSelect}
            onUpload={handleUpload}
            isProcessing={isProcessing}
            selectedFile={selectedFile}
            onClearSelection={clearSelection}
          />
        </TabsContent>
        
        <TabsContent value="video" className="mt-4">
          <VideoUploader
            onFileSelect={handleFileSelect}
            onUpload={handleUpload}
            isProcessing={isProcessing}
            selectedFile={selectedFile}
            onClearSelection={clearSelection}
          />
        </TabsContent>
      </Tabs>
      
      {isProcessing && (
        <>
          <ProgressIndicator 
            isProcessing={isProcessing} 
            progress={progress} 
            type={activeTab as "image" | "video"} 
          />
          <ProcessingSteps 
            progress={progress} 
            type={activeTab as "image" | "video"} 
          />
        </>
      )}

      <FileValidationError error={error} />
    </div>
  );
};

export default UploadArea;
