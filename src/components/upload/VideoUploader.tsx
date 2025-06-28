
import { useState, useRef, ChangeEvent, DragEvent } from "react";
import { Video } from "lucide-react";
import UploadDropzone from "./UploadDropzone";
import FilePreview from "./FilePreview";
import { validateVideoFile } from "@/services/validation";

interface VideoUploaderProps {
  onFileSelect: (file: File) => void;
  onUpload: () => void;
  isProcessing: boolean;
  selectedFile: File | null;
  onClearSelection: () => void;
}

const VideoUploader = ({
  onFileSelect,
  onUpload,
  isProcessing,
  selectedFile,
  onClearSelection
}: VideoUploaderProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [previewVideo, setPreviewVideo] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        processFile(file);
      }
    }
  };

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      processFile(file);
    }
  };

  const processFile = (file: File) => {
    const validation = validateVideoFile(file);
    if (!validation.valid) return;

    onFileSelect(file);
    setPreviewVideo(URL.createObjectURL(file));
  };

  const clearSelection = () => {
    if (previewVideo) {
      URL.revokeObjectURL(previewVideo);
      setPreviewVideo(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    onClearSelection();
  };

  return (
    <div>
      <UploadDropzone
        isDragging={isDragging}
        hasFile={!!previewVideo}
        fileInputRef={fileInputRef}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        isProcessing={isProcessing}
        accept="video/mp4,video/quicktime,video/x-msvideo,video/webm"
        icon={<Video className="h-4 w-4 mr-2" />}
        title="Upload a video"
        browseBtnText="Browse Files"
      >
        {previewVideo && selectedFile && (
          <FilePreview
            preview={previewVideo}
            fileName={selectedFile.name}
            isProcessing={isProcessing}
            onClear={clearSelection}
            onAnalyze={onUpload}
            isVideo={true}
          />
        )}
      </UploadDropzone>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileSelect}
        accept="video/mp4,video/quicktime,video/x-msvideo,video/webm"
        className="hidden"
      />
    </div>
  );
};

export default VideoUploader;
