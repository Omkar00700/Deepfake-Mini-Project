
import { useState, useRef, ChangeEvent, DragEvent } from "react";
import { Image as ImageIcon } from "lucide-react";
import UploadDropzone from "./UploadDropzone";
import FilePreview from "./FilePreview";
import { validateImageFile } from "@/services/validation";

interface ImageUploaderProps {
  onFileSelect: (file: File) => void;
  onUpload: () => void;
  isProcessing: boolean;
  selectedFile: File | null;
  onClearSelection: () => void;
}

const ImageUploader = ({
  onFileSelect,
  onUpload,
  isProcessing,
  selectedFile,
  onClearSelection
}: ImageUploaderProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
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
      if (file.type.startsWith('image/')) {
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
    const validation = validateImageFile(file);
    if (!validation.valid) return;

    onFileSelect(file);
    const reader = new FileReader();
    reader.onload = () => {
      setPreviewImage(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const clearSelection = () => {
    setPreviewImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    onClearSelection();
  };

  return (
    <div>
      <UploadDropzone
        isDragging={isDragging}
        hasFile={!!previewImage}
        fileInputRef={fileInputRef}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        isProcessing={isProcessing}
        accept="image/jpeg,image/jpg,image/png"
        icon={<ImageIcon className="h-4 w-4 mr-2" />}
        title="Upload an image"
        browseBtnText="Browse Files"
      >
        {previewImage && selectedFile && (
          <FilePreview
            preview={previewImage}
            fileName={selectedFile.name}
            isProcessing={isProcessing}
            onClear={clearSelection}
            onAnalyze={onUpload}
            isVideo={false}
          />
        )}
      </UploadDropzone>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileSelect}
        accept="image/jpeg,image/jpg,image/png"
        className="hidden"
      />
    </div>
  );
};

export default ImageUploader;
