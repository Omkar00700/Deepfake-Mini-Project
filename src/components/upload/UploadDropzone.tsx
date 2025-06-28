
import { DragEvent, useRef, ReactNode } from "react";
import { Upload } from "lucide-react";
import { Button } from "@/components/ui/button";

interface UploadDropzoneProps {
  isDragging: boolean;
  hasFile: boolean;
  fileInputRef: React.RefObject<HTMLInputElement>;
  onDragEnter: (e: DragEvent<HTMLDivElement>) => void;
  onDragLeave: (e: DragEvent<HTMLDivElement>) => void;
  onDragOver: (e: DragEvent<HTMLDivElement>) => void;
  onDrop: (e: DragEvent<HTMLDivElement>) => void;
  isProcessing: boolean;
  accept: string;
  icon: ReactNode;
  title: string;
  browseBtnText: string;
  children?: ReactNode;
}

const UploadDropzone = ({
  isDragging,
  hasFile,
  fileInputRef,
  onDragEnter,
  onDragLeave,
  onDragOver,
  onDrop,
  isProcessing,
  accept,
  icon,
  title,
  browseBtnText,
  children
}: UploadDropzoneProps) => {
  return (
    <div
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
      className={`
        upload-area relative rounded-xl border-2 border-dashed p-8 transition-all duration-300
        ${
          isDragging
            ? "border-primary bg-primary/5 scale-[1.02]"
            : "border-border hover:border-primary/50 hover:bg-secondary/50"
        }
        ${hasFile ? "bg-secondary/30" : ""}
      `}
    >
      <input
        type="file"
        ref={fileInputRef}
        accept={accept}
        className="hidden"
      />

      {!hasFile ? (
        <div className="flex flex-col items-center justify-center py-6">
          <div className="w-16 h-16 mb-4 rounded-full bg-primary/10 flex items-center justify-center">
            {icon || <Upload className="h-8 w-8 text-primary" />}
          </div>
          <h3 className="text-lg font-medium mb-2">{title}</h3>
          <p className="text-muted-foreground text-center mb-6 max-w-md">
            Drag and drop or click below to browse files
          </p>
          <Button
            onClick={() => fileInputRef.current?.click()}
            className="relative overflow-hidden group"
            disabled={isProcessing}
          >
            <span className="relative z-10 flex items-center">
              {icon}
              {browseBtnText}
            </span>
            <span className="absolute inset-0 bg-gradient-to-r from-primary via-accent to-primary bg-[length:200%_100%] group-hover:animate-gradient" />
          </Button>
        </div>
      ) : (
        children
      )}
    </div>
  );
};

export default UploadDropzone;
