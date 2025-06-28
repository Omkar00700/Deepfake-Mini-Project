
import { AlertCircle } from "lucide-react";

interface FileValidationErrorProps {
  error: string | null;
}

const FileValidationError = ({ error }: FileValidationErrorProps) => {
  if (!error) return null;
  
  return (
    <div className="mt-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md flex items-start">
      <AlertCircle className="h-5 w-5 text-destructive shrink-0 mr-2 mt-0.5" />
      <p className="text-sm text-destructive">{error}</p>
    </div>
  );
};

export default FileValidationError;
