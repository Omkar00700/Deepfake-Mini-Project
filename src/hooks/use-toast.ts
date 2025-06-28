
import { toast } from "sonner";

export { toast };

// Simple wrapper for consistency with previous API
export function useToast() {
  return {
    toast,
    dismiss: toast.dismiss,
  };
}
