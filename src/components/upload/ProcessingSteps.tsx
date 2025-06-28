
import { CheckCircle, Clock, AlertCircle } from "lucide-react";

interface ProcessingStepsProps {
  progress: number;
  type: "image" | "video";
}

const ProcessingSteps = ({ progress, type }: ProcessingStepsProps) => {
  const steps = type === "video" 
    ? ["Upload", "Extract Frames", "Face Detection", "Analysis", "Complete"] 
    : ["Upload", "Preprocessing", "Face Detection", "Analysis", "Complete"];
  
  const currentStep = 
    progress < 20 ? 0 :
    progress < 40 ? 1 :
    progress < 60 ? 2 :
    progress < 90 ? 3 : 4;
  
  return (
    <div className="flex justify-between w-full mt-2 px-1">
      {steps.map((step, index) => (
        <div key={step} className="flex flex-col items-center">
          <div className={`
            w-6 h-6 rounded-full flex items-center justify-center
            ${index < currentStep ? 'bg-primary text-primary-foreground' : 
              index === currentStep ? 'border-2 border-primary' : 
              'border-2 border-muted'}
          `}>
            {index < currentStep ? (
              <CheckCircle className="h-3 w-3" />
            ) : index === currentStep ? (
              <Clock className="h-3 w-3 text-primary animate-pulse" />
            ) : (
              <span className="text-xs text-muted-foreground">{index + 1}</span>
            )}
          </div>
          <span className="text-[10px] mt-1 text-muted-foreground font-medium">
            {step}
          </span>
        </div>
      ))}
    </div>
  );
};

export default ProcessingSteps;
