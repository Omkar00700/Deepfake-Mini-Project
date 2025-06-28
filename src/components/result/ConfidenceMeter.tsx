
import React from 'react';
import { cva } from 'class-variance-authority';

const meterVariants = cva("h-full rounded-full transition-all duration-500", {
  variants: {
    level: {
      low: "bg-emerald-500",
      medium: "bg-yellow-500",
      high: "bg-red-500",
    }
  },
  defaultVariants: {
    level: "medium"
  }
});

interface ConfidenceMeterProps {
  probability: number;
  confidence?: number;
  uncertainty?: number;
}

const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({ 
  probability,
  confidence = 0.5,
  uncertainty
}) => {
  // Determine if it's a deepfake based on probability threshold
  const isDeepfake = probability > 0.5;
  
  // Calculate percentage for display
  const percentage = Math.round(probability * 100);
  
  // Determine level for color
  const getLevel = () => {
    if (isDeepfake) {
      if (probability > 0.8) return "high";
      return "medium";
    } else {
      if (probability < 0.2) return "low";
      return "medium";
    }
  };

  // Format confidence as percentage
  const confidencePercentage = Math.round(confidence * 100);
  
  return (
    <div className="w-full flex flex-col items-center">
      <div className="text-center mb-2">
        <div className="text-sm text-muted-foreground mb-1">Detection Result</div>
        <div className={`text-2xl font-bold ${isDeepfake ? 'text-red-500' : 'text-emerald-500'}`}>
          {isDeepfake ? 'Deepfake' : 'Authentic'}
        </div>
      </div>
      
      <div className="w-32 h-32 relative rounded-full border-8 flex items-center justify-center">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-3xl font-bold">{percentage}%</div>
        </div>
        <svg className="absolute inset-0" width="100%" height="100%" viewBox="0 0 100 100">
          <circle 
            cx="50" 
            cy="50" 
            r="40" 
            fill="none" 
            stroke="#e2e8f0" 
            strokeWidth="8" 
          />
          <circle
            cx="50"
            cy="50"
            r="40"
            fill="none"
            stroke={isDeepfake ? '#ef4444' : '#10b981'}
            strokeWidth="8"
            strokeDasharray={`${percentage * 2.51} 251`}
            strokeDashoffset="0"
            transform="rotate(-90 50 50)"
          />
        </svg>
      </div>
      
      <div className="mt-4 text-center">
        <div className="text-sm text-muted-foreground">Confidence</div>
        <div className="font-medium">{confidencePercentage}%</div>
        
        {uncertainty !== undefined && (
          <div className="mt-1 text-xs text-muted-foreground">
            Uncertainty: {Math.round(uncertainty * 100)}%
          </div>
        )}
      </div>
    </div>
  );
};

export default ConfidenceMeter;
