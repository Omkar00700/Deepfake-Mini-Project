
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { submitFeedback } from "@/services/feedback-service";
import { toast } from "sonner";

interface FeedbackFormProps {
  detectionId: string;
  predictedLabel: "real" | "deepfake";
  confidence: number;
  onSubmitSuccess: () => void;
}

const FeedbackForm = ({ detectionId, predictedLabel, confidence, onSubmitSuccess }: FeedbackFormProps) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [actualLabel, setActualLabel] = useState<"real" | "deepfake">(predictedLabel);
  const [additionalComments, setAdditionalComments] = useState("");
  const [region, setRegion] = useState<string>("");
  
  const handleSubmit = async () => {
    if (isCorrect === null) {
      toast.error("Please indicate if the detection was correct");
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      const success = await submitFeedback({
        detection_id: detectionId,
        correct: isCorrect,
        actual_label: actualLabel,
        confidence: confidence,
        metadata: {
          comments: additionalComments,
          region: region || undefined
        }
      });
      
      if (success) {
        toast.success("Thank you for your feedback!");
        onSubmitSuccess();
      } else {
        toast.error("Failed to submit feedback. Please try again.");
      }
    } catch (error) {
      toast.error("An error occurred while submitting feedback");
      console.error("Feedback submission error:", error);
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-sm font-medium mb-2">Help us improve our detection</h4>
        <p className="text-sm text-muted-foreground mb-4">
          Your feedback helps us continuously improve our deepfake detection system.
        </p>
      </div>
      
      <div className="space-y-3">
        <div>
          <p className="text-sm font-medium mb-2">
            Was this detection result correct?
          </p>
          <RadioGroup
            onValueChange={(value) => setIsCorrect(value === "yes")}
            className="flex space-x-4"
          >
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="yes" id="correct-yes" />
              <Label htmlFor="correct-yes">Yes</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="no" id="correct-no" />
              <Label htmlFor="correct-no">No</Label>
            </div>
          </RadioGroup>
        </div>
        
        {isCorrect === false && (
          <div>
            <p className="text-sm font-medium mb-2">
              What is the correct classification?
            </p>
            <RadioGroup
              defaultValue={actualLabel}
              onValueChange={(value) => setActualLabel(value as "real" | "deepfake")}
              className="flex space-x-4"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="real" id="label-real" />
                <Label htmlFor="label-real">Real</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="deepfake" id="label-fake" />
                <Label htmlFor="label-fake">Deepfake</Label>
              </div>
            </RadioGroup>
          </div>
        )}
        
        <div>
          <p className="text-sm font-medium mb-2">
            Geographic region the subject is from (optional):
          </p>
          <RadioGroup
            onValueChange={setRegion}
            className="grid grid-cols-2 gap-2"
          >
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="south_asia" id="region-south-asia" />
              <Label htmlFor="region-south-asia">South Asia (India, etc.)</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="east_asia" id="region-east-asia" />
              <Label htmlFor="region-east-asia">East Asia</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="western" id="region-western" />
              <Label htmlFor="region-western">Western</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="middle_east" id="region-middle-east" />
              <Label htmlFor="region-middle-east">Middle East</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="african" id="region-african" />
              <Label htmlFor="region-african">African</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="other" id="region-other" />
              <Label htmlFor="region-other">Other</Label>
            </div>
          </RadioGroup>
        </div>
        
        <div>
          <Label htmlFor="comments" className="text-sm font-medium">
            Additional comments (optional):
          </Label>
          <Textarea
            id="comments"
            placeholder="Any additional details about this detection..."
            value={additionalComments}
            onChange={(e) => setAdditionalComments(e.target.value)}
            className="mt-1"
            rows={3}
          />
        </div>
      </div>
      
      <div className="flex justify-end">
        <Button
          onClick={handleSubmit}
          disabled={isSubmitting || isCorrect === null}
        >
          {isSubmitting ? "Submitting..." : "Submit Feedback"}
        </Button>
      </div>
    </div>
  );
};

export default FeedbackForm;
