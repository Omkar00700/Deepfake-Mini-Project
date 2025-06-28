
import { API_URL, isUsingMockApi } from "./config";
import { handleApiError } from "./error-handler";

interface FeedbackData {
  detection_id: string;
  correct: boolean;
  actual_label: "real" | "deepfake";
  confidence?: number;
  metadata?: {
    comments?: string;
    region?: string;
    [key: string]: any;
  };
}

export const submitFeedback = async (feedbackData: FeedbackData): Promise<boolean> => {
  // If we're using the mock API, just simulate feedback submission
  if (isUsingMockApi()) {
    console.log("Mock API: Submitting feedback", feedbackData);
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 500));
    return true;
  }

  try {
    const response = await fetch(`${API_URL}/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(feedbackData),
    });

    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || `Failed to submit feedback (${response.status})`);
    }

    return data.success;
  } catch (error) {
    handleApiError(error);
    return false;
  }
};

export const getFeedbackStatus = async (): Promise<any> => {
  // If we're using the mock API, return mock status
  if (isUsingMockApi()) {
    return {
      enabled: true,
      retraining_enabled: true,
      samples_collected: 250,
      last_evaluation_time: Date.now() - 86400000, // 1 day ago
      last_retraining_time: Date.now() - 172800000, // 2 days ago
    };
  }

  try {
    const response = await fetch(`${API_URL}/feedback/status`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to get feedback status (${response.status})`);
    }

    return await response.json();
  } catch (error) {
    handleApiError(error);
    return null;
  }
};
