
import { DetectionResult } from "@/types";
import { API_URL, isUsingMockApi } from "./config";
import { mockGetHistory } from "./mock-api";

export const getHistory = async (): Promise<DetectionResult[]> => {
  // If we're using the mock API, return mock data
  if (isUsingMockApi()) {
    return await mockGetHistory();
  }

  try {
    const response = await fetch(`${API_URL}/history`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch history (${response.status})`);
    }

    const data = await response.json();
    return data.history as DetectionResult[];
  } catch (error) {
    console.error('History fetch error:', error);
    return [];
  }
};
