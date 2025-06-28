
import { DetectionResult, ApiStatus } from "@/types";

// Simulates a delay for async operations
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Mock database in memory
let mockDetectionHistory: DetectionResult[] = [];

// Generate a random probability between 0.1 and 0.9
const getRandomProbability = () => {
  return Math.round((0.1 + Math.random() * 0.8) * 100) / 100;
};

// Generate a random confidence value
const getRandomConfidence = () => {
  return Math.round((0.6 + Math.random() * 0.3) * 100) / 100;
};

// Generate fake regions for face detection
const generateFakeRegions = (count = 1) => {
  const regions = [];
  for (let i = 0; i < count; i++) {
    regions.push({
      x: Math.round(Math.random() * 80),
      y: Math.round(Math.random() * 80),
      width: Math.round(Math.random() * 100) + 100,
      height: Math.round(Math.random() * 100) + 100,
      confidence: getRandomConfidence(),
      label: "face"
    });
  }
  return regions;
};

// Simulate the /api/status endpoint
export const mockCheckApiStatus = async (): Promise<ApiStatus> => {
  await delay(500); // Simulate network delay
  
  return {
    status: 'online',
    message: 'Mock API is running (simulation)',
    version: '1.0.0-mock',
    database: {
      connected: true,
      type: 'SQLite (Mock)',
      version: '3.0.0-simulated'
    },
    config: {
      using_supabase: false
    }
  };
};

// Simulate image upload and detection
export const mockUploadImage = async (file: File) => {
  await delay(1500); // Simulate processing time
  
  const probability = getRandomProbability();
  const confidence = getRandomConfidence();
  const regions = generateFakeRegions(probability > 0.5 ? 2 : 1);
  
  const result: DetectionResult = {
    id: Date.now(),
    imageName: file.name,
    probability,
    confidence,
    timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19),
    detectionType: 'image',
    processingTime: Math.round(Math.random() * 800) + 400,
    regions
  };
  
  // Save to mock history
  mockDetectionHistory.unshift(result);
  
  return {
    success: true,
    message: 'Image processed successfully (mock)',
    result,
    statusCode: 200
  };
};

// Simulate video upload and detection
export const mockUploadVideo = async (file: File) => {
  await delay(3000); // Simulate longer processing time for video
  
  const frameCount = Math.round(Math.random() * 100) + 50;
  const probability = getRandomProbability();
  const confidence = getRandomConfidence();
  const regions = generateFakeRegions(probability > 0.5 ? 3 : 2);
  
  const result: DetectionResult = {
    id: Date.now(),
    imageName: file.name,
    probability,
    confidence,
    timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19),
    detectionType: 'video',
    frameCount,
    processingTime: Math.round(Math.random() * 2000) + 1000,
    regions
  };
  
  // Save to mock history
  mockDetectionHistory.unshift(result);
  
  return {
    success: true,
    message: 'Video processed successfully (mock)',
    result,
    statusCode: 200
  };
};

// Simulate getting detection history
export const mockGetHistory = async (): Promise<DetectionResult[]> => {
  await delay(300);
  return [...mockDetectionHistory];
};
