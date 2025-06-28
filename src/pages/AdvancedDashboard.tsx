import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { API_URL, isUsingMockApi } from '@/services/config';
import { toast } from 'sonner';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import Header from '@/components/Header';
import Footer from '@/components/Footer';

interface ModelPerformance {
  id: string;
  name: string;
  type: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  avg_inference_time: number;
  memory_usage: string;
}

interface DashboardData {
  total_detections: number;
  verdicts: {
    deepfakes: number;
    suspicious: number;
    authentic: number;
  };
  models: Record<string, number>;
  avg_confidence: number;
  avg_processing_time: number;
  recent_detections: any[];
}

interface DetectionSettings {
  ensemble_method: string;
  temporal_analysis: boolean;
  indian_face_enhancement: boolean;
  adversarial_detection: boolean;
  confidence_threshold: number;
  use_quantized_models: boolean;
  parallel_processing: boolean;
  max_workers: number;
}

const COLORS = ['#FF6B6B', '#FFD166', '#06D6A0', '#118AB2', '#073B4C'];

const AdvancedDashboard = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [modelPerformance, setModelPerformance] = useState<Record<string, ModelPerformance>>({});
  const [detectionSettings, setDetectionSettings] = useState<DetectionSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    fetchDashboardData();
    fetchModelPerformance();
    fetchDetectionSettings();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // If using mock API, return mock data
      if (isUsingMockApi()) {
        const mockData: DashboardData = {
          total_detections: 42,
          verdicts: {
            deepfakes: 18,
            suspicious: 12,
            authentic: 12
          },
          models: {
            efficientnet: 20,
            xception: 15,
            indian_specialized: 7
          },
          avg_confidence: 0.78,
          avg_processing_time: 1250,
          recent_detections: [
            {
              id: "mock-1",
              filename: "sample1.jpg",
              probability: 0.85,
              confidence: 0.92,
              model: "efficientnet"
            },
            {
              id: "mock-2",
              filename: "sample2.jpg",
              probability: 0.32,
              confidence: 0.88,
              model: "xception"
            },
            {
              id: "mock-3",
              filename: "sample3.jpg",
              probability: 0.67,
              confidence: 0.75,
              model: "indian_specialized"
            }
          ]
        };
        setDashboardData(mockData);
        setLoading(false);
        return;
      }

      // Fetch dashboard data from API
      const response = await fetch(`${API_URL}/api/dashboard`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to fetch dashboard data: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || "Failed to fetch dashboard data");
      }
      
      setDashboardData(data.dashboard);
    } catch (error) {
      setError(error instanceof Error ? error.message : "An unknown error occurred");
      toast.error("Failed to fetch dashboard data");
    } finally {
      setLoading(false);
    }
  };

  const fetchModelPerformance = async () => {
    try {
      // If using mock API, return mock data
      if (isUsingMockApi()) {
        const mockPerformance: Record<string, ModelPerformance> = {
          efficientnet: {
            id: "efficientnet",
            name: "EfficientNet",
            type: "image",
            accuracy: 0.92,
            precision: 0.90,
            recall: 0.89,
            f1_score: 0.895,
            avg_inference_time: 0.25,
            memory_usage: "120 MB"
          },
          xception: {
            id: "xception",
            name: "Xception",
            type: "image",
            accuracy: 0.94,
            precision: 0.93,
            recall: 0.92,
            f1_score: 0.925,
            avg_inference_time: 0.35,
            memory_usage: "180 MB"
          },
          indian_specialized: {
            id: "indian_specialized",
            name: "Indian Specialized",
            type: "image",
            accuracy: 0.96,
            precision: 0.95,
            recall: 0.94,
            f1_score: 0.945,
            avg_inference_time: 0.40,
            memory_usage: "220 MB"
          },
          temporal_cnn: {
            id: "temporal_cnn",
            name: "Temporal CNN",
            type: "video",
            accuracy: 0.91,
            precision: 0.89,
            recall: 0.90,
            f1_score: 0.895,
            avg_inference_time: 0.75,
            memory_usage: "280 MB"
          },
          audio_analyzer: {
            id: "audio_analyzer",
            name: "Audio Analyzer",
            type: "audio",
            accuracy: 0.88,
            precision: 0.86,
            recall: 0.85,
            f1_score: 0.855,
            avg_inference_time: 0.45,
            memory_usage: "150 MB"
          }
        };
        setModelPerformance(mockPerformance);
        return;
      }

      // Fetch model performance from API
      const response = await fetch(`${API_URL}/api/model-performance`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to fetch model performance: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || "Failed to fetch model performance");
      }
      
      setModelPerformance(data.performance);
    } catch (error) {
      toast.error("Failed to fetch model performance");
    }
  };

  const fetchDetectionSettings = async () => {
    try {
      // If using mock API, return mock data
      if (isUsingMockApi()) {
        const mockSettings: DetectionSettings = {
          ensemble_method: "weighted_average",
          temporal_analysis: true,
          indian_face_enhancement: true,
          adversarial_detection: true,
          confidence_threshold: 0.5,
          use_quantized_models: false,
          parallel_processing: true,
          max_workers: 4
        };
        setDetectionSettings(mockSettings);
        return;
      }

      // Fetch detection settings from API
      const response = await fetch(`${API_URL}/api/detection-settings`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to fetch detection settings: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || "Failed to fetch detection settings");
      }
      
      setDetectionSettings(data.settings);
    } catch (error) {
      toast.error("Failed to fetch detection settings");
    }
  };

  const updateDetectionSettings = async (newSettings: Partial<DetectionSettings>) => {
    try {
      // If using mock API, update local state
      if (isUsingMockApi()) {
        setDetectionSettings(prev => ({
          ...prev!,
          ...newSettings
        }));
        toast.success("Settings updated successfully");
        return;
      }

      // Update detection settings via API
      const response = await fetch(`${API_URL}/api/detection-settings`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newSettings)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to update detection settings: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || "Failed to update detection settings");
      }
      
      setDetectionSettings(data.settings);
      toast.success("Settings updated successfully");
    } catch (error) {
      toast.error("Failed to update detection settings");
    }
  };

  const getVerdictData = () => {
    if (!dashboardData) return [];
    
    return [
      { name: "Deepfakes", value: dashboardData.verdicts.deepfakes },
      { name: "Suspicious", value: dashboardData.verdicts.suspicious },
      { name: "Authentic", value: dashboardData.verdicts.authentic }
    ];
  };

  const getModelUsageData = () => {
    if (!dashboardData) return [];
    
    return Object.entries(dashboardData.models).map(([model, count]) => ({
      name: model,
      count
    }));
  };

  const getModelPerformanceData = () => {
    return Object.values(modelPerformance).map(model => ({
      name: model.name,
      accuracy: model.accuracy,
      precision: model.precision,
      recall: model.recall,
      f1_score: model.f1_score
    }));
  };

  const getModelTimeData = () => {
    return Object.values(modelPerformance).map(model => ({
      name: model.name,
      time: model.avg_inference_time
    }));
  };

  if (loading) {
    return (
      <div className="flex flex-col min-h-screen">
        <Header />
        <main className="flex-1 container mx-auto p-4">
          <div className="flex flex-col items-center justify-center h-full">
            <h2 className="text-2xl font-bold mb-4">Loading Dashboard...</h2>
            <Progress value={50} className="w-[60%] mb-4" />
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col min-h-screen">
        <Header />
        <main className="flex-1 container mx-auto p-4">
          <div className="flex flex-col items-center justify-center h-full">
            <h2 className="text-2xl font-bold mb-4 text-red-500">Error Loading Dashboard</h2>
            <p className="text-gray-700 mb-4">{error}</p>
            <Button onClick={fetchDashboardData}>Retry</Button>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      <main className="flex-1 container mx-auto p-4">
        <h1 className="text-3xl font-bold mb-6">Advanced Dashboard</h1>
        
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="models">Model Performance</TabsTrigger>
            <TabsTrigger value="settings">Detection Settings</TabsTrigger>
            <TabsTrigger value="detections">Recent Detections</TabsTrigger>
          </TabsList>
          
          <TabsContent value="overview">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Total Detections</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">{dashboardData?.total_detections}</div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Deepfakes Detected</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-red-500">{dashboardData?.verdicts.deepfakes}</div>
                  <div className="text-sm text-gray-500">
                    {dashboardData && ((dashboardData.verdicts.deepfakes / dashboardData.total_detections) * 100).toFixed(1)}% of total
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Avg. Confidence</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">{dashboardData && (dashboardData.avg_confidence * 100).toFixed(1)}%</div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Avg. Processing Time</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">{dashboardData && dashboardData.avg_processing_time.toFixed(0)} ms</div>
                </CardContent>
              </Card>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <Card>
                <CardHeader>
                  <CardTitle>Detection Verdicts</CardTitle>
                  <CardDescription>Distribution of detection results</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={getVerdictData()}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          outerRadius={100}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {getVerdictData().map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Model Usage</CardTitle>
                  <CardDescription>Number of detections by model</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={getModelUsageData()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="count" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          <TabsContent value="models">
            <Card>
              <CardHeader>
                <CardTitle>Model Performance Metrics</CardTitle>
                <CardDescription>Accuracy, precision, recall, and F1 score for each model</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getModelPerformanceData()} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" domain={[0, 1]} />
                      <YAxis dataKey="name" type="category" width={150} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="accuracy" fill="#8884d8" name="Accuracy" />
                      <Bar dataKey="precision" fill="#82ca9d" name="Precision" />
                      <Bar dataKey="recall" fill="#ffc658" name="Recall" />
                      <Bar dataKey="f1_score" fill="#ff8042" name="F1 Score" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Inference Time</CardTitle>
                <CardDescription>Average inference time per model (seconds)</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getModelTimeData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="time" fill="#8884d8" name="Inference Time (s)" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Model Details</CardTitle>
                <CardDescription>Detailed information about each model</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b">
                        <th className="py-3 px-4 text-left">Model</th>
                        <th className="py-3 px-4 text-left">Type</th>
                        <th className="py-3 px-4 text-left">Accuracy</th>
                        <th className="py-3 px-4 text-left">Precision</th>
                        <th className="py-3 px-4 text-left">Recall</th>
                        <th className="py-3 px-4 text-left">F1 Score</th>
                        <th className="py-3 px-4 text-left">Inference Time</th>
                        <th className="py-3 px-4 text-left">Memory Usage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.values(modelPerformance).map((model) => (
                        <tr key={model.id} className="border-b hover:bg-muted/50">
                          <td className="py-3 px-4">{model.name}</td>
                          <td className="py-3 px-4">{model.type}</td>
                          <td className="py-3 px-4">{(model.accuracy * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4">{(model.precision * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4">{(model.recall * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4">{(model.f1_score * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4">{model.avg_inference_time.toFixed(3)} s</td>
                          <td className="py-3 px-4">{model.memory_usage}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="settings">
            {detectionSettings && (
              <Card>
                <CardHeader>
                  <CardTitle>Detection Settings</CardTitle>
                  <CardDescription>Configure advanced detection settings</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-lg font-medium mb-4">Detection Methods</h3>
                      
                      <div className="mb-4">
                        <label className="block text-sm font-medium mb-1">Ensemble Method</label>
                        <select
                          className="w-full p-2 border rounded-md"
                          value={detectionSettings.ensemble_method}
                          onChange={(e) => updateDetectionSettings({ ensemble_method: e.target.value })}
                        >
                          <option value="weighted_average">Weighted Average</option>
                          <option value="max_confidence">Max Confidence</option>
                          <option value="voting">Voting</option>
                        </select>
                      </div>
                      
                      <div className="mb-4">
                        <label className="block text-sm font-medium mb-1">Confidence Threshold</label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={detectionSettings.confidence_threshold}
                          onChange={(e) => updateDetectionSettings({ confidence_threshold: parseFloat(e.target.value) })}
                          className="w-full"
                        />
                        <div className="text-sm text-gray-500 mt-1">
                          {(detectionSettings.confidence_threshold * 100).toFixed(0)}%
                        </div>
                      </div>
                      
                      <div className="mb-4">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={detectionSettings.temporal_analysis}
                            onChange={(e) => updateDetectionSettings({ temporal_analysis: e.target.checked })}
                            className="mr-2"
                          />
                          <span>Enable Temporal Analysis</span>
                        </label>
                        <div className="text-sm text-gray-500 mt-1">
                          Analyze temporal patterns in videos
                        </div>
                      </div>
                      
                      <div className="mb-4">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={detectionSettings.indian_face_enhancement}
                            onChange={(e) => updateDetectionSettings({ indian_face_enhancement: e.target.checked })}
                            className="mr-2"
                          />
                          <span>Enable Indian Face Enhancement</span>
                        </label>
                        <div className="text-sm text-gray-500 mt-1">
                          Specialized detection for Indian faces
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-medium mb-4">Performance Settings</h3>
                      
                      <div className="mb-4">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={detectionSettings.adversarial_detection}
                            onChange={(e) => updateDetectionSettings({ adversarial_detection: e.target.checked })}
                            className="mr-2"
                          />
                          <span>Enable Adversarial Detection</span>
                        </label>
                        <div className="text-sm text-gray-500 mt-1">
                          Detect adversarial examples
                        </div>
                      </div>
                      
                      <div className="mb-4">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={detectionSettings.use_quantized_models}
                            onChange={(e) => updateDetectionSettings({ use_quantized_models: e.target.checked })}
                            className="mr-2"
                          />
                          <span>Use Quantized Models</span>
                        </label>
                        <div className="text-sm text-gray-500 mt-1">
                          Faster inference with slightly lower accuracy
                        </div>
                      </div>
                      
                      <div className="mb-4">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={detectionSettings.parallel_processing}
                            onChange={(e) => updateDetectionSettings({ parallel_processing: e.target.checked })}
                            className="mr-2"
                          />
                          <span>Enable Parallel Processing</span>
                        </label>
                        <div className="text-sm text-gray-500 mt-1">
                          Process multiple models in parallel
                        </div>
                      </div>
                      
                      <div className="mb-4">
                        <label className="block text-sm font-medium mb-1">Max Workers</label>
                        <input
                          type="number"
                          min="1"
                          max="8"
                          value={detectionSettings.max_workers}
                          onChange={(e) => updateDetectionSettings({ max_workers: parseInt(e.target.value) })}
                          className="w-full p-2 border rounded-md"
                          disabled={!detectionSettings.parallel_processing}
                        />
                        <div className="text-sm text-gray-500 mt-1">
                          Number of parallel workers
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button onClick={() => fetchDetectionSettings()}>Reset to Defaults</Button>
                </CardFooter>
              </Card>
            )}
          </TabsContent>
          
          <TabsContent value="detections">
            <Card>
              <CardHeader>
                <CardTitle>Recent Detections</CardTitle>
                <CardDescription>Latest detection results</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b">
                        <th className="py-3 px-4 text-left">ID</th>
                        <th className="py-3 px-4 text-left">Filename</th>
                        <th className="py-3 px-4 text-left">Probability</th>
                        <th className="py-3 px-4 text-left">Confidence</th>
                        <th className="py-3 px-4 text-left">Verdict</th>
                        <th className="py-3 px-4 text-left">Model</th>
                        <th className="py-3 px-4 text-left">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dashboardData?.recent_detections.map((detection) => (
                        <tr key={detection.id} className="border-b hover:bg-muted/50">
                          <td className="py-3 px-4">{detection.id.substring(0, 8)}...</td>
                          <td className="py-3 px-4">{detection.filename}</td>
                          <td className="py-3 px-4">
                            <div className="flex items-center">
                              <div
                                className={`w-3 h-3 rounded-full mr-2 ${
                                  detection.probability > 0.7 ? 'bg-red-500' : 
                                  detection.probability > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                                }`}
                              />
                              {(detection.probability * 100).toFixed(1)}%
                            </div>
                          </td>
                          <td className="py-3 px-4">{(detection.confidence * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4">{detection.verdict}</td>
                          <td className="py-3 px-4">{Array.isArray(detection.model) ? detection.model.join(", ") : detection.model}</td>
                          <td className="py-3 px-4">
                            <div className="flex space-x-2">
                              <Button variant="outline" size="sm" asChild>
                                <a href={`${API_URL}/api/visualization/${detection.id}`} target="_blank" rel="noopener noreferrer">
                                  View
                                </a>
                              </Button>
                              <Button variant="outline" size="sm" asChild>
                                <a href={`${API_URL}/api/report/${detection.id}?format=pdf`} download>
                                  Report
                                </a>
                              </Button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline" onClick={fetchDashboardData}>Refresh Data</Button>
                <Button variant="default" asChild>
                  <a href="/history">View All History</a>
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
      <Footer />
    </div>
  );
};

export default AdvancedDashboard;import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { API_URL, isUsingMockApi } from '@/services/config';
import { toast } from 'sonner';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import Header from '@/components/Header';
import Footer from '@/components/Footer';

interface ModelPerformance {
  id: string;
  name: string;
  type: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  avg_inference_time: number;
  memory_usage: string;
}

interface DashboardData {
  total_detections: number;
  verdicts: {
    deepfakes: number;
    suspicious: number;
    authentic: number;
  };
  models: Record<string, number>;
  avg_confidence: number;
  avg_processing_time: number;
  recent_detections: any[];
}

interface DetectionSettings {
  ensemble_method: string;
  temporal_analysis: boolean;
  indian_face_enhancement: boolean;
  adversarial_detection: boolean;
  confidence_threshold: number;
  use_quantized_models: boolean;
  parallel_processing: boolean;
  max_workers: number;
}

const COLORS = ['#FF6B6B', '#FFD166', '#06D6A0', '#118AB2', '#073B4C'];

const AdvancedDashboard = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [modelPerformance, setModelPerformance] = useState<Record<string, ModelPerformance>>({});
  const [detectionSettings, setDetectionSettings] = useState<DetectionSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    fetchDashboardData();
    fetchModelPerformance();
    fetchDetectionSettings();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // If using mock API, return mock data
      if (isUsingMockApi()) {
        const mockData: DashboardData = {
          total_detections: 42,
          verdicts: {
            deepfakes: 18,
            suspicious: 12,
            authentic: 12
          },
          models: {
            efficientnet: 20,
            xception: 15,
            indian_specialized: 7
          },
          avg_confidence: 0.78,
          avg_processing_time: 1250,
          recent_detections: [
            {
              id: "mock-1",
              filename: "sample1.jpg",
              probability: 0.85,
              confidence: 0.92,
              model: "efficientnet"
            },
            {
              id: "mock-2",
              filename: "sample2.jpg",
              probability: 0.32,
              confidence: 0.88,
              model: "xception"
            },
            {
              id: "mock-3",
              filename: "sample3.jpg",
              probability: 0.67,
              confidence: 0.75,
              model: "indian_specialized"
            }
          ]
        };
        setDashboardData(mockData);
        setLoading(false);
        return;
      }

      // Fetch dashboard data from API
      const response = await fetch(`${API_URL}/api/dashboard`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to fetch dashboard data: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || "Failed to fetch dashboard data");
      }
      
      setDashboardData(data.dashboard);
    } catch (error) {
      setError(error instanceof Error ? error.message : "An unknown error occurred");
      toast.error("Failed to fetch dashboard data");
    } finally {
      setLoading(false);
    }
  };

  const fetchModelPerformance = async () => {
    try {
      // If using mock API, return mock data
      if (isUsingMockApi()) {
        const mockPerformance: Record<string, ModelPerformance> = {
          efficientnet: {
            id: "efficientnet",
            name: "EfficientNet",
            type: "image",
            accuracy: 0.92,
            precision: 0.90,
            recall: 0.89,
            f1_score: 0.895,
            avg_inference_time: 0.25,
            memory_usage: "120 MB"
          },
          xception: {
            id: "xception",
            name: "Xception",
            type: "image",
            accuracy: 0.94,
            precision: 0.93,
            recall: 0.92,
            f1_score: 0.925,
            avg_inference_time: 0.35,
            memory_usage: "180 MB"
          },
          indian_specialized: {
            id: "indian_specialized",
            name: "Indian Specialized",
            type: "image",
            accuracy: 0.96,
            precision: 0.95,
            recall: 0.94,
            f1_score: 0.945,
            avg_inference_time: 0.40,
            memory_usage: "220 MB"
          },
          temporal_cnn: {
            id: "temporal_cnn",
            name: "Temporal CNN",
            type: "video",
            accuracy: 0.91,
            precision: 0.89,
            recall: 0.90,
            f1_score: 0.895,
            avg_inference_time: 0.75,
            memory_usage: "280 MB"
          },
          audio_analyzer: {
            id: "audio_analyzer",
            name: "Audio Analyzer",
            type: "audio",
            accuracy: 0.88,
            precision: 0.86,
            recall: 0.85,
            f1_score: 0.855,
            avg_inference_time: 0.45,
            memory_usage: "150 MB"
          }
        };
        setModelPerformance(mockPerformance);
        return;
      }

      // Fetch model performance from API
      const response = await fetch(`${API_URL}/api/model-performance`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to fetch model performance: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || "Failed to fetch model performance");
      }
      
      setModelPerformance(data.performance);
    } catch (error) {
      toast.error("Failed to fetch model performance");
    }
  };

  const fetchDetectionSettings = async () => {
    try {
      // If using mock API, return mock data
      if (isUsingMockApi()) {
        const mockSettings: DetectionSettings = {
          ensemble_method: "weighted_average",
          temporal_analysis: true,
          indian_face_enhancement: true,
          adversarial_detection: true,
          confidence_threshold: 0.5,
          use_quantized_models: false,
          parallel_processing: true,
          max_workers: 4
        };
        setDetectionSettings(mockSettings);
        return;
      }

      // Fetch detection settings from API
      const response = await fetch(`${API_URL}/api/detection-settings`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to fetch detection settings: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || "Failed to fetch detection settings");
      }
      
      setDetectionSettings(data.settings);
    } catch (error) {
      toast.error("Failed to fetch detection settings");
    }
  };

  const updateDetectionSettings = async (newSettings: Partial<DetectionSettings>) => {
    try {
      // If using mock API, update local state
      if (isUsingMockApi()) {
        setDetectionSettings(prev => ({
          ...prev!,
          ...newSettings
        }));
        toast.success("Settings updated successfully");
        return;
      }

      // Update detection settings via API
      const response = await fetch(`${API_URL}/api/detection-settings`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newSettings)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to update detection settings: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || "Failed to update detection settings");
      }
      
      setDetectionSettings(data.settings);
      toast.success("Settings updated successfully");
    } catch (error) {
      toast.error("Failed to update detection settings");
    }
  };

  const getVerdictData = () => {
    if (!dashboardData) return [];
    
    return [
      { name: "Deepfakes", value: dashboardData.verdicts.deepfakes },
      { name: "Suspicious", value: dashboardData.verdicts.suspicious },
      { name: "Authentic", value: dashboardData.verdicts.authentic }
    ];
  };

  const getModelUsageData = () => {
    if (!dashboardData) return [];
    
    return Object.entries(dashboardData.models).map(([model, count]) => ({
      name: model,
      count
    }));
  };

  const getModelPerformanceData = () => {
    return Object.values(modelPerformance).map(model => ({
      name: model.name,
      accuracy: model.accuracy,
      precision: model.precision,
      recall: model.recall,
      f1_score: model.f1_score
    }));
  };

  const getModelTimeData = () => {
    return Object.values(modelPerformance).map(model => ({
      name: model.name,
      time: model.avg_inference_time
    }));
  };

  if (loading) {
    return (
      <div className="flex flex-col min-h-screen">
        <Header />
        <main className="flex-1 container mx-auto p-4">
          <div className="flex flex-col items-center justify-center h-full">
            <h2 className="text-2xl font-bold mb-4">Loading Dashboard...</h2>
            <Progress value={50} className="w-[60%] mb-4" />
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col min-h-screen">
        <Header />
        <main className="flex-1 container mx-auto p-4">
          <div className="flex flex-col items-center justify-center h-full">
            <h2 className="text-2xl font-bold mb-4 text-red-500">Error Loading Dashboard</h2>
            <p className="text-gray-700 mb-4">{error}</p>
            <Button onClick={fetchDashboardData}>Retry</Button>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      <main className="flex-1 container mx-auto p-4">
        <h1 className="text-3xl font-bold mb-6">Advanced Dashboard</h1>
        
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="models">Model Performance</TabsTrigger>
            <TabsTrigger value="settings">Detection Settings</TabsTrigger>
            <TabsTrigger value="detections">Recent Detections</TabsTrigger>
          </TabsList>
          
          <TabsContent value="overview">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Total Detections</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">{dashboardData?.total_detections}</div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Deepfakes Detected</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-red-500">{dashboardData?.verdicts.deepfakes}</div>
                  <div className="text-sm text-gray-500">
                    {dashboardData && ((dashboardData.verdicts.deepfakes / dashboardData.total_detections) * 100).toFixed(1)}% of total
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Avg. Confidence</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">{dashboardData && (dashboardData.avg_confidence * 100).toFixed(1)}%</div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Avg. Processing Time</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">{dashboardData && dashboardData.avg_processing_time.toFixed(0)} ms</div>
                </CardContent>
              </Card>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <Card>
                <CardHeader>
                  <CardTitle>Detection Verdicts</CardTitle>
                  <CardDescription>Distribution of detection results</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={getVerdictData()}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          outerRadius={100}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {getVerdictData().map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Model Usage</CardTitle>
                  <CardDescription>Number of detections by model</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={getModelUsageData()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="count" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          <TabsContent value="models">
            <Card>
              <CardHeader>
                <CardTitle>Model Performance Metrics</CardTitle>
                <CardDescription>Accuracy, precision, recall, and F1 score for each model</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getModelPerformanceData()} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" domain={[0, 1]} />
                      <YAxis dataKey="name" type="category" width={150} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="accuracy" fill="#8884d8" name="Accuracy" />
                      <Bar dataKey="precision" fill="#82ca9d" name="Precision" />
                      <Bar dataKey="recall" fill="#ffc658" name="Recall" />
                      <Bar dataKey="f1_score" fill="#ff8042" name="F1 Score" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Inference Time</CardTitle>
                <CardDescription>Average inference time per model (seconds)</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getModelTimeData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="time" fill="#8884d8" name="Inference Time (s)" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Model Details</CardTitle>
                <CardDescription>Detailed information about each model</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b">
                        <th className="py-3 px-4 text-left">Model</th>
                        <th className="py-3 px-4 text-left">Type</th>
                        <th className="py-3 px-4 text-left">Accuracy</th>
                        <th className="py-3 px-4 text-left">Precision</th>
                        <th className="py-3 px-4 text-left">Recall</th>
                        <th className="py-3 px-4 text-left">F1 Score</th>
                        <th className="py-3 px-4 text-left">Inference Time</th>
                        <th className="py-3 px-4 text-left">Memory Usage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.values(modelPerformance).map((model) => (
                        <tr key={model.id} className="border-b hover:bg-muted/50">
                          <td className="py-3 px-4">{model.name}</td>
                          <td className="py-3 px-4">{model.type}</td>
                          <td className="py-3 px-4">{(model.accuracy * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4">{(model.precision * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4">{(model.recall * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4">{(model.f1_score * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4">{model.avg_inference_time.toFixed(3)} s</td>
                          <td className="py-3 px-4">{model.memory_usage}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="settings">
            {detectionSettings && (
              <Card>
                <CardHeader>
                  <CardTitle>Detection Settings</CardTitle>
                  <CardDescription>Configure advanced detection settings</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-lg font-medium mb-4">Detection Methods</h3>
                      
                      <div className="mb-4">
                        <label className="block text-sm font-medium mb-1">Ensemble Method</label>
                        <select
                          className="w-full p-2 border rounded-md"
                          value={detectionSettings.ensemble_method}
                          onChange={(e) => updateDetectionSettings({ ensemble_method: e.target.value })}
                        >
                          <option value="weighted_average">Weighted Average</option>
                          <option value="max_confidence">Max Confidence</option>
                          <option value="voting">Voting</option>
                        </select>
                      </div>
                      
                      <div className="mb-4">
                        <label className="block text-sm font-medium mb-1">Confidence Threshold</label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={detectionSettings.confidence_threshold}
                          onChange={(e) => updateDetectionSettings({ confidence_threshold: parseFloat(e.target.value) })}
                          className="w-full"
                        />
                        <div className="text-sm text-gray-500 mt-1">
                          {(detectionSettings.confidence_threshold * 100).toFixed(0)}%
                        </div>
                      </div>
                      
                      <div className="mb-4">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={detectionSettings.temporal_analysis}
                            onChange={(e) => updateDetectionSettings({ temporal_analysis: e.target.checked })}
                            className="mr-2"
                          />
                          <span>Enable Temporal Analysis</span>
                        </label>
                        <div className="text-sm text-gray-500 mt-1">
                          Analyze temporal patterns in videos
                        </div>
                      </div>
                      
                      <div className="mb-4">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={detectionSettings.indian_face_enhancement}
                            onChange={(e) => updateDetectionSettings({ indian_face_enhancement: e.target.checked })}
                            className="mr-2"
                          />
                          <span>Enable Indian Face Enhancement</span>
                        </label>
                        <div className="text-sm text-gray-500 mt-1">
                          Specialized detection for Indian faces
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-medium mb-4">Performance Settings</h3>
                      
                      <div className="mb-4">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={detectionSettings.adversarial_detection}
                            onChange={(e) => updateDetectionSettings({ adversarial_detection: e.target.checked })}
                            className="mr-2"
                          />
                          <span>Enable Adversarial Detection</span>
                        </label>
                        <div className="text-sm text-gray-500 mt-1">
                          Detect adversarial examples
                        </div>
                      </div>
                      
                      <div className="mb-4">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={detectionSettings.use_quantized_models}
                            onChange={(e) => updateDetectionSettings({ use_quantized_models: e.target.checked })}
                            className="mr-2"
                          />
                          <span>Use Quantized Models</span>
                        </label>
                        <div className="text-sm text-gray-500 mt-1">
                          Faster inference with slightly lower accuracy
                        </div>
                      </div>
                      
                      <div className="mb-4">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={detectionSettings.parallel_processing}
                            onChange={(e) => updateDetectionSettings({ parallel_processing: e.target.checked })}
                            className="mr-2"
                          />
                          <span>Enable Parallel Processing</span>
                        </label>
                        <div className="text-sm text-gray-500 mt-1">
                          Process multiple models in parallel
                        </div>
                      </div>
                      
                      <div className="mb-4">
                        <label className="block text-sm font-medium mb-1">Max Workers</label>
                        <input
                          type="number"
                          min="1"
                          max="8"
                          value={detectionSettings.max_workers}
                          onChange={(e) => updateDetectionSettings({ max_workers: parseInt(e.target.value) })}
                          className="w-full p-2 border rounded-md"
                          disabled={!detectionSettings.parallel_processing}
                        />
                        <div className="text-sm text-gray-500 mt-1">
                          Number of parallel workers
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button onClick={() => fetchDetectionSettings()}>Reset to Defaults</Button>
                </CardFooter>
              </Card>
            )}
          </TabsContent>
          
          <TabsContent value="detections">
            <Card>
              <CardHeader>
                <CardTitle>Recent Detections</CardTitle>
                <CardDescription>Latest detection results</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b">
                        <th className="py-3 px-4 text-left">ID</th>
                        <th className="py-3 px-4 text-left">Filename</th>
                        <th className="py-3 px-4 text-left">Probability</th>
                        <th className="py-3 px-4 text-left">Confidence</th>
                        <th className="py-3 px-4 text-left">Verdict</th>
                        <th className="py-3 px-4 text-left">Model</th>
                        <th className="py-3 px-4 text-left">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dashboardData?.recent_detections.map((detection) => (
                        <tr key={detection.id} className="border-b hover:bg-muted/50">
                          <td className="py-3 px-4">{detection.id.substring(0, 8)}...</td>
                          <td className="py-3 px-4">{detection.filename}</td>
                          <td className="py-3 px-4">
                            <div className="flex items-center">
                              <div
                                className={`w-3 h-3 rounded-full mr-2 ${
                                  detection.probability > 0.7 ? 'bg-red-500' : 
                                  detection.probability > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                                }`}
                              />
                              {(detection.probability * 100).toFixed(1)}%
                            </div>
                          </td>
                          <td className="py-3 px-4">{(detection.confidence * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4">{detection.verdict}</td>
                          <td className="py-3 px-4">{Array.isArray(detection.model) ? detection.model.join(", ") : detection.model}</td>
                          <td className="py-3 px-4">
                            <div className="flex space-x-2">
                              <Button variant="outline" size="sm" asChild>
                                <a href={`${API_URL}/api/visualization/${detection.id}`} target="_blank" rel="noopener noreferrer">
                                  View
                                </a>
                              </Button>
                              <Button variant="outline" size="sm" asChild>
                                <a href={`${API_URL}/api/report/${detection.id}?format=pdf`} download>
                                  Report
                                </a>
                              </Button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline" onClick={fetchDashboardData}>Refresh Data</Button>
                <Button variant="default" asChild>
                  <a href="/history">View All History</a>
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
      <Footer />
    </div>
  );
};

export default AdvancedDashboard;