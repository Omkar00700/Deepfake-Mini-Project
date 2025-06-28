import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { API_URL, isUsingMockApi } from '@/services/config';
import { toast } from 'sonner';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

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

const COLORS = ['#FF6B6B', '#FFD166', '#06D6A0', '#118AB2', '#073B4C'];

const Dashboard = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDashboardData();
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
        throw new Error(data.error || 'Failed to fetch dashboard data');
      }

      setDashboardData(data.dashboard);
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      toast.error(`Failed to load dashboard: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data
  const prepareVerdictChartData = () => {
    if (!dashboardData) return [];
    
    return [
      { name: 'Deepfakes', value: dashboardData.verdicts.deepfakes },
      { name: 'Suspicious', value: dashboardData.verdicts.suspicious },
      { name: 'Authentic', value: dashboardData.verdicts.authentic }
    ];
  };

  const prepareModelChartData = () => {
    if (!dashboardData) return [];
    
    return Object.entries(dashboardData.models).map(([name, value]) => ({
      name: name === 'efficientnet' ? 'EfficientNet' : 
            name === 'xception' ? 'Xception' : 
            name === 'indian_specialized' ? 'Indian Specialized' : name,
      value
    }));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <h2 className="text-2xl font-semibold mb-4">Loading Dashboard</h2>
          <Progress value={75} className="w-[300px] mb-2" />
          <p className="text-muted-foreground">Fetching analytics data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <h2 className="text-2xl font-semibold mb-4 text-destructive">Dashboard Error</h2>
          <p className="mb-4">{error}</p>
          <Button onClick={fetchDashboardData}>Retry</Button>
        </div>
      </div>
    );
  }

  if (!dashboardData) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <h2 className="text-2xl font-semibold mb-4">No Data Available</h2>
          <p className="mb-4">No detection data is available for the dashboard.</p>
          <Button onClick={fetchDashboardData}>Refresh</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Deepfake Detection Dashboard</h1>
      
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Total Detections</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-4xl font-bold">{dashboardData.total_detections}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Deepfakes Detected</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-4xl font-bold text-red-500">{dashboardData.verdicts.deepfakes}</div>
            <div className="text-sm text-muted-foreground">
              {dashboardData.total_detections > 0 
                ? `${((dashboardData.verdicts.deepfakes / dashboardData.total_detections) * 100).toFixed(1)}% of total`
                : '0% of total'}
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Avg. Confidence</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-4xl font-bold">{(dashboardData.avg_confidence * 100).toFixed(1)}%</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Avg. Processing Time</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-4xl font-bold">{dashboardData.avg_processing_time.toFixed(0)}ms</div>
          </CardContent>
        </Card>
      </div>
      
      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <Card>
          <CardHeader>
            <CardTitle>Detection Results</CardTitle>
            <CardDescription>Distribution of detection verdicts</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={prepareVerdictChartData()}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {prepareVerdictChartData().map((entry, index) => (
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
            <CardTitle>Models Used</CardTitle>
            <CardDescription>Distribution of detection models</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={prepareModelChartData()}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value" fill="#8884d8" name="Detections" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Recent Detections */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Detections</CardTitle>
          <CardDescription>Latest detection results</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-3 px-4">ID</th>
                  <th className="text-left py-3 px-4">Filename</th>
                  <th className="text-left py-3 px-4">Probability</th>
                  <th className="text-left py-3 px-4">Confidence</th>
                  <th className="text-left py-3 px-4">Model</th>
                  <th className="text-left py-3 px-4">Actions</th>
                </tr>
              </thead>
              <tbody>
                {dashboardData.recent_detections.map((detection) => (
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
                    <td className="py-3 px-4">{detection.model}</td>
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
    </div>
  );
};

export default Dashboard;