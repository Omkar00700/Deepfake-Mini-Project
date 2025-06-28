import React, { useState, useRef } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { API_URL, isUsingMockApi } from '@/services/config';
import { toast } from 'sonner';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const COLORS = ['#FF6B6B', '#FFD166', '#06D6A0', '#118AB2', '#073B4C'];

interface AnalysisResult {
  analysis_id: string;
  probability: number;
  confidence: number;
  verdict: string;
  cross_modal_score: number;
  metadata_score: number;
  modality_results: {
    image?: {
      probability: number;
      confidence: number;
      verdict: string;
      frame_results?: any[];
    };
    audio?: {
      probability: number;
      confidence: number;
      verdict: string;
      segment_results?: any[];
    };
  };
  metadata_findings?: string[];
  file_type: string;
  filename: string;
  processing_time: number;
}

const MultimodalAnalysis = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState("upload");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      toast.error("Please select a file to analyze");
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 95) {
          clearInterval(progressInterval);
          return prev;
        }
        return prev + 5;
      });
    }, 200);

    try {
      // If using mock API, return mock data
      if (isUsingMockApi()) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 2000));

        const mockResult: AnalysisResult = {
          analysis_id: "mock-analysis-123",
          probability: 0.75,
          confidence: 0.85,
          verdict: "deepfake",
          cross_modal_score: 0.82,
          metadata_score: 0.65,
          modality_results: {
            image: {
              probability: 0.78,
              confidence: 0.88,
              verdict: "deepfake",
              frame_results: [
                { frame_idx: 0, probability: 0.76, confidence: 0.87 },
                { frame_idx: 1, probability: 0.79, confidence: 0.89 },
                { frame_idx: 2, probability: 0.77, confidence: 0.86 }
              ]
            },
            audio: {
              probability: 0.68,
              confidence: 0.75,
              verdict: "suspicious",
              segment_results: [
                { segment_idx: 0, start_time: 0, end_time: 1.5, probability: 0.65, confidence: 0.72 },
                { segment_idx: 1, start_time: 1.5, end_time: 3.0, probability: 0.70, confidence: 0.78 },
                { segment_idx: 2, start_time: 3.0, end_time: 4.5, probability: 0.69, confidence: 0.76 }
              ]
            }
          },
          metadata_findings: [
            "File was edited with Photoshop",
            "EXIF data is missing, which is unusual for camera photos",
            "File was modified 3 days after creation"
          ],
          file_type: file.type.includes("video") ? "video" : file.type.includes("audio") ? "audio" : "image",
          filename: file.name,
          processing_time: 2.35
        };

        setAnalysisResult(mockResult);
        setActiveTab("results");
        clearInterval(progressInterval);
        setUploadProgress(100);
        setIsUploading(false);
        return;
      }

      // Create form data
      const formData = new FormData();
      formData.append('file', file);

      // Add settings if needed
      const settings = {
        cross_modal_verification: true,
        metadata_analysis: true
      };
      formData.append('settings', JSON.stringify(settings));

      // Send to API
      const response = await fetch(`${API_URL}/api/multimodal-analyze`, {
        method: 'POST',
        body: formData
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Upload failed: ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Analysis failed");
      }

      setAnalysisResult(data.result);
      setActiveTab("results");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "An unknown error occurred");
    } finally {
      clearInterval(progressInterval);
      setIsUploading(false);
    }
  };

  const resetAnalysis = () => {
    setFile(null);
    setAnalysisResult(null);
    setActiveTab("upload");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const getModalityData = () => {
    if (!analysisResult) return [];

    const data = [];

    if (analysisResult.modality_results.image) {
      data.push({
        name: "Image Analysis",
        probability: analysisResult.modality_results.image.probability,
        confidence: analysisResult.modality_results.image.confidence
      });
    }

    if (analysisResult.modality_results.audio) {
      data.push({
        name: "Audio Analysis",
        probability: analysisResult.modality_results.audio.probability,
        confidence: analysisResult.modality_results.audio.confidence
      });
    }

    data.push({
      name: "Metadata Analysis",
      probability: analysisResult.metadata_score,
      confidence: 0.8  // Assumed confidence for metadata
    });

    return data;
  };

  const getVerdictData = () => {
    if (!analysisResult) return [];

    return [
      { name: "Deepfake", value: analysisResult.probability },
      { name: "Authentic", value: 1 - analysisResult.probability }
    ];
  };

  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      <main className="flex-1 container mx-auto p-4">
        <h1 className="text-3xl font-bold mb-6">Multi-Modal Analysis</h1>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="upload">Upload</TabsTrigger>
            <TabsTrigger value="results" disabled={!analysisResult}>Results</TabsTrigger>
            <TabsTrigger value="modalities" disabled={!analysisResult}>Modality Analysis</TabsTrigger>
            <TabsTrigger value="metadata" disabled={!analysisResult}>Metadata</TabsTrigger>
          </TabsList>

          <TabsContent value="upload">
            <Card>
              <CardHeader>
                <CardTitle>Upload Media for Analysis</CardTitle>
                <CardDescription>
                  Upload an image, video, or audio file for multi-modal deepfake analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:bg-muted/50 transition-colors"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    className="hidden"
                    accept="image/*,video/*,audio/*"
                  />
                  <div className="mb-4">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-12 w-12 mx-auto text-gray-400"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                      />
                    </svg>
                  </div>
                  <p className="text-lg font-medium">
                    Drag and drop your file here, or click to browse
                  </p>
                  <p className="text-sm text-gray-500 mt-2">
                    Supports images (JPG, PNG), videos (MP4, AVI), and audio files (MP3, WAV)
                  </p>
                </div>

                {file && (
                  <div className="mt-4 p-4 bg-muted rounded-lg">
                    <div className="flex items-center">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-6 w-6 mr-2 text-green-500"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                      <div className="flex-1">
                        <p className="font-medium">{file.name}</p>
                        <p className="text-sm text-gray-500">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          setFile(null);
                          if (fileInputRef.current) {
                            fileInputRef.current.value = "";
                          }
                        }}
                      >
                        Remove
                      </Button>
                    </div>
                  </div>
                )}

                {isUploading && (
                  <div className="mt-4">
                    <p className="text-sm font-medium mb-1">Uploading and analyzing...</p>
                    <Progress value={uploadProgress} className="h-2" />
                    <p className="text-xs text-gray-500 mt-1">{uploadProgress}%</p>
                  </div>
                )}
              </CardContent>
              <CardFooter>
                <Button onClick={handleUpload} disabled={!file || isUploading}>
                  {isUploading ? "Analyzing..." : "Start Analysis"}
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="results">
            {analysisResult && (
              <Card>
                <CardHeader>
                  <CardTitle>Analysis Results</CardTitle>
                  <CardDescription>
                    Multi-modal analysis results for {analysisResult.filename}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="bg-muted p-4 rounded-lg">
                      <h3 className="text-lg font-medium mb-2">Verdict</h3>
                      <div className="text-3xl font-bold mb-1 capitalize">
                        {analysisResult.verdict}
                      </div>
                      <div className="text-sm text-gray-500">
                        {analysisResult.verdict === "deepfake"
                          ? "High confidence this media is manipulated"
                          : analysisResult.verdict === "suspicious"
                          ? "Some signs of manipulation detected"
                          : "No clear signs of manipulation detected"}
                      </div>
                    </div>

                    <div className="bg-muted p-4 rounded-lg">
                      <h3 className="text-lg font-medium mb-2">Probability</h3>
                      <div className="text-3xl font-bold mb-1">
                        {(analysisResult.probability * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-500">
                        Likelihood of being a deepfake
                      </div>
                    </div>

                    <div className="bg-muted p-4 rounded-lg">
                      <h3 className="text-lg font-medium mb-2">Confidence</h3>
                      <div className="text-3xl font-bold mb-1">
                        {(analysisResult.confidence * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-500">
                        Confidence in the analysis result
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div>
                      <h3 className="text-lg font-medium mb-4">Verdict Distribution</h3>
                      <div className="h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={getVerdictData()}
                              cx="50%"
                              cy="50%"
                              labelLine={false}
                              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                              outerRadius={80}
                              fill="#8884d8"
                              dataKey="value"
                            >
                              {getVerdictData().map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={index === 0 ? "#FF6B6B" : "#06D6A0"} />
                              ))}
                            </Pie>
                            <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-medium mb-4">Cross-Modal Analysis</h3>
                      <div className="h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={getModalityData()}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                            <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
                            <Legend />
                            <Bar dataKey="probability" name="Probability" fill="#8884d8" />
                            <Bar dataKey="confidence" name="Confidence" fill="#82ca9d" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  <div className="mb-6">
                    <h3 className="text-lg font-medium mb-4">Analysis Details</h3>
                    <div className="bg-muted p-4 rounded-lg">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <p className="text-sm font-medium">File Type</p>
                          <p className="text-lg capitalize">{analysisResult.file_type}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Processing Time</p>
                          <p className="text-lg">{analysisResult.processing_time.toFixed(2)} seconds</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Cross-Modal Score</p>
                          <p className="text-lg">{(analysisResult.cross_modal_score * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Metadata Score</p>
                          <p className="text-lg">{(analysisResult.metadata_score * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" onClick={resetAnalysis}>
                    Analyze Another File
                  </Button>
                  <Button asChild>
                    <a
                      href={`${API_URL}/api/report/${analysisResult.analysis_id}?format=pdf`}
                      download
                    >
                      Download Report
                    </a>
                  </Button>
                </CardFooter>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="modalities">
            {analysisResult && (
              <div className="space-y-6">
                {analysisResult.modality_results.image && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Image Analysis</CardTitle>
                      <CardDescription>
                        Results from image-based deepfake detection
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <div className="bg-muted p-4 rounded-lg">
                          <h3 className="text-lg font-medium mb-2">Verdict</h3>
                          <div className="text-2xl font-bold mb-1 capitalize">
                            {analysisResult.modality_results.image.verdict}
                          </div>
                        </div>
                        <div className="bg-muted p-4 rounded-lg">
                          <h3 className="text-lg font-medium mb-2">Probability</h3>
                          <div className="text-2xl font-bold mb-1">
                            {(analysisResult.modality_results.image.probability * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="bg-muted p-4 rounded-lg">
                          <h3 className="text-lg font-medium mb-2">Confidence</h3>
                          <div className="text-2xl font-bold mb-1">
                            {(analysisResult.modality_results.image.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      {analysisResult.modality_results.image.frame_results && (
                        <div>
                          <h3 className="text-lg font-medium mb-4">Frame Analysis</h3>
                          <div className="overflow-x-auto">
                            <table className="w-full border-collapse">
                              <thead>
                                <tr className="border-b">
                                  <th className="py-2 px-4 text-left">Frame</th>
                                  <th className="py-2 px-4 text-left">Probability</th>
                                  <th className="py-2 px-4 text-left">Confidence</th>
                                </tr>
                              </thead>
                              <tbody>
                                {analysisResult.modality_results.image.frame_results.map((frame) => (
                                  <tr key={frame.frame_idx} className="border-b">
                                    <td className="py-2 px-4">Frame {frame.frame_idx + 1}</td>
                                    <td className="py-2 px-4">{(frame.probability * 100).toFixed(1)}%</td>
                                    <td className="py-2 px-4">{(frame.confidence * 100).toFixed(1)}%</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}

                {analysisResult.modality_results.audio && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Audio Analysis</CardTitle>
                      <CardDescription>
                        Results from audio-based deepfake detection
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <div className="bg-muted p-4 rounded-lg">
                          <h3 className="text-lg font-medium mb-2">Verdict</h3>
                          <div className="text-2xl font-bold mb-1 capitalize">
                            {analysisResult.modality_results.audio.verdict}
                          </div>
                        </div>
                        <div className="bg-muted p-4 rounded-lg">
                          <h3 className="text-lg font-medium mb-2">Probability</h3>
                          <div className="text-2xl font-bold mb-1">
                            {(analysisResult.modality_results.audio.probability * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="bg-muted p-4 rounded-lg">
                          <h3 className="text-lg font-medium mb-2">Confidence</h3>
                          <div className="text-2xl font-bold mb-1">
                            {(analysisResult.modality_results.audio.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      {analysisResult.modality_results.audio.segment_results && (
                        <div>
                          <h3 className="text-lg font-medium mb-4">Segment Analysis</h3>
                          <div className="overflow-x-auto">
                            <table className="w-full border-collapse">
                              <thead>
                                <tr className="border-b">
                                  <th className="py-2 px-4 text-left">Segment</th>
                                  <th className="py-2 px-4 text-left">Time Range</th>
                                  <th className="py-2 px-4 text-left">Probability</th>
                                  <th className="py-2 px-4 text-left">Confidence</th>
                                </tr>
                              </thead>
                              <tbody>
                                {analysisResult.modality_results.audio.segment_results.map((segment) => (
                                  <tr key={segment.segment_idx} className="border-b">
                                    <td className="py-2 px-4">Segment {segment.segment_idx + 1}</td>
                                    <td className="py-2 px-4">
                                      {segment.start_time.toFixed(1)}s - {segment.end_time.toFixed(1)}s
                                    </td>
                                    <td className="py-2 px-4">{(segment.probability * 100).toFixed(1)}%</td>
                                    <td className="py-2 px-4">{(segment.confidence * 100).toFixed(1)}%</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}

                <Card>
                  <CardHeader>
                    <CardTitle>Cross-Modal Verification</CardTitle>
                    <CardDescription>
                      Analysis of consistency between different modalities
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="mb-6">
                      <h3 className="text-lg font-medium mb-2">Cross-Modal Score</h3>
                      <div className="text-2xl font-bold mb-1">
                        {(analysisResult.cross_modal_score * 100).toFixed(1)}%
                      </div>
                      <p className="text-sm text-gray-500">
                        Higher scores indicate greater consistency between modalities, suggesting more reliable results.
                      </p>
                    </div>

                    <div className="bg-muted p-4 rounded-lg">
                      <h3 className="text-lg font-medium mb-2">Interpretation</h3>
                      <p>
                        {analysisResult.cross_modal_score > 0.8
                          ? "High consistency between modalities. The analysis results are highly reliable."
                          : analysisResult.cross_modal_score > 0.6
                          ? "Good consistency between modalities. The analysis results are generally reliable."
                          : analysisResult.cross_modal_score > 0.4
                          ? "Moderate consistency between modalities. Some discrepancies exist between different analysis methods."
                          : "Low consistency between modalities. Different analysis methods produced conflicting results."}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>

          <TabsContent value="metadata">
            {analysisResult && (
              <Card>
                <CardHeader>
                  <CardTitle>Metadata Analysis</CardTitle>
                  <CardDescription>
                    Analysis of file metadata for signs of manipulation
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="mb-6">
                    <h3 className="text-lg font-medium mb-2">Metadata Score</h3>
                    <div className="text-2xl font-bold mb-1">
                      {(analysisResult.metadata_score * 100).toFixed(1)}%
                    </div>
                    <p className="text-sm text-gray-500">
                      Higher scores indicate more signs of potential manipulation in the file metadata.
                    </p>
                  </div>

                  {analysisResult.metadata_findings && analysisResult.metadata_findings.length > 0 && (
                    <div className="mb-6">
                      <h3 className="text-lg font-medium mb-4">Key Findings</h3>
                      <ul className="space-y-2">
                        {analysisResult.metadata_findings.map((finding, index) => (
                          <li key={index} className="bg-muted p-3 rounded-lg flex items-start">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-5 w-5 mr-2 text-yellow-500 mt-0.5 flex-shrink-0"
                              viewBox="0 0 20 20"
                              fill="currentColor"
                            >
                              <path
                                fillRule="evenodd"
                                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                                clipRule="evenodd"
                              />
                            </svg>
                            {finding}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="bg-muted p-4 rounded-lg">
                    <h3 className="text-lg font-medium mb-2">Interpretation</h3>
                    <p>
                      {analysisResult.metadata_score > 0.7
                        ? "Significant metadata anomalies detected. This file shows strong signs of manipulation based on metadata analysis."
                        : analysisResult.metadata_score > 0.4
                        ? "Some metadata anomalies detected. This file shows potential signs of manipulation based on metadata analysis."
                        : "Few or no metadata anomalies detected. This file does not show significant signs of manipulation based on metadata analysis."}
                    </p>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button asChild>
                    <a
                      href={`${API_URL}/api/report/${analysisResult.analysis_id}?format=pdf`}
                      download
                    >
                      Download Full Report
                    </a>
                  </Button>
                </CardFooter>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </main>
      <Footer />
    </div>
  );
};

export default MultimodalAnalysis;import React, { useState, useRef } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { API_URL, isUsingMockApi } from '@/services/config';
import { toast } from 'sonner';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const COLORS = ['#FF6B6B', '#FFD166', '#06D6A0', '#118AB2', '#073B4C'];

interface AnalysisResult {
  analysis_id: string;
  probability: number;
  confidence: number;
  verdict: string;
  cross_modal_score: number;
  metadata_score: number;
  modality_results: {
    image?: {
      probability: number;
      confidence: number;
      verdict: string;
      frame_results?: any[];
    };
    audio?: {
      probability: number;
      confidence: number;
      verdict: string;
      segment_results?: any[];
    };
  };
  metadata_findings?: string[];
  file_type: string;
  filename: string;
  processing_time: number;
}

const MultimodalAnalysis = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState("upload");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      toast.error("Please select a file to analyze");
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 95) {
          clearInterval(progressInterval);
          return prev;
        }
        return prev + 5;
      });
    }, 200);

    try {
      // If using mock API, return mock data
      if (isUsingMockApi()) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 2000));

        const mockResult: AnalysisResult = {
          analysis_id: "mock-analysis-123",
          probability: 0.75,
          confidence: 0.85,
          verdict: "deepfake",
          cross_modal_score: 0.82,
          metadata_score: 0.65,
          modality_results: {
            image: {
              probability: 0.78,
              confidence: 0.88,
              verdict: "deepfake",
              frame_results: [
                { frame_idx: 0, probability: 0.76, confidence: 0.87 },
                { frame_idx: 1, probability: 0.79, confidence: 0.89 },
                { frame_idx: 2, probability: 0.77, confidence: 0.86 }
              ]
            },
            audio: {
              probability: 0.68,
              confidence: 0.75,
              verdict: "suspicious",
              segment_results: [
                { segment_idx: 0, start_time: 0, end_time: 1.5, probability: 0.65, confidence: 0.72 },
                { segment_idx: 1, start_time: 1.5, end_time: 3.0, probability: 0.70, confidence: 0.78 },
                { segment_idx: 2, start_time: 3.0, end_time: 4.5, probability: 0.69, confidence: 0.76 }
              ]
            }
          },
          metadata_findings: [
            "File was edited with Photoshop",
            "EXIF data is missing, which is unusual for camera photos",
            "File was modified 3 days after creation"
          ],
          file_type: file.type.includes("video") ? "video" : file.type.includes("audio") ? "audio" : "image",
          filename: file.name,
          processing_time: 2.35
        };

        setAnalysisResult(mockResult);
        setActiveTab("results");
        clearInterval(progressInterval);
        setUploadProgress(100);
        setIsUploading(false);
        return;
      }

      // Create form data
      const formData = new FormData();
      formData.append('file', file);

      // Add settings if needed
      const settings = {
        cross_modal_verification: true,
        metadata_analysis: true
      };
      formData.append('settings', JSON.stringify(settings));

      // Send to API
      const response = await fetch(`${API_URL}/api/multimodal-analyze`, {
        method: 'POST',
        body: formData
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Upload failed: ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Analysis failed");
      }

      setAnalysisResult(data.result);
      setActiveTab("results");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "An unknown error occurred");
    } finally {
      clearInterval(progressInterval);
      setIsUploading(false);
    }
  };

  const resetAnalysis = () => {
    setFile(null);
    setAnalysisResult(null);
    setActiveTab("upload");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const getModalityData = () => {
    if (!analysisResult) return [];

    const data = [];

    if (analysisResult.modality_results.image) {
      data.push({
        name: "Image Analysis",
        probability: analysisResult.modality_results.image.probability,
        confidence: analysisResult.modality_results.image.confidence
      });
    }

    if (analysisResult.modality_results.audio) {
      data.push({
        name: "Audio Analysis",
        probability: analysisResult.modality_results.audio.probability,
        confidence: analysisResult.modality_results.audio.confidence
      });
    }

    data.push({
      name: "Metadata Analysis",
      probability: analysisResult.metadata_score,
      confidence: 0.8  // Assumed confidence for metadata
    });

    return data;
  };

  const getVerdictData = () => {
    if (!analysisResult) return [];

    return [
      { name: "Deepfake", value: analysisResult.probability },
      { name: "Authentic", value: 1 - analysisResult.probability }
    ];
  };

  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      <main className="flex-1 container mx-auto p-4">
        <h1 className="text-3xl font-bold mb-6">Multi-Modal Analysis</h1>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="upload">Upload</TabsTrigger>
            <TabsTrigger value="results" disabled={!analysisResult}>Results</TabsTrigger>
            <TabsTrigger value="modalities" disabled={!analysisResult}>Modality Analysis</TabsTrigger>
            <TabsTrigger value="metadata" disabled={!analysisResult}>Metadata</TabsTrigger>
          </TabsList>

          <TabsContent value="upload">
            <Card>
              <CardHeader>
                <CardTitle>Upload Media for Analysis</CardTitle>
                <CardDescription>
                  Upload an image, video, or audio file for multi-modal deepfake analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:bg-muted/50 transition-colors"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    className="hidden"
                    accept="image/*,video/*,audio/*"
                  />
                  <div className="mb-4">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-12 w-12 mx-auto text-gray-400"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                      />
                    </svg>
                  </div>
                  <p className="text-lg font-medium">
                    Drag and drop your file here, or click to browse
                  </p>
                  <p className="text-sm text-gray-500 mt-2">
                    Supports images (JPG, PNG), videos (MP4, AVI), and audio files (MP3, WAV)
                  </p>
                </div>

                {file && (
                  <div className="mt-4 p-4 bg-muted rounded-lg">
                    <div className="flex items-center">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-6 w-6 mr-2 text-green-500"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                      <div className="flex-1">
                        <p className="font-medium">{file.name}</p>
                        <p className="text-sm text-gray-500">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          setFile(null);
                          if (fileInputRef.current) {
                            fileInputRef.current.value = "";
                          }
                        }}
                      >
                        Remove
                      </Button>
                    </div>
                  </div>
                )}

                {isUploading && (
                  <div className="mt-4">
                    <p className="text-sm font-medium mb-1">Uploading and analyzing...</p>
                    <Progress value={uploadProgress} className="h-2" />
                    <p className="text-xs text-gray-500 mt-1">{uploadProgress}%</p>
                  </div>
                )}
              </CardContent>
              <CardFooter>
                <Button onClick={handleUpload} disabled={!file || isUploading}>
                  {isUploading ? "Analyzing..." : "Start Analysis"}
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="results">
            {analysisResult && (
              <Card>
                <CardHeader>
                  <CardTitle>Analysis Results</CardTitle>
                  <CardDescription>
                    Multi-modal analysis results for {analysisResult.filename}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="bg-muted p-4 rounded-lg">
                      <h3 className="text-lg font-medium mb-2">Verdict</h3>
                      <div className="text-3xl font-bold mb-1 capitalize">
                        {analysisResult.verdict}
                      </div>
                      <div className="text-sm text-gray-500">
                        {analysisResult.verdict === "deepfake"
                          ? "High confidence this media is manipulated"
                          : analysisResult.verdict === "suspicious"
                          ? "Some signs of manipulation detected"
                          : "No clear signs of manipulation detected"}
                      </div>
                    </div>

                    <div className="bg-muted p-4 rounded-lg">
                      <h3 className="text-lg font-medium mb-2">Probability</h3>
                      <div className="text-3xl font-bold mb-1">
                        {(analysisResult.probability * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-500">
                        Likelihood of being a deepfake
                      </div>
                    </div>

                    <div className="bg-muted p-4 rounded-lg">
                      <h3 className="text-lg font-medium mb-2">Confidence</h3>
                      <div className="text-3xl font-bold mb-1">
                        {(analysisResult.confidence * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-500">
                        Confidence in the analysis result
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div>
                      <h3 className="text-lg font-medium mb-4">Verdict Distribution</h3>
                      <div className="h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={getVerdictData()}
                              cx="50%"
                              cy="50%"
                              labelLine={false}
                              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                              outerRadius={80}
                              fill="#8884d8"
                              dataKey="value"
                            >
                              {getVerdictData().map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={index === 0 ? "#FF6B6B" : "#06D6A0"} />
                              ))}
                            </Pie>
                            <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-medium mb-4">Cross-Modal Analysis</h3>
                      <div className="h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={getModalityData()}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                            <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
                            <Legend />
                            <Bar dataKey="probability" name="Probability" fill="#8884d8" />
                            <Bar dataKey="confidence" name="Confidence" fill="#82ca9d" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  <div className="mb-6">
                    <h3 className="text-lg font-medium mb-4">Analysis Details</h3>
                    <div className="bg-muted p-4 rounded-lg">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <p className="text-sm font-medium">File Type</p>
                          <p className="text-lg capitalize">{analysisResult.file_type}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Processing Time</p>
                          <p className="text-lg">{analysisResult.processing_time.toFixed(2)} seconds</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Cross-Modal Score</p>
                          <p className="text-lg">{(analysisResult.cross_modal_score * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Metadata Score</p>
                          <p className="text-lg">{(analysisResult.metadata_score * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" onClick={resetAnalysis}>
                    Analyze Another File
                  </Button>
                  <Button asChild>
                    <a
                      href={`${API_URL}/api/report/${analysisResult.analysis_id}?format=pdf`}
                      download
                    >
                      Download Report
                    </a>
                  </Button>
                </CardFooter>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="modalities">
            {analysisResult && (
              <div className="space-y-6">
                {analysisResult.modality_results.image && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Image Analysis</CardTitle>
                      <CardDescription>
                        Results from image-based deepfake detection
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <div className="bg-muted p-4 rounded-lg">
                          <h3 className="text-lg font-medium mb-2">Verdict</h3>
                          <div className="text-2xl font-bold mb-1 capitalize">
                            {analysisResult.modality_results.image.verdict}
                          </div>
                        </div>
                        <div className="bg-muted p-4 rounded-lg">
                          <h3 className="text-lg font-medium mb-2">Probability</h3>
                          <div className="text-2xl font-bold mb-1">
                            {(analysisResult.modality_results.image.probability * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="bg-muted p-4 rounded-lg">
                          <h3 className="text-lg font-medium mb-2">Confidence</h3>
                          <div className="text-2xl font-bold mb-1">
                            {(analysisResult.modality_results.image.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      {analysisResult.modality_results.image.frame_results && (
                        <div>
                          <h3 className="text-lg font-medium mb-4">Frame Analysis</h3>
                          <div className="overflow-x-auto">
                            <table className="w-full border-collapse">
                              <thead>
                                <tr className="border-b">
                                  <th className="py-2 px-4 text-left">Frame</th>
                                  <th className="py-2 px-4 text-left">Probability</th>
                                  <th className="py-2 px-4 text-left">Confidence</th>
                                </tr>
                              </thead>
                              <tbody>
                                {analysisResult.modality_results.image.frame_results.map((frame) => (
                                  <tr key={frame.frame_idx} className="border-b">
                                    <td className="py-2 px-4">Frame {frame.frame_idx + 1}</td>
                                    <td className="py-2 px-4">{(frame.probability * 100).toFixed(1)}%</td>
                                    <td className="py-2 px-4">{(frame.confidence * 100).toFixed(1)}%</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}

                {analysisResult.modality_results.audio && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Audio Analysis</CardTitle>
                      <CardDescription>
                        Results from audio-based deepfake detection
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <div className="bg-muted p-4 rounded-lg">
                          <h3 className="text-lg font-medium mb-2">Verdict</h3>
                          <div className="text-2xl font-bold mb-1 capitalize">
                            {analysisResult.modality_results.audio.verdict}
                          </div>
                        </div>
                        <div className="bg-muted p-4 rounded-lg">
                          <h3 className="text-lg font-medium mb-2">Probability</h3>
                          <div className="text-2xl font-bold mb-1">
                            {(analysisResult.modality_results.audio.probability * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="bg-muted p-4 rounded-lg">
                          <h3 className="text-lg font-medium mb-2">Confidence</h3>
                          <div className="text-2xl font-bold mb-1">
                            {(analysisResult.modality_results.audio.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      {analysisResult.modality_results.audio.segment_results && (
                        <div>
                          <h3 className="text-lg font-medium mb-4">Segment Analysis</h3>
                          <div className="overflow-x-auto">
                            <table className="w-full border-collapse">
                              <thead>
                                <tr className="border-b">
                                  <th className="py-2 px-4 text-left">Segment</th>
                                  <th className="py-2 px-4 text-left">Time Range</th>
                                  <th className="py-2 px-4 text-left">Probability</th>
                                  <th className="py-2 px-4 text-left">Confidence</th>
                                </tr>
                              </thead>
                              <tbody>
                                {analysisResult.modality_results.audio.segment_results.map((segment) => (
                                  <tr key={segment.segment_idx} className="border-b">
                                    <td className="py-2 px-4">Segment {segment.segment_idx + 1}</td>
                                    <td className="py-2 px-4">
                                      {segment.start_time.toFixed(1)}s - {segment.end_time.toFixed(1)}s
                                    </td>
                                    <td className="py-2 px-4">{(segment.probability * 100).toFixed(1)}%</td>
                                    <td className="py-2 px-4">{(segment.confidence * 100).toFixed(1)}%</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}

                <Card>
                  <CardHeader>
                    <CardTitle>Cross-Modal Verification</CardTitle>
                    <CardDescription>
                      Analysis of consistency between different modalities
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="mb-6">
                      <h3 className="text-lg font-medium mb-2">Cross-Modal Score</h3>
                      <div className="text-2xl font-bold mb-1">
                        {(analysisResult.cross_modal_score * 100).toFixed(1)}%
                      </div>
                      <p className="text-sm text-gray-500">
                        Higher scores indicate greater consistency between modalities, suggesting more reliable results.
                      </p>
                    </div>

                    <div className="bg-muted p-4 rounded-lg">
                      <h3 className="text-lg font-medium mb-2">Interpretation</h3>
                      <p>
                        {analysisResult.cross_modal_score > 0.8
                          ? "High consistency between modalities. The analysis results are highly reliable."
                          : analysisResult.cross_modal_score > 0.6
                          ? "Good consistency between modalities. The analysis results are generally reliable."
                          : analysisResult.cross_modal_score > 0.4
                          ? "Moderate consistency between modalities. Some discrepancies exist between different analysis methods."
                          : "Low consistency between modalities. Different analysis methods produced conflicting results."}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>

          <TabsContent value="metadata">
            {analysisResult && (
              <Card>
                <CardHeader>
                  <CardTitle>Metadata Analysis</CardTitle>
                  <CardDescription>
                    Analysis of file metadata for signs of manipulation
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="mb-6">
                    <h3 className="text-lg font-medium mb-2">Metadata Score</h3>
                    <div className="text-2xl font-bold mb-1">
                      {(analysisResult.metadata_score * 100).toFixed(1)}%
                    </div>
                    <p className="text-sm text-gray-500">
                      Higher scores indicate more signs of potential manipulation in the file metadata.
                    </p>
                  </div>

                  {analysisResult.metadata_findings && analysisResult.metadata_findings.length > 0 && (
                    <div className="mb-6">
                      <h3 className="text-lg font-medium mb-4">Key Findings</h3>
                      <ul className="space-y-2">
                        {analysisResult.metadata_findings.map((finding, index) => (
                          <li key={index} className="bg-muted p-3 rounded-lg flex items-start">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-5 w-5 mr-2 text-yellow-500 mt-0.5 flex-shrink-0"
                              viewBox="0 0 20 20"
                              fill="currentColor"
                            >
                              <path
                                fillRule="evenodd"
                                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                                clipRule="evenodd"
                              />
                            </svg>
                            {finding}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="bg-muted p-4 rounded-lg">
                    <h3 className="text-lg font-medium mb-2">Interpretation</h3>
                    <p>
                      {analysisResult.metadata_score > 0.7
                        ? "Significant metadata anomalies detected. This file shows strong signs of manipulation based on metadata analysis."
                        : analysisResult.metadata_score > 0.4
                        ? "Some metadata anomalies detected. This file shows potential signs of manipulation based on metadata analysis."
                        : "Few or no metadata anomalies detected. This file does not show significant signs of manipulation based on metadata analysis."}
                    </p>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button asChild>
                    <a
                      href={`${API_URL}/api/report/${analysisResult.analysis_id}?format=pdf`}
                      download
                    >
                      Download Full Report
                    </a>
                  </Button>
                </CardFooter>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </main>
      <Footer />
    </div>
  );
};

export default MultimodalAnalysis;