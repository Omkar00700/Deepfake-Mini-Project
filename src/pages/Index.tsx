
import { useState } from "react";
import { motion } from "framer-motion";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import UploadArea from "@/components/UploadArea";
import ResultDisplay from "@/components/ResultDisplay";
import ApiStatus from "@/components/ApiStatus";
import { DetectionResult } from "@/types";
import { Layers, Shield, Cpu } from "lucide-react";

const Index = () => {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleUploadComplete = (newResult: DetectionResult) => {
    setResult(newResult);
  };

  const handleReset = () => {
    setResult(null);
  };

  const MotionLayeredDiv = ({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay }}
    >
      {children}
    </motion.div>
  );

  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      
      <main className="flex-1 pt-24 pb-16">
        <div className="container">
          <MotionLayeredDiv>
            <div className="text-center max-w-3xl mx-auto mb-12">
              <div className="inline-block mb-3 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium">
                Advanced Deepfake Detection
              </div>
              <h1 className="text-4xl md:text-5xl font-bold mb-4">
                Protect Authenticity with Deep<span className="text-primary">Defend</span>
              </h1>
              <p className="text-lg text-muted-foreground">
                Our AI-powered technology detects deepfakes with high precision, 
                specially optimized for Indian faces.
              </p>
              <div className="mt-4">
                <ApiStatus className="justify-center" showAlert={true} />
              </div>
            </div>
          </MotionLayeredDiv>

          <MotionLayeredDiv delay={0.1}>
            {!result ? (
              <UploadArea 
                onUploadComplete={handleUploadComplete} 
                isProcessing={isProcessing}
                setIsProcessing={setIsProcessing}
              />
            ) : (
              <ResultDisplay result={result} onReset={handleReset} />
            )}
          </MotionLayeredDiv>

          {!result && !isProcessing && (
            <MotionLayeredDiv delay={0.2}>
              <div className="mt-24 grid grid-cols-1 md:grid-cols-3 gap-8">
                <div className="p-6 rounded-lg glass-panel">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                    <Shield className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-medium mb-2">
                    Specialized Detection
                  </h3>
                  <p className="text-muted-foreground">
                    Our model is specifically fine-tuned for Indian faces, providing more accurate results.
                  </p>
                </div>
                
                <div className="p-6 rounded-lg glass-panel">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                    <Cpu className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-medium mb-2">
                    Advanced Algorithm
                  </h3>
                  <p className="text-muted-foreground">
                    Powered by deep learning models trained on extensive datasets for reliable detection.
                  </p>
                </div>
                
                <div className="p-6 rounded-lg glass-panel">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                    <Layers className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-medium mb-2">
                    Detailed Analysis
                  </h3>
                  <p className="text-muted-foreground">
                    Get comprehensive probability scores and maintain a history of all your detection results.
                  </p>
                </div>
              </div>
            </MotionLayeredDiv>
          )}
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;
