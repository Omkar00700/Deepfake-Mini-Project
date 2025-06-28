
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import HistoryTable from "@/components/HistoryTable";
import { Button } from "@/components/ui/button";
import { getHistory } from "@/services/api";
import { DetectionResult } from "@/types";
import { History as HistoryIcon, RefreshCw } from "lucide-react";

const History = () => {
  const [historyData, setHistoryData] = useState<DetectionResult[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const loadHistory = async () => {
    setIsLoading(true);
    try {
      const data = await getHistory();
      setHistoryData(data);
    } catch (error) {
      console.error("Failed to load history:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      
      <main className="flex-1 pt-24 pb-16">
        <div className="container">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-5xl mx-auto"
          >
            <div className="flex flex-col md:flex-row md:items-center justify-between mb-8">
              <div className="mb-4 md:mb-0">
                <div className="flex items-center">
                  <HistoryIcon className="h-6 w-6 mr-2 text-primary" />
                  <h1 className="text-3xl font-bold">Detection History</h1>
                </div>
                <p className="text-muted-foreground mt-1">
                  View your previous deepfake detection results
                </p>
              </div>
              
              <Button
                variant="outline"
                size="sm"
                onClick={loadHistory}
                disabled={isLoading}
                className="flex items-center"
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? "animate-spin" : ""}`} />
                Refresh
              </Button>
            </div>
            
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-20">
                <div className="w-12 h-12 rounded-full border-4 border-primary border-t-transparent animate-spin mb-4"></div>
                <p className="text-muted-foreground">Loading history...</p>
              </div>
            ) : (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
              >
                <HistoryTable history={historyData} />
              </motion.div>
            )}
          </motion.div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default History;
