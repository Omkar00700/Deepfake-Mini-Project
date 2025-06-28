
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { DetectionResult } from "@/types";
import { 
  AlertCircle, 
  CheckCircle, 
  AlertTriangle, 
  FileImage, 
  ArrowUpDown, 
  ChevronDown, 
  ChevronUp,
  Video,
  Image
} from "lucide-react";

interface HistoryTableProps {
  history: DetectionResult[];
}

type SortKey = "timestamp" | "probability";
type SortOrder = "asc" | "desc";

const HistoryTable = ({ history }: HistoryTableProps) => {
  const navigate = useNavigate();
  const [sortKey, setSortKey] = useState<SortKey>("timestamp");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");

  if (!history.length) {
    return (
      <div className="text-center py-12">
        <div className="bg-muted w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
          <FileImage className="w-8 h-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-medium mb-2">No detection history</h3>
        <p className="text-muted-foreground mb-6">
          Upload an image or video to see detection results
        </p>
        <Button onClick={() => navigate("/")}>Go to Upload</Button>
      </div>
    );
  }

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortOrder("desc");
    }
  };

  const sortedHistory = [...history].sort((a, b) => {
    if (sortKey === "timestamp") {
      const dateA = new Date(a.timestamp).getTime();
      const dateB = new Date(b.timestamp).getTime();
      return sortOrder === "asc" ? dateA - dateB : dateB - dateA;
    } else {
      return sortOrder === "asc"
        ? a.probability - b.probability
        : b.probability - a.probability;
    }
  });

  const getResultIcon = (probability: number) => {
    const probabilityPercentage = Math.round(probability * 100);
    if (probabilityPercentage > 70) {
      return <AlertCircle className="h-5 w-5 text-destructive" />;
    }
    if (probabilityPercentage > 40) {
      return <AlertTriangle className="h-5 w-5 text-amber-500" />;
    }
    return <CheckCircle className="h-5 w-5 text-emerald-600" />;
  };

  const getTypeIcon = (type: string) => {
    if (type === 'video') {
      return <Video className="h-4 w-4 text-blue-500 mr-1" />;
    }
    return <Image className="h-4 w-4 text-purple-500 mr-1" />;
  };

  return (
    <Card className="glass-panel">
      <div className="rounded-md overflow-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[100px]">Status</TableHead>
              <TableHead>File Name</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>
                <div
                  className="flex items-center cursor-pointer"
                  onClick={() => handleSort("probability")}
                >
                  Probability
                  <span className="ml-1">
                    {sortKey === "probability" ? (
                      sortOrder === "asc" ? (
                        <ChevronUp className="h-4 w-4" />
                      ) : (
                        <ChevronDown className="h-4 w-4" />
                      )
                    ) : (
                      <ArrowUpDown className="h-4 w-4" />
                    )}
                  </span>
                </div>
              </TableHead>
              <TableHead>
                <div
                  className="flex items-center cursor-pointer"
                  onClick={() => handleSort("timestamp")}
                >
                  Timestamp
                  <span className="ml-1">
                    {sortKey === "timestamp" ? (
                      sortOrder === "asc" ? (
                        <ChevronUp className="h-4 w-4" />
                      ) : (
                        <ChevronDown className="h-4 w-4" />
                      )
                    ) : (
                      <ArrowUpDown className="h-4 w-4" />
                    )}
                  </span>
                </div>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedHistory.map((item, index) => {
              const probabilityPercentage = Math.round(item.probability * 100);
              const detectionType = item.detectionType || 'image'; // Default to image for backward compatibility
              
              return (
                <TableRow key={index}>
                  <TableCell>{getResultIcon(item.probability)}</TableCell>
                  <TableCell className="font-medium">{item.imageName}</TableCell>
                  <TableCell>
                    <div className="flex items-center">
                      {getTypeIcon(detectionType)}
                      <span className="capitalize">{detectionType}</span>
                      {detectionType === 'video' && item.frameCount && (
                        <span className="text-xs text-muted-foreground ml-1">
                          ({item.frameCount} frames)
                        </span>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div
                      className={`px-2 py-1 rounded-full text-xs font-medium inline-flex items-center ${
                        probabilityPercentage > 70
                          ? "bg-destructive/10 text-destructive"
                          : probabilityPercentage > 40
                          ? "bg-amber-500/10 text-amber-500"
                          : "bg-emerald-600/10 text-emerald-600"
                      }`}
                    >
                      {probabilityPercentage}%
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {new Date(item.timestamp).toLocaleString()}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </Card>
  );
};

export default HistoryTable;
