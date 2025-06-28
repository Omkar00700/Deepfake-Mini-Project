
import React from 'react';
import { Button } from '@/components/ui/button';
import { ReportFormat, ReportFormatSelectorProps } from '@/types';
import { Loader2 } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

/**
 * Component for selecting report formats and triggering downloads
 */
const ReportFormatSelector: React.FC<ReportFormatSelectorProps> = ({
  formats,
  onSelect,
  disabled,
  isGenerating,
  selectedFormat = 'pdf',
  onGenerate
}) => {
  // Handle format change - only updates the format selection
  const handleFormatChange = (value: string) => {
    const format = value as 'pdf' | 'json' | 'csv';
    onSelect(format);
  };

  // Handle generate button click - explicitly triggers download
  const handleGenerateClick = () => {
    if (onGenerate) {
      onGenerate();
    } else {
      // Fallback for backward compatibility
      onSelect(selectedFormat as 'pdf' | 'json' | 'csv');
    }
  };

  return (
    <div className="flex space-x-2">
      <Select
        disabled={disabled || isGenerating}
        value={selectedFormat}
        onValueChange={handleFormatChange}
      >
        <SelectTrigger className="w-[140px]">
          <SelectValue placeholder="Select format" />
        </SelectTrigger>
        <SelectContent>
          {formats.map((format) => (
            <SelectItem key={format.id} value={format.value}>
              <div className="flex items-center">
                {format.icon}
                {format.label}
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Button
        className="flex-1"
        disabled={disabled || isGenerating}
        onClick={handleGenerateClick}
      >
        {isGenerating ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Generating...
          </>
        ) : (
          <>Download Report</>
        )}
      </Button>
    </div>
  );
};

export default ReportFormatSelector;
