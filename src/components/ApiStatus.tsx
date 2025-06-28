
import { useEffect, useState } from 'react';
import { AlertCircle, CheckCircle, WifiOff, Database, AlertTriangle, Loader2, Server, ExternalLink } from 'lucide-react';
import { checkApiStatus } from '@/services/status-service';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { API_URL, isUsingMockApi } from '@/services/config';
import { Button } from './ui/button';

interface ApiStatusProps {
  className?: string;
  showAlert?: boolean;
}

interface DatabaseInfo {
  connected: boolean;
  type: string;
  version?: string;
  error?: string;
}

const ApiStatus = ({ className = '', showAlert = false }: ApiStatusProps) => {
  const [isOnline, setIsOnline] = useState<boolean | null>(null);
  const [isChecking, setIsChecking] = useState(true);
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [dbInfo, setDbInfo] = useState<DatabaseInfo | null>(null);
  const [usingSupabase, setUsingSupabase] = useState<boolean>(false);
  const [checkCount, setCheckCount] = useState(0);
  const [lastCheckTime, setLastCheckTime] = useState<Date | null>(null);
  const [isMockApi, setIsMockApi] = useState<boolean>(isUsingMockApi());

  const checkStatus = async () => {
    setIsChecking(true);
    try {
      const status = await checkApiStatus();
      setIsOnline(status.status === 'online');
      setStatusMessage(status.message || "API is online");
      setLastCheckTime(new Date());
      
      if (status.database) {
        setDbInfo(status.database);
      }
      
      if (status.config && status.config.using_supabase !== undefined) {
        setUsingSupabase(status.config.using_supabase);
      }
    } catch (error) {
      setIsOnline(false);
      setStatusMessage("API is offline - could not connect");
      setDbInfo(null);
      setLastCheckTime(new Date());
    } finally {
      setIsChecking(false);
      setCheckCount(prev => prev + 1);
    }
  };

  useEffect(() => {
    // Initial check
    checkStatus();
    
    // Check API status every 30 seconds
    const interval = setInterval(checkStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);

  if (isChecking && checkCount === 0) {
    return (
      <div className={`flex items-center text-xs text-muted-foreground ${className}`}>
        <Loader2 className="h-3 w-3 mr-1 animate-spin text-muted-foreground" />
        <span>Checking API status...</span>
      </div>
    );
  }

  const isLovablePreview = window.location.hostname.includes('lovable');

  if (isOnline === false) {
    // If we're in Lovable preview, switch to mock API mode
    if (isLovablePreview && !isMockApi) {
      return (
        <Alert variant="destructive" className="mt-4">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <div className="space-y-2">
              <p>Lovable preview mode doesn't support running Python APIs directly. The app will use simulated data.</p>
              <Button 
                variant="default" 
                size="sm" 
                className="h-7 text-xs" 
                onClick={() => window.location.reload()}
              >
                <Server className="h-3 w-3 mr-1" /> Switch to Simulated Mode
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      );
    }
    
    return (
      <>
        <div className={`flex items-center text-xs text-destructive ${className}`}>
          <WifiOff className="h-3 w-3 mr-1 text-destructive" />
          <span>API offline - detection service unavailable</span>
        </div>
        
        {showAlert && (
          <Alert variant="destructive" className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              <div className="space-y-2">
                <p>Cannot connect to the API at {API_URL}. Please ensure the backend server is running at localhost:5000 or the configured URL.</p>
                
                {isLovablePreview && !isMockApi && (
                  <div className="p-2 rounded-md bg-background/50 text-xs">
                    <p className="font-medium mb-1">Lovable Preview Environment Detected</p>
                    <p>Lovable preview mode doesn't support running Python directly. The app will use simulated data.</p>
                    <Button 
                      variant="default" 
                      size="sm" 
                      className="h-7 text-xs mt-2" 
                      onClick={() => window.location.reload()}
                    >
                      <Server className="h-3 w-3 mr-1" /> Switch to Simulated Mode
                    </Button>
                  </div>
                )}
                
                <div className="flex items-center justify-between">
                  <span className="text-xs">
                    Last check: {lastCheckTime ? lastCheckTime.toLocaleTimeString() : 'never'}
                  </span>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="h-7 text-xs" 
                    onClick={checkStatus}
                  >
                    <Server className="h-3 w-3 mr-1" /> Check Again
                  </Button>
                </div>
              </div>
            </AlertDescription>
          </Alert>
        )}
      </>
    );
  }

  // Mock API banner (only show when in Lovable preview)
  const mockApiBanner = isMockApi && isLovablePreview && (
    <div className="text-xs px-2 py-1 bg-amber-100 text-amber-800 rounded mt-1 inline-flex gap-1 items-center">
      <AlertTriangle className="h-3 w-3" />
      Simulated API Mode
    </div>
  );

  if (isOnline === true && dbInfo && !dbInfo.connected) {
    return (
      <>
        <div className={`flex items-center text-xs text-amber-500 ${className}`}>
          <AlertTriangle className="h-3 w-3 mr-1 text-amber-500" />
          <span>API online but database disconnected</span>
          
          {dbInfo && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="ml-2 flex items-center gap-1">
                    <Database className="h-3 w-3 text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">
                      {usingSupabase ? 'Supabase' : dbInfo.type}
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Database: {dbInfo.type}</p>
                  {dbInfo.version && <p>Version: {dbInfo.version}</p>}
                  {dbInfo.error && <p className="text-destructive">Error: {dbInfo.error}</p>}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          {mockApiBanner}
        </div>
        
        {showAlert && (
          <Alert variant="default" className="mt-4 border-amber-500">
            <AlertTriangle className="h-4 w-4 text-amber-500" />
            <AlertDescription>
              API is online but cannot connect to the database. Some features may not work correctly.
              {lastCheckTime && (
                <div className="text-xs mt-1">
                  Last check: {lastCheckTime.toLocaleTimeString()}
                </div>
              )}
            </AlertDescription>
          </Alert>
        )}
      </>
    );
  }

  return (
    <div className={`flex flex-col items-center text-xs text-emerald-600 ${className}`}>
      <div className="flex items-center">
        <CheckCircle className="h-3 w-3 mr-1 text-emerald-600" />
        <span>API online - ready for detection</span>
        
        {dbInfo && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="ml-2 flex items-center gap-1">
                  <Database className="h-3 w-3 text-muted-foreground" />
                  <span className={`text-xs ${usingSupabase ? 'text-blue-500' : 'text-muted-foreground'}`}>
                    {usingSupabase ? 'Supabase' : dbInfo.type}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Database: {dbInfo.type}</p>
                {dbInfo.version && <p>Version: {dbInfo.version}</p>}
                {dbInfo.error && <p className="text-destructive">Error: {dbInfo.error}</p>}
                {lastCheckTime && <p className="text-xs">Last check: {lastCheckTime.toLocaleTimeString()}</p>}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
      {mockApiBanner}
    </div>
  );
};

export default ApiStatus;
