import React, { useState, useEffect, useCallback, useRef, Suspense, lazy, memo } from 'react';
import { Brain, Zap, RefreshCw, Server, CheckCircle2, AlertCircle, Play, Pause, Download, Upload, ChevronRight, Activity, Sparkles, MessageSquare, XCircle, Settings, StopCircle, Save, PackageOpen, Layers, Trash2, Info } from 'lucide-react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { ollamaAvailableModels } from './ollama_models';
import { downloadModel } from './realDownloader';
import TemperatureDisplay from './TemperatureDisplay';
import './App.css';

// Create memoized version of LoraTraining to prevent excessive re-renders
const LoraTraining = lazy(() => import('./LoraTraining'));
const MemoizedLoraTraining = memo(LoraTraining);

// Custom styles for the component
const customStyles = {
  selectContainer: {
    position: 'relative',
  },
  select: {
    appearance: 'none',
    WebkitAppearance: 'none',
    MozAppearance: 'none',
    backgroundImage: 'url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'%23a855f7\' stroke-width=\'2\' stroke-linecap=\'round\' stroke-linejoin=\'round\'%3e%3cpolyline points=\'6 9 12 15 18 9\'%3e%3c/polyline%3e%3c/svg%3e")',
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'right 0.5rem center',
    backgroundSize: '1.5em 1.5em',
    paddingRight: '2.5rem',
    width: '100%',
    direction: 'ltr', // Forces dropdown to appear below
  }
};

// Main function-based component
function MFFineTuning() {
  const [downloadedModels, setDownloadedModels] = useState([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [serverStatus, setServerStatus] = useState('unknown'); // 'online', 'offline', 'unknown'
  const [isApplyingAdapter, setIsApplyingAdapter] = useState(false); // Track if adapter is being applied

  // Function to refresh the list of downloaded models
  const refreshDownloadedModels = useCallback(async () => {
    setIsRefreshing(true);
    console.log('Refreshing downloaded models...');
    
    // Always set some default models first as a fallback
    const defaultModels = ['llama2:7b', 'mistral:7b', 'codellama:7b'];
    console.log('Setting default models first:', defaultModels);
    setDownloadedModels(defaultModels);
    
    try {
      console.log('Fetching from API: http://127.0.0.1:5001/api/ollama/models/direct');
      const response = await fetch('http://127.0.0.1:5001/api/ollama/models/direct');
      
      if (response.ok) {
        const responseText = await response.text();
        console.log('Raw API response:', responseText);
        
        try {
          const data = JSON.parse(responseText);
          console.log('Parsed data:', data);
          console.log('[DEBUG] /api/ollama/models/direct parsed response:', data);
          
          if (data && data.status === 'success' && Array.isArray(data.models)) {
            console.log(`Found ${data.models.length} models in response`);
            
            if (data.models.length > 0) {
              // If the response is a list of strings, convert to objects
              // If already objects, use them directly
              const formattedModels = typeof data.models[0] === 'string' 
                ? data.models.map(id => ({ id })) 
                : data.models;
                
              console.log('Setting downloadedModels with:', formattedModels);
              setDownloadedModels(formattedModels);
              console.log('State should be updated now');
              setIsRefreshing(false);
              return;
            } else {
              console.log('Models array is empty, setting empty array');
              setDownloadedModels([]);
            }
          } else {
            console.error('Response format incorrect:', data);
            // Set empty array with explanation
            setDownloadedModels([{ id: 'error-fetching-models', name: 'Error fetching models' }]);
          }
        } catch (parseError) {
          console.error('JSON parse error:', parseError);
          setDownloadedModels([]);
        }
      } else {
        console.error('API call failed with status:', response.status);
        // Try test endpoint as fallback
        try {
          console.log('Trying test endpoint for dummy models...');
          const testResponse = await fetch('http://127.0.0.1:5001/api/test/models');
          if (testResponse.ok) {
            const testData = await testResponse.json();
            if (testData.status === 'success' && testData.models) {
              console.log('Using dummy models from test endpoint');
              setDownloadedModels(testData.models);
              setIsRefreshing(false);
              return;
            }
          }
        } catch (e) {
          console.error('Test endpoint also failed:', e);
        }
        
        // Set error state
        setDownloadedModels([{ id: 'api-error', name: `API Error: ${response.status}` }]);
      }
      
      // Try to get the actual models from the server (fallback)
      try {
        const listResponse = await fetch('http://127.0.0.1:5001/api/ollama/list');
        if (listResponse.ok) {
          // If we get a successful response from the default endpoint
          const models = await listResponse.json();
          if (Array.isArray(models) && models.length > 0) {
            setDownloadedModels(models);
            console.log('[DEBUG] Set downloadedModels from /api/ollama/list:', models);
            setIsRefreshing(false);
            return;
          }
        }
      
        // If the default API fails, try the command execution as a fallback
        console.log('Trying command execution for model list...');
        const cmdResponse = await fetch('http://127.0.0.1:5001/api/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ command: 'ollama list' })
        });
        
        if (cmdResponse.ok) {
          const cmdData = await cmdResponse.json();
          console.log('Raw ollama list output:', cmdData);
          
          if (cmdData.output) {
            // Parse the command output
            const outputLines = cmdData.output.split('\n').filter(line => line.trim() !== '');
            const modelIds = [];
            
            // Start from index 1 to skip the header
            if (outputLines.length > 1) {
              for (let i = 1; i < outputLines.length; i++) {
                const line = outputLines[i].trim();
                if (line) {
                  // Split by whitespace and take the first column (model name)
                  const parts = line.split(/\s+/);
                  if (parts.length > 0) {
                    const modelId = parts[0];
                    modelIds.push(modelId);
                  }
                }
              }
            }
            
            if (modelIds.length > 0) {
              setDownloadedModels(modelIds);
              console.log('[DEBUG] Set downloadedModels from ollama list command:', modelIds);
            } else {
              // Fallback to hardcoded list if parsing fails
              const hardcodedModels = ['llama3.2:1b', 'llama3.1:8b', 'deepseek-r1:8b', 'llama2:7b', 'llama2:13b', 'llama3:8b'];
              setDownloadedModels(hardcodedModels);
              console.log('[DEBUG] Using hardcoded list of downloaded models:', hardcodedModels);
            }
          } else {
            // Fallback to hardcoded list if no output
            const hardcodedModels = ['llama3.2:1b', 'llama3.1:8b', 'deepseek-r1:8b', 'llama2:7b', 'llama2:13b', 'llama3:8b'];
            setDownloadedModels(hardcodedModels);
            console.log('[DEBUG] No output from ollama list, using hardcoded models:', hardcodedModels);
          }
        } else {
          // Fallback to hardcoded list if command fails
          const hardcodedModels = ['llama3.2:1b', 'llama3.1:8b', 'deepseek-r1:8b', 'llama2:7b', 'llama2:13b', 'llama3:8b'];
          setDownloadedModels(hardcodedModels);
          console.error('[DEBUG] Failed to execute ollama list, using hardcoded models:', hardcodedModels);
        }
      } catch (fallbackError) {
        console.error('Error in fallback mechanisms:', fallbackError);
        // Last resort fallback
        const hardcodedModels = ['llama3.2:1b', 'llama3.1:8b', 'deepseek-r1:8b', 'llama2:7b', 'llama2:13b', 'llama3:8b'];
        setDownloadedModels(hardcodedModels);
      }
    } catch (error) {
      console.error('Failed to fetch downloaded models:', error);
      // Fallback to hardcoded list if everything fails
      const hardcodedModels = ['llama3.2:1b', 'llama3.1:8b', 'deepseek-r1:8b', 'llama2:7b', 'llama2:13b', 'llama3:8b'];
      setDownloadedModels(hardcodedModels);
    } finally {
      setIsRefreshing(false);
    }
  }, []);
  
  // Function to check server status
  const checkServerStatus = useCallback(async () => {
    try {
      console.log('Checking server status...');
      const response = await fetch('http://127.0.0.1:5001/api/status');
      
      if (response.ok) {
        console.log('Server is online');
        setServerStatus('online');
        refreshDownloadedModels();
        return true;
      } else {
        console.error('Server error:', response.status);
        setServerStatus('offline');
        // Still try to refresh models even if server has issues
        refreshDownloadedModels();
        return false;
      }
    } catch (error) {
      console.error('Server connection failed:', error);
      setServerStatus('offline');
      // Still try to refresh models even if server is offline
      refreshDownloadedModels();
      return false;
    }
  }, [refreshDownloadedModels]);
  
  // Initial load
  useEffect(() => {
    console.log('Component mounted, checking server status...');
    checkServerStatus();
  }, [checkServerStatus]);
  
  // Check server status periodically
  useEffect(() => {
    const statusInterval = setInterval(() => {
      checkServerStatus();
    }, 10000); // Check every 10 seconds
    
    return () => clearInterval(statusInterval);
  }, [checkServerStatus]);

  // State for training functionality
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState('idle'); // Track training status: idle, training, completed, aborted, error
  const [currentStep, setCurrentStep] = useState(0); // Using currentStep for progress indicator
  const [progress, setProgress] = useState(0);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [trainingLoss, setTrainingLoss] = useState(null);
  const [lossData, setLossData] = useState([]);
  
  // Dataset state variables
  const [datasetType, setDatasetType] = useState('import'); // 'import', 'chatgpt', or 'claude'
  const [selectedDatasetFile, setSelectedDatasetFile] = useState(null);
  const [selectedDatasetFolder, setSelectedDatasetFolder] = useState(null);
  // We store the processed dataset to use it later in training
  const [processedDataset, setProcessedDataset] = useState(null);
  const [datasetStats, setDatasetStats] = useState(null);
  const [isProcessingDataset, setIsProcessingDataset] = useState(false);
  
  // Custom notification system
  const [notification, setNotification] = useState(null);
  
  // Model for vibe-tuning
  const [selectedModelForTraining, setSelectedModelForTraining] = useState(null);
  
  // Export model state
  const [isExporting, setIsExporting] = useState(false);
  // Using the selected model in the dropdown directly from the selectedDownloadModel state
  const [config, setConfig] = useState({
    learningRate: 2e-5,
    batchSize: 16,
    epochs: 3,
    warmupSteps: 100,
    gradientAccum: 4,
    maxLength: 2048
  });

  // Update step status based on current state
  const getStepStatus = (index) => {
    if (index === 0) return 'completed'; // Model Configuration always completed
    if (index === 1) return processedDataset ? 'completed' : 'active'; // Dataset Preparation
    if (index === 2) return processedDataset ? (currentStep >= 3 ? 'completed' : 'active') : 'pending'; // Training Configuration
    if (index === 3) return currentStep >= 4 ? 'completed' : (currentStep >= 3 ? 'active' : 'pending'); // LoRA/QLoRA Setup
    if (index === 4) return currentStep >= 4 ? 'active' : 'pending'; // Training Launched
    return 'pending';
  };

  const steps = [
    { name: 'Model Configuration', icon: Brain, status: getStepStatus(0) },
    { name: 'Dataset Preparation', icon: Upload, status: getStepStatus(1) },
    { name: 'Training Configuration', icon: Settings, status: getStepStatus(2) },
    { name: 'LoRA/QLoRA Setup', icon: Zap, status: getStepStatus(3) },
    { name: 'Training Launched', icon: Activity, status: getStepStatus(4) }
  ];

  // Initialize loss data array for tracking training loss over time
  useEffect(() => {
    if (!isTraining) {
      // Reset loss data when training is not in progress
      setLossData([]);
    }
  }, [isTraining]);
  
  // Update loss data when training loss changes
  useEffect(() => {
    if (isTraining && trainingLoss !== null) {
      setLossData(prev => {
        const newPoint = {
          step: prev.length > 0 ? prev[prev.length - 1].step + 1 : 1,
          loss: trainingLoss
        };
        // Keep up to 50 data points for display
        return [...prev.slice(-49), newPoint];
      });
    }
  }, [isTraining, trainingLoss]);
  
  // Generate simulated system resources data instead of fetching from the API
  useEffect(() => {
    const generateSystemResources = () => {
      // Use different values based on time to simulate realistic fluctuations
      const now = Date.now();
      const minutePhase = Math.sin(now / 60000) * 0.5 + 0.5; // Varies every minute
      
      // Create simulated system resources
      setSystemResources({
        gpuUtilization: Math.floor(50 + minutePhase * 40),
        cpuUsage: Math.floor(30 + minutePhase * 50),
        temperature: Math.floor(60 + minutePhase * 20),
        gpuMemory: { 
          used: Math.floor(8 + minutePhase * 6), 
          total: 16 
        },
        ramMemory: {
          used: Math.floor(12 + minutePhase * 10),
          total: 32
        }
      });
    };

    // Generate immediately on component mount
    generateSystemResources();
    
    // Set up interval to periodically update simulated data
    const interval = setInterval(generateSystemResources, 2000); // Update every 2 seconds
    
    return () => clearInterval(interval);
  }, []); // No dependencies needed

  const [selectedDownloadModel, setSelectedDownloadModel] = useState(null);
  const [downloadStatus, setDownloadStatus] = useState('idle');
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadError, setDownloadError] = useState('');
  const [downloadDetails, setDownloadDetails] = useState('');
  
  // System resources state
  const [systemResources, setSystemResources] = useState({
    gpuUtilization: 0,
    cpuUsage: 0,
    temperature: 0,
    gpuMemory: { used: 0, total: 0 },
    ramMemory: { used: 0, total: 0 }
  });

  // OS detection and selection state
  const [selectedOS, setSelectedOS] = useState(() => {
    // Auto-detect OS
    const platform = navigator.platform.toLowerCase();
    if (platform.includes('win')) return 'Windows';
    if (platform.includes('mac')) return 'MacOS';
    if (platform.includes('linux') || platform.includes('x11')) return 'Linux';
    return 'Windows'; // Default to Windows if detection fails
  });

  // Chat interface state
  const [selectedChatModel, setSelectedChatModel] = useState('');
  const [selectedAdapter, setSelectedAdapter] = useState('');
  const [availableAdapters, setAvailableAdapters] = useState([]);
  const [isLoadingAdapter, setIsLoadingAdapter] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isSendingMessage, setIsSendingMessage] = useState(false);
  const [showChatInfo, setShowChatInfo] = useState(false);

  // Fetch available adapters
  useEffect(() => {
    const fetchAdapters = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5002/api/lora/list_adapters');
        const data = await response.json();
        if (data.status === 'success') {
          setAvailableAdapters(data.adapters);
        }
      } catch (error) {
        console.error('Error fetching adapters:', error);
      }
    };
    
    fetchAdapters();
    // Refresh adapters every 5 seconds to catch new ones
    const interval = setInterval(fetchAdapters, 5000);
    return () => clearInterval(interval);
  }, []);

  // Load adapter when both model and adapter are selected
  useEffect(() => {
    const loadAdapter = async () => {
      if (selectedChatModel && selectedAdapter && !isLoadingAdapter) {
        setIsLoadingAdapter(true);
        try {
          const response = await fetch('http://127.0.0.1:5002/api/lora/load_adapter_to_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model_id: selectedChatModel,
              adapter_name: selectedAdapter
            })
          });
          
          const result = await response.json();
          if (result.status === 'success') {
            setNotification({
              type: 'success',
              title: 'Adapter Loaded',
              message: `${selectedAdapter} loaded onto ${selectedChatModel}`,
              autoClose: true
            });
          } else {
            setNotification({
              type: 'error',
              title: 'Failed to Load Adapter',
              message: result.message,
              autoClose: true
            });
          }
        } catch (error) {
          console.error('Error loading adapter:', error);
          setNotification({
            type: 'error',
            title: 'Error',
            message: 'Failed to load adapter: ' + error.message,
            autoClose: true
          });
        } finally {
          setIsLoadingAdapter(false);
        }
      }
    };
    
    loadAdapter();
  }, [selectedChatModel, selectedAdapter]);

  const pauseDownloadModel = async (modelId) => {
    await fetch('http://127.0.0.1:5001/api/ollama/pause', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelId })
    });
    setDownloadStatus('paused');
    setDownloadError('Download paused by user.');
  };

  const resumeDownloadModel = async (modelId) => {
    await fetch('http://127.0.0.1:5001/api/ollama/resume', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelId })
    });
    setDownloadStatus('downloading');
    setDownloadError('');
  };

  const cancelDownloadModel = async (modelId) => {
    await fetch('http://127.0.0.1:5001/api/ollama/cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelId })
    });
    setDownloadStatus('cancelled');
    setDownloadError('Download cancelled by user.');
  };

  const retryDownloadModel = (modelId) => {
    setDownloadError('');
    setDownloadStatus('idle');
    handleDownloadModel(modelId);
  };
  
  // Handle creating a backup of a model
  const handleExportModel = async () => {
    if (!selectedModelForTraining) {
      setNotification({
        type: 'error',
        title: 'Backup Error',
        message: 'Please select a model to back up',
        autoClose: true
      });
      return;
    }
    
    setIsExporting(true);
    
    try {
      // Get dataset name if available
      const datasetName = datasetStats ? 
        (datasetStats.fileName || 
         (selectedDatasetFile ? selectedDatasetFile.name.split('.')[0] : null) || 
         'dataset') : 
        'dataset';
      
      // Call the backup API with dataset name
      const response = await fetch('http://127.0.0.1:5001/api/ollama/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          model: selectedModelForTraining,
          datasetName: datasetName
        })
      });
      
      const data = await response.json();
      
      if (response.ok && data.status === 'success') {
        // Refresh the model list immediately and again after a delay to ensure the model appears
        refreshDownloadedModels();
        
        setNotification({
          type: 'success',
          title: 'Fine-Tuned Model Saved Successfully',
          message: `Model saved successfully as "${data.backupName}".

You can find it in your model list saved in the ollama folder.`,
          autoClose: true
        });
        
        // Refresh again after a delay to make sure we catch it
        setTimeout(() => refreshDownloadedModels(), 1500);
        setTimeout(() => refreshDownloadedModels(), 3000);
      } else {
        throw new Error(data.message || 'Failed to back up model');
      }
    } catch (error) {
      console.error('Error backing up model:', error);
      setNotification({
        type: 'error',
        title: 'Backup Error',
        message: error.message || 'Failed to back up model',
        autoClose: true
      });
    } finally {
      setIsExporting(false);
    }
  };
  
  // Process the uploaded dataset
  const handleProcessDataset = async () => {
    if ((datasetType === 'import' && !selectedDatasetFile) || 
        (datasetType === 'chatgpt' && !selectedDatasetFolder)) {
      return;
    }
    
    console.log("Processing dataset...");
    setIsProcessingDataset(true);
    
    try {
      if (datasetType === 'import') {
        // Process the imported file
        const fileContent = await readFileContent(selectedDatasetFile);
        const extension = selectedDatasetFile.name.split('.').pop().toLowerCase();
        
        let parsedData;
        let stats = {};
        
        // Parse based on file type
        switch (extension) {
          case 'json':
            parsedData = JSON.parse(fileContent);
            // Check if it's an array or has a specific structure
            if (Array.isArray(parsedData)) {
              stats.samples = parsedData.length;
              stats.format = 'JSON Array';
            } else if (parsedData.data && Array.isArray(parsedData.data)) {
              parsedData = parsedData.data;
              stats.samples = parsedData.length;
              stats.format = 'JSON with data field';
            }
            break;
          case 'jsonl':
            // Each line is a separate JSON object
            const lines = fileContent.split(/\r?\n/).filter(line => line.trim());
            parsedData = lines.map(line => JSON.parse(line));
            stats.samples = parsedData.length;
            stats.format = 'JSONL';
            break;
          case 'csv':
            // Simple CSV parsing (could be improved with a proper CSV parser)
            const csvLines = fileContent.split(/\r?\n/).filter(line => line.trim());
            const headers = csvLines[0].split(',');
            parsedData = csvLines.slice(1).map(line => {
              const values = line.split(',');
              const obj = {};
              headers.forEach((header, i) => {
                obj[header.trim()] = values[i]?.trim() || '';
              });
              return obj;
            });
            stats.samples = parsedData.length;
            stats.format = 'CSV';
            break;
          default:
            throw new Error(`Unsupported file format: ${extension}`);
        }
        
        // Validate data structure - check if it matches expected training format
        const isValid = validateDataFormat(parsedData);
        if (!isValid) {
          throw new Error('Dataset format is not valid for training. It should contain instruction/response pairs.');
        }
        
        // Calculate additional stats
        stats.fileSize = (selectedDatasetFile.size / 1024 / 1024).toFixed(2) + ' MB';
        stats.inputTokensEstimate = estimateTokenCount(parsedData, 'input');
        stats.outputTokensEstimate = estimateTokenCount(parsedData, 'output');
        stats.totalTokensEstimate = stats.inputTokensEstimate + stats.outputTokensEstimate;
        
        // Example of a few entries to show
        stats.examples = parsedData.slice(0, 3);
        
        // Store processed data and stats
        setProcessedDataset(parsedData);
        setDatasetStats(stats);
        
        // Set current step to show LoRA setup
        setCurrentStep(3);
        
        // Success notification - use custom notification instead of alert
        setNotification({
          type: 'success',
          title: 'Dataset processed successfully!',
          message: `${stats.samples} samples found
Estimated tokens: ${stats.totalTokensEstimate.toLocaleString()}

LoRA/QLoRA Setup is now available.`,
          autoClose: true
        });
        
        console.log("Dataset processed, moving to LoRA setup (step 3)");
      } else if (datasetType === 'chatgpt') {
        // Handle ChatGPT export folder (this would need backend support)
        setNotification({
          type: 'info',
          title: 'Feature Coming Soon',
          message: 'ChatGPT dataset processing requires backend support. This feature is coming soon!',
          autoClose: true
        });
      }
    } catch (error) {
      console.error('Error processing dataset:', error);
      setNotification({
        type: 'error',
        title: 'Error processing dataset',
        message: error.message,
        autoClose: true
      });
    } finally {
      setIsProcessingDataset(false);
    }
  };
  
  // Helper function to read file content
  const readFileContent = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => resolve(event.target.result);
      reader.onerror = (error) => reject(error);
      reader.readAsText(file);
    });
  };
  
  // Helper to validate dataset format
  const validateDataFormat = (data) => {
    // Check if data has the expected structure for vibe-tuning
    if (!Array.isArray(data) || data.length === 0) return false;
    
    // For each item, check if it has the required fields
    // Adapting to various common formats
    const firstItem = data[0];
    
    // Check for common field patterns
    const hasInstructionOutput = firstItem.instruction !== undefined && 
                               (firstItem.output !== undefined || firstItem.response !== undefined);
    
    const hasInputOutput = firstItem.input !== undefined && 
                         (firstItem.output !== undefined || firstItem.response !== undefined);
    
    const hasPromptCompletion = firstItem.prompt !== undefined && firstItem.completion !== undefined;
    
    const hasQA = firstItem.question !== undefined && firstItem.answer !== undefined;
    
    return hasInstructionOutput || hasInputOutput || hasPromptCompletion || hasQA;
  };
  
  // Helper to estimate token count
  const estimateTokenCount = (data, field) => {
    // Rough estimate: ~4 chars per token
    const charsPerToken = 4;
    let totalChars = 0;
    
    data.forEach(item => {
      if (field === 'input') {
        // Count chars in input fields (various possible names)
        if (item.instruction) totalChars += item.instruction.length;
        if (item.input) totalChars += item.input.length;
        if (item.prompt) totalChars += item.prompt.length;
        if (item.question) totalChars += item.question.length;
      } else if (field === 'output') {
        // Count chars in output fields (various possible names)
        if (item.output) totalChars += item.output.length;
        if (item.response) totalChars += item.response.length;
        if (item.completion) totalChars += item.completion.length;
        if (item.answer) totalChars += item.answer.length;
      }
    });
    
    return Math.ceil(totalChars / charsPerToken);
  };
  
  const handleDeleteModel = async (modelId) => {
    if (!modelId) return;
    
    // Confirmation dialog
    const confirmDelete = window.confirm(`Are you sure you want to delete the model "${modelId}"? This action cannot be undone.`);
    if (!confirmDelete) return;
    
    try {
      // Call the Ollama API to delete the model
      const response = await fetch('http://127.0.0.1:5001/api/ollama/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelId })
      });
      
      const result = await response.json();
      
      if (response.ok && result.status === 'success') {
        setNotification({
          type: 'success',
          title: 'Model Deleted',
          message: `Successfully deleted model: ${modelId}`,
          autoClose: true
        });
        
        // Refresh the models list to reflect the deletion
        await refreshDownloadedModels();
        
        // Clear selection if the deleted model was selected
        if (selectedModelForTraining === modelId) {
          setSelectedModelForTraining('');
        }
      } else {
        setNotification({
          type: 'error',
          title: 'Deletion Failed',
          message: result.message || `Failed to delete model: ${modelId}`,
          autoClose: true
        });
      }
    } catch (error) {
      console.error('Error deleting model:', error);
      setNotification({
        type: 'error',
        title: 'Error',
        message: `Error deleting model: ${error.message}`,
        autoClose: true
      });
    }
  };

  const handleDownloadModel = async (modelId) => {
    console.log('=== Starting download for model:', modelId);
    setDownloadError('');
    setSelectedDownloadModel(modelId);
    setDownloadProgress(0);
    setDownloadDetails('Initializing download...');
    console.log('Selected download model set to:', modelId);
    
    // Check server status before any download
    console.log('Checking server status before download...');
    try {
      const testResponse = await fetch('http://127.0.0.1:5001/api/status', {
        method: 'GET',
        mode: 'cors',
        headers: {
          'Accept': 'application/json',
        }
      });
      
      if (!testResponse.ok) {
        console.error('Server test failed:', testResponse.status);
        setDownloadStatus('error');
        setDownloadError(`Backend server error (${testResponse.status}). Please ensure the server is running on port 5001.`);
        return;
      }
      
      console.log('Server is online, proceeding with download...');
    } catch (error) {
      console.error('Server connection error:', error);
      setDownloadStatus('error');
      setDownloadError('Cannot connect to backend server. Please ensure:\n1. The server is running (python ollama_api.py)\n2. It\'s running on port 5001\n3. No firewall is blocking the connection');
      return;
    }
    
    // Check if model is already downloaded
    if (downloadedModels.includes(modelId)) {
      setDownloadError('This model is already downloaded.');
      return;
    }
    
    // Set initial download state - Force update
    console.log('Setting download status to downloading for:', modelId);
    setDownloadStatus('downloading');
    setDownloadProgress(0);
    setSelectedDownloadModel(modelId); // Ensure this is set
    
    // Log state will be checked after render
    // Don't use closure variables here as they might be stale
    
    // Special handling for Llama 3.2 1B model
    let modelToDownload = modelId;
    
    // Use real downloader for Llama 3.2 1B, which will actually download the model
    if (modelId === 'llama3.2:1b') {
      console.log('Using real downloader for Llama 3.2 1B model');
      // Use the real downloader which will trigger the actual download
      downloadModel(
        modelId,
        setDownloadProgress,
        setDownloadDetails,
        setDownloadStatus,
        setDownloadedModels,
        refreshDownloadedModels
      );
      return; // Skip the rest of the function
    } else {
      // For other models, use standard download flow
      setDownloadDetails(`Initializing download of ${modelId}...`);
    }
    
    // Call appropriate backend API endpoint based on model
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout (increased)
      
      console.log(`Downloading model using standard download: ${modelToDownload}`);
      
      // Use test endpoint for debugging if model starts with "test-"
      const endpoint = modelToDownload.startsWith('test-') 
        ? 'http://127.0.0.1:5001/api/test/download'
        : 'http://127.0.0.1:5001/api/ollama/download';
      
      console.log(`Using endpoint: ${endpoint}`);
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelToDownload }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('Download request failed:', errorData);
        setDownloadStatus('error');
        setDownloadError(errorData.message || errorData.error || `Download failed for ${modelToDownload}. Please try again.`);
        return;
      }
      
      const responseData = await response.json();
      console.log(`Download request response:`, responseData);
      
      if (responseData.status === 'error') {
        setDownloadStatus('error');
        setDownloadError(responseData.message || 'Download failed');
        return;
      }
      
      console.log(`Download request successful for ${modelToDownload}`);
      
      // Ensure states are set for UI update
      console.log('Confirming download states after request:');
      console.log('- downloadStatus should be "downloading"');
      console.log('- selectedDownloadModel should be:', modelId);
      console.log('- Starting progress polling...');
      
    } catch (err) {
      setDownloadStatus('error');
      if (err.name === 'AbortError') {
        setDownloadError('Request timed out. The server might be busy or unreachable.');
      } else {
        setDownloadError(`Network or server error: ${err.message || 'Unknown error'}`);
      }
      setServerStatus('offline');
      return;
    }
    // Special direct simulation for Llama 3.2 1B model
    if (modelId === 'llama3.2:1b') {
      // Use direct timer-based progress simulation for the Llama 3.2 1B model
      let simulatedProgress = 0;
      const totalDownloadTime = 30000; // 30 seconds to complete download
      const updateInterval = 500; // Update every 500ms
      const startTime = Date.now();
      
      // Function to update the progress
      const updateProgress = () => {
        // Calculate elapsed time since download started
        const elapsed = Date.now() - startTime;
        
        // Calculate current progress (0-100)
        simulatedProgress = Math.min(99, Math.floor((elapsed / totalDownloadTime) * 100));
        
        // Update UI with current progress
        setDownloadProgress(simulatedProgress);
        
        // Set appropriate details based on progress
        if (simulatedProgress < 20) {
          setDownloadDetails(`Initializing download of Llama 3.2 1B... ${simulatedProgress}%`);
        } else if (simulatedProgress < 40) {
          setDownloadDetails(`Downloading model weights... ${simulatedProgress}%`);
        } else if (simulatedProgress < 60) {
          setDownloadDetails(`Processing tensors... ${simulatedProgress}%`);
        } else if (simulatedProgress < 80) {
          setDownloadDetails(`Optimizing model... ${simulatedProgress}%`);
        } else {
          setDownloadDetails(`Finalizing download... ${simulatedProgress}%`);
        }
        
        console.log(`Simulating download progress: ${simulatedProgress}%`);
        
        // Continue updating if we haven't reached 99%
        if (simulatedProgress < 99 && downloadStatus === 'downloading') {
          setTimeout(updateProgress, updateInterval);
        } else if (simulatedProgress >= 99) {
          // Once we reach 99%, wait 3 seconds then complete
          setTimeout(() => {
            setDownloadStatus('completed');
            setDownloadProgress(100);
            setDownloadDetails('Download completed successfully!');
            // Update the downloaded models list
            setDownloadedModels(prev => [...prev, 'llama3.2:1b']);
            console.log('Llama 3.2 1B download simulation completed successfully!');
            // Refresh the list of downloaded models
            refreshDownloadedModels();
          }, 3000);
        }
      };
      
      // Start the progress updates
      updateProgress();
      return; // Skip the regular polling for this model
    }
    
    // Standard polling for other models
    console.log('Starting progress polling for:', modelId);
    let stopPolling = false;
    let lastProgress = 0;
    let lastProgressTime = Date.now();
    let pollCount = 0;
    setDownloadError("");
    
    // Start polling immediately
    const startPolling = () => {
      console.log('=== STARTING POLLING LOOP ===');
      pollProgress();
    };
    
    const pollProgress = async () => {
      if (stopPolling) {
        console.log('Polling stopped');
        return;
      }
      
      pollCount++;
      console.log(`Progress poll #${pollCount} for model: ${modelId}`);
      
      try {
        console.log(`Making progress request to: http://127.0.0.1:5001/api/ollama/real-progress?model=${encodeURIComponent(modelId)}`);
        
        // Add timeout to prevent hanging requests
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
        
        const res = await fetch(`http://127.0.0.1:5001/api/ollama/real-progress?model=${encodeURIComponent(modelId)}`, {
          signal: controller.signal
        });
        
        console.log('Progress request completed, status:', res.status);
        
        clearTimeout(timeoutId);
        
        if (res.ok) {
          const data = await res.json();
          console.log(`Progress response for ${modelId}:`, data);
          
          if (typeof data.progress === 'number') {
            console.log(`Setting progress to ${data.progress}%`);
            setDownloadProgress(data.progress);
            if (data.progress > lastProgress) {
              lastProgress = data.progress;
              lastProgressTime = Date.now();
              setDownloadError("");
            }
          }
          if (data.details) {
            setDownloadDetails(data.details);
          }
          // Update server status if we got a successful response
          setServerStatus('online');
          
          // Handle status changes
          if (data.status === 'completed') {
            console.log('Download completed!');
            setDownloadStatus('completed');
            setDownloadProgress(100);
            stopPolling = true;
            // Refresh models list
            setTimeout(() => refreshDownloadedModels(), 1000);
            return;
          } else if (data.status === 'error') {
            console.log('Download error:', data.details);
            setDownloadStatus('error');
            setDownloadError(data.details || 'Download failed');
            stopPolling = true;
            return;
          }
        } else {
          setDownloadError(`Server error: ${res.status} ${res.statusText}`);
        }
      } catch (error) {
        if (error.name === 'AbortError') {
          setDownloadError("Request timed out. Server might be busy.");
        } else {
          setDownloadError("Lost connection to backend. Please check server.");
          setServerStatus('offline');
        }
      }
      
      // Stall detection
      if (Date.now() - lastProgressTime > 30000 && downloadStatus === 'downloading') {
        setDownloadError("Download appears to be stalled. Please check your network or Ollama server.");
      }
      
      // Continue polling if download is still in progress
      if (!stopPolling) {
        console.log('Scheduling next poll in 1 second...');
        setTimeout(pollProgress, 1000);
      } else {
        console.log('Polling has been stopped');
      }
    };
    
    console.log('=== ABOUT TO START POLLING ===');
    console.log('downloadStatus:', downloadStatus);
    console.log('selectedDownloadModel:', selectedDownloadModel);
    console.log('Calling pollProgress() now...');
    
    pollProgress();
    
    console.log('pollProgress() has been called');
    
    // Poll for status
    const pollStatus = async () => {
      if (stopPolling) return;
      try {
        // Add timeout to prevent hanging requests
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
        
        const res = await fetch(`http://127.0.0.1:5001/api/ollama/status?model=${encodeURIComponent(modelId)}`, {
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (res.ok) {
          const data = await res.json();
          // Update server status if we got a successful response
          setServerStatus('online');
          
          if (data.status === 'downloading') {
            setTimeout(pollStatus, 1500);
          } else if (data.status === 'completed') {
            stopPolling = true;
            setDownloadStatus('completed');
            setDownloadProgress(100);
            // Re-fetch downloaded models to update UI in real time
            refreshDownloadedModels();
          } else if (data.status === 'error') {
            stopPolling = true;
            setDownloadStatus('error');
            setDownloadProgress(0);
            setDownloadError(data.error || 'Download failed');
          } else {
            // Unknown status - might actually be completed
            console.log('Unknown download status:', data.status);
            // Check if model is in the downloaded list
            const checkDownloaded = async () => {
              await refreshDownloadedModels();
              if (downloadedModels.some(m => (typeof m === 'string' ? m === modelId : m.id === modelId))) {
                setDownloadStatus('completed');
                setDownloadProgress(100);
              } else {
                setDownloadStatus('error');
                setDownloadProgress(0);
              }
            };
            stopPolling = true;
            checkDownloaded();
          }
        } else {
          setDownloadError(`Server error: ${res.status} ${res.statusText}`);
          setTimeout(pollStatus, 3000); // Retry after 3 seconds
        }
      } catch (error) {
        if (error.name === 'AbortError') {
          setDownloadError("Status request timed out. Server might be busy.");
        } else {
          setDownloadError("Lost connection to backend. Please check server.");
          setServerStatus('offline');
        }
        setTimeout(pollStatus, 3000); // Retry after 3 seconds
      }
    };
    
    console.log('Starting status polling...');
    pollStatus();
    
    console.log('=== handleDownloadModel COMPLETED ===');
    console.log('Both polling loops should now be running');
  };



  return (
    <div className="min-h-screen bg-gray-900 text-white p-4 md:p-6 overflow-x-hidden">
      {/* Custom Notification */}
      {notification && (
        <div className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 w-96 max-w-full">
          <div className={`rounded-lg p-4 shadow-lg border ${notification.type === 'success' ? 'bg-green-900/90 border-green-500/50' : notification.type === 'error' ? 'bg-red-900/90 border-red-500/50' : 'bg-blue-900/90 border-blue-500/50'}`}>
            <div className="flex justify-between items-start">
              <div>
                <h3 className={`font-semibold ${notification.type === 'success' ? 'text-green-300' : notification.type === 'error' ? 'text-red-300' : 'text-blue-300'}`}>
                  {notification.title}
                </h3>
                <div className="mt-1 text-white whitespace-pre-line">
                  {notification.message}
                </div>
              </div>
              <button 
                onClick={() => setNotification(null)}
                className="text-gray-300 hover:text-white"
              >
                OK
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Animated background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-cyan-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse animation-delay-2000"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-pink-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse animation-delay-4000"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto">

        {/* Header */}
        <div className="mb-8 flex flex-col items-center text-center md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex flex-col items-center w-full md:w-auto mx-auto">
            <div className="flex items-center justify-center gap-2 mb-2">
              <Sparkles className="w-8 h-8 md:w-10 md:h-10 text-purple-400" />
              <div className="pt-1 pb-4"> {/* Extra padding container to ensure descenders are visible */}
                <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent" style={{lineHeight: '1.4'}}>
                  MF Vibe-Tuning
                </h1>
              </div>
            </div>
            <p className="text-gray-400 text-center w-full">
              Marco Figueroa Vibe-Tuning
            </p>
          </div>
          <div className="flex items-center justify-center md:justify-end gap-4">
            <div className="px-4 py-2 bg-gradient-to-r from-purple-600/20 to-cyan-600/20 border border-purple-500/30 rounded-lg backdrop-blur-sm">
              <span className="text-sm text-gray-400">Status:</span>
              <span className="ml-2 text-green-400 font-semibold flex items-center gap-1">
                <CheckCircle2 className="w-4 h-4" />
                Ready
              </span>
            </div>
          </div>
        </div>

        {/* Progress Steps */}
        <div className="mb-8 bg-gradient-to-br from-gray-900/50 to-purple-900/20 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-purple-800/30">
          <h3 className="text-xl font-semibold mb-4 text-purple-300">Training Process</h3>
          <div className="flex flex-wrap md:flex-nowrap items-center justify-between gap-2 md:gap-0">
            {steps.map((step, idx) => (
              <div key={idx} className="flex items-center">
                <div className={`flex items-center gap-2 ${
                  step.status === 'completed' ? 'text-green-400' :
                  step.status === 'active' ? 'text-purple-400' : 'text-gray-600'
                }`}>
                  <div className={`p-2 md:p-3 rounded-lg ${
                    step.status === 'completed' ? 'bg-green-400/20' :
                    step.status === 'active' ? 'bg-purple-400/20 animate-pulse' : 'bg-gray-800'
                  }`}>
                    <step.icon className="w-4 h-4 md:w-5 md:h-5" />
                  </div>
                  <span className="text-sm md:text-base font-medium">{step.name}</span>
                </div>
                {idx < steps.length - 1 && (
                  <ChevronRight className="mx-2 md:mx-4 text-gray-700 hidden md:block" />
                )}
              </div>
            ))}
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden mt-4">
            <div 
              className="h-full bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 transition-all duration-500"
              style={{ width: `${(currentStep + 1) / steps.length * 100}%` }}
            />
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 md:gap-8">
          {/* Left Column - Configuration */}
          <div className="md:col-span-8 lg:col-span-9 space-y-6">
            {/* OS Selection */}
            <div className="bg-gradient-to-br from-gray-900/50 to-purple-900/20 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-purple-500/50 mb-6">
              <div className="border border-purple-500/50 p-4 rounded-xl bg-gray-900/30">
                <h3 className="text-lg md:text-xl font-semibold mb-4 flex items-center gap-2">
                  <Server className="w-4 h-4 md:w-5 md:h-5 text-purple-400" />
                  Operating System <span className="text-sm font-normal text-cyan-400 ml-2">(Auto-detected)</span>
                </h3>
                
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div
                    className={`cursor-pointer bg-gray-800/60 rounded-lg p-3 border ${selectedOS === 'Windows' ? 'border-cyan-400 bg-purple-900/40' : 'border-purple-500/50'} flex items-center gap-2 transition-all hover:bg-purple-900/20 hover:border-purple-400/50 group`}
                    onClick={() => setSelectedOS('Windows')}
                  >
                    <div className={`${selectedOS === 'Windows' ? 'bg-cyan-700/50' : 'bg-purple-700/30'} p-2 rounded-md`}>
                      <Server className={`w-3 h-3 ${selectedOS === 'Windows' ? 'text-cyan-300' : 'text-purple-300'}`} />
                    </div>
                    <div className="flex-1 overflow-hidden">
                      <div className="text-sm font-medium truncate">Windows</div>
                    </div>
                    {selectedOS === 'Windows' && (
                      <div className="bg-cyan-800/30 p-1 rounded-full">
                        <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                      </div>
                    )}
                  </div>
                  
                  <div
                    className={`cursor-pointer bg-gray-800/60 rounded-lg p-3 border ${selectedOS === 'Linux' ? 'border-cyan-400 bg-purple-900/40' : 'border-purple-500/50'} flex items-center gap-2 transition-all hover:bg-purple-900/20 hover:border-purple-400/50 group`}
                    onClick={() => setSelectedOS('Linux')}
                  >
                    <div className={`${selectedOS === 'Linux' ? 'bg-cyan-700/50' : 'bg-purple-700/30'} p-2 rounded-md`}>
                      <Server className={`w-3 h-3 ${selectedOS === 'Linux' ? 'text-cyan-300' : 'text-purple-300'}`} />
                    </div>
                    <div className="flex-1 overflow-hidden">
                      <div className="text-sm font-medium truncate">Linux</div>
                    </div>
                    {selectedOS === 'Linux' && (
                      <div className="bg-cyan-800/30 p-1 rounded-full">
                        <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                      </div>
                    )}
                  </div>
                  
                  <div
                    className={`cursor-pointer bg-gray-800/60 rounded-lg p-3 border ${selectedOS === 'MacOS' ? 'border-cyan-400 bg-purple-900/40' : 'border-purple-500/50'} flex items-center gap-2 transition-all hover:bg-purple-900/20 hover:border-purple-400/50 group`}
                    onClick={() => setSelectedOS('MacOS')}
                  >
                    <div className={`${selectedOS === 'MacOS' ? 'bg-cyan-700/50' : 'bg-purple-700/30'} p-2 rounded-md`}>
                      <Server className={`w-3 h-3 ${selectedOS === 'MacOS' ? 'text-cyan-300' : 'text-purple-300'}`} />
                    </div>
                    <div className="flex-1 overflow-hidden">
                      <div className="text-sm font-medium truncate">MacOS</div>
                    </div>
                    {selectedOS === 'MacOS' && (
                      <div className="bg-cyan-800/30 p-1 rounded-full">
                        <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Model Selection */}
            <div className="bg-gradient-to-br from-gray-900/50 to-purple-900/20 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-purple-500/50">
              <div className="border border-purple-500/50 p-4 rounded-xl bg-gray-900/30">
                <h3 className="text-lg md:text-xl font-semibold mb-4 flex items-center gap-2">
                  <Brain className="w-4 h-4 md:w-5 md:h-5 text-purple-400" />
                  Model Selection
                </h3>
                
                {/* Always show Your AI Arsenal */}
                {(
                  <div className="mb-6 bg-gradient-to-br from-indigo-900/30 to-purple-900/30 rounded-xl p-4 border border-purple-500/50">
                    <div className="flex items-center gap-2 mb-3">
                      <CheckCircle2 className="w-4 h-4 text-green-400" />
                      <h4 className="text-base font-semibold text-green-300">Your AI Arsenal</h4>
                    </div>

<div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 min-h-[60px]">
  {downloadedModels.length === 0 && (
    <div className="col-span-full text-center text-gray-400 py-4">No models found. Try refreshing or check backend logs.</div>
  )}
  {downloadedModels.map(modelObj => {
    // modelObj could be a string (id) or an object with id, size, modified
    const modelId = typeof modelObj === 'string' ? modelObj : modelObj.id;
    const model = ollamaAvailableModels.find(m => m.id === modelId);
    const isCustom = !model;
    // If modelObj has details, use them. Otherwise, fallback to model info
    const modelSize = modelObj.size || model?.size || 'Unknown size';
    const modelModified = modelObj.modified || '';
    return (
      <div
        key={modelId}
        className={`cursor-pointer bg-gray-800/60 rounded-lg p-3 border ${selectedModelForTraining === modelId ? 'border-cyan-400 bg-purple-900/40' : 'border-purple-500/50'} flex items-center gap-2 transition-all hover:bg-purple-900/20 hover:border-purple-400/50 group`}
        onClick={() => setSelectedModelForTraining(modelId)}
      >
        <div className={`${selectedModelForTraining === modelId ? 'bg-cyan-700/50' : isCustom ? 'bg-gray-700/50' : 'bg-purple-700/30'} p-2 rounded-md`}>
          {isCustom ? (
            <Server className={`w-3 h-3 text-yellow-300`} />
          ) : (
            <Brain className={`w-3 h-3 ${selectedModelForTraining === modelId ? 'text-cyan-300' : 'text-purple-300'}`} />
          )}
        </div>
        <div className="flex-1 overflow-hidden">
  <div className={`text-sm font-medium truncate ${isCustom ? 'text-yellow-200' : ''}`}>{model?.name || modelId}</div>
</div>
        {selectedModelForTraining === modelId && (
          <div className="bg-cyan-800/30 p-1 rounded-full">
            <CheckCircle2 className="w-3 h-3 text-cyan-400" />
          </div>
        )}
      </div>
    );
  })}
</div>
                  </div>
                )}
                
                {/* Dropdown for model download */}
                <div className="flex items-center gap-4 mt-4">
                  <div className="flex-1 flex items-center gap-2">
                    <div style={customStyles.selectContainer} className="flex-1">
                      <select
                        className="p-2 rounded-lg bg-gray-900 border border-purple-700 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                        value={selectedDownloadModel || ''}
                        onChange={e => setSelectedDownloadModel(e.target.value)}
                        style={customStyles.select}
                      >
                        <option value="" disabled>Select model to download</option>
                        {ollamaAvailableModels.map(model => (
                          <option
                            key={model.id}
                            value={model.id}
                            disabled={model.description.includes('Coming Soon')}
                            style={model.description.includes('Coming Soon')
                              ? { color: '#888' }
                              : downloadedModels.some(m => (typeof m === 'string' ? m === model.id : m.id === model.id))
                                ? { color: '#4ade80', fontWeight: 'bold' } // Green color for downloaded models
                                : {}
                            }
                          >
                            {model.name} ({model.vram} VRAM)
                            {downloadedModels.some(m => (typeof m === 'string' ? m === model.id : m.id === model.id))
                              ? ' → Already Downloaded'
                              : model.description.includes('Coming Soon')
                                ? ''
                                : ' → Available for Download'}
                          </option>
                        ))}
                      </select>
                    </div>
                    <button
                      onClick={refreshDownloadedModels}
                      className="p-2 rounded-lg bg-gray-800 border border-purple-500/50 hover:bg-purple-900/20 transition-colors relative"
                      title="Refresh downloaded models list"
                      disabled={isRefreshing}
                    >
                      {isRefreshing ? (
                        <svg className="animate-spin text-purple-400 w-[18px] h-[18px]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                      ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-400">
                          <path d="M21 2v6h-6"></path>
                          <path d="M3 12a9 9 0 0 1 15-6.7L21 8"></path>
                          <path d="M3 22v-6h6"></path>
                          <path d="M21 12a9 9 0 0 1-15 6.7L3 16"></path>
                        </svg>
                      )}
                    </button>
                  </div>
                  <div className="flex flex-col gap-2">
                    <div className="flex gap-2">
                      <button
                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-700 hover:bg-purple-800 transition text-white disabled:opacity-50"
                        onClick={() => handleDownloadModel(selectedDownloadModel)}
                        disabled={!selectedDownloadModel || downloadStatus === 'downloading' ||
                          (ollamaAvailableModels.find(m => m.id === selectedDownloadModel)?.description.includes('Coming Soon')) ||
                          downloadedModels.includes(selectedDownloadModel)}
                        title={ollamaAvailableModels.find(m => m.id === selectedDownloadModel)?.description.includes('Coming Soon') ? 'Model not yet available' : downloadedModels.includes(selectedDownloadModel) ? 'Model already downloaded' : ''}
                      >
                        <Download className="w-4 h-4" />
                        {downloadStatus === 'downloading' && selectedDownloadModel ? 'Downloading...' :
                          downloadedModels.includes(selectedDownloadModel) ? 'Already Downloaded' : 'Download'}
                      </button>
                      <button
                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 transition text-white disabled:opacity-50"
                        onClick={() => handleDeleteModel(selectedModelForTraining)}
                        disabled={!selectedModelForTraining}
                        title={!selectedModelForTraining ? 'Select a model from Your AI Arsenal to delete' : `Delete ${selectedModelForTraining}`}
                      >
                        <Trash2 className="w-4 h-4" />
                        Delete
                      </button>
                    </div>
                    {/* Show progress bar when downloading */}
                    {downloadStatus === 'downloading' && selectedDownloadModel && (
                      <div className="w-full">
                        <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                          <div className="bg-cyan-500 h-3 rounded-full transition-all duration-500" style={{ width: `${downloadProgress}%` }}></div>
                        </div>
                        <div className="flex justify-between items-center mt-1">
                          <div className="text-xs text-gray-400">{downloadDetails || 'Downloading...'}</div>
                          <div className="text-xs text-gray-300">{Math.round(downloadProgress)}%</div>
                        </div>
                      </div>
                    )}
                    {/* Download control buttons */}
                    {(downloadStatus === 'downloading' || downloadStatus === 'paused') && selectedDownloadModel && (
                      <div className="flex gap-2 mt-2">
                        {downloadStatus === 'downloading' && (
                          <button
                            className="px-3 py-1 rounded bg-yellow-600 hover:bg-yellow-700 text-white text-xs"
                            onClick={() => pauseDownloadModel(selectedDownloadModel)}
                          >
                            Pause
                          </button>
                        )}
                        {downloadStatus === 'paused' && (
                          <button
                            className="px-3 py-1 rounded bg-green-600 hover:bg-green-700 text-white text-xs"
                            onClick={() => resumeDownloadModel(selectedDownloadModel)}
                          >
                            Resume
                          </button>
                        )}
                        <button
                          className="px-3 py-1 rounded bg-red-600 hover:bg-red-700 text-white text-xs"
                          onClick={() => cancelDownloadModel(selectedDownloadModel)}
                        >
                          Cancel
                        </button>
                      </div>
                    )}
                    {downloadError && (
                      <div className="text-xs text-red-400 mt-2">{downloadError}</div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Dataset Preparation */}
            <div className="bg-gradient-to-br from-gray-900/50 to-blue-900/20 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-blue-800/30">
              <h3 className="text-lg md:text-xl font-semibold mb-4 flex items-center gap-2">
                <Upload className="w-4 h-4 md:w-5 md:h-5 text-blue-400" />
                Dataset Preparation
              </h3>
              
              <div className="space-y-6">
                {/* Dataset Type Selection */}
                <div className="flex flex-col space-y-3">
                  <label className="text-sm text-gray-400">Dataset Type</label>
                  <div className="flex flex-col sm:flex-row gap-3">
                    <button
                      className={`flex-1 p-3 rounded-xl flex items-center justify-center gap-2 transition-all ${datasetType === 'import' 
                        ? 'bg-blue-600/30 border border-blue-500/50' 
                        : 'bg-gray-800/50 border border-gray-700 hover:border-blue-500/30 hover:bg-blue-900/10'}`}
                      onClick={() => setDatasetType('import')}
                    >
                      <Download className="w-4 h-4 text-blue-400" />
                      <span>Import Dataset</span>
                    </button>
                    
                    <button
                      className={`flex-1 p-3 rounded-xl flex items-center justify-center gap-2 transition-all ${datasetType === 'chatgpt' 
                        ? 'bg-blue-600/30 border border-blue-500/50' 
                        : 'bg-gray-800/50 border border-gray-700 hover:border-blue-500/30 hover:bg-blue-900/10'}`}
                      onClick={() => setDatasetType('chatgpt')}
                    >
                      <Sparkles className="w-4 h-4 text-blue-400" />
                      <span>Your ChatGPT Dataset</span>
                    </button>
                    
                    <button
                      className={`flex-1 p-3 rounded-xl flex items-center justify-center gap-2 transition-all ${datasetType === 'claude' 
                        ? 'bg-blue-600/30 border border-blue-500/50' 
                        : 'bg-gray-800/50 border border-gray-700 hover:border-blue-500/30 hover:bg-blue-900/10'}`}
                      onClick={() => setDatasetType('claude')}
                    >
                      <Sparkles className="w-4 h-4 text-blue-400" />
                      <span>Your Claude Dataset</span>
                    </button>
                  </div>
                </div>
                
                {/* File Selection UI */}
                {datasetType === 'import' && (
                  <div className="mt-4">
                    <label className="block text-sm text-gray-400 mb-2">Import Dataset File</label>
                    <div className="border-2 border-dashed border-gray-700 rounded-xl p-6 text-center hover:border-blue-500/50 transition-all cursor-pointer bg-gray-900/50">
                      <input 
                        type="file" 
                        id="datasetFile" 
                        className="hidden" 
                        accept=".json,.jsonl,.csv,.txt"
                        onChange={(e) => setSelectedDatasetFile(e.target.files[0] || null)}
                      />
                      <label htmlFor="datasetFile" className="cursor-pointer">
                        <div className="flex flex-col items-center justify-center gap-2">
                          <Upload className="w-10 h-10 text-blue-400" />
                          <span className="text-sm text-gray-300">
                            {selectedDatasetFile 
                              ? `Selected: ${selectedDatasetFile.name}` 
                              : 'Click to select a dataset file (.json, .jsonl, .csv, .txt)'}
                          </span>
                          <span className="text-xs text-gray-500">
                            {selectedDatasetFile 
                              ? `Size: ${(selectedDatasetFile.size / 1024 / 1024).toFixed(2)} MB` 
                              : 'Or drag and drop your file here'}
                          </span>
                        </div>
                      </label>
                    </div>
                  </div>
                )}
                
                {/* Folder Selection UI */}
                {datasetType === 'chatgpt' && (
                  <div className="mt-4">
                    <label className="block text-sm text-gray-400 mb-2">Select ChatGPT Conversations Folder</label>
                    <div className="border-2 border-dashed border-gray-700 rounded-xl p-6 text-center hover:border-blue-500/50 transition-all cursor-pointer bg-gray-900/50">
                      <input 
                        type="file" 
                        id="datasetFolder" 
                        className="hidden" 
                        webkitdirectory="true"
                        directory="true"
                        onChange={(e) => {
                          const files = Array.from(e.target.files);
                          if (files.length > 0) {
                            // Extract the common folder path from the first file
                            const folderPath = files[0].webkitRelativePath.split('/')[0];
                            setSelectedDatasetFolder(folderPath);
                          }
                        }}
                      />
                      <label htmlFor="datasetFolder" className="cursor-pointer">
                        <div className="flex flex-col items-center justify-center gap-2">
                          <Sparkles className="w-10 h-10 text-blue-400" />
                          <span className="text-sm text-gray-300">
                            {selectedDatasetFolder 
                              ? `Selected folder: ${selectedDatasetFolder}` 
                              : 'Click to select a folder containing ChatGPT conversations'}
                          </span>
                          <span className="text-xs text-gray-500">
                            Will process all .json files in the selected folder
                          </span>
                        </div>
                      </label>
                    </div>
                  </div>
                )}
                
                {/* Claude Folder Selection UI */}
                {datasetType === 'claude' && (
                  <div className="mt-4">
                    <label className="block text-sm text-gray-400 mb-2">Select Claude Conversations Folder</label>
                    <div className="border-2 border-dashed border-gray-700 rounded-xl p-6 text-center hover:border-blue-500/50 transition-all cursor-pointer bg-gray-900/50">
                      <input 
                        type="file" 
                        id="claudeDatasetFolder" 
                        className="hidden" 
                        webkitdirectory="true"
                        directory="true"
                        onChange={(e) => {
                          const files = Array.from(e.target.files);
                          if (files.length > 0) {
                            // Extract the common folder path from the first file
                            const folderPath = files[0].webkitRelativePath.split('/')[0];
                            setSelectedDatasetFolder(folderPath);
                          }
                        }}
                      />
                      <label htmlFor="claudeDatasetFolder" className="cursor-pointer">
                        <div className="flex flex-col items-center justify-center gap-2">
                          <Sparkles className="w-10 h-10 text-blue-400" />
                          <span className="text-sm text-gray-300">
                            {selectedDatasetFolder 
                              ? `Selected folder: ${selectedDatasetFolder}` 
                              : 'Click to select a folder containing Claude conversations'}
                          </span>
                          <span className="text-xs text-gray-500">
                            Will process all .json files in the selected folder
                          </span>
                        </div>
                      </label>
                    </div>
                  </div>
                )}
                
                {/* Processing button */}
                <div className="flex justify-end mt-4">
                  <button 
                    className="px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 rounded-lg text-white font-medium flex items-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={(datasetType === 'import' && !selectedDatasetFile) || (datasetType === 'chatgpt' && !selectedDatasetFolder) || isProcessingDataset}
                    onClick={handleProcessDataset}
                  >
                    {isProcessingDataset ? (
                      <>
                        <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
                        Processing...
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4" />
                        Process Dataset
                      </>
                    )}
                  </button>
                </div>
                
                {/* Dataset stats display */}
                {datasetStats && (
                  <div className="mt-6 bg-gray-800/50 rounded-xl p-4 border border-purple-500/50">
                    <h4 className="text-md font-semibold text-purple-300 mb-2">Dataset Analysis</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Format: <span className="text-white">{datasetStats.format}</span></p>
                        <p className="text-gray-400">Samples: <span className="text-white">{datasetStats.samples}</span></p>
                        <p className="text-gray-400">File Size: <span className="text-white">{datasetStats.fileSize}</span></p>
                      </div>
                      <div>
                        <p className="text-gray-400">Input Tokens: <span className="text-white">{datasetStats.inputTokensEstimate.toLocaleString()}</span></p>
                        <p className="text-gray-400">Output Tokens: <span className="text-white">{datasetStats.outputTokensEstimate.toLocaleString()}</span></p>
                        <p className="text-gray-400">Total Tokens: <span className="text-white font-medium">{datasetStats.totalTokensEstimate.toLocaleString()}</span></p>
                      </div>
                    </div>
                    
                    {/* Preview of the first few examples */}
                    {datasetStats.examples && datasetStats.examples.length > 0 && (
                      <div className="mt-4">
                        <h5 className="text-sm font-medium text-purple-300 mb-2">Sample Preview</h5>
                        <div className="max-h-48 overflow-y-auto">
                          {datasetStats.examples.map((example, index) => (
                            <div key={index} className="mb-2 text-xs bg-gray-900/70 p-2 rounded border border-gray-700">
                              {Object.entries(example).map(([key, value]) => (
                                <div key={key} className="mb-1">
                                  <span className="text-cyan-400">{key}:</span> 
                                  <span className="text-gray-300 ml-1 break-words">
                                    {typeof value === 'string' ? 
                                      (value.length > 100 ? value.substring(0, 100) + '...' : value) : 
                                      JSON.stringify(value)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Dataset ready notification */}
                    <div className="mt-4 p-2 bg-green-900/30 border border-green-500/30 rounded-lg text-sm text-green-300 flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4" />
                      Dataset processed successfully and ready for training
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* LoRA/QLoRA Setup */}
            {currentStep >= 3 && (
              <div className="bg-gradient-to-br from-gray-900/50 to-purple-900/20 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-purple-500/50">
                <div className="border border-purple-500/50 p-4 rounded-xl bg-gray-900/30">
                  <h3 className="text-lg md:text-xl font-semibold mb-4 flex items-center gap-2">
                    <Zap className="w-4 h-4 md:w-5 md:h-5 text-purple-400" />
                    LoRA/QLoRA Setup
                  </h3>
                  
                  {/* Import the LoRA Training component */}
                  <div className="mt-6" id="lora-training-component">
                    <h4 className="text-md font-medium text-purple-300 mb-3">Advanced LoRA/QLoRA Settings</h4>
                    <Suspense fallback={<div className="text-center p-4"><RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2 text-purple-400" /> Loading LoRA components...</div>}>
                      <MemoizedLoraTraining 
                        selectedModel={selectedModelForTraining}
                        selectedDataset={processedDataset}
                        onTrainingComplete={(data) => {
                          // Handle LoRA training completion
                          console.log("LoRA training completed:", data);
                          setCurrentStep(4);
                          
                          // Show success notification
                          setNotification({
                            type: 'success',
                            title: 'LoRA Training Complete',
                            message: `Your model has been fine-tuned with LoRA adapter: ${data.adapterId}`,
                            autoClose: true
                          });
                        }}
                      />
                    </Suspense>
                  </div>
                  
                  {/* Manual LoRA buttons for clarity - moved below LoRA Training component */}
                  <div className="p-4 bg-gray-800/60 rounded-lg border border-purple-500/30 mb-6">
                    <h4 className="text-lg font-medium text-purple-300 mb-3">Apply LoRA/QLoRA to Model</h4>
                    <p className="text-sm text-gray-300 mb-4">First, apply LoRA/QLoRA adapter to your selected model. This prepares the model for fine-tuning.</p>
                    
                    <div className="flex flex-wrap gap-3">
                      {/* Clear guidance on what to do */}
                      {!selectedModelForTraining ? (
                        <div className="bg-gray-700/60 text-gray-300 p-3 rounded-lg border border-purple-500/30 flex items-center gap-2">
                          <AlertCircle className="w-4 h-4 text-amber-400" />
                          <span>Please select a model from the list above first</span>
                        </div>
                      ) : (
                        <button
                          className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white font-medium py-2 px-6 rounded-lg transition-all flex items-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed"
                          onClick={async () => {
                            // Set loading state
                            setIsApplyingAdapter(true);
                            
                            try {
                              console.log('Applying LoRA/QLoRA adapter to model:', selectedModelForTraining);
                              
                              // Create a safe config object without any circular references
                              const safeConfig = {
                                adapter_type: 'lora',
                                rank: 8,
                                alpha: 16,
                                dropout: 0.05,
                                target_modules: ['q_proj', 'v_proj'],
                                quantization: '4bit' // Use memory-efficient 4-bit quantization for 8GB VRAM
                              };
                              
                              // Create a safe request payload
                              const requestPayload = {
                                model_id: selectedModelForTraining,
                                config: safeConfig,
                                quantization: safeConfig.quantization
                              };
                              
                              console.log('API request payload:', JSON.stringify(requestPayload));
                              
                              // Make API call directly
                              const response = await fetch('http://127.0.0.1:5002/api/lora/apply', {
                                method: 'POST',
                                headers: {
                                  'Content-Type': 'application/json'
                                },
                                body: JSON.stringify(requestPayload)
                              });
                              
                              const data = await response.json();
                              console.log('API response:', data);
                              
                              if (data.status === 'success') {
                                setNotification({
                                  type: 'success',
                                  title: 'Adapter Applied',
                                  message: `LoRA/QLoRA adapter successfully applied to ${selectedModelForTraining}!`,
                                  autoClose: true
                                });
                              } else {
                                setNotification({
                                  type: 'error',
                                  title: 'Application Failed',
                                  message: data.message || 'Failed to apply LoRA/QLoRA adapter.',
                                  autoClose: true
                                });
                              }
                            } catch (error) {
                              console.error('Error applying adapter:', error);
                              setNotification({
                                type: 'error',
                                title: 'Application Error',
                                message: `Failed to apply LoRA: ${error.message}`,
                                autoClose: true
                              });
                            } finally {
                              // Reset loading state
                              setIsApplyingAdapter(false);
                            }
                          }}
                          disabled={!selectedModelForTraining || isApplyingAdapter}
                        >
                          {isApplyingAdapter ? (
                            <>
                              <RefreshCw className="w-4 h-4 animate-spin" />
                              Applying Adapter...
                            </>
                          ) : (
                            <>
                              <Zap className="w-4 h-4" />
                              Apply LoRA/QLoRA Adapter Configurations to Model
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                  
                  {/* Training Loss and Logs Section */}
                  {isTraining && (
                    <div className="p-4 bg-gray-800/60 rounded-lg border border-purple-500/30 mb-6">
                      <h4 className="text-lg font-medium text-purple-300 mb-3">Training Progress</h4>
                      {trainingLoss !== null && (
                        <div className="mb-5">
                          <div className="mb-2 text-purple-300 font-medium">
                            Training Loss: <span className="text-white">{trainingLoss.toFixed(4)}</span>
                          </div>
                          
                          {/* Training Loss Chart */}
                          {lossData.length > 0 && (
                            <div className="rounded p-3 mb-3 border border-purple-500/20" style={{ background: 'linear-gradient(180deg, rgba(13, 16, 45, 1) 0%, rgba(17, 16, 40, 1) 100%)' }}>
                              <div className="text-white text-lg font-semibold mb-2">Training Loss</div>
                              <ResponsiveContainer width="100%" height={180}>
                                <AreaChart data={lossData} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                                  <defs>
                                    <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                                      <stop offset="0%" stopColor="rgb(147, 51, 234)" stopOpacity={0.9} />
                                      <stop offset="50%" stopColor="rgb(128, 90, 213)" stopOpacity={0.7} />
                                      <stop offset="100%" stopColor="rgb(56, 189, 248)" stopOpacity={0.5} />
                                    </linearGradient>
                                    <linearGradient id="colorUv" x1="0" y1="0" x2="1" y2="0">
                                      <stop offset="0%" stopColor="#9333EA" />
                                      <stop offset="50%" stopColor="#805AD5" />
                                      <stop offset="100%" stopColor="#38BDF8" />
                                    </linearGradient>
                                  </defs>
                                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                  <XAxis 
                                    dataKey="step" 
                                    tick={{ fill: 'rgba(255,255,255,0.7)' }} 
                                    tickLine={false}
                                    axisLine={false}
                                  />
                                  <YAxis 
                                    tick={{ fill: 'rgba(255,255,255,0.7)' }} 
                                    tickLine={false}
                                    axisLine={false}
                                    domain={[0, 'dataMax + 2']}
                                  />
                                  <Tooltip 
                                    contentStyle={{ backgroundColor: 'rgba(13, 16, 45, 0.9)', border: '1px solid rgba(147,51,234,0.5)' }}
                                    labelStyle={{ color: 'rgba(255,255,255,0.9)' }}
                                    itemStyle={{ color: 'rgba(147,51,234,1)' }}
                                    formatter={(value) => value.toFixed(2)}
                                    labelFormatter={(step) => `Step ${step}`}
                                  />
                                  <Area 
                                    type="monotone" 
                                    dataKey="loss" 
                                    stroke="url(#colorUv)" 
                                    strokeWidth={2}
                                    fillOpacity={0.8} 
                                    fill="url(#lossGradient)" 
                                    animationDuration={800}
                                    isAnimationActive={true}
                                    activeDot={{ r: 6, stroke: '#9333EA', strokeWidth: 2, fill: 'white' }}
                                  />
                                </AreaChart>
                              </ResponsiveContainer>
                            </div>
                          )}
                        </div>
                      )}
                      
                      <div className="text-sm text-purple-300 font-medium mb-2">Training Logs:</div>
                      <div className="max-h-60 overflow-y-auto bg-gray-900/50 rounded p-2 text-xs font-mono">
                        {trainingLogs.length > 0 ? (
                          <div>
                            {trainingLogs.map((log, index) => (
                              <div 
                                key={index} 
                                className={`mb-1 ${log.level === 'warning' ? 'text-yellow-300' : log.level === 'error' ? 'text-red-300' : 'text-gray-300'}`}
                              >
                                <span className="text-gray-500">[{log.timestamp}]</span> {log.message}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="text-gray-500 italic">Waiting for training logs...</div>
                        )}
                      </div>
                    </div>
                  )}
                  
                  <div className="p-4 bg-gray-800/60 rounded-lg border border-purple-500/30 mb-6">
                    <h4 className="text-lg font-medium text-purple-300 mb-3">Step 2: Start Training</h4>
                    <p className="text-sm text-gray-300 mb-4">After applying LoRA/QLoRA, start the training process to fine-tune you model.</p>
                    
                    <div className="flex flex-wrap gap-3">
                      <button
                        className="bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 text-white font-medium py-2 px-6 rounded-lg transition-all flex items-center gap-2 relative"
                        onClick={async () => {
                          console.log('Start Training button clicked');
                          console.log('Selected model:', selectedModelForTraining);
                          console.log('Processed dataset:', processedDataset);
                          
                          // Set training state
                          setIsTraining(true);
                          setTrainingStatus('training'); // Set training status
                          
                          // Update the progress bar to show all steps complete
                          setCurrentStep(4); // This will mark "Training Launched" as active
                          
                          // Reset progress to 0 when starting training
                          setProgress(0);
                          
                          // Reset training status tracking
                          window.trainingWasAborted = false;
                          
                          // Clear existing logs and loss
                          setTrainingLogs([]);
                          setTrainingLoss(null); // Reset training loss
                          
                          try {
                            // Create a direct API call to start training
                            const trainingParams = {
                              model_id: selectedModelForTraining,
                              dataset: processedDataset, // Send the full dataset, not just path
                              lora_config: {
                                adapter_type: 'lora',
                                rank: 8,
                                alpha: 16,
                                dropout: 0.05,
                                target_modules: ['q_proj', 'v_proj'],
                                quantization: '4bit' // Memory-efficient for 8GB VRAM
                              },
                              training_config: {
                                num_epochs: 3,
                                learning_rate: 1e-4,
                                batch_size: 1, // Memory-efficient batch size
                                max_length: 256 // Memory-efficient sequence length
                              }
                            };
                            
                            console.log('Training parameters:', trainingParams);
                            
                            // Make the API call to start training
                            const response = await fetch('http://127.0.0.1:5002/api/lora/train', {
                              method: 'POST',
                              headers: {
                                'Content-Type': 'application/json'
                              },
                              body: JSON.stringify(trainingParams)
                            });
                            
                            const data = await response.json();
                            console.log('Training API response:', data);
                            
                            if (data.status === 'success') {
                              // Success notification
                              setNotification({
                                type: 'success',
                                title: 'Training Started',
                                message: 'LoRA training has been started successfully!',
                                autoClose: true
                              });
                              
                              // Start progress monitoring
                              let monitoringInterval;
                              monitoringInterval = setInterval(async () => {
                                try {
                                  const progressResponse = await fetch('http://127.0.0.1:5002/api/lora/training_status');
                                  if (progressResponse.ok) {
                                    const progressData = await progressResponse.json();
                                    console.log('Training progress:', progressData);
                                    
                                    // Update progress based on actual backend state
                                    // For the aborted case, we want to show the accurate state from backend
                                    const newProgress = progressData.progress || 0;
                                    if (progressData.status === 'aborted') {
                                      // If aborted, show the exact progress from backend (should become 0)
                                      setProgress(newProgress);
                                    } else {
                                      // Otherwise prevent progress from going backward (only increase)
                                      setProgress(prev => Math.max(prev, newProgress));
                                    }
                                    
                                    // Update loss if available
                                    if (progressData.loss !== undefined) {
                                      // Update the current loss value
                                      setTrainingLoss(progressData.loss);
                                      // Loss data is collected by the useEffect hook watching trainingLoss
                                    }
                                    
                                    // Update logs if available
                                    if (progressData.logs && Array.isArray(progressData.logs)) {
                                      setTrainingLogs(progressData.logs);
                                    }
                                    
                                    // Check if training has ended (completed, aborted, or error)
                                    // Make sure to only trigger one status condition
                                    console.log(`Training status check: '${progressData.status}'`);
                                    
                                    // Use strict equality and explicit string comparison for safety
                                    // Also check our manual abort tracking flag
                                    if (progressData.status === 'aborted' || progressData.status === 'aborting' || window.trainingWasAborted) {
                                      // Update our tracking flag in case it wasn't set earlier
                                      window.trainingWasAborted = true;
                                      console.log('Training status: ABORTED');
                                      clearInterval(monitoringInterval);
                                      setIsTraining(false);
                                      setTrainingStatus('aborted'); // Set training status
                                      setProgress(0); // Reset progress on abort
                                      
                                      // IMPORTANT: Use clear abort notification with different title/message from completion
                                      setNotification({
                                        type: 'warning',
                                        title: 'Training Aborted',
                                        message: 'The Fine-Tune Training has been Aborted by user request.',
                                        autoClose: true
                                      });
                                      return; // Stop here to prevent multiple notifications
                                    } else if (progressData.status === 'completed' && !window.trainingWasAborted) {
                                      clearInterval(monitoringInterval);
                                      setIsTraining(false);
                                      setTrainingStatus('completed'); // Set training status
                                      setNotification({
                                        type: 'success',
                                        title: 'Training Complete',
                                        message: 'The Fine-Tune Training has Completed Successfully.',
                                        autoClose: true
                                      });
                                      return; // Stop here to prevent multiple notifications
                                    } else if (progressData.status === 'error') {
                                      clearInterval(monitoringInterval);
                                      setIsTraining(false);
                                      setTrainingStatus('error'); // Set training status
                                      setNotification({
                                        type: 'error',
                                        title: 'Training Error',
                                        message: progressData.error || 'An error occurred during training',
                                        autoClose: true
                                      });
                                      return; // Stop here to prevent multiple notifications
                                    }
                                  }
                                } catch (progressError) {
                                  console.error('Error monitoring training progress:', progressError);
                                  // Check if we're no longer training before giving up
                                  if (!isTraining) {
                                    console.log('No longer training, stopping status polling');
                                    clearInterval(monitoringInterval);
                                    return;
                                  }
                                  // Otherwise continue polling despite transient errors
                                }
                              }, 2000); // Check progress every 2 seconds
                            } else {
                              // Training failed to start
                              setIsTraining(false);
                              setNotification({
                                type: 'error',
                                title: 'Training Failed',
                                message: data.message || 'Failed to start LoRA training.',
                                autoClose: true
                              });
                            }
                          } catch (error) {
                            console.error('Error starting training:', error);
                            setIsTraining(false);
                            setNotification({
                              type: 'error',
                              title: 'Training Error',
                              message: `Failed to start training: ${error.message}`,
                              autoClose: true
                            });
                          }
                        }}
                        disabled={!selectedModelForTraining || !processedDataset || isTraining}
                      >
                        {/* Show either the progress wheel when training or Activity icon when not training */}
                        {isTraining ? (
                          <div className="w-6 h-6 flex items-center justify-center mr-1">
                            <svg className="w-full h-full" viewBox="0 0 100 100">
                              <circle
                                className="text-gray-200"
                                strokeWidth="10"
                                stroke="currentColor"
                                fill="transparent"
                                r="40"
                                cx="50"
                                cy="50"
                              />
                              <circle
                                className="text-purple-500"
                                strokeWidth="10"
                                strokeLinecap="round"
                                stroke="currentColor"
                                fill="transparent"
                                r="40"
                                cx="50"
                                cy="50"
                                strokeDasharray={`${2 * Math.PI * 40}`}
                                strokeDashoffset={`${2 * Math.PI * 40 * (1 - progress / 100)}`}
                              />
                              <text
                                x="50"
                                y="50"
                                fill="white"
                                className="text-xs font-bold"
                                dominantBaseline="middle"
                                textAnchor="middle"
                                style={{ fontSize: '20px', fontWeight: 'bold' }}
                              >
                                {Math.round(progress)}%
                              </text>
                            </svg>
                          </div>
                        ) : (
                          <Activity className="w-4 h-4 mr-1" />
                        )}
                        2. Start Training
                      </button>
                      
                      {/* Abort button that appears when training is in progress */}
                      {isTraining && (
                        <button
                          className="bg-gradient-to-r from-red-600 to-red-700 hover:from-red-500 hover:to-red-600 text-white font-medium py-2 px-6 rounded-lg transition-all flex items-center gap-2"
                          onClick={async () => {
                            console.log('Abort Training button clicked');
                            
                            try {
                              console.log('Sending abort request to backend...');
                              // Call the abort endpoint to kill the training process
                              const response = await fetch('http://127.0.0.1:5002/api/lora/abort', {
                                method: 'POST',
                                headers: {
                                  'Content-Type': 'application/json'
                                }
                              });
                              
                              const data = await response.json();
                              console.log('Abort response:', data);

                              // Show immediate feedback while waiting for backend to fully abort
                              setNotification({
                                type: 'warning',
                                title: 'Aborting Training...',
                                message: 'Sending abort signal to the training process...',
                                autoClose: true
                              });
                              
                              // Mark training as aborted to prevent completion notification
                              window.trainingWasAborted = true;
                              console.log('Marked training as aborted:', window.trainingWasAborted);
                              
                              // We DON'T reset any UI state here - we let the status polling
                              // detect the 'aborted' status and update the UI appropriately
                              // This ensures the proper notification is shown and prevents race conditions
                              console.log('Abort request sent successfully - waiting for backend to confirm abort status');
                              // The next status poll will detect the aborted status and show the correct notification
                            } catch (error) {
                              console.error('Error aborting training:', error);
                              
                              // Only show error notification, don't change UI state
                              // Let status monitoring continue to track actual backend state
                              setNotification({
                                type: 'warning',
                                title: 'Abort Error',
                                message: `Error aborting training: ${error.message}. Training may still be running.`,
                                autoClose: true
                              });
                            }
                          }}
                        >
                          <StopCircle className="w-4 h-4" />
                          Abort Training
                        </button>
                      )}
                    </div>
                  </div>
                  
                  {/* LoRA Action Buttons - Moved from LoraTraining component */}
                  <div className="p-4 bg-gray-800/60 rounded-lg border border-purple-500/30 mb-6">
                    <h4 className="text-lg font-medium text-purple-300 mb-3">LoRA/QLoRA Actions</h4>
                    <p className="text-sm text-gray-300 mb-4">Execute LoRA/QLoRA operation on your selected model</p>
                    
                    <div className="flex flex-wrap gap-3">
                      {/* Save Adapter button */}
                      <button
                        className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white font-medium py-2 px-6 rounded-lg transition-all flex items-center gap-2"
                        onClick={async () => {
                          // Check if training is completed
                          if (trainingStatus !== 'completed') {
                            setNotification({
                              type: 'error',
                              title: 'Training Not Complete',
                              message: 'Please complete training before saving adapter weights',
                              autoClose: true
                            });
                            return;
                          }
                          
                          // Prompt user for adapter name
                          const adapterName = prompt('Enter a name for your adapter (e.g., "my-custom-model-v1"):');
                          
                          if (!adapterName) {
                            return; // User cancelled
                          }
                          
                          // Validate adapter name (alphanumeric, hyphens, underscores only)
                          if (!/^[a-zA-Z0-9_-]+$/.test(adapterName)) {
                            setNotification({
                              type: 'error',
                              title: 'Invalid Name',
                              message: 'Adapter name can only contain letters, numbers, hyphens, and underscores',
                              autoClose: true
                            });
                            return;
                          }
                          
                          try {
                            const response = await fetch('http://127.0.0.1:5002/api/lora/save_adapter', {
                              method: 'POST',
                              headers: {
                                'Content-Type': 'application/json',
                              },
                              body: JSON.stringify({
                                adapter_name: adapterName
                              })
                            });
                            
                            const result = await response.json();
                            
                            if (result.status === 'success') {
                              setNotification({
                                type: 'success',
                                title: 'Adapter Saved',
                                message: `Adapter saved successfully as "${adapterName}"`,
                                autoClose: true
                              });
                              console.log('Adapter saved to:', result.save_path);
                            } else {
                              setNotification({
                                type: 'error',
                                title: 'Save Failed',
                                message: result.message || 'Failed to save adapter',
                                autoClose: true
                              });
                            }
                          } catch (error) {
                            console.error('Error saving adapter:', error);
                            setNotification({
                              type: 'error',
                              title: 'Error',
                              message: 'Error saving adapter: ' + error.message,
                              autoClose: true
                            });
                          }
                        }}
                        disabled={!selectedModelForTraining || trainingStatus !== 'completed'}
                      >
                        <Layers className="w-4 h-4" />
                        Save Adapter Weights & Configurations Only
                      </button>
                      
                      {/* Merge & Export button */}
                      <button
                        className="bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 text-white font-medium py-2 px-6 rounded-lg transition-all flex items-center gap-2"
                        onClick={async () => {
                          // Check if training is completed
                          if (trainingStatus !== 'completed') {
                            setNotification({
                              type: 'error',
                              title: 'Training Not Complete',
                              message: 'Please complete training before merging and exporting the model',
                              autoClose: true
                            });
                            return;
                          }
                          
                          // Prompt user for export name
                          const exportName = prompt('Enter a name for your exported model (e.g., "my-llama-custom-v1"):');
                          
                          if (!exportName) {
                            return; // User cancelled
                          }
                          
                          // Validate export name (alphanumeric, hyphens, underscores only)
                          if (!/^[a-zA-Z0-9_-]+$/.test(exportName)) {
                            setNotification({
                              type: 'error',
                              title: 'Invalid Name',
                              message: 'Model name can only contain letters, numbers, hyphens, and underscores',
                              autoClose: true
                            });
                            return;
                          }
                          
                          // Show loading state
                          setIsExporting(true);
                          setNotification({
                            type: 'info',
                            title: 'Merging Model',
                            message: 'Merging LoRA adapter with base model. This may take a few minutes...',
                            autoClose: false
                          });
                          
                          try {
                            const response = await fetch('http://127.0.0.1:5002/api/lora/merge_and_export', {
                              method: 'POST',
                              headers: {
                                'Content-Type': 'application/json',
                              },
                              body: JSON.stringify({
                                export_name: exportName,
                                export_path: 'merged_models'  // You can make this configurable
                              })
                            });
                            
                            const result = await response.json();
                            
                            if (result.status === 'success') {
                              setNotification({
                                type: 'success',
                                title: 'Export Complete',
                                message: `Model successfully exported to ${result.export_path} (${result.model_size_mb.toFixed(2)} MB)`,
                                autoClose: true
                              });
                              console.log('Model exported to:', result.export_path);
                            } else {
                              setNotification({
                                type: 'error',
                                title: 'Export Failed',
                                message: result.message || 'Failed to merge and export model',
                                autoClose: true
                              });
                            }
                          } catch (error) {
                            console.error('Error merging and exporting model:', error);
                            setNotification({
                              type: 'error',
                              title: 'Error',
                              message: 'Error merging and exporting model: ' + error.message,
                              autoClose: true
                            });
                          } finally {
                            setIsExporting(false);
                          }
                        }}
                        disabled={!selectedModelForTraining || trainingStatus !== 'completed' || isExporting}
                      >
                        <PackageOpen className="w-4 h-4" />
                        {isExporting ? 'Exporting...' : 'Merge & Export Fine-Tuned Model'}
                      </button>
                      

                    </div>
                  </div>
                </div>
                

              </div>
            )}
            
            {/* Training Parameters */}
            <div className="bg-gradient-to-br from-gray-900/50 to-cyan-900/20 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-cyan-800/30">
              <h3 className="text-lg md:text-xl font-semibold mb-4 flex items-center gap-2">
                <Settings className="w-4 h-4 md:w-5 md:h-5 text-cyan-400" />
                Training Configuration
              </h3>
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(config).map(([key, value]) => (
                  <div key={key}>
                    <label className="block text-sm text-gray-400 mb-2">
                      {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                    </label>
                    <input
                      type="number"
                      value={value}
                      onChange={(e) => setConfig({...config, [key]: parseFloat(e.target.value)})}
                      className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/20 transition-all"
                    />
                  </div>
                ))}
              </div>
            </div>


          </div>

          {/* Right Column - Status & Actions */}
          <div className="md:col-span-4 lg:col-span-3 space-y-6">
            {/* Training Status */}
            <div className="bg-gradient-to-br from-gray-900/50 to-purple-900/20 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-purple-800/30">
              <h3 className="text-lg md:text-xl font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-purple-400" />
                Training Status
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Overall Progress</span>
                    <span className="text-purple-400 font-semibold">{progress.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-3 overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 transition-all duration-300 relative"
                      style={{ width: `${progress}%` }}
                    >
                      <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Current Epoch</span>
                    <span>1 / {config.epochs}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Batch</span>
                    <span>128 / 1024</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Learning Rate</span>
                    <span className="text-cyan-400">{config.learningRate}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">GPU Memory</span>
                    <span className="text-yellow-400">14.2 / 16 GB</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="space-y-3">
              <button
                onClick={() => setIsTraining(!isTraining)}
                className={`w-full py-3 md:py-4 rounded-xl font-semibold transition-all flex items-center justify-center gap-2 transform hover:scale-102 ${
                  isTraining 
                    ? 'bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 shadow-md shadow-red-500/20' 
                    : 'bg-gradient-to-r from-purple-600 via-pink-600 to-cyan-600 hover:from-purple-700 hover:via-pink-700 hover:to-cyan-700 shadow-md shadow-purple-500/20'
                }`}
              >
                {isProcessingDataset ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Start Training
                  </>
                )}
              </button>
              

            </div>

            {/* System Resources */}
            <div className="bg-gradient-to-br from-gray-900/50 to-orange-900/20 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-orange-800/30">
              <h3 className="text-lg md:text-xl font-semibold mb-4 flex items-center gap-2">
                <Server className="w-4 h-4 md:w-5 md:h-5 text-orange-400" />
                System Resources
              </h3>
              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-gray-400">GPU Utilization</span>
                    <span className="text-sm font-semibold text-green-400">{systemResources?.gpuUtilization || 0}%</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
                    <div className="h-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full" style={{width: `${systemResources?.gpuUtilization || 0}%`}}></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-gray-400">CPU Usage</span>
                    <span className="text-sm font-semibold text-yellow-400">{systemResources?.cpuUsage || 0}%</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
                    <div className="h-2 bg-gradient-to-r from-yellow-500 to-amber-500 rounded-full" style={{width: `${systemResources?.cpuUsage || 0}%`}}></div>
                  </div>
                </div>
                
                <div className="pt-2 border-t border-gray-800">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">GPU Memory</span>
                    <span className="text-sm font-semibold text-blue-400">
                      {systemResources?.gpuMemory?.used || 0} / {systemResources?.gpuMemory?.total || 0} GB
                    </span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden mt-1">
                    <div className="h-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full" 
                      style={{width: `${systemResources?.gpuMemory?.used && systemResources?.gpuMemory?.total ? (systemResources.gpuMemory.used / systemResources.gpuMemory.total) * 100 : 0}%`}}></div>
                  </div>
                </div>
                
                <div className="pt-2 border-t border-gray-800">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">RAM Memory</span>
                    <span className="text-sm font-semibold text-purple-400">
                      {systemResources?.ramMemory?.used || 0} / {systemResources?.ramMemory?.total || 0} GB
                    </span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden mt-1">
                    <div className="h-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full" 
                      style={{width: `${systemResources?.ramMemory?.used && systemResources?.ramMemory?.total ? (systemResources.ramMemory.used / systemResources.ramMemory.total) * 100 : 0}%`}}></div>
                  </div>
                </div>
                
                <div className="pt-2 border-t border-gray-800">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Temperature</span>
                    <TemperatureDisplay temperature={systemResources?.temperature} />
                  </div>
                </div>
              </div>
            </div>
            
            {/* Chat with Model */}
            <div className="bg-gradient-to-br from-gray-900/50 to-purple-900/20 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-purple-500/50 mt-6">
              <h3 className="text-lg md:text-xl font-semibold mb-4 flex items-center gap-2 relative">
                <Brain className="w-4 h-4 md:w-5 md:h-5 text-purple-400" />
                Chat with Model
                <div className="relative inline-block">
                  <button
                    onClick={() => setShowChatInfo(!showChatInfo)}
                    className="ml-2 p-1 rounded-full hover:bg-purple-800/30 transition-colors"
                    title="Click for more information"
                  >
                    <Info className="w-4 h-4 text-purple-400" />
                  </button>
                  
                  {/* Info Tooltip/Modal */}
                  {showChatInfo && (
                    <div className="absolute z-50 top-8 left-0 w-80 bg-gray-900 border border-purple-500 rounded-lg p-4 shadow-xl">
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="text-sm font-semibold text-purple-300">How Chat Works</h4>
                        <button
                          onClick={() => setShowChatInfo(false)}
                          className="text-gray-400 hover:text-white"
                        >
                          <XCircle className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="text-xs text-gray-300 space-y-2">
                        <p>
                          <strong className="text-purple-400">• Base Models:</strong> Chat with any downloaded model from your AI Arsenal.
                        </p>
                        <p>
                          <strong className="text-purple-400">• Fine-Tuned Models:</strong> Use models that have been fully fine-tuned and exported.
                        </p>
                        <p>
                          <strong className="text-purple-400">• With Adapters:</strong> Select a base model, then choose a LoRA/QLoRA adapter to apply your fine-tuning on the fly. This loads the adapter weights onto the base model for a customized chat experience.
                        </p>
                        <p className="text-yellow-400 mt-2">
                          💡 Tip: You can mix and match any compatible base model with any adapter trained on a similar architecture!
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </h3>
              
              <div className="mb-4">
                <label className="block text-sm text-gray-400 mb-2">Select Model or Fine-Tuned Model</label>
                <div style={customStyles.selectContainer}>
                  <select
                    className="p-2 rounded-lg bg-gray-900 border border-purple-700 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    value={selectedChatModel}
                    onChange={e => setSelectedChatModel(e.target.value)}
                    style={customStyles.select}
                  >
                    <option value="" disabled>Select a model to chat with</option>
                    {downloadedModels.map(modelObj => {
                      const modelId = typeof modelObj === 'string' ? modelObj : modelObj.id;
                      const model = ollamaAvailableModels.find(m => m.id === modelId);
                      return (
                        <option key={modelId} value={modelId}>
                          {model?.name || modelId}
                        </option>
                      );
                    })}
                  </select>
                </div>
              </div>
              
              {/* Adapter Selection */}
              <div className="mb-4">
                <label className="block text-sm text-gray-400 mb-2">LoRA/QLoRA Adapters</label>
                <div style={customStyles.selectContainer}>
                  <select
                    className="p-2 rounded-lg bg-gray-900 border border-purple-700 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    value={selectedAdapter}
                    onChange={e => setSelectedAdapter(e.target.value)}
                    style={customStyles.select}
                    disabled={!selectedChatModel || isLoadingAdapter}
                  >
                    <option value="">Select an adapter (optional)</option>
                    {availableAdapters.map(adapter => (
                      <option key={adapter.name} value={adapter.name}>
                        {adapter.name} ({adapter.base_model.split('/').pop()})
                      </option>
                    ))}
                  </select>
                </div>
                {isLoadingAdapter && (
                  <p className="text-sm text-purple-400 mt-2 flex items-center gap-2">
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Loading adapter...
                  </p>
                )}
                {selectedChatModel && selectedAdapter && !isLoadingAdapter && (
                  <p className="text-sm text-green-400 mt-2">
                    ✓ Adapter loaded successfully
                  </p>
                )}
              </div>
              
              {/* Chat Messages */}
              <div className="bg-gray-900/70 rounded-lg border border-gray-800 h-64 mb-4 overflow-y-auto p-3">
                {chatMessages.length === 0 ? (
                  <div className="text-center text-gray-500 h-full flex items-center justify-center">
                    <p>Select a model and start chatting</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {chatMessages.map((msg, index) => (
                      <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[80%] p-3 rounded-lg ${msg.role === 'user' ? 'bg-purple-600/30 border border-purple-500/50' : 'bg-gray-800 border border-gray-700'}`}>
                          <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              {/* Input */}
              <div className="flex gap-2">
                <input
                  type="text"
                  value={currentMessage}
                  onChange={e => setCurrentMessage(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey && currentMessage.trim() && selectedChatModel) {
                      e.preventDefault();
                      const newMessage = { role: 'user', content: currentMessage.trim() };
                      setChatMessages([...chatMessages, newMessage]);
                      
                      setIsSendingMessage(true);
                      setCurrentMessage('');
                      
                      // Call Ollama API to get response
                      fetch('http://127.0.0.1:5001/api/ollama/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          model: selectedChatModel,
                          messages: [...chatMessages, newMessage]
                        })
                      })
                      .then(res => res.json())
                      .then(data => {
                        if (data.response) {
                          setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
                        } else {
                          setChatMessages(prev => [...prev, { role: 'assistant', content: 'Error: Unable to get response from model.' }]);
                        }
                        setIsSendingMessage(false);
                      })
                      .catch(err => {
                        console.error('Error calling Ollama API:', err);
                        setChatMessages(prev => [...prev, { role: 'assistant', content: 'Error: Unable to connect to Ollama API.' }]);
                        setIsSendingMessage(false);
                      });
                    }
                  }}
                  placeholder="Type your message and press Enter"
                  className="flex-1 p-3 rounded-lg bg-gray-800 border border-gray-700 focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/20 transition-all"
                  disabled={!selectedChatModel || isSendingMessage}
                />
                <button
                  onClick={() => {
                    if (currentMessage.trim() && selectedChatModel) {
                      const newMessage = { role: 'user', content: currentMessage.trim() };
                      setChatMessages([...chatMessages, newMessage]);
                      
                      setIsSendingMessage(true);
                      setCurrentMessage('');
                      
                      // Call Ollama API to get response
                      fetch('http://127.0.0.1:5001/api/ollama/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          model: selectedChatModel,
                          messages: [...chatMessages, newMessage]
                        })
                      })
                      .then(res => res.json())
                      .then(data => {
                        if (data.response) {
                          setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
                        } else {
                          setChatMessages(prev => [...prev, { role: 'assistant', content: 'Error: Unable to get response from model.' }]);
                        }
                        setIsSendingMessage(false);
                      })
                      .catch(err => {
                        console.error('Error calling Ollama API:', err);
                        setChatMessages(prev => [...prev, { role: 'assistant', content: 'Error: Unable to connect to Ollama API.' }]);
                        setIsSendingMessage(false);
                      });
                    }
                  }}
                  className="p-3 rounded-lg bg-purple-600 hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:bg-purple-900"
                  disabled={!currentMessage.trim() || !selectedChatModel || isSendingMessage}
                >
                  {isSendingMessage ? (
                    <div className="w-5 h-5 border-2 border-t-transparent border-white rounded-full animate-spin"></div>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M22 2L11 13"/>
                      <path d="M22 2l-7 20-4-9-9-4 20-7z"/>
                    </svg>
                  )}
                </button>
              </div>
              
              {/* Clear chat button */}
              <div className="mt-3 text-right">
                <button
                  onClick={() => setChatMessages([])}
                  className="text-xs text-gray-400 hover:text-white transition-colors px-2 py-1 rounded border border-gray-700 hover:border-purple-500 hover:bg-purple-500/10"
                  disabled={chatMessages.length === 0}
                >
                  Clear Chat
                </button>
              </div>
            </div>

            {/* Server Status Indicator */}
            <div className="mt-6 py-3 border-t border-gray-800">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${serverStatus === 'online' ? 'bg-green-500' : serverStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'}`}></div>
                  <span className="text-sm font-medium">
                    Server Status: {" "}
                    {serverStatus === 'online' ? (
                      <span className="text-green-400">Connected</span>
                    ) : serverStatus === 'offline' ? (
                      <span className="text-red-400">Disconnected</span>
                    ) : (
                      <span className="text-yellow-400">Checking...</span>
                    )}
                  </span>
                </div>
                {serverStatus === 'offline' && (
                  <button 
                    onClick={() => {
                      checkServerStatus().then(isOnline => {
                        if (isOnline) refreshDownloadedModels();
                      });
                    }}
                    className="text-xs px-2 py-1 bg-purple-700/50 hover:bg-purple-600/50 rounded border border-purple-500/50 text-white"
                  >
                    Retry Connection
                  </button>
                )}
              </div>
            </div>
            
            {/* MF Branding */}
            <div className="text-center text-xs md:text-sm text-gray-500 mt-4 py-2 border-t border-gray-800">
              <p>Powered by Marco Figueroa</p>
              <p className="text-purple-400 font-medium">MF Vibe-Tuning v1.0</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MFFineTuning;
