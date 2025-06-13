import React, { useState, useEffect } from 'react';
import { Cpu, Zap, HardDrive, Server, RefreshCw, BarChart, XCircle, Check } from 'lucide-react';
import LoraConfig from './LoraConfig';
import ApplyAdapterButton from './ApplyAdapterButton';

const LoraTraining = ({ selectedModel, selectedDataset, onTrainingComplete }) => {
  const [hardwareInfo, setHardwareInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [loraConfig, setLoraConfig] = useState(null);
  const [error, setError] = useState(null);
  const [notification, setNotification] = useState(null);
  const [isQLoraEnabled, setIsQLoraEnabled] = useState(false);
  const [hardwareTier, setHardwareTier] = useState('detecting');
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [lossData, setLossData] = useState([]);
  const [adapterId, setAdapterId] = useState(null);
  const [adapterApplied, setAdapterApplied] = useState(false);
  const [selectedModelForAdapter, setSelectedModelForAdapter] = useState('');

  // Fetch hardware info on component mount, only once
  useEffect(() => {
    console.log('Hardware detection: Fetching hardware info on mount');
    fetchHardwareInfo();
    
    // Fetch hardware info every 30 seconds to keep it updated without causing flashing
    const interval = setInterval(() => {
      console.log('Hardware detection: Periodic refresh');
      fetchHardwareInfo();
    }, 30000); // 30 seconds interval
    
    return () => clearInterval(interval);
  }, []);

  // Track if a fetch is in progress to prevent multiple concurrent requests
  const [isFetchingHardware, setIsFetchingHardware] = useState(false);
  
  // Fetch hardware information with debouncing
  const fetchHardwareInfo = async () => {
    // Prevent concurrent fetches
    if (isFetchingHardware) {
      console.log('Hardware detection: Skipping fetch as one is already in progress');
      return;
    }
    
    setIsFetchingHardware(true);
    
    // Only show loading state for initial fetch
    if (!hardwareInfo) {
      setIsLoading(true);
    }
    
    try {
      console.log('Hardware detection: Fetching data from API');
      const response = await fetch('http://127.0.0.1:5002/api/lora/hardware');
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success') {
          // Only update state if data has actually changed
          const newHardwareJSON = JSON.stringify(data.hardware);
          const oldHardwareJSON = hardwareInfo ? JSON.stringify(hardwareInfo) : '';
          
          if (newHardwareJSON !== oldHardwareJSON) {
            console.log('Hardware detection: Updating state with new hardware info');
            setHardwareInfo(data.hardware);
            
            // Configure LoRA/QLoRA based on available VRAM
            if (data.hardware.gpu.available) {
              const totalGpuMemory = data.hardware.gpu.devices.reduce(
                (sum, device) => sum + parseFloat(device.total_memory_gb), 0
              );
              
              // Define VRAM tiers for different configurations
              if (totalGpuMemory >= 16) {
                // High-end GPU (16GB+ VRAM): RTX 3090, 4090, etc.
                setHardwareTier('high');
                setIsQLoraEnabled(true); // Still use QLoRA for consistency and quality
              } else if (totalGpuMemory >= 8) {
                // Mid-range GPU (8-16GB VRAM): RTX 3070, 3080, 2080 SUPER, etc.
                setHardwareTier('medium');
                setIsQLoraEnabled(true);
              } else if (totalGpuMemory >= 4) {
                // Entry-level GPU (4-8GB VRAM): GTX 1650, RTX 2060, etc.
                setHardwareTier('low');
                setIsQLoraEnabled(true);
              } else {
                // Insufficient VRAM: Intel integrated, old GPUs, etc.
                setHardwareTier('minimal');
                setIsQLoraEnabled(false);
              }
            } else {
              // No GPU available
              setHardwareTier('cpu');
              setIsQLoraEnabled(false);
            }
          } else {
            console.log('Hardware detection: No changes in hardware info');
          }
        }
      }
    } catch (error) {
      console.error('Error fetching hardware info:', error);
      setError('Failed to fetch hardware information. Please check if the server is running.');
    } finally {
      setIsLoading(false);
      setIsFetchingHardware(false);
    }
  };

  // Handle LoRA config updates
  const handleConfigChange = (config) => {
    setLoraConfig(config);
  };

  // Apply LoRA to model
  const applyLoraToModel = async (specificAdapterId = null) => {
    console.log("=== Applying LoRA to model ===");
    console.log("Selected model:", selectedModel);
    console.log("LoRA config:", loraConfig);
    console.log("Specific adapter ID:", specificAdapterId);
    
    if (!selectedModel) {
      setNotification({
        type: 'error',
        message: 'Please select a model first'
      });
      return;
    }

    setIsApplying(true);
    setError(null);
    
    try {
      // Ensure we have a valid config object
      const config = loraConfig || {
        adapter_type: 'lora',
        rank: 8,
        alpha: 16,
        dropout: 0.05,
        target_modules: ['q_proj', 'v_proj'],
        quantization: 'none'
      };
      
      // Create a safe copy of the config without any potential circular references
      const safeConfig = config ? JSON.parse(JSON.stringify({
        adapter_type: config.adapter_type || 'lora',
        rank: config.rank || 8,
        alpha: config.alpha || 16,
        dropout: config.dropout || 0.05,
        target_modules: config.target_modules || ['q_proj', 'v_proj'],
        quantization: config.quantization || 'none'
      })) : null;
      
      console.log("Sending API request to apply LoRA...");
      console.log("Request payload:", {
        model_id: selectedModel,
        quantization: safeConfig?.quantization !== 'none' ? safeConfig?.quantization : null
      });
      
      // Create a safe request payload
      const requestPayload = {
        model_id: selectedModel,
        quantization: safeConfig?.quantization || 'none'
      };
      
      // Add either adapter_id or config based on the specificAdapterId parameter
      if (specificAdapterId) {
        requestPayload.adapter_id = specificAdapterId;
      } else if (safeConfig) {
        requestPayload.config = safeConfig;
      }
      
      const response = await fetch('http://127.0.0.1:5002/api/lora/apply', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestPayload),
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success') {
          setModelInfo(data.model_info);
          
          // Set adapter applied flag if we're applying a specific adapter
          if (specificAdapterId) {
            setAdapterApplied(true);
            setNotification({
              type: 'success',
              message: `LoRA adapter ${specificAdapterId} applied successfully!`
            });
          } else {
            setNotification({
              type: 'success',
              message: 'LoRA applied successfully!'
            });
          }
        } else {
          setError(data.message || 'Failed to apply LoRA');
          setNotification({
            type: 'error',
            message: data.message || 'Failed to apply LoRA'
          });
          if (specificAdapterId) {
            setAdapterApplied(false);
          }
        }
      } else {
        setError('Server returned an error. Please check the logs.');
      }
    } catch (error) {
      console.error('Error applying LoRA:', error);
      setError(`Failed to apply LoRA: ${error.message}`);
    } finally {
      setIsApplying(false);
    }
  };

  // Start training with LoRA
  const startTraining = async () => {
    console.log("=== Starting LoRA training ===");
    console.log("Selected model:", selectedModel);
    console.log("Dataset:", selectedDataset ? `${selectedDataset.length} samples` : 'none');
    
    if (!selectedModel) {
      setNotification({
        type: 'error',
        message: 'Please select a model first'
      });
      return;
    }
    
    if (!selectedDataset) {
      setNotification({
        type: 'error',
        message: 'Please upload and process a dataset first'
      });
      return;
    }
    
    // If model info is not set, show warning but don't block
    if (!modelInfo) {
      console.warn("No model info available - LoRA adapter may not be applied");
    }
    
    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingLogs([]);
    setLossData([]);
    setError(null);
    
    try {
      // Set default config if not provided
      const config = loraConfig || {
        adapter_type: 'lora',
        rank: 8,
        alpha: 16,
        dropout: 0.05,
        target_modules: ['q_proj', 'v_proj'],
        quantization: 'none'
      };
      
      const requestBody = {
        model_id: selectedModel,
        dataset: selectedDataset,
        lora_config: config,
        training_config: {
          num_epochs: 3,
          learning_rate: 1e-4,
          batch_size: 4,
          max_length: 1024
        }
      };
      
      console.log("Starting training with config:", requestBody);
      
      // Call training API
      const response = await fetch('http://127.0.0.1:5002/api/lora/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success') {
          // Set up polling for training status
          const pollInterval = setInterval(async () => {
            try {
              const statusResponse = await fetch('http://127.0.0.1:5002/api/lora/training_status');
              if (statusResponse.ok) {
                const statusData = await statusResponse.json();
                
                // Update progress
                setTrainingProgress(statusData.progress || 0);
                
                // Add new logs
                if (statusData.logs && statusData.logs.length > 0) {
                  setTrainingLogs(prevLogs => {
                    // Only add logs we don't already have
                    const existingTimestamps = new Set(prevLogs.map(log => log.timestamp));
                    const newLogs = statusData.logs.filter(log => !existingTimestamps.has(log.timestamp));
                    return [...prevLogs, ...newLogs];
                  });
                }
                
                // Update loss data for chart
                if (statusData.loss) {
                  setLossData(prevData => {
                    const existingSteps = new Set(prevData.map(point => point.step));
                    const newPoints = Object.entries(statusData.loss)
                      .map(([step, value]) => ({ step: parseInt(step), value }))
                      .filter(point => !existingSteps.has(point.step));
                    
                    return [...prevData, ...newPoints].sort((a, b) => a.step - b.step);
                  });
                }
                
                // Check if training is complete
                if (statusData.status === 'completed') {
                  clearInterval(pollInterval);
                  setIsTraining(false);
                  setAdapterId(statusData.adapter_id);
                  
                  // Notify parent component
                  if (onTrainingComplete) {
                    onTrainingComplete({
                      adapterId: statusData.adapter_id,
                      loss: statusData.loss
                    });
                  }
                  
                  setNotification({
                    type: 'success',
                    message: `Training completed! Adapter ID: ${statusData.adapter_id}`
                  });
                }
                
                // Check if there was an error
                if (statusData.status === 'error') {
                  clearInterval(pollInterval);
                  setIsTraining(false);
                  setError(statusData.error || 'An error occurred during training');
                }
              }
            } catch (error) {
              console.error('Error polling training status:', error);
            }
          }, 2000); // Poll every 2 seconds
          
          // Clear interval if component unmounts
          return () => clearInterval(pollInterval);
        } else {
          setError(data.message || 'Failed to start training');
          setIsTraining(false);
        }
      } else {
        setError('Server returned an error. Please check the logs.');
        setIsTraining(false);
      }
    } catch (error) {
      console.error('Error starting training:', error);
      setError(`Failed to start training: ${error.message}`);
      setIsTraining(false);
    }
  };

  // Render LoRA Training component
  return (
    <div className="p-4">
      {/* Hardware Info Section */}
      <div className="mb-6 bg-gray-900/40 rounded-lg p-4 border border-gray-700/50">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-md font-medium text-purple-300 flex items-center gap-2">
            <Server size={16} className="text-purple-400" />
            Hardware Detection
          </h3>
          <button 
            onClick={fetchHardwareInfo} 
            className="p-1 bg-gray-700/70 hover:bg-gray-600/70 rounded text-xs"
            title="Refresh hardware detection"
          >
            <RefreshCw size={14} className={isLoading ? "animate-spin" : ""} />
          </button>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <RefreshCw size={20} className="animate-spin text-purple-400 mr-2" />
            <span className="text-gray-300">Detecting hardware...</span>
          </div>
        ) : hardwareInfo ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* GPU Info */}
            <div className="bg-gray-800/40 rounded-lg p-3 border border-gray-700/50">
              <h4 className="text-sm font-medium text-blue-300 mb-2">
                <span className="font-medium">GPU</span>
              </h4>
              {hardwareInfo.gpu.available ? (
                <div className="space-y-2">
                  <div className="text-sm text-green-400">
                    {hardwareInfo.gpu.count} GPU(s) Available
                  </div>
                  {hardwareInfo.gpu.devices.map((device, idx) => (
                    <div key={idx} className="text-xs text-gray-400 flex justify-between">
                      <span>{device.name}</span>
                      <span>{device.total_memory_gb} GB</span>
                    </div>
                  ))}
                  <div className="text-xs text-purple-300 mt-1">
                    {isQLoraEnabled ? 
                      "✓ QLoRA enabled (4-bit quantization available)" : 
                      "⚠ QLoRA disabled (insufficient VRAM)"
                    }
                  </div>
                  <div className="text-xs text-blue-300 mt-1">
                    {hardwareTier === 'high' && "Hardware Tier: High-End GPU (16GB+ VRAM) - Using optimized settings"}
                    {hardwareTier === 'medium' && "Hardware Tier: Mid-Range GPU (8-16GB VRAM) - Using balanced settings"}
                    {hardwareTier === 'low' && "Hardware Tier: Entry GPU (4-8GB VRAM) - Using memory-efficient settings"}
                    {hardwareTier === 'minimal' && "Hardware Tier: Limited GPU (<4GB VRAM) - Using minimal settings"}
                    {hardwareTier === 'cpu' && "Hardware Tier: CPU-Only - Using minimal settings"}
                    {hardwareTier === 'detecting' && "Hardware Tier: Detecting..."}
                  </div>
                </div>
              ) : (
                <div className="text-sm text-red-400">
                  No CUDA-compatible GPU detected
                </div>
              )}
            </div>
            
            {/* CPU Info */}
            <div className="bg-gray-800/40 rounded-lg p-3 border border-gray-700/50">
              <h4 className="text-sm font-medium text-purple-300 mb-2">
                <span className="font-medium">CPU</span>
              </h4>
              <div className="text-sm text-gray-300">
                {hardwareInfo.cpu.count} CPU Core(s) Available
              </div>
              <div className="text-xs text-gray-400 mt-2">
                {hardwareInfo.gpu.available ? 
                  "Training will use GPU acceleration" : 
                  "Training will use CPU only (slower)"
                }
              </div>
            </div>
          </div>
        ) : (
          <div className="text-sm text-gray-400 p-3">
            <p>Failed to detect hardware capabilities. Please check if the server is running.</p>
            <button 
              onClick={fetchHardwareInfo}
              className="mt-2 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-md text-xs flex items-center gap-1"
            >
              <RefreshCw size={14} /> Retry Detection
            </button>
          </div>
        )}
      </div>
      
      {error && (
        <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-3 mb-4 text-sm text-red-300">
          {error}
        </div>
      )}
      
      {notification && (
        <div className={`${
          notification.type === 'success' ? 'bg-green-500/20 border-green-500/50 text-green-300' : 
          'bg-red-500/20 border-red-500/50 text-red-300'
        } border rounded-lg p-3 mb-4 text-sm`}>
          {notification.message}
        </div>
      )}

      {/* LoRA/QLoRA Configuration */}
      <div className="bg-gray-900/40 rounded-lg p-4 border border-gray-700/50 mb-6">
        <h3 className="text-md font-medium text-purple-300 mb-3 flex items-center gap-2">
          <Zap size={16} className="text-purple-400" />
          LoRA/QLoRA Configuration
        </h3>
        
        <LoraConfig 
          onConfigChange={handleConfigChange}
          isQLoraEnabled={isQLoraEnabled}
          hardwareTier={hardwareTier}
        />
      </div>

      {/* Model info after LoRA application */}
      {modelInfo && (
        <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-3 mb-6">
          <h3 className="text-md font-medium text-blue-300 mb-2 flex items-center gap-2">
            <Server size={16} className="text-blue-400" />
            Model with LoRA Adapter
          </h3>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-400">Base Model: <span className="text-white">{modelInfo.base_model_name}</span></p>
              <p className="text-gray-400">Model Type: <span className="text-white">{modelInfo.model_type}</span></p>
            </div>
            <div>
              <p className="text-gray-400">Trainable Parameters: <span className="text-green-300">{modelInfo.trainable_params.toLocaleString()}</span></p>
              <p className="text-gray-400">Percentage: <span className="text-green-300">{modelInfo.trainable_percentage.toFixed(4)}%</span></p>
            </div>
          </div>
        </div>
      )}

      {/* Training section */}
      {(isTraining || trainingLogs.length > 0 || lossData.length > 0) && (
        <div className="bg-gray-900/40 rounded-lg p-4 border border-gray-700/50 mb-6">
          <div className="mb-6">
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-lg font-semibold text-purple-300 flex items-center gap-2">
                <BarChart size={18} className="text-purple-400" />
                Training Loss
              </h3>
            </div>
            {lossData.length > 0 ? (
              <div className="h-48 bg-gray-900/70 rounded-lg p-3 border border-gray-700/50">
                {/* Simple loss chart */}
                <div className="h-full flex items-end">
                  {lossData.map((point, idx) => (
                    <div 
                      key={idx} 
                      className="bg-purple-500 hover:bg-purple-400 transition-all mx-1 last:bg-green-500 last:hover:bg-green-400"
                      style={{ 
                        height: `${Math.max(5, 100 - (point.value * 100))}%`, 
                        width: `${100 / Math.min(30, lossData.length)}%` 
                      }}
                      title={`Step ${point.step}: Loss ${point.value.toFixed(4)}`}
                    />
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-48 bg-gray-900/70 rounded-lg p-3 border border-gray-700/50 flex items-center justify-center text-gray-400 text-sm">
                Loss data will appear here during training
              </div>
            )}
          </div>
          
          <div>
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-lg font-semibold text-purple-300 flex items-center gap-2">
                <BarChart size={18} className="text-purple-400" />
                Training Logs
              </h3>
            </div>
            <div className="bg-gray-900/70 rounded-lg p-2 max-h-40 overflow-y-auto font-mono text-xs">
              {trainingLogs.map((log, idx) => (
                <div key={idx} className="mb-1 last:mb-0 text-gray-300">
                  <span className="text-gray-500">[{new Date(log.timestamp).toLocaleTimeString()}]</span>{" "}
                  <span>{log.message}</span>{" "}
                  {log.loss && <span className="text-yellow-400">loss: {log.loss}</span>}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons - Moved to mfFT.jsx */}
      <div className="hidden">
        <button
          className="hidden"
          onClick={applyLoraToModel}
          disabled={isApplying || !selectedModel}
          data-action="apply-lora"
        >
          Apply LoRA Configurations to Model
        </button>

        <button
          className="hidden"
          onClick={startTraining}
          disabled={isTraining || !modelInfo || !selectedDataset}
          data-action="start-training"
        >
          Start Training
        </button>
        
        <button
          className="hidden"
          onClick={() => startTraining()}
          data-action="debug-start-training"
        >
          Debug: Force Start Training
        </button>
      </div>

      {/* Training Progress */}
      {isTraining && (
        <div className="mt-4">
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300"
              style={{ width: `${trainingProgress}%` }}
            />
          </div>
          <div className="text-sm text-gray-400 mt-1 text-right">
            {trainingProgress}% Complete
          </div>
        </div>
      )}

      {/* Completed Status */}
      {adapterId && (
        <div className="bg-green-900/30 rounded-lg p-3 border border-green-500/40 text-green-300 mt-4">
          <div className="flex items-center gap-2">
            <span className="font-medium">LoRA Training Complete!</span>
            <span className="text-sm text-green-400">Adapter ID: {adapterId}</span>
          </div>
          <div className="text-sm mt-1">
            Your model has been fine-tuned with LoRA. You can now apply this adapter to any model.
          </div>
          
          {/* Adapter Applied Status */}
          {adapterApplied && (
            <div className="mt-2 bg-blue-900/30 py-2 px-3 rounded-lg border border-blue-500/40 text-blue-300">
              <div className="flex items-center gap-2">
                <Check size={16} className="text-blue-400" />
                <span className="font-medium">LoRA Adapter Applied Successfully!</span>
              </div>
              <div className="text-sm mt-1">
                Your model is now enhanced with the LoRA adapter. Test it with some queries!
              </div>
            </div>
          )}
          
          {/* Current Model Apply Button */}
          <div className="mt-3 space-y-2">
            <div className="bg-gray-800/60 p-2 rounded-lg border border-gray-700">
              <div className="text-sm font-medium mb-2 text-gray-300">Apply to Current Model</div>
              <div className="text-xs text-gray-400 mb-2">
                Apply this adapter to your currently selected model: <span className="text-purple-300 font-mono">{selectedModel || "No model selected"}</span>
              </div>
              
              <ApplyAdapterButton 
                selectedModel={selectedModel} 
                adapterId={adapterId} 
                setNotification={setNotification} 
              />
            </div>
            
            <div className="bg-gray-800/60 p-2 rounded-lg border border-gray-700">
              <div className="text-sm font-medium mb-2 text-gray-300">Apply to Any Model</div>
              <div className="text-xs text-gray-400 mb-2">
                You can also apply this adapter to any other compatible model in your Ollama library.
              </div>
              
              <div className="grid grid-cols-1 gap-2">
                {/* Model Select Dropdown */}
                <select 
                  className="bg-gray-900 w-full rounded-lg border border-gray-700 p-2 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500 mb-2"
                  onChange={(e) => setSelectedModelForAdapter(e.target.value)}
                  value={selectedModelForAdapter || ""}
                >
                  <option value="">Select a different model...</option>
                  <option value="llama3.2:1b">Llama 3.2 (1B)</option>
                  <option value="llama3.2:8b">Llama 3.2 (8B)</option>
                  <option value="llama3:70b">Llama 3 (70B)</option>
                  <option value="mistral:7b">Mistral (7B)</option>
                  <option value="mixtral:8x7b">Mixtral (8x7B)</option>
                </select>
                
                {selectedModelForAdapter && (
                  <ApplyAdapterButton 
                    selectedModel={selectedModelForAdapter} 
                    adapterId={adapterId} 
                    setNotification={setNotification} 
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LoraTraining;
