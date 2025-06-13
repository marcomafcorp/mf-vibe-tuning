import React, { useState, useEffect, memo } from 'react';
import { HelpCircle } from 'lucide-react';

const LoraConfig = ({ onConfigChange, initialConfig, isAdvancedMode = false, isQLoraEnabled = false, hardwareTier = 'detecting' }) => {
  // Default configuration - will be adjusted based on hardware capabilities and tiers
  const getDefaultConfig = (qloraEnabled, tier) => {
    // Base configuration that works for all tiers
    const baseConfig = {
      adapter_type: qloraEnabled ? 'qlora' : 'lora',
      dropout: 0.05,
      target_modules: ['q_proj', 'v_proj']
    };
    
    // Tier-specific optimizations
    switch(tier) {
      case 'high': // 16GB+ VRAM (RTX 3090, 4090, etc.)
        return {
          ...baseConfig,
          rank: 16,        // Higher rank for better adaptation
          alpha: 32,       // Higher alpha for more expressive updates
          target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        };
      
      case 'medium': // 8-16GB VRAM (RTX 3070, 3080, 2080 SUPER, etc.)
        return {
          ...baseConfig,
          rank: 12,        // Moderate rank
          alpha: 24,       // Moderate alpha
          target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        };
      
      case 'low': // 4-8GB VRAM (GTX 1650, RTX 2060, etc.)
        return {
          ...baseConfig,
          rank: 8,         // Lower rank to save memory
          alpha: 16,       // Proportional alpha
          target_modules: ['q_proj', 'v_proj']
        };
      
      case 'minimal': // <4GB VRAM
      case 'cpu': // CPU only
        return {
          ...baseConfig,
          adapter_type: 'lora', // Force standard LoRA for CPU
          rank: 4,         // Minimal rank
          alpha: 8,        // Minimal alpha
          target_modules: ['q_proj']
        };
        
      case 'detecting':
      default:
        return {
          ...baseConfig,
          rank: 8,
          alpha: 16
        };
    }
  };
  
  // Get the appropriate default config based on hardware tier
  const defaultConfig = getDefaultConfig(isQLoraEnabled, hardwareTier);

  // State for configuration
  const [config, setConfig] = useState(initialConfig || defaultConfig);
  const [isLoading, setIsLoading] = useState(false);
  const [availableTargets, setAvailableTargets] = useState({});
  const [selectedTargetGroup, setSelectedTargetGroup] = useState('llama');
  const [showAdvanced, setShowAdvanced] = useState(isAdvancedMode);
  const [selectedModules, setSelectedModules] = useState(config.target_modules);

  // Quantization options
  const [quantizationOptions, setQuantizationOptions] = useState(['none']);
  const [selectedQuantization, setSelectedQuantization] = useState(isQLoraEnabled ? '4bit' : 'none');
  
  // Update adapter type, quantization, and other settings when hardware capabilities change
  useEffect(() => {
    if (!initialConfig) {
      // Get optimized default settings for the current hardware tier
      const optimizedConfig = getDefaultConfig(isQLoraEnabled, hardwareTier);
      
      // Update settings based on hardware tier
      setConfig(prev => ({
        ...prev,
        adapter_type: optimizedConfig.adapter_type,
        rank: optimizedConfig.rank,
        alpha: optimizedConfig.alpha,
        target_modules: optimizedConfig.target_modules
      }));
      
      // Set appropriate quantization based on hardware tier
      if (isQLoraEnabled) {
        if (hardwareTier === 'high') {
          // For high-end GPUs, 8-bit offers better quality at acceptable memory cost
          setSelectedQuantization('8bit');
        } else {
          // For other GPUs, use 4-bit for maximum memory efficiency
          setSelectedQuantization('4bit');
        }
      } else {
        setSelectedQuantization('none');
      }
      
      // Update selected modules to match the optimized config
      setSelectedModules(optimizedConfig.target_modules);
    }
  }, [isQLoraEnabled, hardwareTier, initialConfig]);
  
  // Fetch available target modules
  useEffect(() => {
    const fetchAvailableTargets = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5002/api/lora/available_targets');
        if (response.ok) {
          const data = await response.json();
          if (data.status === 'success') {
            setAvailableTargets(data.available_targets);
          }
        }
      } catch (error) {
        console.error('Error fetching available targets:', error);
      }
    };

    const fetchQuantizationOptions = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5002/api/lora/quantization_options');
        if (response.ok) {
          const data = await response.json();
          if (data.status === 'success') {
            setQuantizationOptions(data.quantization_options);
          }
        }
      } catch (error) {
        console.error('Error fetching quantization options:', error);
      }
    };

    fetchAvailableTargets();
    if (isQLoraEnabled) {
      fetchQuantizationOptions();
    }
  }, [isQLoraEnabled]);

  // Update parent component when config changes
  useEffect(() => {
    if (onConfigChange) {
      onConfigChange({ ...config, quantization: selectedQuantization });
    }
  }, [config, selectedQuantization, onConfigChange]);

  // Handle input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    
    // Convert numeric values
    const processedValue = ['rank', 'alpha'].includes(name) 
      ? parseInt(value, 10) 
      : name === 'dropout' 
        ? parseFloat(value) 
        : value;
    
    setConfig(prev => ({
      ...prev,
      [name]: processedValue
    }));
  };

  // Handle target module selection
  const handleModuleToggle = (module) => {
    setSelectedModules(prev => {
      if (prev.includes(module)) {
        return prev.filter(m => m !== module);
      } else {
        return [...prev, module];
      }
    });

    setConfig(prev => ({
      ...prev,
      target_modules: selectedModules
    }));
  };

  // Handle quantization selection
  const handleQuantizationChange = (e) => {
    setSelectedQuantization(e.target.value);
  };

  // Handle target group selection
  const handleTargetGroupChange = (e) => {
    setSelectedTargetGroup(e.target.value);
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Basic Settings */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Adapter Type
            </label>
            <select
              name="adapter_type"
              value={config.adapter_type}
              onChange={handleInputChange}
              className="bg-gray-800/80 w-full rounded-lg border border-gray-700 p-2 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
            >
              <option value="lora">LoRA</option>
              <option value="qlora" disabled={!isQLoraEnabled}>QLoRA {!isQLoraEnabled && "(Requires 4GB+ VRAM)"}</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Rank (r)
              <span className="ml-1 text-xs text-gray-500">(Higher = more capacity, but more parameters)</span>
            </label>
            <input
              type="number"
              name="rank"
              value={config.rank}
              onChange={handleInputChange}
              min="1"
              max="256"
              className="bg-gray-800/80 w-full rounded-lg border border-gray-700 p-2 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Alpha (α)
              <span className="ml-1 text-xs text-gray-500">(Typically 2x rank)</span>
            </label>
            <input
              type="number"
              name="alpha"
              value={config.alpha}
              onChange={handleInputChange}
              min="1"
              max="512"
              className="bg-gray-800/80 w-full rounded-lg border border-gray-700 p-2 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
            />
          </div>

          {isQLoraEnabled && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Quantization
                <span className="ml-1 text-xs text-gray-500">(4-bit uses less VRAM but 8-bit may be more stable)</span>
              </label>
              <select
                value={selectedQuantization}
                onChange={handleQuantizationChange}
                className="bg-gray-800/80 w-full rounded-lg border border-gray-700 p-2 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
              >
                {quantizationOptions.map(option => (
                  <option key={option} value={option}>
                    {option === '4bit' ? '4-bit Quantization (Most Memory Efficient)' : 
                     option === '8bit' ? '8-bit Quantization (Better Quality)' : 
                     'No Quantization (Standard LoRA)'}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* Advanced Settings */}
        <div className={`space-y-4 ${showAdvanced ? 'block' : 'hidden md:block'}`}>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Dropout
              <span className="ml-1 text-xs text-gray-500">(Regularization, typically 0.05-0.1)</span>
            </label>
            <input
              type="number"
              name="dropout"
              value={config.dropout}
              onChange={handleInputChange}
              min="0"
              max="0.5"
              step="0.01"
              className="bg-gray-800/80 w-full rounded-lg border border-gray-700 p-2 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Model Architecture
              <span className="ml-1 text-xs text-gray-500">(For target modules)</span>
            </label>
            <select
              value={selectedTargetGroup}
              onChange={handleTargetGroupChange}
              className="bg-gray-800/80 w-full rounded-lg border border-gray-700 p-2 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
            >
              {Object.keys(availableTargets).map(group => (
                <option key={group} value={group}>{group}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="flex items-center justify-between text-sm font-medium text-gray-300 mb-1">
              <span>Target Modules</span>
              <button 
                className="text-xs text-purple-400 hover:text-purple-300 flex items-center"
                title="What parts of the model to adapt with LoRA"
              >
                <HelpCircle size={12} className="mr-1" />
                Help
              </button>
            </label>
            <div className="bg-gray-800/80 rounded-lg border border-gray-700 p-2 max-h-40 overflow-y-auto">
              {availableTargets[selectedTargetGroup]?.map(module => (
                <div key={module} className="flex items-center mb-1 last:mb-0">
                  <input
                    type="checkbox"
                    id={`module-${module}`}
                    checked={selectedModules.includes(module)}
                    onChange={() => handleModuleToggle(module)}
                    className="mr-2 accent-purple-500 bg-gray-700 border-gray-600"
                  />
                  <label htmlFor={`module-${module}`} className="text-sm text-gray-300">
                    {module}
                  </label>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 bg-purple-900/20 rounded-lg p-3 border border-purple-500/30">
        <div className="text-sm text-purple-300">
          <span className="font-medium">Efficiency Stats:</span> {" "}
          <span className="text-gray-300">
            {isQLoraEnabled && selectedQuantization !== 'none' ? 
              `Using ${selectedQuantization} quantization with ${config.rank} rank` : 
              `Using full precision with ${config.rank} rank`}
            {" "} - approximately{" "}
            <span className="text-green-300 font-medium">
              {isQLoraEnabled && selectedQuantization === '4bit' ? 
                '0.5%' : 
              selectedQuantization === '8bit' ? 
                '1%' : 
                `${(config.rank * selectedModules.length * 2 / 7680 * 100).toFixed(2)}%`}
            </span> of full model parameters
          </span>
        </div>
      </div>
    </div>
  );
};

// Export a memoized version to prevent unnecessary re-renders
export default memo(LoraConfig);
