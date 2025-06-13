import React, { useState } from 'react';
import { Zap, Check } from 'lucide-react';

const ApplyAdapterButton = ({ selectedModel, adapterId, setNotification }) => {
  const [isApplying, setIsApplying] = useState(false);
  const [adapterApplied, setAdapterApplied] = useState(false);

  const applyAdapter = async () => {
    if (!selectedModel) {
      setNotification({
        message: 'Please select a model first',
        type: 'error'
      });
      return;
    }

    setIsApplying(true);
    
    try {
      console.log(`Applying LoRA adapter ${adapterId} to model ${selectedModel}`);
      
      const response = await fetch('http://127.0.0.1:5002/api/lora/apply', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: selectedModel,
          adapter_id: adapterId
        }),
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setAdapterApplied(true);
        setNotification({
          message: `LoRA adapter ${adapterId} applied successfully to ${selectedModel}!`,
          type: 'success'
        });
      } else {
        setNotification({
          message: `Failed to apply adapter: ${data.message || 'Unknown error'}`,
          type: 'error'
        });
      }
    } catch (error) {
      console.error('Error applying adapter:', error);
      setNotification({
        message: 'Error applying adapter. Check console for details.',
        type: 'error'
      });
    } finally {
      setIsApplying(false);
    }
  };

  return (
    <button
      className="bg-gradient-to-r from-green-600 to-green-700 hover:from-green-500 hover:to-green-600 text-white font-medium py-2 px-6 rounded-lg transition-all flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
      onClick={applyAdapter}
      disabled={isApplying || adapterApplied || !selectedModel}
    >
      {isApplying ? (
        <>
          <span className="animate-spin">⟳</span>
          Applying Adapter...
        </>
      ) : adapterApplied ? (
        <>
          <Check size={16} />
          Adapter Applied
        </>
      ) : (
        <>
          <Zap size={16} />
          Apply LoRA Configurations to {selectedModel || "Selected Model"}
        </>
      )}
    </button>
  );
};

export default ApplyAdapterButton;
