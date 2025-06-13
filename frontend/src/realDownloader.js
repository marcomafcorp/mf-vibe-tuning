// realDownloader.js - Direct model downloading without simulation

// Function to trigger actual model download and track real progress
export async function downloadModel(
  modelId, 
  setDownloadProgress, 
  setDownloadDetails, 
  setDownloadStatus,
  setDownloadedModels,
  refreshDownloadedModels
) {
  // Convert model ID to the correct format for Ollama
  const actualModelId = modelId === 'llama3.2:1b' ? 'llama3.2:1b' : modelId;
  
  console.log(`[REAL DOWNLOADER] Starting DIRECT API download for ${actualModelId}`);
  setDownloadStatus('downloading');
  setDownloadProgress(0);
  setDownloadDetails(`Initiating download of ${modelId}...`);
  
  try {
    // Direct API call to server to start the download
    const response = await fetch('http://127.0.0.1:5001/api/ollama/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: actualModelId })
    });
    
    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('Server response:', data);
    
    // Start polling for real progress
    const pollComplete = await pollRealProgress(
      actualModelId,
      setDownloadProgress,
      setDownloadDetails,
      setDownloadStatus,
      setDownloadedModels,
      refreshDownloadedModels
    );
    
    if (pollComplete) {
      // Add the model to downloaded models
      setDownloadedModels(prev => {
        if (!prev.includes(modelId)) {
          return [...prev, modelId];
        }
        return prev;
      });
      
      // Refresh the models list
      setTimeout(() => {
        refreshDownloadedModels();
        console.log(`[REAL DOWNLOADER] Download completed for ${modelId}`);
      }, 1000);
    }
    
    return pollComplete;
  } catch (error) {
    console.error('[REAL DOWNLOADER] Error:', error);
    setDownloadStatus('error');
    setDownloadDetails(`Download API call failed: ${error.message}. Please check the server status.`);
    return false;
  }
}

// Function to poll the server for real download progress
async function pollRealProgress(
  modelId,
  setDownloadProgress,
  setDownloadDetails,
  setDownloadStatus,
  setDownloadedModels,
  refreshDownloadedModels
) {
  return new Promise((resolve) => {
    let completed = false;
    let errorCount = 0;
    const maxErrors = 5;
    
    const checkProgress = async () => {
      try {
        // Call the real-progress endpoint
        const response = await fetch(`http://127.0.0.1:5001/api/ollama/real-progress?model=${encodeURIComponent(modelId)}`);
        
        if (response.ok) {
          const data = await response.json();
          console.log('Progress data:', data);
          
          // Update UI with real progress
          const progress = data.progress || 0;
          setDownloadProgress(progress);
          
          if (data.details) {
            setDownloadDetails(data.details);
          }
          
          if (data.status === 'completed') {
            setDownloadStatus('completed');
            setDownloadProgress(100);
            setDownloadDetails('Download completed successfully!');
            completed = true;
            
            // Immediately add the model to downloaded models list
            setDownloadedModels(prev => {
              if (!prev.includes(modelId)) {
                return [modelId, ...prev];
              }
              return prev;
            });
            
            // Force a refresh of the model list immediately
            refreshDownloadedModels();
            
            resolve(true);
            return;
          } else if (data.status === 'error') {
            setDownloadStatus('error');
            setDownloadDetails(`Download failed: ${data.details || 'Unknown error'}`);
            completed = true;
            resolve(false);
            return;
          }
          
          // Reset error count on successful response
          errorCount = 0;
        } else {
          errorCount++;
          console.error(`Error checking progress (${errorCount}/${maxErrors}):`, response.status);
          
          if (errorCount >= maxErrors) {
            setDownloadStatus('error');
            setDownloadDetails('Lost connection to server. Please check your backend.');
            completed = true;
            resolve(false);
            return;
          }
        }
      } catch (error) {
        errorCount++;
        console.error(`Error checking progress (${errorCount}/${maxErrors}):`, error);
        
        if (errorCount >= maxErrors) {
          setDownloadStatus('error');
          setDownloadDetails(`Connection error: ${error.message}`);
          completed = true;
          resolve(false);
          return;
        }
      }
      
      // Continue polling if not completed
      if (!completed) {
        setTimeout(checkProgress, 1000);
      }
    };
    
    // Start checking progress
    checkProgress();
  });
}
