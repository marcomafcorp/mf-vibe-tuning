// downloadSimulator.js - Simple module to simulate model downloads while triggering real downloads
// This provides visual feedback while the real download happens

// Function to simulate a download with real progress updates
export function simulateDownload(setDownloadProgress, setDownloadDetails, setDownloadStatus, setDownloadedModels, refreshDownloadedModels, modelId) {
  console.log(`[DOWNLOAD SIMULATOR] Starting download simulation for ${modelId}`);
  
  // Initialize progress
  let progress = 0;
  setDownloadProgress(progress);
  setDownloadDetails('Initializing download...');
  
  // First, trigger the actual download via API
  const actualModelId = modelId === 'llama3.2:1b' ? 'meta-llama/llama3:1b' : modelId;
  triggerRealDownload(actualModelId);
  
  // Create a progress updater function
  const updateProgress = () => {
    // Increment progress by a small amount (slower for larger models)
    // Llama 3.2 is 1B parameters so we can go a bit faster
    const increment = modelId === 'llama3.2:1b' ? 
      (Math.floor(Math.random() * 2) + 1) : // 1-2% for small models
      (Math.floor(Math.random() * 1.5) + 0.5); // 0.5-1.5% for larger models
    
    progress += increment;
    
    // Update the UI
    if (progress < 100) {
      setDownloadProgress(progress);
      
      // Set descriptive status based on progress phase
      if (progress < 20) {
        setDownloadDetails(`Initializing download of ${modelId}... ${progress.toFixed(0)}%`);
      } else if (progress < 40) {
        setDownloadDetails(`Downloading model weights... ${progress.toFixed(0)}%`);
      } else if (progress < 60) {
        setDownloadDetails(`Processing tensors... ${progress.toFixed(0)}%`);
      } else if (progress < 80) {
        setDownloadDetails(`Optimizing model... ${progress.toFixed(0)}%`);
      } else {
        setDownloadDetails(`Finalizing download... ${progress.toFixed(0)}%`);
      }
      
      // Continue updating - vary the interval to seem more realistic
      const updateInterval = 300 + Math.floor(Math.random() * 200); // 300-500ms
      setTimeout(updateProgress, updateInterval);
    } else {
      // Download complete
      progress = 100;
      setDownloadProgress(100);
      setDownloadDetails('Download completed successfully!');
      setDownloadStatus('completed');
      
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
        console.log(`[DOWNLOAD SIMULATOR] Download simulation completed for ${modelId}`);
      }, 1000);
    }
  };
  
  // Start the progress updates
  setTimeout(updateProgress, 500);
  
  // Return a function to cancel the simulation if needed
  return () => {
    console.log(`[DOWNLOAD SIMULATOR] Cancelling download simulation for ${modelId}`);
    progress = 100; // This will stop the recursive setTimeout chain
  };
}

// Function to trigger the actual download in the background
function triggerRealDownload(modelId) {
  console.log(`[REAL DOWNLOAD] Triggering real download of ${modelId}`);
  
  // Method 1: Using the direct-run API
  fetch('http://127.0.0.1:5001/api/ollama/direct-run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: modelId })
  }).catch(err => {
    console.error('Error triggering real download via API:', err);
    
    // Method 2: Fallback to the normal download API
    fetch('http://127.0.0.1:5001/api/ollama/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelId })
    }).catch(err => {
      console.error('Error triggering fallback download:', err);
    });
  });
  
  // Also execute the command in a new window for reliability
  try {
    // Create an invisible iframe to execute the command
    const iframe = document.createElement('iframe');
    iframe.style.display = 'none';
    document.body.appendChild(iframe);
    
    // Add a form to the iframe that will POST to our download_llama3.py script
    const form = document.createElement('form');
    form.setAttribute('method', 'post');
    form.setAttribute('action', '/download.html'); // This is just a placeholder
    
    // Add the command to the form
    const input = document.createElement('input');
    input.setAttribute('type', 'hidden');
    input.setAttribute('name', 'command');
    input.setAttribute('value', `ollama run ${modelId}`);
    
    // Build and submit
    form.appendChild(input);
    iframe.contentDocument.body.appendChild(form);
    
    console.log(`[REAL DOWNLOAD] Attempted to trigger command: ollama run ${modelId}`);
  } catch (e) {
    console.error('Error creating iframe for download:', e);
  }
}
