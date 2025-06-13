from flask import Flask, request, jsonify
import subprocess
import threading
import os
import sys
import json
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import time
import requests

# Set UTF-8 as default encoding to handle special characters
import codecs
import locale

# Force UTF-8 encoding for stdout and stderr
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Set environment variable for subprocess calls
os.environ['PYTHONIOENCODING'] = 'utf-8'

app = Flask(__name__)
CORS(app)

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({'status': 'ok', 'message': 'Backend is running'})

@app.route('/api/test/models', methods=['GET'])
def test_models():
    """Test endpoint that returns dummy models"""
    dummy_models = [
        {
            'id': 'llama2:7b',
            'hash': 'test123456',
            'size': '3.8 GB',
            'modified': '2 days ago'
        },
        {
            'id': 'mistral:7b',
            'hash': 'test789012',
            'size': '4.1 GB',
            'modified': '1 week ago'
        }
    ]
    return jsonify({
        'status': 'success',
        'models': dummy_models
    })

@app.route('/api/test/download', methods=['POST'])
def test_download():
    """Test endpoint that simulates a download"""
    data = request.get_json()
    model = data.get('model', 'test-model')
    
    print(f"Test download endpoint called for model: {model}")
    
    # Simulate download by updating progress
    def simulate_download():
        import time
        for i in range(0, 101, 10):
            download_progress[model] = i
            download_status[model] = 'downloading'
            download_details[model] = f'Test download progress: {i}%'
            time.sleep(0.5)
        
        download_progress[model] = 100
        download_status[model] = 'completed'
        download_details[model] = 'Test download completed!'
    
    # Start simulation in background
    thread = threading.Thread(target=simulate_download)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Test download started'
    })

# Global variables for tracking downloads
current_downloads = {}
download_locks = {}

# Track real download progress
download_progress = {}
download_status = {}  # model -> percent (0-100)
download_cancelled = {}  # model -> bool
download_details = {}  # model -> str (last status line)

def run_ollama_pull(model):
    import re
    try:
        # Log the download attempt
        print(f"Starting download for model: {model}")
        
        # Create consistent model key for tracking
        model_key = model.replace('/', '_').replace(':', '_')
        
        # Format the model name correctly for Ollama
        formatted_model = model
        if model == 'llama3.2:1b':
            # Try the alternative format for Llama 3.2 1B
            formatted_model = 'meta-llama/llama3:1b'
            print(f"Using alternative model format for Llama 3.2 1B: {formatted_model}")
        
        # Store status using both model and model_key for compatibility
        download_status[model] = 'downloading'
        download_status[model_key] = 'downloading'
        download_progress[model] = 0
        download_progress[model_key] = 0
        download_cancelled[model] = False
        download_cancelled[model_key] = False
        download_details[model] = 'Initializing download...'
        download_details[model_key] = 'Initializing download...'
        
        # Log the actual command being executed
        print(f"Executing command: ollama pull {formatted_model}")
        
        # Use text mode with line buffering to properly handle progress
        process = subprocess.Popen(
            ["ollama", "pull", formatted_model], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=0  # Unbuffered to get real-time updates
        )
        
        # Read output character by character to handle \r properly
        current_line = ""
        while True:
            char = process.stdout.read(1)
            if not char:
                # Process any remaining line
                if current_line:
                    process_line(current_line, model, model_key)
                break
            
            if char == '\r':
                # Carriage return - process current line and reset
                if current_line:
                    process_line(current_line, model, model_key)
                current_line = ""
            elif char == '\n':
                # Newline - process current line and reset
                if current_line:
                    process_line(current_line, model, model_key)
                current_line = ""
            else:
                # Add character to current line
                current_line += char
            
            # Check for cancellation
            if download_cancelled.get(model_key, download_cancelled.get(model)):
                process.terminate()
                if download_status.get(model_key, download_status.get(model)) == 'paused':
                    download_details[model] = 'Paused by user.'
                    download_details[model_key] = 'Paused by user.'
                else:
                    download_details[model] = 'Cancelled by user.'
                    download_details[model_key] = 'Cancelled by user.'
                download_progress[model] = 0
                download_progress[model_key] = 0
                return
        
        # Wait for process to complete
        process.wait()
        print(f"Process completed with return code: {process.returncode}")
        
        if download_cancelled.get(model):
            if download_status[model] == 'paused':
                download_details[model] = 'Paused by user.'
                download_details[model_key] = 'Paused by user.'
            else:
                download_details[model] = 'Cancelled by user.'
                download_details[model_key] = 'Cancelled by user.'
            download_progress[model] = 0
            download_progress[model_key] = 0
        elif process.returncode == 0:
            download_status[model] = 'completed'
            download_status[model_key] = 'completed'
            download_progress[model] = 100
            download_progress[model_key] = 100
            download_details[model] = 'Download completed.'
            download_details[model_key] = 'Download completed.'
            print(f"Download completed successfully for {model}")
        else:
            download_status[model] = 'error: download failed'
            download_status[model_key] = 'error: download failed'
            download_progress[model] = 0
            download_progress[model_key] = 0
            download_details[model] = f'Download failed with code {process.returncode}.'
            download_details[model_key] = f'Download failed with code {process.returncode}.'
            print(f"Download failed for {model} with return code {process.returncode}")
    except Exception as e:
        error_msg = str(e)
        print(f"Exception during download: {error_msg}")
        download_status[model] = f'error: {error_msg}'
        download_status[model_key] = f'error: {error_msg}'
        download_progress[model] = 0
        download_progress[model_key] = 0
        download_details[model] = f'Error: {error_msg}'
        download_details[model_key] = f'Error: {error_msg}'

def process_line(line_text, model, model_key):
    """Process a single line of output from ollama pull"""
    import re
    
    line_text = line_text.strip()
    if not line_text:
        return
    
    # Log raw output for debugging
    print(f"[OLLAMA RAW OUTPUT] {repr(line_text)}")
    
    # First, handle the specific pattern the user is seeing: "pulling [K?25h?2026l?2026h?25l"
    # Remove everything between [ and the next letter/space
    line_text = re.sub(r'\[K\?[0-9]+[hl](\?[0-9]+[hl])*', '', line_text)
    
    # Remove ANSI escape sequences and terminal control characters
    # This includes cursor movement, colors, etc.
    ansi_escape = re.compile(r'''
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI, followed by parameter bytes
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    ''', re.VERBOSE)
    clean_line = ansi_escape.sub('', line_text)
    
    # Remove other control characters like \x08 (backspace), \r (carriage return)
    clean_line = re.sub(r'[\x00-\x1F\x7F]', '', clean_line)
    
    # Remove terminal query sequences like ?25h, ?25l, ?2026h, ?2026l
    clean_line = re.sub(r'\?[0-9]+[hl]', '', clean_line)
    
    # Remove [K which is "erase to end of line"
    clean_line = re.sub(r'\[K', '', clean_line)
    
    # Clean up any remaining escape sequences
    clean_line = re.sub(r'\[[\d;]*[A-Za-z]', '', clean_line)
    
    # Remove multiple spaces
    clean_line = re.sub(r'\s+', ' ', clean_line).strip()
    
    print(f"[OLLAMA CLEAN OUTPUT] {clean_line}")
    
    # Try to find percentage FIRST before determining display text
    percent = None
    percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', clean_line)
    
    if percent_match:
        try:
            percent = float(percent_match.group(1))
            download_progress[model] = percent
            download_progress[model_key] = percent
            print(f"Progress updated: {percent}%")
        except ValueError:
            print(f"Could not parse percentage from: {percent_match.group(1)}")
    
    # Parse what type of message this is and create user-friendly text
    if 'pulling manifest' in clean_line.lower():
        display_text = "Downloading model manifest"
        if percent is not None:
            display_text = f"Downloading model manifest ({percent:.0f}%)"
    elif 'pulling' in clean_line.lower() and ('sha256:' in clean_line or percent is not None):
        # This is a layer download
        if percent is not None:
            display_text = f"Downloading model ({percent:.0f}%)"
        else:
            display_text = "Downloading model layers"
    elif 'verifying' in clean_line.lower():
        display_text = "Verifying download"
        if percent is not None:
            display_text = f"Verifying download ({percent:.0f}%)"
    elif 'writing' in clean_line.lower():
        display_text = "Writing model to disk"
        if percent is not None:
            display_text = f"Writing model to disk ({percent:.0f}%)"
    elif 'success' in clean_line.lower() or 'complete' in clean_line.lower():
        display_text = "Download complete!"
    elif percent is not None:
        # If we have a percentage but couldn't identify the stage, just show download progress
        display_text = f"Downloading model ({percent:.0f}%)"
    else:
        # Default to a simple message
        display_text = "Downloading model"
    
    # Update details with clean, user-friendly text
    download_details[model] = display_text
    download_details[model_key] = display_text
    
    # Check for errors
    if 'error' in clean_line.lower() or 'failed' in clean_line.lower():
        print(f"Error detected in output: {clean_line}")

@app.route('/api/ollama/download', methods=['POST'])
def download_model():
    data = request.get_json()
    model = data.get('model')
    print(f"Download endpoint called with model: {model}")
    if not model:
        return jsonify({'error': 'No model specified'}), 400
    
    # Check if Ollama is running first
    try:
        # Test if ollama is accessible
        test_result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=5)
        if test_result.returncode != 0:
            print("Ollama command failed")
            return jsonify({
                'status': 'error',
                'message': 'Ollama is not properly installed or not in PATH'
            }), 500
    except FileNotFoundError:
        print("Ollama not found")
        return jsonify({
            'status': 'error',
            'message': 'Ollama is not installed. Please install from https://ollama.ai'
        }), 500
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error checking Ollama: {str(e)}'
        }), 500
    
    # Create model key for compatibility
    model_key = model.replace('/', '_').replace(':', '_')
    
    # Check if already downloading before resetting
    if download_status.get(model) == 'downloading' or download_status.get(model_key) == 'downloading':
        return jsonify({'status': 'already_downloading'})
    
    # Reset for retry - use both keys
    download_cancelled[model] = False
    download_cancelled[model_key] = False
    download_progress[model] = 0
    download_progress[model_key] = 0
    download_status[model] = 'downloading'
    download_status[model_key] = 'downloading'
    
    print(f"Download status set to 'downloading' for model: {model} and key: {model_key}")
    print(f"Current download_status: {download_status}")
    
    # Start download in a thread
    thread = threading.Thread(target=run_ollama_pull, args=(model,))
    thread.start()
    return jsonify({'status': 'started'})

@app.route('/api/ollama/status', methods=['GET'])
def get_status():
    model = request.args.get('model')
    status = download_status.get(model, 'idle')
    details = download_details.get(model, '')
    return jsonify({'status': status, 'details': details})

@app.route('/api/ollama/list', methods=['GET'])
def list_models():
    try:
        print("list_models endpoint called")
        result = subprocess.run(["ollama", "list", "--json"], capture_output=True, text=True, encoding='utf-8', errors='replace')
        print(f"ollama list command returned code: {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        
        if result.returncode != 0:
            print(f"Error running ollama list: {result.stderr}")
            return jsonify([])
            
        import json
        try:
            models = json.loads(result.stdout)
            print(f"Parsed models: {models}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw output was: {result.stdout}")
            return jsonify([])
            
        # models is a list of dicts with 'name' keys like 'llama3:8b', 'deepseek-llm:7b', etc.
        ids = [m['name'] for m in models if 'name' in m]
        print(f"Returning model IDs: {ids}")
        return jsonify(ids)
    except Exception as e:
        print(f"Exception in list_models: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([])

@app.route('/api/ollama/progress', methods=['GET'])
def get_progress():
    model = request.args.get('model')
    percent = download_progress.get(model, 0)
    details = download_details.get(model, '')
    return jsonify({'progress': percent, 'details': details})

@app.route('/api/ollama/cancel', methods=['POST'])
def cancel_download():
    data = request.get_json()
    model = data.get('model')
    if not model:
        return jsonify({'error': 'No model specified'}), 400
    download_cancelled[model] = True
    download_status[model] = 'cancelled'
    return jsonify({'status': 'cancelled'})

@app.route('/api/ollama/pause', methods=['POST'])
def pause_download():
    data = request.get_json()
    model = data.get('model')
    if not model:
        return jsonify({'error': 'No model specified'}), 400
    download_cancelled[model] = True
    download_status[model] = 'paused'
    return jsonify({'status': 'paused'})

@app.route('/api/ollama/resume', methods=['POST'])
def resume_download():
    data = request.get_json()
    model = data.get('model')
    if not model:
        return jsonify({'error': 'No model specified'}), 400
    download_cancelled[model] = False
    download_status[model] = 'downloading'
    # Resume is just another pull; Ollama resumes partial downloads
    thread = threading.Thread(target=run_ollama_pull, args=(model,))
    thread.start()
    return jsonify({'status': 'resumed'})

@app.route('/api/ollama/pull', methods=['POST'])
def run_ollama_pull_direct():
    try:
        data = request.json
        if not data or 'model' not in data:
            return jsonify({'error': 'No model specified'}), 400

        model = data['model']
        model_key = model.replace('/', '_').replace(':', '_')
        
        # Log the direct pull request
        print(f"Received direct pull request for model: {model}")
        
        # Function to check if Ollama is running
        def is_ollama_running():
            try:
                # This command should be cross-platform
                check_cmd = "ollama list"
                result = subprocess.run(
                    check_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5  # Short timeout
                )
                return result.returncode == 0
            except Exception:
                return False
        
        # Check if Ollama is running
        if not is_ollama_running():
            print("Ollama is not running. Attempting to start it...")
            try:
                # Try to start Ollama (different commands for different platforms)
                start_cmd = "ollama serve"
                subprocess.Popen(
                    start_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                # Give it time to start
                time.sleep(5)
            except Exception as e:
                print(f"Failed to start Ollama: {str(e)}")
        
        # Initialize progress tracking
        download_progress[model_key] = 0
        download_status[model_key] = 'downloading'
        download_details[model_key] = 'Initializing download...'
        
        # Start a thread to handle the download
        def download_thread(model_name):
            try:
                model_key = model_name.replace('/', '_').replace(':', '_')
                # Set progress for both model and model_key
                download_progress[model_name] = 0
                download_progress[model_key] = 0
                download_status[model_name] = 'downloading'
                download_status[model_key] = 'downloading'
                
                # Use popen to get real-time output
                pull_cmd = f"ollama pull {model_name}"
                print(f"Executing: {pull_cmd}")
                
                # Fix for Windows encoding issues - don't use text mode
                process = subprocess.Popen(
                    pull_cmd, 
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    # Don't use text=True or universal_newlines=True on Windows
                )
                
                # Read output line by line to track progress
                # Use error handling for decoding
                for raw_line in iter(process.stdout.readline, b''):
                    try:
                        # Try to decode with utf-8 and ignore errors
                        line = raw_line.decode('utf-8', errors='ignore').strip()
                        if not line:
                            continue
                        
                        print(f"[{model_name}] {line}")
                        
                        # Clean the line of ANSI escape sequences
                        import re
                        # First, handle the specific pattern: "pulling [K?25h?2026l?2026h?25l"
                        line = re.sub(r'\[K\?[0-9]+[hl](\?[0-9]+[hl])*', '', line)
                        clean_line = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', line)
                        clean_line = re.sub(r'[\x00-\x1F\x7F]', '', clean_line)
                        clean_line = re.sub(r'\?[0-9]+[hl]', '', clean_line)
                        clean_line = re.sub(r'\[K', '', clean_line)
                        clean_line = re.sub(r'\s+', ' ', clean_line).strip()
                        
                        # Try to parse progress percentage first
                        percent = None
                        if '%' in clean_line:
                            percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', clean_line)
                            if percent_match:
                                try:
                                    percent = float(percent_match.group(1))
                                    download_progress[model_name] = percent
                                    download_progress[model_key] = percent
                                except Exception:
                                    pass
                        
                        # Create user-friendly message
                        if 'pulling manifest' in clean_line.lower():
                            display_text = "Downloading model manifest"
                            if percent is not None:
                                display_text = f"Downloading model manifest ({percent:.0f}%)"
                        elif 'pulling' in clean_line.lower():
                            if percent is not None:
                                display_text = f"Downloading model ({percent:.0f}%)"
                            else:
                                display_text = "Downloading model"
                        elif 'verifying' in clean_line.lower():
                            display_text = "Verifying download"
                            if percent is not None:
                                display_text = f"Verifying download ({percent:.0f}%)"
                        elif 'writing' in clean_line.lower():
                            display_text = "Writing model to disk"
                            if percent is not None:
                                display_text = f"Writing model to disk ({percent:.0f}%)"
                        elif 'success' in clean_line.lower() or 'complete' in clean_line.lower():
                            display_text = "Download complete!"
                        elif percent is not None:
                            display_text = f"Downloading model ({percent:.0f}%)"
                        else:
                            display_text = "Downloading model"
                        
                        download_details[model_name] = display_text
                        download_details[model_key] = display_text
                        
                        # Try to parse progress percentage for legacy code
                        if '%' in line and percent is None:
                            try:
                                # Extract percentage from lines like "downloading: 45.23%"
                                percent_str = line.split('%')[0].split(' ')[-1]
                                percent = float(percent_str)
                                download_progress[model_name] = percent
                                download_progress[model_key] = percent
                                print(f"Parsed progress: {percent}%")
                            except Exception as e:
                                print(f"Error parsing progress: {e}")
                    except Exception as e:
                        print(f"Error decoding output: {e}")
                        download_details[model_key] = f"Reading download progress..."
                
                # Wait for process to complete
                return_code = process.wait()
                
                if return_code == 0:
                    download_status[model_name] = 'completed'
                    download_status[model_key] = 'completed'
                    download_progress[model_name] = 100
                    download_progress[model_key] = 100
                    download_details[model_name] = 'Download completed successfully!'
                    download_details[model_key] = 'Download completed successfully!'
                    print(f"Download completed successfully for {model_name}")
                else:
                    download_status[model_name] = 'error'
                    download_status[model_key] = 'error'
                    download_details[model_name] = f"Download failed with code {return_code}"
                    download_details[model_key] = f"Download failed with code {return_code}"
                    print(f"Download failed for {model_name} with code {return_code}")
                
                return return_code == 0
            except Exception as e:
                print(f"Error in download thread: {str(e)}")
                download_status[model_key] = 'error'
                download_details[model_key] = f"Error: {str(e)}"
                return False
        
        # Start the download in a background thread
        thread = threading.Thread(target=download_thread, args=(model,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'downloading', 
            'model': model,
            'message': 'Download started in background with real progress tracking.'
        })
    except Exception as e:
        print(f"Error in pull endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
# Endpoint to check real download progress
@app.route('/api/ollama/real-progress', methods=['GET'])
def check_real_progress():
    try:
        model = request.args.get('model')
        if not model:
            return jsonify({'error': 'No model specified'}), 400
            
        model_key = model.replace('/', '_').replace(':', '_')
        
        # Debug logging
        print(f"Real-progress endpoint called for model: {model}, model_key: {model_key}")
        print(f"Current download_progress keys: {list(download_progress.keys())}")
        print(f"Current download_status keys: {list(download_status.keys())}")
        
        # Get progress data - check both key formats for compatibility
        progress = download_progress.get(model_key, download_progress.get(model, 0))
        status = download_status.get(model_key, download_status.get(model, 'unknown'))
        details = download_details.get(model_key, download_details.get(model, ''))
        
        # Map status values for frontend compatibility
        if status == 'completed' or progress == 100:
            status = 'completed'
        elif status == 'downloading' or (progress > 0 and progress < 100):
            status = 'downloading'
        elif status in ['error', 'error: download failed']:
            status = 'error'
        
        # Log the response
        response_data = {
            'model': model,
            'progress': progress,
            'status': status,
            'details': details
        }
        print(f"Real-progress response: {response_data}")
        
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in progress check endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint to backup a model using the cp command
@app.route('/api/ollama/export', methods=['POST'])
def backup_model():
    try:
        data = request.json
        if not data or 'model' not in data:
            return jsonify({'error': 'No model specified'}), 400

        model = data['model']
        model_key = model.replace('/', '_').replace(':', '_')
        
        # Get dataset name if provided
        dataset_name = ''
        if 'datasetName' in data and data['datasetName']:
            # Clean up dataset name to be safe for a model name
            dataset_name = data['datasetName']
            dataset_name = dataset_name.replace(' ', '_').replace('.', '_')
            dataset_name = '_' + dataset_name
        
        # Generate backup model name with timestamp
        timestamp = time.strftime("%Y%m%d%H%M%S")
        backup_name = f"{model_key}{dataset_name}"
        
        # Run ollama cp command to create a backup
        copy_cmd = f"ollama cp {model} {backup_name}"
        print(f"Executing: {copy_cmd}")
        
        process = subprocess.run(
            copy_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if process.returncode == 0:
            # Return success response
            return jsonify({
                'status': 'success',
                'message': f'Model backed up successfully as {backup_name}',
                'backupName': backup_name
            })
        else:
            print(f"Backup failed with error: {process.stderr}")
            return jsonify({
                'status': 'error',
                'message': f'Backup failed: {process.stderr}'
            }), 500
    except Exception as e:
        print(f"Error in backup endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

# No longer needed since we're using ollama cp instead of export

# Direct model list endpoint with complete model details
@app.route('/api/ollama/models/direct', methods=['GET'])
def list_models_direct():
    try:
        print("list_models_direct endpoint called")
        # Run ollama list command
        result = subprocess.run(
            'ollama list',
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        print(f"ollama list command returned code: {result.returncode}")
        print(f"stdout: {repr(result.stdout)}")
        print(f"stderr: {repr(result.stderr)}")
        
        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            models = []
            
            print(f"Output has {len(output_lines)} lines")
            if len(output_lines) > 0:
                print(f"First line (header): {repr(output_lines[0])}")
            
            # Skip header line
            if len(output_lines) > 1:
                for i in range(1, len(output_lines)):
                    line = output_lines[i].strip()
                    if line:
                        # Format is: NAME ID SIZE MODIFIED
                        # Example: llama3:8b 365c0bd3c000 4.7 GB 16 hours ago
                        # But NAME might have spaces, so we need to handle carefully
                        
                        # First identify where the ID column starts (always a hex value)
                        parts = line.split()
                        id_index = -1
                        for j, part in enumerate(parts):
                            if len(part) >= 12 and all(c in '0123456789abcdef' for c in part[:12]):
                                id_index = j
                                break
                        
                        if id_index > 0:  # Found a valid ID column
                            # Everything before id_index is the model name
                            model_id = ' '.join(parts[:id_index]).strip()
                            model_hash = parts[id_index]
                            
                            # Size is usually 2 parts (e.g., "4.7 GB")
                            size = parts[id_index+1]
                            if id_index+2 < len(parts) and parts[id_index+2] == 'GB':
                                size += ' ' + parts[id_index+2]
                                modified_start = id_index + 3
                            else:
                                modified_start = id_index + 2
                            
                            # Everything after size is the modified time
                            modified = ' '.join(parts[modified_start:]).strip()
                            
                            # Create a complete model object
                            model_obj = {
                                'id': model_id,  # This is what the frontend uses to identify models
                                'hash': model_hash,
                                'size': size,
                                'modified': modified
                            }
                            models.append(model_obj)
            
            # Log the models we found
            print(f"Found {len(models)} models with details: {models}")
            
            # If no models found but command succeeded, return empty array with success
            if len(models) == 0:
                print("No models found, but command succeeded")
            
            return jsonify({
                'status': 'success',
                'models': models
            })
        else:
            error_msg = result.stderr or "Unknown error"
            print(f"Ollama list command failed with error: {error_msg}")
            
            # Check if Ollama is not installed
            if "not found" in error_msg.lower() or "not recognized" in error_msg.lower():
                return jsonify({
                    'status': 'error',
                    'message': 'Ollama is not installed. Please install Ollama from https://ollama.ai',
                    'models': []
                })
            # Check if Ollama is not running
            elif "connection" in error_msg.lower() or "refused" in error_msg.lower():
                return jsonify({
                    'status': 'error',
                    'message': 'Ollama is not running. Please start Ollama first.',
                    'models': []
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Command failed: {error_msg}',
                    'models': []
                })
    except subprocess.TimeoutExpired:
        print("Ollama list command timed out")
        return jsonify({
            'status': 'error',
            'message': 'Ollama command timed out. Please check if Ollama is running.',
            'models': []
        })
    except FileNotFoundError:
        print("ERROR: Ollama command not found")
        return jsonify({
            'status': 'error',
            'message': 'Ollama is not installed or not in PATH. Please install Ollama from https://ollama.ai',
            'models': []
        })
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Check if it's a "command not found" error
        error_msg = str(e).lower()
        if "no such file" in error_msg or "not found" in error_msg:
            return jsonify({
                'status': 'error',
                'message': 'Ollama is not installed. Please install Ollama from https://ollama.ai',
                'models': []
            })
        
        return jsonify({
            'status': 'error', 
            'message': str(e),
            'models': []
        })

# Keep the direct-run endpoint for backward compatibility
@app.route('/api/ollama/direct-run', methods=['POST'])
def run_ollama_direct():
    # Redirect to the pull endpoint
    return run_ollama_pull_direct()

@app.route('/api/execute', methods=['POST'])
def execute_command():
    data = request.get_json()
    command = data.get('command')
    if not command:
        return jsonify({'error': 'No command specified'}), 400
    
    # Only allow specific safe commands
    allowed_commands = {
        'ollama list': ['ollama', 'list'],
        'ollama list -v': ['ollama', 'list', '-v'],
    }
    
    if command not in allowed_commands:
        return jsonify({'error': 'Command not allowed'}), 403
    
    try:
        print(f"Executing command: {allowed_commands[command]}")
        result = subprocess.run(allowed_commands[command], capture_output=True, text=True)
        print(f"Command output: {result.stdout}")
        print(f"Command error: {result.stderr}")
        print(f"Command return code: {result.returncode}")
        return jsonify({
            'output': result.stdout,
            'error': result.stderr,
            'returncode': result.returncode
        })
    except Exception as e:
        print(f"Command execution error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify server is running"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'timestamp': str(datetime.datetime.now())
    })

@app.route('/api/system/resources', methods=['GET'])
def get_system_resources():
    """Get system resources like CPU, GPU, and memory usage"""
    try:
        # Mock data for demo purposes
        # In a real implementation, you would use libraries like psutil, GPUtil, etc.
        return jsonify({
            'gpuUtilization': random.randint(20, 80),
            'cpuUsage': random.randint(10, 90),
            'temperature': random.randint(50, 85),
            'gpuMemory': {
                'used': round(random.uniform(2, 10), 1),
                'total': 16
            },
            'ramMemory': {
                'used': round(random.uniform(8, 24), 1),
                'total': 32
            }
        })
    except Exception as e:
        print(f"Error getting system resources: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ollama/chat', methods=['POST'])
def chat_with_model():
    """Chat with an Ollama model"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        model = data.get('model')
        messages = data.get('messages', [])
        
        if not model:
            return jsonify({'error': 'No model specified'}), 400
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
            
        print(f"Chat request for model {model} with {len(messages)} messages")
        
        # Format messages for Ollama API
        formatted_messages = []
        for msg in messages:
            if 'role' in msg and 'content' in msg:
                formatted_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        # Call Ollama API (assumes Ollama is running locally on default port)
        ollama_url = "http://localhost:11434/api/chat"
        payload = {
            "model": model,
            "messages": formatted_messages,
            "stream": False  # We don't want streaming for this implementation
        }
        
        response = requests.post(ollama_url, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            return jsonify({
                'status': 'success',
                'response': response_data.get('message', {}).get('content', ''),
                'model': model
            })
        else:
            error_message = f"Ollama API error: {response.status_code} - {response.text}"
            print(error_message)
            return jsonify({
                'status': 'error',
                'error': error_message
            }), 500
    except requests.exceptions.ConnectionError:
        error_message = "Could not connect to Ollama API. Make sure Ollama is running."
        print(error_message)
        return jsonify({
            'status': 'error',
            'error': error_message
        }), 503
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/ollama/delete', methods=['POST'])
def delete_model():
    """Delete a model from Ollama"""
    try:
        data = request.json
        model_name = data.get('model')
        
        if not model_name:
            return jsonify({
                'status': 'error',
                'message': 'Model name is required'
            }), 400
        
        print(f"Attempting to delete model: {model_name}")
        
        # Execute ollama rm command to delete the model
        result = subprocess.run(
            ['ollama', 'rm', model_name],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            print(f"Successfully deleted model: {model_name}")
            return jsonify({
                'status': 'success',
                'message': f'Model {model_name} deleted successfully'
            })
        else:
            error_msg = result.stderr or result.stdout or 'Unknown error'
            print(f"Failed to delete model {model_name}: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to delete model: {error_msg}'
            }), 500
            
    except Exception as e:
        print(f"Error in delete_model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    import datetime
    import random
    print("Starting Ollama API server on port 5001...")
    app.run(port=5001, debug=True, host='0.0.0.0')
