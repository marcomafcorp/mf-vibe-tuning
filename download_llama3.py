# Direct Llama 3.2 1B downloader script
import subprocess
import sys
import os

def download_llama3():
    print("Starting direct download of Llama 3.2 1B model...")
    
    # The command to run - this will download if not present
    cmd = "ollama run meta-llama/llama3:1b"
    
    try:
        # Run the command directly
        process = subprocess.Popen(
            cmd, 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("Download process started. This will take some time.")
        print("You can see the progress in this terminal window.")
        print("Once complete, the model will appear in your 'ollama list' output.")
        print("\nPress Ctrl+C when you want to stop the model after it downloads.\n")
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Get final status
        rc = process.poll()
        return rc == 0
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_llama3()
    if success:
        print("\nDownload completed successfully!")
    else:
        print("\nDownload encountered issues. Please check the logs above.")
