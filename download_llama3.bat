@echo off
echo Starting direct download of Llama 3.2 1B model...
echo.
echo This will download the model using the direct ollama pull command.
echo.

REM Run the ollama pull command
ollama pull meta-llama/llama3:1b

echo.
echo Download process completed.
echo Check 'ollama list' to verify the model is installed.
echo.
pause
