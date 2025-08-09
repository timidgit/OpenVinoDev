@echo off
REM OpenVINO GenAI Project Consolidation Script
REM Usage: consolidate.bat [demo|samples|full|docs]
REM 
REM Options:
REM   demo    - Consolidate demo scripts and configs only
REM   samples - Include official OpenVINO GenAI samples  
REM   full    - Complete codebase including source code
REM   docs    - Documentation and README files only

cd /d "%~dp0"

set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

if "%1"=="docs" (
    echo === OpenVINO GenAI Documentation Consolidation ===
    set OUTPUT_FILE=OpenVINO_GenAI_Docs_%TIMESTAMP%.txt
    goto :consolidate_docs
) else if "%1"=="samples" (
    echo === OpenVINO GenAI Samples Consolidation ===
    set OUTPUT_FILE=OpenVINO_GenAI_Samples_%TIMESTAMP%.txt
    goto :consolidate_samples
) else if "%1"=="full" (
    echo === OpenVINO GenAI Full Consolidation ===
    set OUTPUT_FILE=OpenVINO_GenAI_Full_%TIMESTAMP%.txt
    goto :consolidate_full
) else (
    echo === OpenVINO GenAI Demo Consolidation ===
    set OUTPUT_FILE=OpenVINO_GenAI_Demo_%TIMESTAMP%.txt
    goto :consolidate_demo
)

:consolidate_demo
echo Consolidating demo applications and configurations...
echo # OpenVINO GenAI Demo Consolidation > "%OUTPUT_FILE%"
echo # Generated: %date% %time% >> "%OUTPUT_FILE%"
echo # >> "%OUTPUT_FILE%"
echo # This consolidation includes the main demo scripts and configuration files >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

echo ## Project Overview >> "%OUTPUT_FILE%"
if exist CLAUDE.md (
    echo ### CLAUDE.md >> "%OUTPUT_FILE%"
    type CLAUDE.md >> "%OUTPUT_FILE%"
    echo. >> "%OUTPUT_FILE%"
)

echo ## Demo Scripts >> "%OUTPUT_FILE%"
for %%f in (gradio_qwen_debug.py gradio_qwen_optimized.py) do (
    if exist %%f (
        echo ### %%f >> "%OUTPUT_FILE%"
        echo ```python >> "%OUTPUT_FILE%"
        type %%f >> "%OUTPUT_FILE%"
        echo ``` >> "%OUTPUT_FILE%"
        echo. >> "%OUTPUT_FILE%"
    )
)

echo ## Utility Scripts >> "%OUTPUT_FILE%"
for %%f in (export_qwen_for_npu.py check_model_config.py) do (
    if exist %%f (
        echo ### %%f >> "%OUTPUT_FILE%"
        echo ```python >> "%OUTPUT_FILE%"
        type %%f >> "%OUTPUT_FILE%"
        echo ``` >> "%OUTPUT_FILE%"
        echo. >> "%OUTPUT_FILE%"
    )
)

goto :finish

:consolidate_samples
echo Consolidating OpenVINO GenAI samples...
echo # OpenVINO GenAI Samples Consolidation > "%OUTPUT_FILE%"
echo # Generated: %date% %time% >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

echo ## Python Samples >> "%OUTPUT_FILE%"
if exist "openvino.genai-master\samples\python" (
    for /r "openvino.genai-master\samples\python" %%f in (*.py) do (
        echo ### %%~nxf >> "%OUTPUT_FILE%"
        echo Path: %%f >> "%OUTPUT_FILE%"
        echo ```python >> "%OUTPUT_FILE%"
        type "%%f" >> "%OUTPUT_FILE%"
        echo ``` >> "%OUTPUT_FILE%"
        echo. >> "%OUTPUT_FILE%"
    )
)

echo ## C++ Samples >> "%OUTPUT_FILE%"
if exist "openvino.genai-master\samples\cpp" (
    for /r "openvino.genai-master\samples\cpp" %%f in (*.cpp *.h *.hpp) do (
        echo ### %%~nxf >> "%OUTPUT_FILE%"
        echo Path: %%f >> "%OUTPUT_FILE%"
        echo ```cpp >> "%OUTPUT_FILE%"
        type "%%f" >> "%OUTPUT_FILE%"
        echo ``` >> "%OUTPUT_FILE%"
        echo. >> "%OUTPUT_FILE%"
    )
)

goto :finish

:consolidate_full
echo Consolidating full OpenVINO GenAI codebase...
echo # OpenVINO GenAI Full Consolidation > "%OUTPUT_FILE%"
echo # Generated: %date% %time% >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

REM Include everything from demo consolidation
call :consolidate_demo_content

echo ## OpenVINO GenAI Source Code >> "%OUTPUT_FILE%"
if exist "openvino.genai-master\src" (
    echo ### Python Bindings >> "%OUTPUT_FILE%"
    for /r "openvino.genai-master\src\python" %%f in (*.cpp *.hpp *.py) do (
        echo #### %%~nxf >> "%OUTPUT_FILE%"
        echo ```cpp >> "%OUTPUT_FILE%"
        type "%%f" >> "%OUTPUT_FILE%"
        echo ``` >> "%OUTPUT_FILE%"
        echo. >> "%OUTPUT_FILE%"
    )
    
    echo ### C++ Core Implementation >> "%OUTPUT_FILE%"
    for /r "openvino.genai-master\src\cpp" %%f in (*.cpp *.hpp) do (
        echo #### %%~nxf >> "%OUTPUT_FILE%"
        echo ```cpp >> "%OUTPUT_FILE%"
        type "%%f" >> "%OUTPUT_FILE%"
        echo ``` >> "%OUTPUT_FILE%"
        echo. >> "%OUTPUT_FILE%"
    )
)

goto :finish

:consolidate_docs
echo Consolidating documentation...
echo # OpenVINO GenAI Documentation > "%OUTPUT_FILE%"
echo # Generated: %date% %time% >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

for %%f in (*.md *.txt) do (
    if exist %%f (
        echo ## %%f >> "%OUTPUT_FILE%"
        type %%f >> "%OUTPUT_FILE%"
        echo. >> "%OUTPUT_FILE%"
    )
)

if exist "openvino.genai-master" (
    for /r "openvino.genai-master" %%f in (README.md *.md) do (
        echo ## %%~nxf (%%~dpf) >> "%OUTPUT_FILE%"
        type "%%f" >> "%OUTPUT_FILE%"
        echo. >> "%OUTPUT_FILE%"
    )
)

goto :finish

:consolidate_demo_content
echo ## Demo Applications >> "%OUTPUT_FILE%"
for %%f in (gradio_qwen_debug.py gradio_qwen_optimized.py export_qwen_for_npu.py check_model_config.py) do (
    if exist %%f (
        echo ### %%f >> "%OUTPUT_FILE%"
        echo ```python >> "%OUTPUT_FILE%"
        type %%f >> "%OUTPUT_FILE%"
        echo ``` >> "%OUTPUT_FILE%"
        echo. >> "%OUTPUT_FILE%"
    )
)

if exist CLAUDE.md (
    echo ### CLAUDE.md >> "%OUTPUT_FILE%"
    type CLAUDE.md >> "%OUTPUT_FILE%"
    echo. >> "%OUTPUT_FILE%"
)
goto :eof

:finish
echo.
echo ✅ Consolidation complete!
echo 📁 Output file: %OUTPUT_FILE%
if exist "%OUTPUT_FILE%" (
    for %%F in ("%OUTPUT_FILE%") do (
        echo 📊 File size: %%~zF bytes
    )
) else (
    echo ❌ Output file not found
)
echo.
echo Usage examples:
echo   consolidate.bat       - Demo scripts only
echo   consolidate.bat samples - Include OpenVINO samples  
echo   consolidate.bat full    - Complete codebase
echo   consolidate.bat docs    - Documentation only
echo.
pause