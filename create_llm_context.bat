@echo off
REM LLM Context Consolidator v2.0.0
REM Consolidates all context files into a single timestamped text file for easy sharing

echo === OpenVINO GenAI Context Consolidation ===
echo.

cd /d "%~dp0"

REM Check if context folder exists
if not exist "context" (
    echo ❌ Context folder not found! Run this script from the project root.
    echo Expected location: %CD%\context\
    pause
    exit /b 1
)

REM Generate timestamp using PowerShell for wider compatibility
for /f "tokens=*" %%a in ('powershell -command "Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'"') do set "timestamp=%%a"

set "OUTPUT_FILE=context\OpenVINO_GenAI_Context_%timestamp%.txt"

echo Creating consolidated context file: %OUTPUT_FILE%
echo.

REM Start building the consolidated file
(
echo ================================================================================
echo OpenVINO GenAI Context Consolidation
echo Generated: %date% %time%
echo Total Files: 16 curated essential files
echo ================================================================================
echo.
echo This file contains all essential OpenVINO GenAI reference files needed
echo for building robust, performance-optimized Gradio chat applications.
echo.
echo Directory Structure:
echo   python_samples/     ^(4 files^) - Python API examples
echo   test_configs/       ^(3 files^) - Configuration patterns
echo   core_cpp/          ^(3 files^) - C++ implementation details  
echo   documentation/     ^(2 files^) - Architecture guides
echo   python_bindings/   ^(3 files^) - C++ to Python bindings
echo   README.md          ^(1 file^)  - Usage guide
echo.
echo ================================================================================
echo.
) > "%OUTPUT_FILE%"

echo 📁 Processing context files...

REM Process each directory
for %%d in (python_samples test_configs core_cpp documentation python_bindings) do (
    if exist "context\%%d" (
        echo 📂 Processing %%d directory...
        (
        echo.
        echo ################################################################################
        echo # %%d DIRECTORY
        echo ################################################################################
        echo.
        ) >> "%OUTPUT_FILE%"
        
        REM Process files in this directory
        for %%f in ("context\%%d\*.*") do (
            echo   📄 Adding %%~nxf...
            (
            echo.
            echo --- FILE: %%~nxf ---
            echo Path: %%d/%%~nxf
            echo.
            ) >> "%OUTPUT_FILE%"
            
            REM Add file content with appropriate syntax highlighting
            if "%%~xf"==".py" (
                echo ```python >> "%OUTPUT_FILE%"
            ) else if "%%~xf"==".cpp" (
                echo ```cpp >> "%OUTPUT_FILE%"
            ) else if "%%~xf"==".md" (
                echo ```markdown >> "%OUTPUT_FILE%"
            ) else (
                echo ``` >> "%OUTPUT_FILE%"
            )
            
            type "%%f" >> "%OUTPUT_FILE%"
            
            (
            echo.
            echo ```
            echo.
            echo ================================================================================
            echo.
            ) >> "%OUTPUT_FILE%"
        )
    )
)

REM Add README.md at the end
if exist "context\README.md" (
    echo 📄 Adding README.md...
    (
    echo.
    echo ################################################################################
    echo # CONTEXT README - USAGE GUIDE
    echo ################################################################################
    echo.
    echo ```markdown
    ) >> "%OUTPUT_FILE%"
    
    type "context\README.md" >> "%OUTPUT_FILE%"
    
    (
    echo ```
    echo.
    echo ================================================================================
    echo # END OF CONTEXT CONSOLIDATION
    echo ================================================================================
    ) >> "%OUTPUT_FILE%"
)

echo.
echo === Consolidation Complete! ===
echo.
echo ✅ Successfully consolidated all context files
echo 📁 Output file: %OUTPUT_FILE%

if exist "%OUTPUT_FILE%" (
    for %%F in ("%OUTPUT_FILE%") do (
        set /a size_mb=%%~zF/1024/1024
        echo 📊 File size: %%~zF bytes ^(~!size_mb! MB^)
    )
    
    echo.
    echo 🎯 This consolidated file contains:
    echo    • All 16 essential OpenVINO GenAI files
    echo    • Complete API reference and examples  
    echo    • C++ implementation details
    echo    • Architecture documentation
    echo    • Usage guides and best practices
    echo.
    echo 📤 Ready for sharing with LLMs or team members!
) else (
    echo ❌ Error: Output file was not created
)

echo.
pause
