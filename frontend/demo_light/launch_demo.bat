@echo off

echo Starting STORM Wiki SiliconFlow Demo...

if "%1"=="debug" (
    echo Debug mode enabled
    python run_siliconflow_demo.py --debug
) else (
    python run_siliconflow_demo.py
)

pause 