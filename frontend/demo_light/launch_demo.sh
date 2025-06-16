#!/bin/bash
echo "Starting STORM Wiki SiliconFlow Demo..."

if [ "$1" = "debug" ]; then
    echo "Debug mode enabled"
    python3 run_siliconflow_demo.py --debug
else
    python3 run_siliconflow_demo.py
fi 