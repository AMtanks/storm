#!/usr/bin/env python3
"""
STORM Wiki Streamlit Demo with SiliconFlow Backend

This script launches the STORM Wiki demo using SiliconFlow API as the LLM backend.

Usage:
    python run_siliconflow_demo.py
"""

import os
import sys
import argparse
import subprocess
import toml
import streamlit.web.bootstrap

# Ensure the parent directory is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def check_api_key():
    """Check if SiliconFlow API key is set in the environment or secrets.toml"""
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    
    if api_key:
        return True
    
    # Check in secrets.toml
    secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            secrets = toml.load(secrets_path)
            if secrets.get("SILICONFLOW_API_KEY"):
                return True
        except Exception as e:
            print(f"Error reading secrets file: {e}")
    
    return False

def setup_environment():
    """Set up the environment for the Streamlit app"""
    # Create the output directory
    output_dir = os.path.join(os.path.dirname(__file__), "DEMO_WORKING_DIR")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for required API keys
    if not check_api_key():
        print("\n" + "="*80)
        print("WARNING: SiliconFlow API key not found!")
        print("Please set SILICONFLOW_API_KEY in your environment or in .streamlit/secrets.toml")
        print("="*80 + "\n")
        
        # Create a template secrets file if it doesn't exist
        secrets_dir = os.path.join(os.path.dirname(__file__), ".streamlit")
        template_path = os.path.join(secrets_dir, "secrets_template.toml")
        secrets_path = os.path.join(secrets_dir, "secrets.toml")
        
        if os.path.exists(template_path) and not os.path.exists(secrets_path):
            os.makedirs(secrets_dir, exist_ok=True)
            with open(template_path, "r") as src, open(secrets_path, "w") as dst:
                dst.write(src.read())
            print(f"Created template secrets file at {secrets_path}")
            print("Please edit this file with your API keys before proceeding.\n")

def main():
    parser = argparse.ArgumentParser(description="Run STORM Wiki Streamlit Demo with SiliconFlow")
    parser.add_argument("--server-port", type=int, default=8501, help="Port to run Streamlit server on")
    parser.add_argument("--server-address", type=str, default="localhost", help="Address to run Streamlit server on")
    args = parser.parse_args()
    
    # Set up the environment
    setup_environment()
    
    # Launch the Streamlit app
    streamlit_script = os.path.join(os.path.dirname(__file__), "storm.py")
    sys.argv = [
        "streamlit", "run",
        streamlit_script,
        "--server.port", str(args.server_port),
        "--server.address", args.server_address,
    ]
    
    print(f"Starting STORM Wiki Streamlit Demo on http://{args.server_address}:{args.server_port}")
    streamlit.web.bootstrap.run()

if __name__ == "__main__":
    main() 