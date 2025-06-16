# STORM Wiki SiliconFlow Demo

This demo provides a Streamlit-based user interface for STORM Wiki, powered by the SiliconFlow API (using Qwen2.5-72B-Instruct) and various search engines.

## Features

- Generate informative and well-structured articles on any topic
- Support for multiple search engines (DuckDuckGo, Bing, You.com, etc.)
- Multi-language support with automatic translation of Chinese topics
- Configurable model parameters and search settings
- Interactive UI showing the research and writing process

## Quick Start

### Windows

1. Make sure you've installed all the requirements:
   ```
   pip install -r requirements.txt
   ```
2. Set up your API keys in `.streamlit/secrets.toml` (or copy from the template)
3. Run the demo by double-clicking `launch_demo.bat` or from command prompt:
   ```
   python run_siliconflow_demo.py
   ```

### Linux/Mac

1. Make sure you've installed all the requirements:
   ```
   pip install -r requirements.txt
   ```
2. Set up your API keys in `.streamlit/secrets.toml` (or copy from the template)
3. Run the demo:
   ```
   ./launch_demo.sh
   ```
   or
   ```
   python3 run_siliconflow_demo.py
   ```

## Configuration

### API Keys

You need to set up the following API keys in `.streamlit/secrets.toml`:

- `SILICONFLOW_API_KEY` (required): Your SiliconFlow API key
- Optional search engine API keys depending on which search engine you want to use

### Advanced Settings

In the "Create New Article" page, you can configure:

- Model parameters (temperature, top_p)
- Pipeline stages (research, outline, article generation, polishing)
- Search engine selection
- Research parameters (conversation turns, perspectives, etc.)

## Troubleshooting

### API Key Issues

If you see errors related to API keys:

1. Make sure you've set up your SiliconFlow API key in `.streamlit/secrets.toml`
2. For search engines other than DuckDuckGo, make sure you've provided the corresponding API key

### Search Engine Issues

If you encounter errors with search engines:

1. Try switching to a different search engine
2. For DuckDuckGo, you can adjust the request delay and retries in the advanced settings

### Performance Tips

- Reduce the "Max Thread Number" if you're experiencing rate limiting
- For longer articles, increase the "Max Perspectives" and "Max Conversation Turns"
- Use the "Remove Duplicate Content" option to improve article quality

## How It Works

The STORM Wiki pipeline has several stages:

1. **Research**: The system explores the topic from different perspectives
2. **Outline Generation**: Creates a structured outline based on research
3. **Article Generation**: Writes a comprehensive article following the outline
4. **Polishing**: Improves the article's structure, readability, and removes duplicates

For Chinese topics, the system first translates them to English for better search results, while preserving the original language for display.

## Development

See the main README.md for more details on how to customize the STORM Wiki pipeline.
