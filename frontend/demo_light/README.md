# STORM Wiki SiliconFlow User Interface

This is a user interface for `STORMWikiRunner` powered by SiliconFlow API and various search engines. It includes the following features:

1. Allowing user to create a new article through the "Create New Article" page.
2. Showing the intermediate steps of STORMWikiRunner in real-time when creating an article.
3. Displaying the written article and references side by side.
4. Allowing user to view previously created articles through the "My Articles" page.
5. Supporting Chinese topic input with automatic translation for better search results.
6. Configurable model parameters and search engines.

<p align="center">
  <img src="assets/create_article.jpg" style="width: 70%; height: auto;">
</p>

<p align="center">
  <img src="assets/article_display.jpg" style="width: 70%; height: auto;">
</p>

## Setup

1. Make sure you have installed `knowledge-storm` or set up the source code correctly.
2. Install additional packages required by the user interface:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up API keys in a `.streamlit/secrets.toml` file with the following structure:

   ```toml
   SILICONFLOW_API_KEY = "your-siliconflow-api-key"

   # Optional: Search engine API keys
   YDC_API_KEY = "your-you-api-key"
   BING_SEARCH_API_KEY = "your-bing-api-key"
   BRAVE_API_KEY = "your-brave-api-key"
   SERPER_API_KEY = "your-serper-api-key"
   TAVILY_API_KEY = "your-tavily-api-key"
   SEARXNG_API_KEY = "your-searxng-api-key"

   # Optional: HTTP proxy for some API calls
   HTTP_PROXY = "your-proxy-url"
   ```

4. Run the following command to start the user interface:
   ```bash
   streamlit run storm.py
   ```
   The user interface will create a `DEMO_WORKING_DIR` directory in the current directory to store the outputs.

## Features

### Multi-language Support

The application supports non-English topics (particularly Chinese) by automatically translating them to English for better search results, while preserving the original language in the final output.

### Configurable Search Engines

You can choose from the following search engines:

- DuckDuckGo (default, no API key required)
- Bing Search
- You.com
- Brave Search
- Serper API
- Tavily Search
- SearXNG

### Advanced Settings

In the "Create New Article" page, you can configure:

- Model parameters (temperature, top_p)
- Pipeline stages (research, outline generation, article generation, polishing)
- Search engine settings
- Research parameters (conversation turns, perspectives, etc.)

## Customization

You can customize the `STORMWikiRunner` powering the user interface according to [the guidelines](https://github.com/stanford-oval/storm?tab=readme-ov-file#customize-storm) in the main README file.

The `STORMWikiRunner` is initialized in `set_storm_runner()` in [demo_util.py](demo_util.py). You can change `STORMWikiRunnerArguments`, `STORMWikiLMConfigs`, or use a different retrieval model according to your need.
