# Using SiliconFlow API with STORM Wiki

This example demonstrates how to use SiliconFlow's API to power the STORM Wiki knowledge curation system.

## Setup

1. **API Key**: The script uses a default API key, but you can set your own by:

   - Setting the `SILICONFLOW_API_KEY` environment variable
   - Adding it to a `secrets.toml` file in the root directory with the format:
     ```toml
     SILICONFLOW_API_KEY = "your-api-key-here"
     ```

2. **Search Engine**: You'll need at least one search engine API key. The script supports:
   - You.com (`YDC_API_KEY`)
   - Bing Search (`BING_SEARCH_API_KEY`)
   - Brave Search (`BRAVE_API_KEY`)
   - Serper (`SERPER_API_KEY`)
   - DuckDuckGo (no API key required)
   - Tavily (`TAVILY_API_KEY`)
   - SearXNG (`SEARXNG_API_KEY`)

## Running the Script

```bash
python examples/storm_examples/run_storm_wiki_siliconflow.py --retriever you --do-research --do-generate-outline --do-generate-article --do-polish-article
```

When prompted, enter the topic you want to research.

## Available Models

The default model is `Qwen/Qwen2.5-72B-Instruct`. SiliconFlow offers various models that you can use by modifying the `model_name` variable in the script.

## Output

The generated content will be saved in the `./results/siliconflow/` directory (or the directory specified with `--output-dir`).

## Parameters

- `--temperature`: Controls randomness (default: 1.0)
- `--top_p`: Controls diversity (default: 0.9)
- `--max-conv-turn`: Maximum conversation turns (default: 3)
- `--max-perspective`: Maximum perspectives to consider (default: 3)
- `--search-top-k`: Top search results to use (default: 3)
- `--retrieve-top-k`: Top references for each section (default: 3)
- `--remove-duplicate`: Remove duplicate content (flag)

For more details, run:

```bash
python examples/storm_examples/run_storm_wiki_siliconflow.py --help
```
