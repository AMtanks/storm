"""
STORM Wiki pipeline powered by Qwen2.5-72B-Instruct hosted by SiliconFlow API and You.com search engine.
You need to set up the following environment variables to run this script:
    - SILICONFLOW_API_KEY: Your SiliconFlow API Key (e.g., sk-lpmnkrmjecjryeruvsdrovodolqnjhsiohbjwgbinqllkomy)
    - YDC_API_KEY: You.com API key; BING_SEARCH_API_KEY: Bing Search API key, SERPER_API_KEY: Serper API key, BRAVE_API_KEY: Brave API key, or TAVILY_API_KEY: Tavily API key
You also need to have a VLLM server running with the Mistral-7B-Instruct-v0.2 model. Specify `--url` and `--port` accordingly.

Output will be structured as below
args.output_dir/
    topic_name/  # topic_name will follow convention of underscore-connected topic name w/o space and slash
        conversation_log.json           # Log of information-seeking conversation
        raw_search_results.json         # Raw search results from search engine
        direct_gen_outline.txt          # Outline directly generated with LLM's parametric knowledge
        storm_gen_outline.txt           # Outline refined with collected information
        url_to_info.json                # Sources that are used in the final article
        storm_gen_article.txt           # Final article generated
        storm_gen_article_polished.txt  # Polished final article (if args.do_polish_article is True)
"""

import os
import re
from argparse import ArgumentParser
import logging

from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)

# Now import lm directly
from knowledge_storm.lm import SiliconFlowModel
from knowledge_storm.rm import (
    YouRM,
    BingSearch,
    BraveRM,
    SerperRM,
    DuckDuckGoSearchRM,
    TavilySearchRM,
    SearXNG,
)
from knowledge_storm.utils import load_api_key

# 添加logger定义
logger = logging.getLogger(__name__)

def translate_topic_to_english(model, topic):
    """
    将输入的主题翻译成英文，如原本就是英文，则直接返回
    """
    # 检测主题是否包含中文字符
    if any('\u4e00' <= char <= '\u9fff' for char in topic):
        logger.info("正在调用翻译模型...")
        prompt = f"将输入的主题翻译成英文，如原本就是英文，则直接返回，只需要返回翻译结果，不要添加任何其他内容：\n\n{topic}"
        
        try:
            # 使用 __call__ 方法而不是 generate
            response = model(prompt)
            if isinstance(response, list) and len(response) > 0:
                english_topic = response[0].strip()
            else:
                english_topic = str(response).strip()
                
            logger.info(f"原始主题: {topic}")
            logger.info(f"英文主题: {english_topic}")
            return english_topic, True  # 返回英文主题和一个标志表示这是翻译的
        except Exception as e:
            logger.error(f"翻译主题时出错: {str(e)}")
            return topic, False  # 如果翻译失败，返回原始主题
    
    return topic, False  # 如果不是中文，直接返回原始主题

def sanitize_topic(topic):
    """
    Sanitize the topic name for use in file names.
    Remove or replace characters that are not allowed in file names.
    """
    # Replace spaces with underscores
    topic = topic.replace(" ", "_")

    # Remove any character that isn't alphanumeric, underscore, or hyphen
    topic = re.sub(r"[^a-zA-Z0-9_-]", "", topic)

    # Ensure the topic isn't empty after sanitization
    if not topic:
        topic = "unnamed_topic"

    return topic


def main(args):
    load_api_key(toml_file_path="secrets.toml")
    lm_configs = STORMWikiLMConfigs()

    # Use default API key if environment variable is not set
    api_key = os.getenv("SILICONFLOW_API_KEY") or "sk-lpmnkrmjecjryeruvsdrovodolqnjhsiohbjwgbinqllkomy"
    
    # 设置Tavily API密钥
    os.environ["TAVILY_API_KEY"] = "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"

    siliconflow_kwargs = {
        "api_key": api_key,
        "api_base": "https://api.siliconflow.cn/v1",
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    # SiliconFlow offers models like "Qwen/Qwen2.5-72B-Instruct"
    model_name = "Qwen/Qwen2.5-72B-Instruct"
    
    conv_simulator_lm = SiliconFlowModel(
        model=model_name, max_tokens=500, **siliconflow_kwargs
    )
    question_asker_lm = SiliconFlowModel(
        model=model_name, max_tokens=500, **siliconflow_kwargs
    )
    outline_gen_lm = SiliconFlowModel(model=model_name, max_tokens=400, **siliconflow_kwargs)
    article_gen_lm = SiliconFlowModel(model=model_name, max_tokens=700, **siliconflow_kwargs)
    article_polish_lm = SiliconFlowModel(
        model=model_name, max_tokens=4000, **siliconflow_kwargs
    )
    
    # 创建一个额外的模型实例用于翻译
    translation_lm = SiliconFlowModel(
        model=model_name, max_tokens=100, **siliconflow_kwargs
    )

    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    # STORM is a knowledge curation system which consumes information from the retrieval module.
    # Currently, the information source is the Internet and we use search engine API as the retrieval module.
    match args.retriever:
        case "bing":
            rm = BingSearch(
                bing_search_api=os.getenv("BING_SEARCH_API_KEY"),
                k=engine_args.search_top_k,
            )
        case "you":
            rm = YouRM(ydc_api_key=os.getenv("YDC_API_KEY"), k=engine_args.search_top_k)
        case "brave":
            rm = BraveRM(
                brave_search_api_key=os.getenv("BRAVE_API_KEY"),
                k=engine_args.search_top_k,
            )
        case "duckduckgo":
            rm = DuckDuckGoSearchRM(
                k=engine_args.search_top_k, 
                safe_search="On", 
                region="us-en",
                request_delay=1.0,  # 基础请求延迟3秒
                max_retries=99,  # 最大重试次数99
                use_multiple_backends=True,  # 启用多后端轮换
                exponential_backoff=True,  # 启用指数退避
                max_delay=5.0,  # 最大延迟5秒
                webpage_helper_max_threads=1  # 减少并发线程数
            )
        case "serper":
            rm = SerperRM(
                serper_search_api_key=os.getenv("SERPER_API_KEY"),
                query_params={"autocorrect": True, "num": 10, "page": 1},
            )
        case "tavily":
            # 设置代理（如果有）
            proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
            
            rm = TavilySearchRM(
                tavily_search_api_key=os.getenv("TAVILY_API_KEY"),
                k=engine_args.search_top_k,
                include_raw_content=True,
                proxy=proxy,
                exclude_domains=["wikipedia.org", "en.wikipedia.org"],
            )
        case "searxng":
            rm = SearXNG(
                searxng_api_key=os.getenv("SEARXNG_API_KEY"), k=engine_args.search_top_k
            )
        case _:
            raise ValueError(
                f'Invalid retriever: {args.retriever}. Choose either "bing", "you", "brave", "duckduckgo", "serper", "tavily", or "searxng"'
            )

    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    original_topic = input("Topic: ")
    
    # 如果输入的是中文，翻译为英文用于搜索
    english_topic, is_translated = translate_topic_to_english(translation_lm, original_topic)
    
    # 使用英文主题进行搜索和处理
    search_topic = english_topic if is_translated else original_topic
    
    # 但使用原始主题（可能是中文）作为文件名
    sanitized_topic = sanitize_topic(original_topic)

    try:
        # 如果是翻译的主题，保存原始中文主题和英文主题的映射关系
        if is_translated:
            # 创建输出目录
            output_dir = os.path.join(args.output_dir, sanitized_topic)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存主题映射关系
            with open(os.path.join(output_dir, "topic_translation.txt"), "w", encoding="utf-8") as f:
                f.write(f"原始主题: {original_topic}\n")
                f.write(f"英文主题: {english_topic}\n")
        
        runner.run(
            topic=search_topic,  # 使用英文主题进行搜索
            do_research=args.do_research,
            do_generate_outline=args.do_generate_outline,
            do_generate_article=args.do_generate_article,
            do_polish_article=args.do_polish_article,
            remove_duplicate=args.remove_duplicate,
            original_topic=original_topic,  # 传入原始主题（可能是中文）
        )
        runner.post_run()
        runner.summary()
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser()
    # global arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/siliconflow",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--max-thread-num",
        type=int,
        default=3,
        help="最大线程数。信息搜索部分和文章生成部分可以通过使用多个线程来加速。如果调用LM API时持续遇到"
        '"超出速率限制"错误，请考虑减少线程数。',
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bing", "you", "brave", "serper", "duckduckgo", "tavily", "searxng"],
        help="The search engine API to use for retrieving information.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature to use."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter."
    )
    # stage of the pipeline
    parser.add_argument(
        "--do-research",
        action="store_true",
        help="If True, simulate conversation to research the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-generate-outline",
        action="store_true",
        help="If True, generate an outline for the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-generate-article",
        action="store_true",
        help="If True, generate an article for the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-polish-article",
        action="store_true",
        help="If True, polish the article by adding a summarization section and (optionally) removing "
        "duplicate content.",
    )
    # hyperparameters for the pre-writing stage
    parser.add_argument(
        "--max-conv-turn",
        type=int,
        default=3,
        help="Maximum number of questions in conversational question asking.",
    )
    parser.add_argument(
        "--max-perspective",
        type=int,
        default=3,
        help="Maximum number of perspectives to consider in perspective-guided question asking.",
    )
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=3,
        help="Top k search results to consider for each search query.",
    )
    # hyperparameters for the writing stage
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=3,
        help="Top k collected references for each section title.",
    )
    parser.add_argument(
        "--remove-duplicate",
        action="store_true",
        help="If True, remove duplicate content from the article.",
    )

    main(parser.parse_args())
