#!/usr/bin/env python
"""
测试DuckDuckGo搜索引擎的速率限制
这个脚本会使用不同的延迟参数测试DuckDuckGo搜索，找出最优的请求频率
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

# 添加开始时间记录
start_time = time.time()
logger = logging.getLogger(__name__)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger.info(f"脚本启动，开始导入模块... 耗时: {time.time() - start_time:.2f}秒")

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
logger.info(f"添加项目根目录到路径... 耗时: {time.time() - start_time:.2f}秒")

# 导入前记录时间
import_start = time.time()
from knowledge_storm.rm import DuckDuckGoSearchRM
logger.info(f"导入DuckDuckGoSearchRM模块完成... 耗时: {time.time() - import_start:.2f}秒")

logger.info(f"初始化完成，总耗时: {time.time() - start_time:.2f}秒")

# 测试查询列表
TEST_QUERIES = [
    "climate change solutions",
    "artificial intelligence ethics",
    "renewable energy technologies",
    "space exploration missions",
    "quantum computing applications",
    "sustainable agriculture practices",
    "blockchain technology use cases",
    "machine learning algorithms",
    "cybersecurity best practices",
    "virtual reality development"
]

def test_single_delay(delay: float, queries: List[str], max_results: int = 3) -> tuple:
    """
    使用指定的延迟测试DuckDuckGo搜索
    
    Args:
        delay: 请求之间的延迟（秒）
        queries: 要搜索的查询列表
        max_results: 每次搜索返回的最大结果数
        
    Returns:
        tuple: (成功查询数, 总查询数, 总耗时)
    """
    logger.info(f"测试延迟: {delay}秒")
    
    # 记录初始化搜索引擎的时间
    init_start = time.time()
    logger.info("开始初始化DuckDuckGoSearchRM...")
    rm = DuckDuckGoSearchRM(k=max_results, request_delay=delay)
    logger.info(f"DuckDuckGoSearchRM初始化完成，耗时: {time.time() - init_start:.2f}秒")
    
    start_time = time.time()
    success_count = 0
    total_count = len(queries)
    
    for i, query in enumerate(queries):
        query_start = time.time()
        try:
            logger.info(f"查询 {i+1}/{total_count}: '{query}'")
            logger.info(f"开始执行forward方法...")
            results = rm.forward(query)
            logger.info(f"forward方法执行完成，耗时: {time.time() - query_start:.2f}秒")
            
            # 检查结果
            if results and len(results) > 0:
                logger.info(f"成功: 找到 {len(results)} 个结果")
                success_count += 1
            else:
                logger.warning(f"警告: 查询成功但没有结果")
                
        except Exception as e:
            logger.error(f"错误: {str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"测试完成 - 延迟: {delay}秒, 成功率: {success_count}/{total_count}, 总耗时: {total_time:.2f}秒")
    return success_count, total_count, total_time

def test_concurrent_requests(delay: float, queries: List[str], max_results: int = 10, max_workers: int = 5) -> tuple:
    """
    使用指定的延迟和并发数测试DuckDuckGo搜索
    
    Args:
        delay: 请求之间的延迟（秒）
        queries: 要搜索的查询列表
        max_results: 每次搜索返回的最大结果数
        max_workers: 最大并发工作线程数
        
    Returns:
        tuple: (成功查询数, 总查询数, 总耗时)
    """
    logger.info(f"测试并发请求 - 延迟: {delay}秒, 最大工作线程: {max_workers}")
    
    # 记录初始化搜索引擎的时间
    init_start = time.time()
    logger.info("开始初始化DuckDuckGoSearchRM...")
    rm = DuckDuckGoSearchRM(k=max_results, request_delay=delay)
    logger.info(f"DuckDuckGoSearchRM初始化完成，耗时: {time.time() - init_start:.2f}秒")
    
    start_time = time.time()
    success_count = 0
    total_count = len(queries)
    
    def process_query(query):
        query_start = time.time()
        try:
            logger.info(f"查询: '{query}'")
            results = rm.forward(query)
            logger.info(f"查询 '{query}' 完成，耗时: {time.time() - query_start:.2f}秒")
            
            # 检查结果
            if results and len(results) > 0:
                logger.info(f"成功: 找到 {len(results)} 个结果")
                return True
            else:
                logger.warning(f"警告: 查询成功但没有结果")
                return False
                
        except Exception as e:
            logger.error(f"错误: {str(e)}")
            return False
    
    # 使用线程池执行并发请求
    logger.info(f"创建线程池，最大工作线程: {max_workers}")
    pool_start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        logger.info(f"线程池创建完成，耗时: {time.time() - pool_start:.2f}秒")
        results = list(executor.map(process_query, queries))
    
    success_count = sum(1 for r in results if r)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"并发测试完成 - 延迟: {delay}秒, 工作线程: {max_workers}, 成功率: {success_count}/{total_count}, 总耗时: {total_time:.2f}秒")
    return success_count, total_count, total_time

def main():
    main_start = time.time()
    logger.info("开始解析命令行参数...")
    
    parser = argparse.ArgumentParser(description="测试DuckDuckGo搜索引擎的速率限制")
    parser.add_argument("--mode", choices=["sequential", "concurrent", "both"], default="both",
                      help="测试模式: sequential(顺序), concurrent(并发), both(两者)")
    parser.add_argument("--delays", type=str, default="1,2,3,5,7,10",
                      help="要测试的延迟值（秒），用逗号分隔")
    parser.add_argument("--workers", type=str, default="2,3,5",
                      help="要测试的并发工作线程数，用逗号分隔")
    parser.add_argument("--max-results", type=int, default=3,
                      help="每次搜索返回的最大结果数")
    parser.add_argument("--queries", type=int, default=5,
                      help="要使用的测试查询数量")
    
    args = parser.parse_args()
    logger.info(f"命令行参数解析完成，耗时: {time.time() - main_start:.2f}秒")
    
    # 解析延迟值和工作线程数
    delays = [float(d) for d in args.delays.split(",")]
    workers = [int(w) for w in args.workers.split(",")]
    
    # 选择查询
    selected_queries = TEST_QUERIES[:args.queries]
    logger.info(f"已选择 {len(selected_queries)} 个查询")
    
    # 运行测试
    if args.mode in ["sequential", "both"]:
        logger.info("开始顺序测试...")
        for delay in delays:
            test_single_delay(delay, selected_queries, args.max_results)
            # 在测试之间添加额外的延迟，避免触发速率限制
            logger.info(f"等待10秒后进行下一次测试...")
            time.sleep(10)
    
    if args.mode in ["concurrent", "both"]:
        logger.info("开始并发测试...")
        for delay in delays:
            for worker_count in workers:
                test_concurrent_requests(delay, selected_queries, args.max_results, worker_count)
                # 在测试之间添加额外的延迟，避免触发速率限制
                logger.info(f"等待20秒后进行下一次测试...")
                time.sleep(20)
    
    logger.info(f"所有测试完成，总耗时: {time.time() - main_start:.2f}秒")

if __name__ == "__main__":
    logger.info("脚本开始执行...")
    main() 