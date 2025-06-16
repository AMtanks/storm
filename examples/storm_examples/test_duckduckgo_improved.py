#!/usr/bin/env python
"""
测试改进后的DuckDuckGo搜索引擎
这个脚本测试DuckDuckGo搜索引擎的改进版本，包括多后端轮换和智能退避策略
"""

import argparse
import logging
import os
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 导入DuckDuckGoSearchRM
from knowledge_storm.rm import DuckDuckGoSearchRM

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

class DuckDuckGoTester:
    """DuckDuckGo搜索引擎测试类"""
    
    def __init__(self, config: Dict[str, Any], queries: List[str]):
        """
        初始化测试器
        
        Args:
            config: 配置字典，包含测试参数
            queries: 测试查询列表
        """
        self.config = config
        self.queries = queries
        self.results = {
            "success_count": 0,
            "total_count": len(queries),
            "success_rate": 0.0,
            "total_time": 0.0,
            "query_results": [],
            "config": config
        }
        
        logger.info(f"初始化测试器，配置: {json.dumps(config, indent=2)}")
        
        # 初始化搜索引擎，确保参数名称与DuckDuckGoSearchRM类一致
        self.rm = DuckDuckGoSearchRM(
            k=config.get("max_results", 3),
            request_delay=config.get("request_delay", 3.0),
            max_retries=config.get("max_retries", 99),
            use_multiple_backends=config.get("use_multiple_backends", True),
            exponential_backoff=config.get("exponential_backoff", True),
            max_delay=config.get("max_delay", 8.0),
            webpage_helper_max_threads=config.get("webpage_helper_max_threads", 1)
        )
    
    def run_sequential_test(self) -> Dict:
        """
        运行顺序测试
        
        Returns:
            测试结果字典
        """
        logger.info(f"开始顺序测试，查询数量: {len(self.queries)}")
        
        start_time = time.time()
        
        for i, query in enumerate(self.queries):
            query_start = time.time()
            query_result = {
                "query": query,
                "success": False,
                "time": 0,
                "result_count": 0,
                "error": None
            }
            
            logger.info(f"查询 {i+1}/{len(self.queries)}: '{query}'")
            
            try:
                results = self.rm.forward(query)
                
                query_time = time.time() - query_start
                query_result["time"] = query_time
                
                if results and len(results) > 0:
                    query_result["success"] = True
                    query_result["result_count"] = len(results)
                    self.results["success_count"] += 1
                    logger.info(f"成功: 找到 {len(results)} 个结果，耗时: {query_time:.2f}秒")
                else:
                    logger.warning(f"警告: 查询成功但没有结果，耗时: {query_time:.2f}秒")
            
            except Exception as e:
                error_msg = str(e)
                query_result["error"] = error_msg
                logger.error(f"错误: {error_msg}")
            
            self.results["query_results"].append(query_result)
            
            # 在测试之间添加额外的延迟，避免触发速率限制
            if i < len(self.queries) - 1:  # 如果不是最后一个查询
                wait_time = self.config.get("test_interval", 5.0)
                logger.info(f"等待 {wait_time} 秒后进行下一次查询...")
                time.sleep(wait_time)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        self.results["total_time"] = total_time
        self.results["success_rate"] = self.results["success_count"] / self.results["total_count"] if self.results["total_count"] > 0 else 0
        
        logger.info(f"顺序测试完成 - 成功率: {self.results['success_rate']:.2%}, 总耗时: {total_time:.2f}秒")
        
        return self.results
    
    def run_concurrent_test(self) -> Dict:
        """
        运行并发测试
        
        Returns:
            测试结果字典
        """
        max_workers = self.config.get("max_workers", 2)
        logger.info(f"开始并发测试，查询数量: {len(self.queries)}, 最大工作线程: {max_workers}")
        
        start_time = time.time()
        
        def process_query(query):
            query_start = time.time()
            query_result = {
                "query": query,
                "success": False,
                "time": 0,
                "result_count": 0,
                "error": None
            }
            
            logger.info(f"查询: '{query}'")
            
            try:
                results = self.rm.forward(query)
                
                query_time = time.time() - query_start
                query_result["time"] = query_time
                
                if results and len(results) > 0:
                    query_result["success"] = True
                    query_result["result_count"] = len(results)
                    logger.info(f"成功: 找到 {len(results)} 个结果，耗时: {query_time:.2f}秒")
                    return True, query_result
                else:
                    logger.warning(f"警告: 查询成功但没有结果，耗时: {query_time:.2f}秒")
                    return False, query_result
            
            except Exception as e:
                error_msg = str(e)
                query_result["error"] = error_msg
                logger.error(f"错误: {error_msg}")
                return False, query_result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_query, query) for query in self.queries]
            
            for future in futures:
                success, query_result = future.result()
                if success:
                    self.results["success_count"] += 1
                self.results["query_results"].append(query_result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        self.results["total_time"] = total_time
        self.results["success_rate"] = self.results["success_count"] / self.results["total_count"] if self.results["total_count"] > 0 else 0
        
        logger.info(f"并发测试完成 - 成功率: {self.results['success_rate']:.2%}, 总耗时: {total_time:.2f}秒")
        
        return self.results
    
    def save_results(self, output_dir: str = "./results/duckduckgo_test"):
        """
        保存测试结果到文件
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode = "concurrent" if self.config.get("max_workers", 1) > 1 else "sequential"
        delay = self.config.get("request_delay", 0)
        
        filename = f"{mode}_delay{delay}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试结果已保存到: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="测试改进后的DuckDuckGo搜索引擎")
    parser.add_argument("--mode", choices=["sequential", "concurrent"], default="sequential",
                      help="测试模式: sequential(顺序), concurrent(并发)")
    parser.add_argument("--delay", type=float, default=3.0,
                      help="请求延迟（秒）")
    parser.add_argument("--max-retries", type=int, default=99,
                      help="最大重试次数")
    parser.add_argument("--max-results", type=int, default=3,
                      help="每次搜索返回的最大结果数")
    parser.add_argument("--max-workers", type=int, default=2,
                      help="并发模式下的最大工作线程数")
    parser.add_argument("--test-interval", type=float, default=3.0,
                      help="顺序测试中查询之间的间隔时间（秒）")
    parser.add_argument("--queries", type=int, default=3,
                      help="要使用的测试查询数量")
    parser.add_argument("--output-dir", type=str, default="./results/duckduckgo_test",
                      help="测试结果输出目录")
    parser.add_argument("--no-multiple-backends", action="store_true",
                      help="禁用多后端轮换")
    parser.add_argument("--no-exponential-backoff", action="store_true",
                      help="禁用指数退避")
    parser.add_argument("--max-delay", type=float, default=8.0,
                      help="最大延迟时间（秒）")
    
    args = parser.parse_args()
    
    # 创建配置字典
    config = {
        "request_delay": args.delay,
        "max_retries": args.max_retries,
        "max_results": args.max_results,
        "max_workers": args.max_workers,
        "test_interval": args.test_interval,
        "use_multiple_backends": not args.no_multiple_backends,
        "exponential_backoff": not args.no_exponential_backoff,
        "max_delay": args.max_delay,
        "webpage_helper_max_threads": 1
    }
    
    # 选择查询
    selected_queries = TEST_QUERIES[:args.queries]
    
    # 初始化测试器
    tester = DuckDuckGoTester(config, selected_queries)
    
    # 运行测试
    if args.mode == "sequential":
        results = tester.run_sequential_test()
    else:
        results = tester.run_concurrent_test()
    
    # 保存结果
    tester.save_results(args.output_dir)
    
    # 打印摘要
    print("\n测试摘要:")
    print(f"模式: {args.mode}")
    print(f"请求延迟: {args.delay}秒")
    print(f"成功率: {results['success_rate']:.2%}")
    print(f"总耗时: {results['total_time']:.2f}秒")
    print(f"查询数量: {results['total_count']}")
    print(f"成功查询: {results['success_count']}")
    
    if results['success_rate'] < 1.0:
        print("\n失败的查询:")
        for result in results['query_results']:
            if not result['success']:
                print(f"- '{result['query']}': {result.get('error', '无错误信息')}")

if __name__ == "__main__":
    main() 