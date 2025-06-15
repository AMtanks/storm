"""
检查代理是否正常工作的脚本
"""

import os
import sys
import requests
import httpx

def check_proxy_with_requests():
    """使用requests库检查代理"""
    print("=== 使用requests库检查代理 ===")
    proxies = {
        "http": os.environ.get("HTTP_PROXY", "http://127.0.0.1:7890"),
        "https": os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7890"),
    }
    
    print(f"当前代理设置: {proxies}")
    
    try:
        # 检查IP地址
        response = requests.get("https://api.ipify.org?format=json", proxies=proxies, timeout=10)
        print(f"当前IP地址: {response.json()['ip']}")
        
        # 尝试访问Google（需要代理才能访问）
        response = requests.get("https://www.google.com", proxies=proxies, timeout=10)
        print(f"Google访问状态码: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"请求失败: {str(e)}")
        return False

def check_proxy_with_httpx():
    """使用httpx库检查代理"""
    print("\n=== 使用httpx库检查代理 ===")
    proxies = os.environ.get("HTTP_PROXY", "http://127.0.0.1:7890")
    
    print(f"当前代理设置: {proxies}")
    
    try:
        # 检查IP地址
        with httpx.Client(proxies=proxies, verify=False) as client:
            response = client.get("https://api.ipify.org?format=json", timeout=10)
            print(f"当前IP地址: {response.json()['ip']}")
            
            # 尝试访问Google（需要代理才能访问）
            response = client.get("https://www.google.com", timeout=10)
            print(f"Google访问状态码: {response.status_code}")
            
        return True
    except Exception as e:
        print(f"请求失败: {str(e)}")
        return False

def check_env_variables():
    """检查环境变量设置"""
    print("\n=== 检查环境变量 ===")
    print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY', '未设置')}")
    print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', '未设置')}")
    print(f"REQUESTS_CA_BUNDLE: {os.environ.get('REQUESTS_CA_BUNDLE', '未设置')}")
    print(f"SSL_CERT_FILE: {os.environ.get('SSL_CERT_FILE', '未设置')}")

if __name__ == "__main__":
    print("代理检查工具")
    print("============")
    
    check_env_variables()
    
    requests_ok = check_proxy_with_requests()
    httpx_ok = check_proxy_with_httpx()
    
    if requests_ok and httpx_ok:
        print("\n✅ 代理设置正常工作！")
        sys.exit(0)
    else:
        print("\n❌ 代理设置存在问题，请检查您的代理配置。")
        sys.exit(1) 