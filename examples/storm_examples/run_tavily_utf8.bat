@echo off
echo 设置代理环境变量...
rem 设置HTTP代理，请根据您的实际情况修改代理地址和端口
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
set REQUESTS_CA_BUNDLE=
set SSL_CERT_FILE=

rem 设置Python默认编码为UTF-8
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo 正在使用Tavily搜索引擎运行STORM Wiki...
python examples/storm_examples/run_storm_wiki_siliconflow.py --retriever tavily --do-research --do-generate-outline --do-generate-article --do-polish-article
pause 