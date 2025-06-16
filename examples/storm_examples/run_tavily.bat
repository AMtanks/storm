@echo off
echo 正在使用Tavily搜索引擎运行STORM Wiki...

conda activate storm

$env:PYTHONPATH = "G:\storm"

set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
set REQUESTS_CA_BUNDLE=
set SSL_CERT_FILE=

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

python examples/storm_examples/run_storm_wiki_siliconflow.py --retriever tavily --do-research --do-generate-outline --do-generate-article --do-polish-article
python examples/storm_examples/run_storm_wiki_siliconflow.py --retriever duckduckgo --do-research --do-generate-outline --do-generate-article --do-polish-article

pause 