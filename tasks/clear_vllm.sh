ray stop --force || true

pkill -9 -f '[v]LLM::Worker' || true
pkill -9 -f '[E]ngineCore' || true
pkill -9 -f '[v]LLMHttpServer' || true
pkill -9 -f 'ray::WorkerDict' || true

sleep 2

ps aux | grep -E 'VLLM|EngineCore|vLLMHttpServer|WorkerDict' | grep -v grep || true
free -h
