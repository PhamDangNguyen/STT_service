uv run vllm serve HuggingFaceTB/SmolLM3-3B \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype float16 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.3 \
  --kv-cache-memory-bytes 512M \
  --enforce-eager