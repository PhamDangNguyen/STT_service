uv run vllm serve HuggingFaceTB/SmolLM3-3B \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.4 \
  --max-num-seqs 1 \
  --enforce-eager