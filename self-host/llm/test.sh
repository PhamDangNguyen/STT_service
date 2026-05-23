    curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "HuggingFaceTB/SmolLM3-3B",
        "messages": [
        {"role": "user", "content": "Xin chào, bạn là model gì?"}
        ],
        "temperature": 0.2,
        "max_tokens": 128
    }'