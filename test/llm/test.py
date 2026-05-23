import requests

url = "http://localhost:8001/v1/chat/completions"

payload = {
    "model": "HuggingFaceTB/SmolLM3-3B",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant., always answer in Vietnamese. /no_think"
        },
        {
            "role": "user",
            "content": "Xin chào, bạn là model gì?"
        }
    ],
    "temperature": 0.2,
    "max_tokens": 300,
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": False}
    }
}

response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    json=payload,
    timeout=60
)

print("Status:", response.status_code)
print(response.json())