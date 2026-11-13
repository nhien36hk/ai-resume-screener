import json
import os
import time

from dotenv import load_dotenv
import requests

load_dotenv()

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = os.getenv("NVIDIA_API_KEY", os.getenv("API_NVIDIA", ""))


def test_minimax():
    print("\n" + "=" * 50)
    print("1. Kiểm tra minimaxai/minimax-m3.1-flash:")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image? Answer in 1 short sentence."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1b.jpg"
                        }
                    }
                ]
            }
        ],
        "model": "minimaxai/minimax-m3.1-flash",
        "max_tokens": 128,
        "stream": True,
        "temperature": 0.2,
    }

    t0 = time.time()
    try:
        res = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=15)
        if res.status_code != 200:
            print(f"Status: {res.status_code} - {res.text}")
            return

        received_any = False
        for line in res.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if "delta" in decoded and not received_any:
                    latency = time.time() - t0
                    print(f"-> Bắt đầu phản hồi (Time-to-first-token): {latency:.2f}s")
                    received_any = True
                if decoded.startswith("data: ") and not decoded.startswith("data: [DONE]"):
                    try:
                        chunk = json.loads(decoded[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        text = delta.get("content") or delta.get("reasoning_content")
                        if text:
                            print(text, end="", flush=True)
                    except Exception:
                        pass
        print(f"\n-> Tổng thời gian: {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"Lỗi: {e}")


def test_glm():
    print("\n" + "=" * 50)
    print("2. Kiểm tra z-ai/glm-5.3-flash:")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "z-ai/glm-5.3-flash",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Which number is larger, 9.11 or 9.8? Answer in 1 short sentence."}
        ],
        "temperature": 0.5,
        "max_tokens": 128,
        "stream": False
    }

    t0 = time.time()
    try:
        res = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        latency = time.time() - t0
        if res.status_code != 200:
            print(f"Status: {res.status_code} - {res.text}")
            return

        data = res.json()
        choice = data["choices"][0]
        content = choice["message"].get("content")
        print(f"-> Thời gian phản hồi: {latency:.2f}s")
        print(f"-> Kết quả: {content}")
    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    test_minimax()
    test_glm()

