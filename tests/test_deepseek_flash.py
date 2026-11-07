# Level 1: Thư viện chuẩn Python (Standard Library)
import json
import sys

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import requests

# Level 3: Module nội bộ trong dự án (Project-Local Imports)


API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = "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"
MODEL_NAME = "deepseek-ai/deepseek-v4.1-flash"


def test_deepseek_flash_text():
    """Kiểm tra gọi API DeepSeek v4.1 Flash với text thông thường và streaming."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    sample_prompt = (
        "Chào bạn, hãy giới thiệu ngắn gọn trong 2 câu bạn là mô hình gì "
        "và bạn có khả năng đọc ảnh (multimodal/vision) không?"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": sample_prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.2,
        "stream": True,
    }

    print(f">>> Đang kết nối tới máy chủ NVIDIA ({MODEL_NAME})...")
    try:
        response = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60)
        if response.status_code != 200:
            print(f"Lỗi API: HTTP {response.status_code}")
            print(response.text)
            return

        print(">>> Kết nối thành công! Đang nhận phản hồi:\n")
        print("-" * 60)

        for line in response.iter_lines():
            if not line:
                continue

            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data: ") and not decoded_line.startswith("data: [DONE]"):
                try:
                    chunk = json.loads(decoded_line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    text_chunk = delta.get("content") or delta.get("reasoning_content")
                    if text_chunk:
                        sys.stdout.write(text_chunk)
                        sys.stdout.flush()
                except Exception:
                    pass

        print("\n" + "-" * 60)
        print(">>> Hoàn thành!")

    except requests.exceptions.Timeout:
        print("Lỗi: Quá thời gian chờ (Timeout) khi kết nối NVIDIA API.")
    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    test_deepseek_flash_text()

