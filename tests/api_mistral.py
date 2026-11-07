# Level 1: Thư viện chuẩn Python (Standard Library)
import json
import sys

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import requests

# Level 3: Module nội bộ trong dự án (Project-Local Imports)


API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = "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"
MODEL_NAME = "mistralai/mistral-large-4-1t-a52b"


def test_mistral_stream():
    """Kiểm tra gọi API Mistral Large 4 với chế độ Streaming (hiển thị từng chữ tức thì)."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "text/event-stream",
    }

    sample_jd = (
        "Role: Senior Software Engineer\n"
        "Requirements: 5+ years in Java development. Strong AWS Cloud architecting. "
        "Bachelor degree in Computer Science. Preferred: Kubernetes certification."
    )

    prompt = f"""Hãy bóc tách thông tin từ đoạn JD sau thành đúng định dạng JSON:
"{sample_jd}"

Yêu cầu xuất JSON ngắn gọn có các trường: job_title, skills, experience_years, degree, certs."""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
        "stream": True,
    }

    print(">>> Đang kết nối tới máy chủ NVIDIA Mistral Large 4...")
    try:
        response = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60)
        if response.status_code != 200:
            print(f"Lỗi API: HTTP {response.status_code}")
            print(response.text)
            return

        print(">>> Kết nối thành công! Đang nhận phản hồi theo thời gian thực (Streaming):\n")
        print("-" * 60)

        for line in response.iter_lines():
            if not line:
                continue

            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data: ") and not decoded_line.startswith("data: [DONE]"):
                try:
                    chunk = json.loads(decoded_line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    # In ra chữ mô hình trả về (cả suy luận và nội dung chính)
                    text_chunk = delta.get("content") or delta.get("reasoning_content")
                    if text_chunk:
                        sys.stdout.write(text_chunk)
                        sys.stdout.flush()
                except Exception:
                    pass

        print("\n" + "-" * 60)
        print(">>> Hoàn thành nhận dữ liệu!")

    except requests.exceptions.Timeout:
        print("Lỗi: Kết nối quá thời gian chờ (Timeout). Hãy kiểm tra lại mạng!")
    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    test_mistral_stream()
