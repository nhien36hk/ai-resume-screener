# Hướng dẫn chạy hệ thống Web (Chương 5)

## 1. Kích hoạt môi trường và cài thư viện (chỉ làm 1 lần)

Đứng tại thư mục gốc `ai-resume-screener`:

```bash
conda activate Devign311

# Cài thư viện Python (lõi + web)
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart pdfplumber python-docx

# Cài thư viện Frontend
cd web/frontend && npm install && cd ../..
```

## 2. Chuẩn bị mô hình M5 từ server và khóa API

- Chép kết quả mô hình M5 từ server về đúng 2 vị trí sau:
  - `artifacts/models/e5_finetuned/jd/seed4/`
  - `artifacts/results/jd/seed4/e5.json`
- Tạo tệp `.env` ở thư mục gốc `ai-resume-screener/.env` để bật trích xuất LLM cho JD/CV mới:
  ```env
  NVIDIA_API_KEY=nvapi-...
  ```

## 3. Khởi chạy hệ thống (mặc định dùng M5)

**Terminal 1 — Backend (FastAPI):**
```bash
conda activate Devign311
uvicorn web.backend.main:app --port 8000
```
*(Nếu chưa chép M5 về và muốn chạy tạm bằng M4: `WEB_MODEL_MODE=m4 uvicorn web.backend.main:app --port 8000`.)*

**Terminal 2 — Frontend (React / Vite):**
```bash
cd web/frontend
npm run dev
```

Mở trình duyệt tại **`http://localhost:5173`**.

*(Tùy chọn: nếu muốn chạy 1 terminal duy nhất không cần `npm run dev`, chạy `cd web/frontend && npm run build` một lần rồi bật `uvicorn`, sau đó truy cập thẳng `http://localhost:8000`.)*

