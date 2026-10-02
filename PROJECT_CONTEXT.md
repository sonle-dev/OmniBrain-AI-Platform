# OMNIBRAIN AI PLATFORM CORE — MASTER SYSTEM CONTEXT & ARCHITECTURE MEMORY

> **CẤU HÌNH BỘ NHỚ HỆ THỐNG (SYSTEM MEMORY FILE)**  
> *Tài liệu này ghi nhớ toàn bộ kiến trúc, sơ đồ file, bảng định tuyến mô hình và quy trình vận hành của dự án OmniBrain AI Core. Tất cả AI Agent và Developer khi làm việc với dự án này CẦN đọc file này trước tiên để nắm toàn bộ ngữ cảnh hệ thống mà không phải quét lại toàn bộ codebase.*

---

## 📌 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)

- **Tên dự án**: `OmniBrain AI Platform Core` (`omnibrain-ai-core`)
- **Bản chất**: Microservice AI Gateway tập trung, độc lập, chuẩn Enterprise, xây dựng bằng **FastAPI**, **LiteLLM**, **NetworkX GraphRAG**, **Semantic Caching** và **Dynamic Tool Calling**.
- **Nguyên lý thiết kế**: Decoupled (Phân tách 100% tầng AI khỏi trang web Client). Mọi ứng dụng Web (LMS, E-commerce, Du lịch) tích hợp thông qua **Universal JavaScript Client SDK** (`static/omnibrain-client.js`) hoặc REST API.
- **Tương thích thiết bị**: 100% Local (Ollama) + Multi-Cloud Fallback (Zero-Key Tolerance).

---

## 🧠 2. KIẾN TRÚC TRIAD AI ENGINE & BẢNG ĐỊNH TUYẾN MÔ HÌNH (ROUTING TABLE)

Hệ thống sử dụng **Intelligent Specialist Dispatcher** trong `app/services/llm_factory.py` và `app/core/config.py` để phân công đúng mô hình chuyên gia theo từng tác vụ (`task`):

| Tác vụ (`task`) | Mô hình Local Ưu tiên (Ollama) | Nhiệm vụ Chuyên biệt | Mô hình Cloud Dự phòng (Fallback) |
| :--- | :--- | :--- | :--- |
| `task="chat"` | **`ollama/qwen2.5:7b`** | Trò chuyện tiếng Việt tự nhiên, tư vấn môn học, giao tiếp thường nhật. | `gemma-4-26b-free` -> `gemini-2.0-flash` -> `deepseek-chat` -> `gpt-4o-mini` |
| `task="reasoning"` | **`ollama/deepseek-r1:8b`** | Suy luận logic toán học, lập trình Python/Java/SQL, sinh đề thi JSON chuẩn 100%. | `gpt-oss-20b-free` -> `gemini-2.0-flash` -> `deepseek-reasoner` -> `gpt-4o` |
| `task="graph_rag"` | **`ollama/nomic-embed-text`** | Trích xuất tri thức tài liệu (PDF, PPTX, TXT) thành Đồ thị tri thức Entity-Relationship. | `gemini-2.0-flash` -> `openrouter` |
| `task="vision"` | **`ollama/llava`** / **`llama3.2-vision`** | Nhìn ảnh chụp màn hình, đọc lỗi code, đọc biểu đồ và trích xuất chữ từ ảnh (OCR). | `gemini-2.0-flash` -> `gpt-4o` |
| `task="embedding"` | **`ollama/nomic-embed-text`** | Chuyển đổi văn bản thành Vector toán học. | Local Nomic Embed |

### 🔄 Quy trình Chuyển vùng An toàn (Graceful Cloud Fallback):
`LoRA Adapter Local ➡️ Base Local Model (Ollama) ➡️ OpenRouter Free ➡️ Gemini ➡️ DeepSeek Cloud ➡️ OpenAI`
- Nếu Ollama local tắt hoặc hết RAM ➡️ Tự động nhảy sang Cloud không gây sập app.
- Nếu không có Cloud API Key ➡️ Tự động chạy 100% Local (Zero-Key Tolerance).

---

## ⚡ 3. CÁC TÍNH NĂNG TỐI ƯU HÓA TỐC ĐỘ (SPEED & LATENCY OPTIMIZATIONS)

1. **Fast-Path Intent Classifier (`app/api/v1/chat.py`)**:
   - Sử dụng Regex nhận diện câu chào hỏi đơn giản (*"xin chào"*, *"hello"*, *"bạn là ai"*) để **Early Exit** bỏ qua các bước tra CSDL/GraphRAG ➡️ Phản hồi tức thì trong **< 300ms (TTFT)**.
2. **Dynamic Real-Time Clock Injection (`app/services/llm_factory.py`)**:
   - Tự động tiêm thông tin ngày, thứ, giờ và múi giờ thực tế (`Asia/Ho_Chi_Minh`, GMT+7) vào System Prompt cho mọi request ➡️ Khắc phục triệt để lỗi ảo giác mốc thời gian (Cutoff date).
3. **Persistent Async Connection Pooling**:
   - Tái sử dụng `httpx.AsyncClient` toàn cục với `keepalive_expiry=600.0s` ➡️ Triệt tiêu chi phí bắt tay kết nối mạng (TCP/SSL handshake).
4. **Ollama VRAM Pinning (`keep_alive: -1`)**:
   - Truyền `extra_body={"keep_alive": -1}` cho mọi request gọi Ollama ➡️ Giữ model luôn nằm sẵn trên VRAM GPU, xóa bỏ thời gian nạp lại model (Cold-Start Penalty).
5. **SSE Streaming & Typewriter SDK**:
   - Phản hồi dạng dòng chảy token qua `text/event-stream` kết hợp hàm `streamChatTypewriter()` trong Client SDK ➡️ Tạo hiệu ứng gõ chữ thời gian thực mượt mà.

---

## 📂 4. SƠ ĐỒ THƯ MỤC CỐT LÕI & VAI TRÒ TỪNG FILE (SITEMAP)

```text
d:/OmniBrain AI Platform/
├── PROJECT_CONTEXT.md          # [FILE NÀY] Bộ nhớ ngữ cảnh Master của toàn bộ dự án
├── README.md                   # Tài liệu giới thiệu & hướng dẫn cài đặt nhanh
├── Dockerfile & docker-compose.yml # Đóng gói container Docker phục vụ triển khai
├── requirements.txt            # Danh sách thư viện Python (FastAPI, LiteLLM, NetworkX, SQLAlchemy...)
├── static/
│   └── omnibrain-client.js     # Universal JavaScript Client SDK (Tích hợp cho mọi Web Client)
├── tests/                      # Bộ Unit & Integration Tests tự động (22 test cases cho FastAPI Core)
└── app/
    ├── main.py                 # FastAPI Entrypoint, Middleware CORS & Route Registration
    ├── core/
    │   ├── config.py            # Cấu hình hệ thống, Bảng định tuyến SPECIALIST_MODELS & API Keys
    │   ├── security.py          # Xác thực API Key (X-API-Key), Hạn mức Token & Rate Limiter (RPM)
    │   └── cache.py             # Semantic Cache Engine (Redis / In-Memory LRU + Cosine Similarity)
    ├── api/
    │   └── v1/
    │       ├── chat.py          # API Chat chính (Fast-Path Router, Dynamic Context & SSE Streaming)
    │       ├── quiz.py          # API Sinh đề thi trắc nghiệm có tự sửa lỗi (Self-Correction Loop)
    │       ├── graph_rag.py     # API Nạp tài liệu & Tra cứu Đồ thị tri thức GraphRAG
    │       ├── external_search.py # API Tìm kiếm web thời gian thực (Tavily/Nara)
    │       ├── agent.py         # API Multi-step AI Agent Executor (ReAct Loop)
    │       ├── vision.py        # API Phân tích hình ảnh & OCR
    │       └── audio.py         # API Chuyển giọng nói thành văn bản (STT) & Tổng hợp giọng nói (TTS)
    ├── services/
    │   ├── llm_factory.py       # Bộ điều phối LLM, Real-Time Clock Injection, Connection Pool & VRAM Pinning
    │   ├── graph_rag_service.py # Dịch vụ Đồ thị tri thức NetworkX & Phân tích tài liệu PDF/PPTX
    │   ├── search_service.py    # Kết nối công cụ Tìm kiếm Web
    │   ├── agent_service.py     # Đăng ký & Thực thi danh sách Tool Calling
    │   └── tools/               # Thư mục Plugin mở rộng tra cứu dữ liệu ứng dụng
    │       └── lms_tools.py     # Safe Read-Only SQL Database Query Engine Template
    └── schemas/
        ├── requests.py          # Request Pydantic Schemas
        └── responses.py         # Response Pydantic Schemas
```

---

## 🛠️ 5. KHUNG KHẢO SÁT & QUY TRÌNH PHÁT TRIỂN MỞ RỘNG (DEV GUIDELINES)

### Cách chạy Server Python:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger API: `http://127.0.0.1:8000/docs`

### Quy tắc khi cập nhật / bổ sung tính năng mới:
1. **Không phá vỡ API Contract**: Giữ nguyên cấu trúc JSON của `/api/v1/chat/completions`.
2. **Thêm Tool tra cứu mới**: Tạo file trong `app/services/tools/` (dạng Safe Read-Only connector) và đăng ký vào `_enrich_lms_context()` trong `app/api/v1/chat.py`.
3. **Bảo trì File `PROJECT_CONTEXT.md`**: Khi có thay đổi lớn về kiến trúc hoặc cấu hình model, CẦN cập nhật file này để giữ cho bộ nhớ ngữ cảnh luôn đúng thời gian thực.
