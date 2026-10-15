# OmniBrain AI Platform Core

**Enterprise-Grade, Multi-Capability AI Gateway & Agent Microservice** built with **FastAPI**, **LiteLLM**, **NetworkX / GraphRAG**, **Semantic Caching**, and **Dynamic Tool Calling**.

OmniBrain acts as a centralized, decoupled **"AI Brain"** service that integrates seamlessly into any web platform (LMS, E-commerce, Tourism, SaaS) via clean REST APIs or the included **Universal JavaScript SDK**.

---

## 🚀 Architectural Highlights

### 1. Triad AI Engine Architecture
- **LoRA-Ready Specialist Adapter**: Task-based intelligent routing. Automatically prioritizes local Ollama models (`ollama/qwen2.5:7b`, `ollama/deepseek-r1:8b`) and custom LoRA fine-tuned adapters. Fallbacks gracefully to cloud APIs without breaking client applications.
- **GraphRAG Knowledge Grounding**: Entity-relationship extraction from uploaded files (PDFs, PPTX Slides, TXT) powered by a pure-Python NetworkX Knowledge Graph index with strict anti-hallucination prompts.
- **Dynamic Database Tool Calling**: Extensible tool registry engine (`app/services/tools/`) enabling safe read-only SQL queries to inject live application data (Courses, Products, Booking) directly into prompt context.

### 2. Sub-Second Latency & Speed Optimizations
- **Fast-Path Intent Classifier**: Early exit regex router that instantly bypasses heavy database/vector lookups for greetings and simple prompts (**< 300ms TTFT**).
- **Dynamic Real-Time Clock Injection**: Automatically injects dynamic timestamp, day of week, and timezone (`Asia/Ho_Chi_Minh`, GMT+7) into system prompts, eliminating LLM temporal cutoff hallucinations.
- **Persistent Async Connection Pooling**: Global `httpx.AsyncClient` with `keepalive_expiry=600s` eliminating TCP/SSL handshake overhead.
- **Ollama VRAM Pinning**: Includes `"keep_alive": -1` in Ollama payloads to keep models permanently loaded in GPU VRAM and eliminate cold-start penalties.
- **SSE Streaming & Typewriter SDK**: Full Server-Sent Events support (`text/event-stream`) in API and SDK typewriter DOM rendering helper (`streamChatTypewriter`).

### 3. Zero-Key Tolerance & Security
- **Graceful Degradation**: Microservice boots and runs 100% locally even if all cloud API keys are missing.
- **Multi-Tenant Rate Control**: API key authentication (`X-API-Key`) with configurable sliding-window rate limiting (RPM) and token usage tracking.
- **Semantic Cache Engine**: Sub-5ms response time for repetitive queries using in-memory LRU cache with Redis fallback.

---

## 📂 Project Structure

```text
omnibrain-ai-core/
├── app/
│   ├── main.py                 # FastAPI Entrypoint, Middleware & CORS
│   ├── core/
│   │   ├── config.py            # Gateway Configuration & Model Routing Tables
│   │   ├── security.py          # Tenant API-Key Auth, Token Limits & Rate Limiter
│   │   └── cache.py             # Semantic Cache Engine (Redis / In-Memory LRU)
│   ├── api/
│   │   └── v1/
│   │       ├── chat.py          # Unified Multi-LLM Chat & SSE Streaming Endpoint
│   │       ├── quiz.py          # Structured Quiz Generator & Parser
│   │       ├── graph_rag.py     # GraphRAG Ingestion & Knowledge Graph Query
│   │       ├── external_search.py # Live Tavily Web Search
│   │       ├── agent.py         # Multi-step ReAct AI Agent & Tool Calling
│   │       ├── vision.py        # Vision OCR & Image Analysis
│   │       └── audio.py         # Whisper STT & Speech Synthesis (TTS)
│   ├── services/
│   │   ├── llm_factory.py       # Specialist Dispatcher, VRAM Pinning & Fallback Chain
│   │   ├── graph_rag_service.py # NetworkX Knowledge Graph RAG Service
│   │   ├── search_service.py    # Live Web Search Connector
│   │   ├── agent_service.py     # Function Calling & Agent Loop Executor
│   │   └── tools/               # Reference Plugin Registry for Live App Connectors
│   │       └── lms_tools.py     # Read-only Database Query Engine Template
│   └── schemas/
│       ├── requests.py          # Pydantic Request Models
│       └── responses.py         # Pydantic Response Models
├── static/
│   └── omnibrain-client.js      # Universal JavaScript Client SDK
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## ⚙️ Quick Start

### 1. Setup Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 2. Install Dependencies & Run Locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API documentation is available at `http://127.0.0.1:8000/docs`.

### 3. Run with Docker Compose
```bash
docker-compose up -d --build
```

---

## 🔌 Universal Client SDK Integration

Import `omnibrain-client.js` in any web application (React, Vue, HTML/JS) to interact with OmniBrain Core:

```html
<!-- Import Universal Client SDK -->
<script src="http://localhost:8000/static/omnibrain-client.js"></script>
<script>
  // Initialize OmniBrain Client
  const ai = new OmniBrainClient("http://localhost:8000", "demo_key");

  // 1. Standard Chat Completion
  async function handleChat() {
      const res = await ai.chat([{ role: "user", content: "Xin chào OmniBrain!" }]);
      console.log(res.text);
  }

  // 2. Sub-Second SSE Streaming with Typewriter Effect
  async function handleStream() {
      await ai.streamChatTypewriter(
          [{ role: "user", content: "Viết bài luận ngắn về AI" }],
          "outputDiv", // Target DOM Element ID
          () => console.log("Stream finished!")
      );
  }

  // 3. Query Knowledge Base (GraphRAG)
  async function queryDocs() {
      const res = await ai.queryGraphRAG("Tóm tắt tài liệu đã tải lên");
      console.log(res.text);
  }
</script>
```

---

## 🧠 Multi-Tenant & Partitioned Knowledge Base (GraphRAG Namespaces)

OmniBrain Core is designed as a centralized multi-app gateway. To prevent **Knowledge Cross-Pollution** across different websites (e.g., an LMS university portal vs. an E-commerce store), the Knowledge Graph is partitioned into isolated **namespaces**:

```text
graphrag_data/
├── lms/          # Dedicated knowledge partition for LMS (curriculum, exams, slides)
│   └── graph.json
├── ecommerce/    # Dedicated knowledge partition for Shop (products, refund policies)
│   └── graph.json
└── custom_web/   # Any other web application partition
    └── graph.json
```

### Key Capabilities:
- **Default Partition**: Uploads and queries without a specified namespace automatically default to `"lms"`.
- **Zero Cross-Pollution**: Updating, modifying, or resetting knowledge for one website has **zero impact** on other websites.
- **Dynamic Creation**: Uploading documents with a new `namespace` automatically creates and indexes that partition without server restart.

### Usage Examples:
- **Upload Document to a Namespace**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/graph-rag/upload?namespace=lms" \
       -H "X-API-Key: demo_key" \
       -F "file=@curriculum.pdf"
  ```
- **Query a Specific Namespace**:
  ```json
  POST /api/v1/graph-rag/query
  {
    "query": "Lịch thi môn Python?",
    "namespace": "lms",
    "mode": "local"
  }
  ```
- **List All Active Namespaces & Node Metrics**:
  ```bash
  GET /api/v1/graph-rag/namespaces
  ```
- **Reset Only a Specific Partition**:
  ```bash
  POST /api/v1/graph-rag/reset?namespace=ecommerce
  ```

---

## 📑 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/chat/completions` | Unified Chat Completion & SSE Streaming (`stream=true`) |
| `POST` | `/api/v1/quiz/generate` | Structured Quiz Generator with Self-Correction |
| `POST` | `/api/v1/graph-rag/query` | Grounded Knowledge Graph Query (supports `namespace`) |
| `POST` | `/api/v1/graph-rag/upload` | Document Ingestion into GraphRAG (supports `namespace`) |
| `GET` | `/api/v1/graph-rag/namespaces` | List all active Knowledge Base partitions & stats |
| `POST` | `/api/v1/graph-rag/reset` | Reset a specific Knowledge Base partition |
| `POST` | `/api/v1/agent/run` | Multi-step AI Agent Tool Execution |
| `POST` | `/api/v1/vision/analyze` | OCR & Image Semantic Analysis (Ollama local prioritized) |
| `POST` | `/api/v1/audio/transcribe` | Audio Speech-to-Text Transcription |
| `POST` | `/api/v1/audio/synthesize` | Text-to-Speech Synthesis |
| `GET` | `/health` | Microservice Health & Service Status Check |

---

## 📄 License
MIT License. Created & maintained for high-performance AI Gateway deployments.
