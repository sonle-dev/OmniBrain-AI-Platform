# OmniBrain AI Platform Core

Centralized, modular, and high-performance AI Gateway microservice built with **FastAPI**, **LiteLLM**, **NetworkX / GraphRAG**, and **Semantic Caching**. 

OmniBrain acts as a centralized "AI Brain" that can be integrated into any client application (LMS, E-commerce, Tourism) using simple REST APIs or the included Universal Javascript Client SDK.

---

## 🌟 Key Features

*   **Modular Architecture:** All AI endpoints (Chat, GraphRAG, Search, Vision, Audio, Agent) are lazily loaded to minimize RAM usage and package overhead.
*   **Self-Healing GraphRAG:** Structured entity-relationship extraction from uploaded files (PDFs, PPTX Slides, TXT) with a custom pure-python NetworkX local-search engine fallback.
*   **Multi-LLM Unified Gateway:** Powered by LiteLLM. Automatically handles fallback models and includes a structured JSON output self-correction loop.
*   **Semantic Cache Engine:** Saves redundant query cost and responds in <5ms. Uses Redis if available, otherwise falls back to local SQLite with NumPy cosine similarity.
*   **Multi-Tenant Budget Control:** Multi-tenant client authentication via `X-API-Key`. Controls token limits and enforces IP/App sliding-window rate limits.
*   **Universal JS SDK:** Client-side JS SDK included for rapid frontend integration.

---

## 📂 Project Structure

```text
omnibrain-ai-core/
├── app/
│   ├── main.py                 # FastAPI Entrypoint, Middleware & CORS
│   ├── core/
│   │   ├── config.py            # Configuration Settings
│   │   ├── security.py          # Tenant API-Key Auth, Token Limits & Rate Limiter
│   │   └── cache.py             # Redis / SQLite Semantic Cache Engine
│   ├── api/
│   │   └── v1/
│   │       ├── chat.py          # Unified Multi-LLM Chat Endpoint
│   │       ├── graph_rag.py     # GraphRAG Document Ingestion & Grounded Query
│   │       ├── external_search.py # Nara / Tavily Live Search
│   │       ├── agent.py         # AI Agent Loop & Function Calling
│   │       ├── vision.py        # Vision OCR & Semantic Analysis
│   │       └── audio.py         # Whisper STT & Text-to-Speech (TTS)
│   ├── services/
│   │   ├── llm_factory.py       # LiteLLM router + Fallback + Self-Correction
│   │   ├── graph_rag_service.py # GraphRAG & Document Extractor Service
│   │   ├── search_service.py    # Live Web Search Connector
│   │   └── agent_service.py     # Dynamic Tool Registry & Executor
│   └── schemas/
│       ├── requests.py          # Pydantic Request Models
│       └── responses.py         # Pydantic Response Models
├── static/
│   └── omnibrain-client.js      # Universal JS Client SDK
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## ⚙️ Quick Start

### 1. Setup Environment
Copy `.env.example` to `.env` and fill in your AI API keys:
```bash
cp .env.example .env
```

### 2. Install Dependencies & Run
Install requirements and start development server:
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to `http://127.0.0.1:8000/docs` to access the interactive Swagger API documentation.

### 3. Run with Docker
```bash
docker-compose up -d --build
```

---

## 🔌 Integrating Client SDK

Simply import the SDK in your web application and initialize it using your app's Client API Key:

```html
<script src="http://localhost:8000/static/omnibrain-client.js"></script>
<script>
  // Initialize SDK
  const client = new OmniBrainClient("http://localhost:8000", "demo_key");

  // Call Chat
  async function testChat() {
      const response = await client.chat([
          { role: "user", content: "Hello OmniBrain!" }
      ]);
      console.log(response.text);
  }
</script>
```
