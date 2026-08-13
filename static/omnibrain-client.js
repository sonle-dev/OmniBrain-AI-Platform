/**
 * OmniBrain AI Core Universal Client SDK
 * Lightweight wrapper for LMS, E-commerce, Tourism web apps to interface with the centralized AI Brain.
 */
class OmniBrainClient {
    /**
     * @param {string} baseURL - Root URL of the OmniBrain Core service (e.g., http://localhost:8000)
     * @param {string} apiKey - The authorization X-API-Key
     */
    constructor(baseURL, apiKey) {
        this.baseURL = baseURL.replace(/\/$/, ""); // Strip trailing slash
        this.apiKey = apiKey;
    }

    /**
     * Helper to make HTTP POST requests
     * @private
     */
    async _request(path, body, isBlob = false) {
        const headers = {
            "Content-Type": "application/json",
            "X-API-Key": this.apiKey
        };

        const response = await fetch(`${this.baseURL}${path}`, {
            method: "POST",
            headers: headers,
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            const errDetail = await response.text();
            throw new Error(`OmniBrain Error [${response.status}]: ${errDetail || response.statusText}`);
        }

        if (isBlob) {
            return await response.blob();
        }
        return await response.json();
    }

    /**
     * Helper to handle file uploads (multipart/form-data)
     * @private
     */
    async _uploadRequest(path, file, formField = "file") {
        const headers = {
            "X-API-Key": this.apiKey
            // Let the browser set Content-Type with multipart boundaries automatically
        };

        const formData = new FormData();
        formData.append(formField, file);

        const response = await fetch(`${this.baseURL}${path}`, {
            method: "POST",
            headers: headers,
            body: formData
        });

        if (!response.ok) {
            const errDetail = await response.text();
            throw new Error(`OmniBrain Upload Error [${response.status}]: ${errDetail || response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Standard Chat Completions
     * @param {Array<{role: string, content: string}>} messages - Conversation history
     * @param {Object} [options] - Optional parameters: model, temperature, max_tokens, bypass_cache
     */
    async chat(messages, options = {}) {
        return this._request("/api/v1/chat/completions", {
            messages,
            model: options.model || null,
            temperature: options.temperature !== undefined ? options.temperature : 0.2,
            max_tokens: options.max_tokens || null,
            bypass_cache: !!options.bypass_cache
        });
    }

    /**
     * Queries internal documents using GraphRAG (Knowledge Graph)
     * @param {string} query - Natural language question
     * @param {string} [mode="local"] - Query mode: 'local' or 'global'
     */
    async queryGraphRAG(query, mode = "local") {
        return this._request("/api/v1/graph-rag/query", {
            query,
            mode
        });
    }

    /**
     * Uploads a document (PDF, Slides PPTX, Text) to update the Knowledge Graph GraphRAG index
     * @param {File|Blob} file - The file to upload
     */
    async uploadDocument(file) {
        return this._uploadRequest("/api/v1/graph-rag/upload", file, "file");
    }

    /**
     * Searches live web data using Nara/Tavily Search Engine
     * @param {string} query - Natural language query
     * @param {number} [maxResults=5] - Number of results to fetch
     */
    async searchWeb(query, maxResults = 5) {
        return this._request("/api/v1/search/query", {
            query,
            max_results: maxResults
        });
    }

    /**
     * Runs an AI Agent loop that dynamically executes tools
     * @param {string} prompt - Instructions/goal for the agent
     * @param {Array<{role: string, content: string}>} [history=[]] - Previous conversations
     * @param {string} [model] - Override model (e.g. gemini/gemini-2.0-flash)
     */
    async executeAgent(prompt, history = [], model = null) {
        return this._request("/api/v1/agent/run", {
            prompt,
            history,
            model
        });
    }

    /**
     * Analyzes an image with vision OCR or semantic analysis
     * @param {string} prompt - Prompt instruction (e.g. "read the text")
     * @param {Object} source - Either { image_url } or { image_b64 }
     */
    async analyzeImage(prompt, source = {}) {
        return this._request("/api/v1/vision/analyze", {
            prompt,
            image_url: source.image_url || null,
            image_b64: source.image_b64 || null
        });
    }

    /**
     * Transcribes audio speech to text (STT)
     * @param {File|Blob} audioFile - Audio file payload (e.g., mp3, wav, m4a)
     */
    async transcribeAudio(audioFile) {
        return this._uploadRequest("/api/v1/audio/transcribe", audioFile, "file");
    }

    /**
     * Synthesizes text to audio speech (TTS)
     * @param {string} text - Text to convert to speech
     * @param {string} [voice="alloy"] - Voice character
     * @param {number} [speed=1.0] - Speech velocity
     * @returns {Promise<Blob>} - Returns standard audio file Blob
     */
    async synthesizeSpeech(text, voice = "alloy", speed = 1.0) {
        return this._request("/api/v1/audio/synthesize", {
            text,
            voice,
            speed
        }, true);
    }
}

// Export for ES Modules/Browser compatibility
if (typeof module !== "undefined" && module.exports) {
    module.exports = OmniBrainClient;
} else {
    window.OmniBrainClient = OmniBrainClient;
}
