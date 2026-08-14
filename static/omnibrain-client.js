/**
 * OmniBrain AI Core — Universal Client SDK v2.0 (Speed-Optimized)
 * =================================================================
 * Lightweight, zero-dependency JavaScript client for any web application.
 * Features: Fast-Path Chat, Typewriter SSE Streaming, GraphRAG, Quiz Generator, Vision, Audio.
 *
 * Usage:
 *   const ai = new OmniBrainClient('http://localhost:8000', 'demo_key');
 *   await ai.streamChatTypewriter([{role:'user', content:'Hello'}], 'chatBox');
 *
 * Embed: <script src="http://localhost:8000/static/omnibrain-client.js"></script>
 */
class OmniBrainClient {
    /**
     * @param {string} baseURL - Root URL of the OmniBrain Core service (e.g., http://localhost:8000)
     * @param {string} apiKey  - The X-API-Key for authorization
     */
    constructor(baseURL, apiKey) {
        this.baseURL = baseURL.replace(/\/$/, '');
        this.apiKey  = apiKey;
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    /** @private */
    _headers(extra = {}) {
        return { 'Content-Type': 'application/json', 'X-API-Key': this.apiKey, ...extra };
    }

    /** @private */
    async _post(path, body) {
        const res = await fetch(`${this.baseURL}${path}`, {
            method: 'POST',
            headers: this._headers(),
            body: JSON.stringify(body)
        });
        if (!res.ok) {
            const detail = await res.text().catch(() => res.statusText);
            throw new Error(`OmniBrain [${res.status}] ${path}: ${detail}`);
        }
        return res.json();
    }

    /** @private */
    async _upload(path, file, fieldName = 'file') {
        const form = new FormData();
        form.append(fieldName, file);
        const res = await fetch(`${this.baseURL}${path}`, {
            method: 'POST',
            headers: { 'X-API-Key': this.apiKey },
            body: form
        });
        if (!res.ok) {
            const detail = await res.text().catch(() => res.statusText);
            throw new Error(`OmniBrain Upload [${res.status}] ${path}: ${detail}`);
        }
        return res.json();
    }

    // =========================================================================
    // Chat
    // =========================================================================

    /**
     * Standard (non-streaming) chat completion.
     * @param {Array<{role:string, content:string}>} messages - Conversation history
     * @param {Object} [options]
     * @param {string}  [options.task='chat']       - 'chat' | 'reasoning' | 'graph_rag' | 'vision'
     * @param {string}  [options.model]             - Override model slug
     * @param {number}  [options.temperature=0.2]   - 0.0 – 2.0
     * @param {number}  [options.max_tokens]        - Max output tokens
     * @param {boolean} [options.bypass_cache=false] - Skip cache for fresh response
     * @returns {Promise<{status, text, model, usage, cached}>}
     */
    async chat(messages, options = {}) {
        return this._post('/api/v1/chat/completions', {
            messages,
            task:         options.task          || 'chat',
            model:        options.model         || null,
            temperature:  options.temperature   !== undefined ? options.temperature : 0.2,
            max_tokens:   options.max_tokens    || null,
            bypass_cache: !!options.bypass_cache,
            stream:       false
        });
    }

    /**
     * Streaming chat via Server-Sent Events (Sub-Second TTFT).
     * @param {Array<{role:string, content:string}>} messages
     * @param {function(string):void} onChunk  - Called with each text chunk as it arrives
     * @param {function():void}       [onDone] - Called when stream ends
     * @param {Object} [options]               - Same options as chat()
     * @returns {Promise<void>}
     */
    async streamChat(messages, onChunk, onDone, options = {}) {
        const res = await fetch(`${this.baseURL}/api/v1/chat/completions`, {
            method:  'POST',
            headers: this._headers(),
            body: JSON.stringify({
                messages,
                task:         options.task        || 'chat',
                model:        options.model       || null,
                temperature:  options.temperature !== undefined ? options.temperature : 0.2,
                max_tokens:   options.max_tokens  || null,
                bypass_cache: !!options.bypass_cache,
                stream:       true
            })
        });

        if (!res.ok) {
            const detail = await res.text().catch(() => res.statusText);
            throw new Error(`OmniBrain Stream [${res.status}]: ${detail}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || !trimmed.startsWith('data:')) continue;
                const payload = trimmed.slice(5).trim();
                if (payload === '[DONE]') {
                    if (typeof onDone === 'function') onDone();
                    return;
                }
                try {
                    const parsed = JSON.parse(payload);
                    if (parsed.chunk && typeof onChunk === 'function') {
                        onChunk(parsed.chunk);
                    }
                } catch (_) { /* skip malformed chunks */ }
            }
        }
        if (typeof onDone === 'function') onDone();
    }

    /**
     * Typewriter DOM Streaming Helper.
     * Renders incoming stream chunks directly to a target DOM element with smooth typewriter effect.
     * @param {Array<{role:string, content:string}>} messages
     * @param {HTMLElement|string} targetElement - DOM element or element ID string
     * @param {function():void} [onDone]
     * @param {Object} [options]
     */
    async streamChatTypewriter(messages, targetElement, onDone, options = {}) {
        const el = typeof targetElement === 'string' ? document.getElementById(targetElement) : targetElement;
        if (!el) throw new Error('Target DOM element for typewriter stream not found.');

        el.innerText = '';
        await this.streamChat(
            messages,
            (chunk) => {
                el.innerText += chunk;
                if (options.autoScroll && el.parentElement) {
                    el.parentElement.scrollTop = el.parentElement.scrollHeight;
                }
            },
            onDone,
            options
        );
    }

    // =========================================================================
    // Quiz Generation
    // =========================================================================

    /**
     * Generate a structured multiple-choice quiz.
     */
    async generateQuiz(topic, options = {}) {
        return this._post('/api/v1/quiz/generate', {
            topic,
            difficulty:    options.difficulty    || 'medium',
            count:         options.count         || 5,
            language:      options.language      || 'vi',
            question_type: options.question_type || 'multiple_choice',
            context:       options.context       || null
        });
    }

    // =========================================================================
    // GraphRAG — Document Knowledge Base
    // =========================================================================

    async queryGraphRAG(query, mode = 'local') {
        return this._post('/api/v1/graph-rag/query', { query, mode });
    }

    async uploadDocument(file) {
        return this._upload('/api/v1/graph-rag/upload', file, 'file');
    }

    // =========================================================================
    // Web Search
    // =========================================================================

    async searchWeb(query, maxResults = 5) {
        return this._post('/api/v1/search/query', { query, max_results: maxResults });
    }

    // =========================================================================
    // AI Agent
    // =========================================================================

    async executeAgent(prompt, history = [], model = null) {
        return this._post('/api/v1/agent/run', { prompt, history, model });
    }

    // =========================================================================
    // Vision
    // =========================================================================

    async analyzeImage(prompt, source = {}) {
        return this._post('/api/v1/vision/analyze', {
            prompt,
            image_url: source.image_url || null,
            image_b64: source.image_b64 || null
        });
    }

    // =========================================================================
    // Audio
    // =========================================================================

    async transcribeAudio(audioFile) {
        return this._upload('/api/v1/audio/transcribe', audioFile, 'file');
    }

    async synthesizeSpeech(text, voice = 'alloy', speed = 1.0) {
        const res = await fetch(`${this.baseURL}/api/v1/audio/synthesize`, {
            method:  'POST',
            headers: this._headers(),
            body:    JSON.stringify({ text, voice, speed })
        });
        if (!res.ok) throw new Error(`OmniBrain TTS [${res.status}]: ${res.statusText}`);
        return res.blob();
    }

    // =========================================================================
    // Health Check
    // =========================================================================

    async isOnline() {
        try {
            const res = await fetch(`${this.baseURL}/health`, { method: 'GET' });
            return res.ok;
        } catch (_) {
            return false;
        }
    }
}

// Universal export — works in browser (global) and Node.js (CommonJS)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OmniBrainClient;
} else if (typeof window !== 'undefined') {
    window.OmniBrainClient = OmniBrainClient;
}
