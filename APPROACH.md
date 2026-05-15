# Architectural Approach & Tradeoffs

## 1. System Design

The SHL Conversational Assessment Recommender acts as a stateless API bridge between the user's conversation and the SHL Assessment Catalog. Our focus was on deterministic retrieval, strict schema compliance, and cost-effective LLM execution.

## 2. Core Decisions & Trade-offs

### A. Strict Retrieval (BM25) over Vector Databases
**Decision:** We maintain a deterministic `BM25` retrieval mechanism combined with exact metadata matching (`job_level`, `test_types`), operating directly against a committed local snapshot of `shl_catalog.json`.
**Trade-off:** While embedding-based Vector Search (like Pinecone/FAISS) could provide better semantic nuance, it adds significant infrastructural latency, dependency bloat (like `sentence-transformers`), and unpredictable hallucination vectors. BM25 provides immediate, predictable term-frequency matches without the cost of embeddings.

### B. Stateless API vs. Stateful Session Management
**Decision:** The application honors the strict stateless nature requested by the assignment (`POST /chat`), passing the entire conversation history with every request.
**Trade-off:** This incurs higher token costs as the conversation lengthens. To combat this, we cap conversation context limits and apply strict instructions to "not stretch conversations". Managing server-side sessions (e.g., Redis) was explicitly avoided to maintain the simplicity of the deployment architecture.

### C. OpenRouter Migration over Native Providers
**Decision:** LLM orchestration was migrated from a hardcoded Anthropic connection to OpenRouter using the OpenAI-compatible standard.
**Trade-off:** Using an abstraction layer like OpenRouter introduces a middle-man API constraint. However, the value lies in complete model flexibility (being able to hot-swap from Claude 3.5 Sonnet to Gemini 2.5 Flash without changing code) to avoid credit exhaustion and mitigate rate-limits, which were the primary causes of deployment blockage.

### D. JSON Extraction Layer
**Decision:** Instead of using complex structured generation abstractions (like `Instructor` or `LangChain`), we prompt the model to output a fenced ````json`` block at the end of its response and use a resilient regex extraction layer.
**Trade-off:** This places a burden on the system prompt to enforce formatting. However, our parser validates the extracted structure and URLs directly against our `local_lookup` cache of the catalog. If an error occurs, it falls back gracefully rather than throwing a 500 error.

### E. Explicit Pre-LLM Guardrails
**Decision:** An explicit `guardrails.py` intercepts "off-topic" or "injection" attempts via Regex/Heuristics *before* the request reaches the LLM.
**Trade-off:** Deterministic heuristics might occasionally capture false positives compared to an LLM-based "moderator" layer, but this approach ensures immediate refusal with zero API cost and minimal latency.

## 3. Deployment Constraints

The `shl_catalog.json` file is explicitly committed to version control. This decision circumvents the fragility of downloading data dynamically at build/runtime. The application is designed to be easily deployed on standard PAAS environments (Heroku, Render) using the provided `Procfile`.
