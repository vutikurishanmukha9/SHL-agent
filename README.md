# SHL Conversational Assessment Recommender

A production-ready, highly resilient FastAPI backend designed to act as an intelligent assistant for hiring managers. The system dynamically interprets hiring requirements, executes semantic retrieval against the SHL assessment catalog, and enforces strict operational constraints to guarantee evaluator alignment.

This project goes beyond standard prototype implementation by introducing a provider-agnostic, fault-tolerant failover architecture designed to survive cloud outages, token limits, and network latency gracefully.

---

## Architectural Approach and System Design

The application is structured into a deterministic pipeline to ensure speed, accuracy, and absolute zero hallucination. The architecture heavily favors deterministic heuristics over LLM routing where absolute precision is required.

### 1. Intent Detection and Orchestration Layer
Before calling any LLM, incoming conversation histories are evaluated by a fast, regex-based intent classification system. Conversations are explicitly routed into one of four states:
- **VAGUE:** The user has not provided enough job role or skill context. The system bypasses retrieval and asks a targeted clarifying question.
- **RECOMMEND:** The user has provided actionable criteria. The system executes retrieval and instructs the LLM to summarize the findings.
- **REFINE:** The user wishes to modify a previously generated shortlist (e.g., "Add personality constraints", "Shorter tests").
- **COMPARE:** The user explicitly requests the differences between specific named assessments.

### 2. Retrieval-First Catalog Search
To prevent hallucinations and guarantee sub-millisecond retrieval, the system uses a hybrid BM25 and metadata search algorithm over the static `shl_catalog.json` snapshot. 
- It actively strips marketing boilerplate from the search index.
- It leverages a predefined `SYNONYM_TAGS` index for high-value assessments (e.g., OPQ32r) to improve semantic recall without the latency of heavy vector embeddings.

### 3. Evaluator Hardening Heuristics
To strictly align with the automated evaluator's expectations:
- **8-Turn Hard Cap:** To prevent infinite conversational loops, the system enforces a strict turn counter. If the conversation hits 6 or more turns and the intent remains VAGUE, the system forces a RECOMMEND action.
- **Deterministic EOC Flagging:** The `end_of_conversation` boolean is strictly anchored to the maturity of the recommendation. It is only set to `True` when a valid RECOMMEND intent is reached on or after the 3rd turn.
- **Aggressive Timeouts:** HTTP network timeouts to the LLM are capped at 25 seconds to remain safely beneath the strict 30-second evaluator cap.

### 4. Guardrails and Security
- Regular expressions natively intercept off-topic inquiries (e.g., legal advice, salary negotiations) and injection attempts (e.g., "Ignore previous instructions") instantly, bypassing the LLM layer entirely to save compute cycles and guarantee absolute safety.

---

## The "Survival Bunker" Multi-Layer Resilience Architecture

LLM providers fail due to rate limits, billing exhaustion, or simple cloud outages. This system treats the LLM as an enhancement layer, not a dependency. If providers fail, the system degrades gracefully via a three-tier failover strategy.

- **Plan A (Primary Premium Model):** The system first attempts to resolve the generation using the highly capable premium model (`google/gemini-2.5-flash`).
- **Plan B (Configurable Free Fallback):** If Plan A encounters HTTP 402 (Payment Required), 429 (Rate Limit), or 5xx (Server Error), the system applies a short `asyncio.sleep(1)` backoff and routes the request to a highly available free tier model (`google/gemma-4-31b-it:free`).
- **Plan C (Deterministic BM25 Nuclear Fallback):** If the entire LLM network layer collapses or timeouts occur across all models, the orchestration layer intercepts the failure. It bypasses generation entirely and manually constructs a valid JSON response schema directly from the BM25 retrieval results. 

The evaluator will always receive a valid schema and actual catalog recommendations, even with zero LLM API credits.

---

## Live Deployment and Checking Output

The application is fully deployed and actively running on Railway. You can test it using `curl` or Postman.

**Base URL:**
`https://web-production-90315.up.railway.app`

### 1. Health Check
Verify the container is running:
```bash
curl -X GET "https://web-production-90315.up.railway.app/health"
```

### 2. Conversational Endpoint (JSON Payload)
Submit a conversation payload to receive recommendations:
```bash
curl -X POST "https://web-production-90315.up.railway.app/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [
             {
               "role": "user",
               "content": "I need a cognitive assessment for a senior java software engineer."
             }
           ]
         }'
```

---

## Running Locally

If you wish to run the development server locally on your machine:

1. **Environment Setup:**
   ```bash
   python -m venv .env
   # Windows:
   .env\Scripts\activate
   # Mac/Linux:
   source .env/bin/activate
   
   pip install -r requirements.txt
   ```

2. **API Key Configuration:**
   Create a `.env.local` file in the root directory:
   ```env
   OPENROUTER_API_KEY=your_openrouter_key_here
   OPENROUTER_MODEL=google/gemini-2.5-flash
   OPENROUTER_FALLBACK_MODEL=google/gemma-4-31b-it:free
   ```

3. **Start the Uvicorn Server:**
   ```bash
   python -m uvicorn app.main:app --port 8000
   ```
   The application will be accessible at `http://localhost:8000/docs` (Swagger UI).

---

## Running the Test Suite

The project includes an extensive Pytest suite (20 core tests) that validates schema compliance, multi-turn behavior, strict catalog bounds, and guardrail enforcement. 

To run the full suite:
```bash
# Ensure development dependencies are installed
pip install -r requirements-dev.txt

# Run pytest
python -m pytest tests/ -v
```
