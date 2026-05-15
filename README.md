# SHL Conversational Assessment Recommender

A FastAPI-based conversational backend that acts as an intelligent assistant for hiring managers to navigate the SHL assessment catalog. The application seamlessly understands hiring requirements, retrieves relevant assessments, and outputs structured recommendations using an OpenAI-compatible LLM routing.

## Key Features

1. **Deterministic Retrieval:** Uses BM25 + strict metadata filtering to search a local catalog snapshot (`shl_catalog.json`).
2. **Schema Compliance:** Guarantees proper structured JSON output with HTTP error fallback responses. The `/chat` endpoint *never* returns 500 errors to the client.
3. **Guardrails Layer:** Fast, pre-LLM check using heuristics and intent extraction to block prompt injections and off-topic questions.
4. **Resilient LLM Integration:** Migrated from Anthropic to an OpenRouter implementation to support cost-effective provider fallback and reliable model deployment.

## Running Locally

1. **Environment Setup:**
   ```bash
   python -m venv .env
   source .env/Scripts/activate  # On Windows
   pip install -r requirements.txt
   ```

2. **API Keys:**
   Create a `.env.local` file with your OpenRouter API key:
   ```env
   OPENROUTER_API_KEY=sk-or-...
   OPENROUTER_MODEL=google/gemini-2.5-flash  # Highly recommended
   ```

3. **Start the Server:**
   ```bash
   uvicorn app.main:app --port 8000
   ```
   The API will be accessible at `http://localhost:8000`.

## Testing

The project includes an extensive Pytest suite handling core schema validations, guardrail verification, multi-turn contexts, and intent matching. To run tests:
```bash
pytest tests/ -v
```
*(Note: tests rely on network calls to OpenRouter. A fast model like Gemini Flash is recommended to avoid rate limits and timeouts).*

## API Endpoints

- `GET /health` : Health check endpoint.
- `POST /chat` : Stateless conversational endpoint. Accepts full conversation history and returns valid JSON matching the evaluator schema.
