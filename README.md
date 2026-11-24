# AI Response Validation System with Hallucination Detection

Evaluates an AI-generated answer against trusted evidence (an HDFC ERGO policy
PDF and an insurance CSV dataset) using five LLM judge agents served by Groq:
**Relevance, Accuracy, Hallucination, Completeness** and a final **Verdict**.

* **Backend:** FastAPI, Groq agents, lightweight RAG (ONNX embeddings via
  `fastembed` + numpy cosine search, combined with keyword search)
* **Frontend:** React + Vite + Tailwind + Recharts

## Project layout

```
backend/
  main.py            API endpoints, CORS, startup
  security.py        rate limiting + admin-token guard
  agents/            the five judge agents + Groq client
  rag/               loader, chunker, embeddings, index store, retriever
  evidence/          chooses PDF vs CSV evidence per question
  dataset_*.py       insurance CSV loading / queries (stdlib only)
  uploads/           policy PDF + CSV dataset
frontend/            React dashboard
build.sh             Render build step (installs + pre-builds search index)
render.yaml          optional Render Blueprint
```

## Run locally

See the bottom of [DEPLOY.md](DEPLOY.md), and `.env.example` for settings.

## Deploy (free) on Render

Step-by-step guide: [DEPLOY.md](DEPLOY.md).

## Tests

```bash
pip install pytest
pytest backend/tests
```
(`test_groq.py`, `test_accuracy.py` etc. in the repo root are manual scripts
that call the live Groq API.)
