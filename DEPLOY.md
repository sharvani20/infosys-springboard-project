# Deploying on Render (free tier)

Two free services: a **Web Service** (FastAPI backend) and a **Static Site**
(React frontend). Free web services have 512 MB RAM, an ephemeral disk and
spin down after 15 minutes of inactivity (waking takes about a minute).

## What changed to fit in 512 MB

| Before | Now |
|---|---|
| `sentence-transformers` (PyTorch) + `chromadb` | `fastembed` (ONNX, no PyTorch) + a tiny numpy index |
| PDF embedded **at every startup** (disk is wiped on restart) | Index **pre-built during the build** (`build.sh`); startup is instant |
| `pandas` re-reading the CSV on every request | stdlib `csv`, parsed once and cached |
| `/api/ingest` readable-path endpoint open to the internet | needs `ADMIN_TOKEN`, only reads inside `backend/uploads/` |
| No limits on `/api/evaluate` (5 Groq calls each) | per-IP rate limit + daily cap |

## 1. Push to GitHub

Make sure `.env` is **not** committed (it is in `.gitignore`). If a real
key was ever committed, revoke it in the Groq console and create a new one.

## 2. Backend (Web Service)

Render Dashboard -> **New -> Web Service** -> pick the repo.

| Field | Value |
|---|---|
| Runtime | Python 3 |
| Instance type | **Free** |
| Build Command | `bash build.sh` |
| Start Command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/api/health` |

Environment variables:

| Key | Value |
|---|---|
| `GROQ_API_KEY` | your Groq key |
| `GROQ_MODEL` | the Groq model id you use locally |
| `FRONTEND_URL` | your static site URL (step 3), no trailing slash |
| `ADMIN_TOKEN` | optional; a long random string to enable `/api/ingest` |
| `RETRIEVAL_MODE` | `hybrid` (default) or `keyword` (lowest memory) |
| `RATE_LIMIT_PER_MINUTE` | default `5` |
| `DAILY_EVAL_CAP` | default `300` |

In the **build log** look for:
`Chunks stored: 99 | embeddings present: True`.

## 3. Frontend (Static Site)

**New -> Static Site** -> same repo.

| Field | Value |
|---|---|
| Build Command | `cd frontend && npm ci && npm run build` |
| Publish Directory | `frontend/dist` |
| `VITE_API_BASE_URL` | your backend URL, e.g. `https://xxxx.onrender.com` (no trailing slash) |
| `NODE_VERSION` | `22` (Vite 8 needs a recent Node) |

`VITE_API_BASE_URL` is baked in at build time: if you change it, redeploy
the static site.

## 4. Connect them

Copy the static site URL into the backend's `FRONTEND_URL`
(several origins may be comma-separated). Render restarts the backend.

## 5. Check

1. `https://<backend>/api/health` -> `{"status":"healthy"}`
2. Open the frontend, run an evaluation.

## Troubleshooting

* **Out of memory again** -> set `RETRIEVAL_MODE=keyword`. The API then never
  loads an embedding model (keyword search over the 99 chunks).
* **CORS error in the browser** -> `FRONTEND_URL` must exactly match the
  frontend origin (scheme + host, no path, no trailing slash).
* **"Too many requests" / daily limit** -> adjust `RATE_LIMIT_PER_MINUTE` /
  `DAILY_EVAL_CAP`. Counters reset when the service restarts or spins down.
* **First request after idle is slow** -> free-tier spin-up (~1 minute). Open
  the site a minute before a demo, or ping `/api/health` with a free cron
  service.
* **Build log says "index pre-build failed"** -> the API still starts:
  it builds the index itself and adds embeddings in the background.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # fill GROQ_API_KEY and GROQ_MODEL
python -m backend.rag.build_index
uvicorn backend.main:app --reload

cd frontend && npm install && npm run dev
```
