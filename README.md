# AI Analyst Agent

A production-grade AI analyst that turns natural-language questions into
**accurate, source-backed insights** by routing across MongoDB, MySQL,
and a FAISS vector index, then asking Claude to *explain* (not compute)
the results.

> **Critical design rule:** the LLM never computes business numbers.
> Aggregations are always run by the database. The LLM only summarizes.

---

## Architecture

```
                 ┌──────────────────────────┐
 user question ─▶│  Routing Service (rules) │
                 └──────────────┬───────────┘
                                │
            ┌───────────────────┼─────────────────────┐
            ▼                   ▼                     ▼
      ┌──────────┐        ┌──────────┐          ┌─────────────┐
      │ Mongo    │        │ MySQL    │          │ FAISS       │
      │ (aggs)   │        │ (aggs)   │          │ (semantic)  │
      └────┬─────┘        └────┬─────┘          └──────┬──────┘
           └────────┬──────────┘                       │
                    ▼                                  │
              ┌──────────────────────────────┐         │
              │ Context Builder              │◀────────┘
              └────────────────┬─────────────┘
                               ▼
                       ┌────────────────┐
                       │ Claude (LLM)   │  ← only explains numbers
                       └────────┬───────┘
                                ▼
                       ┌────────────────┐
                       │ Charts (mpl/   │  ← built from DB data only
                       │ plotly)        │
                       └────────┬───────┘
                                ▼
                          AnalystResponse
```

### Components

| Module | File | Responsibility |
| --- | --- | --- |
| Ingestion | `services/ingestion_service.py` | Excel/CSV/JSON → Mongo → row text → FAISS |
| Embeddings | `services/embedding_service.py` | Local `sentence-transformers`, L2-normalized |
| Vector store | `services/vector_service.py` | FAISS `IndexFlatIP` + JSON metadata sidecar |
| MongoDB | `services/mongo_service.py` | Aggregation pipelines (sum/avg/count/group-by) |
| MySQL | `services/mysql_service.py` | Read-only SQL + safe aggregation builder |
| Routing | `services/routing_service.py` | Analytical / Semantic / Hybrid classifier |
| Context | `services/context_service.py` | Builds the LLM prompt body |
| LLM | `services/agent_service.py` | Anthropic Claude wrapper with retries |
| Charts | `services/chart_service.py` | matplotlib (PNG + base64) + Plotly JSON |
| Orchestrator | `services/analyst_service.py` | Single pipeline: route → fetch → explain → chart |
| API | `api/routes.py`, `main.py` | FastAPI surface |

---

## Project layout

```
ai_analyst_agent/
├── api/
│   └── routes.py
├── services/
│   ├── agent_service.py
│   ├── analyst_service.py
│   ├── chart_service.py
│   ├── context_service.py
│   ├── embedding_service.py
│   ├── ingestion_service.py
│   ├── mongo_service.py
│   ├── mysql_service.py
│   ├── routing_service.py
│   └── vector_service.py
├── models/
│   ├── enums.py
│   └── schemas.py
├── utils/
│   ├── config.py
│   ├── exceptions.py
│   └── logger.py
├── scripts/
│   ├── init_mysql.py
│   └── seed_data.py
├── data/
│   ├── uploads/
│   ├── charts/
│   └── faiss_index/
├── main.py
├── requirements.txt
└── .env
```

---

## Quickstart

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env .env
# fill in ANTHROPIC_API_KEY (and DB creds if you're not using defaults)
```

### 2. Start the dependencies (local Docker example)

```bash
docker run -d --name mongo -p 27017:27017 mongo:7
docker run -d --name mysql -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD= -e MYSQL_ALLOW_EMPTY_PASSWORD=1 \
  -e MYSQL_DATABASE=ai_analyst mysql:8
```

### 3. (Optional) Seed sample data

```bash
python -m scripts.init_mysql     # creates `sales` table in MySQL
python -m scripts.seed_data      # creates Excel + ingests to Mongo + FAISS
```

### 3b. AWQAF JSON (flatten in memory, files unchanged)

Loads root `contents.json` into `awqaf_catalog`, and each `AWQAF-DATA/<service>/`
folder into `awqaf_<service>`: `YYYY.json`, `all_records.json`, `glossary.json`,
plus ``directory.json`` (``by_emirate`` + ``mosques[]``), any JSON whose root
is `{ "centers": [ ... ] }` (Quran centers exports), and all ``YYYY.json``
variants (top-level ``months``; ``data[]`` occupancy / petition-style;
``campaigns[]`` / ``mosques[]`` / ``countries[]`` / ``centers[]``; zakat
``channels[]`` / ``projects[]`` with nested ``months``). FAISS vectors link to
Mongo `document_id` as usual.

```bash
python -m scripts.ingest_awqaf --replace          # full reload
python -m scripts.ingest_awqaf --dry-run          # count rows only
python -m scripts.ingest_awqaf --replace --reset-faiss   # clear FAISS first
```

The flatten rules live in `services/awqaf_normalize.py` and the writer
(`Mongo + FAISS`) lives in `services/awqaf_ingest.py`. **Both** the disk
script above **and** the HTTP endpoint `POST /api/v1/awqaf/ingest` share
this single pipeline:

```bash
curl -s http://localhost:8000/api/v1/awqaf/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "service": "hajj-package-service",
    "source_file": "2026.json",
    "payload": { "year": 2026, "dataset": "...", "months": { "january": {"website_transactions": 57}}},
    "replace": false,
    "dry_run": true
  }' | jq
```

### 4. Run the API

```bash
python main.py
# or: uvicorn main:app --reload
```

Open **http://localhost:8000/** for the chat UI, or http://localhost:8000/docs for Swagger.

---

## API

### `POST /api/v1/ingest`

Upload an Excel/CSV/JSON file. Stores **one Mongo document per tabular row**
and indexes a **text line built from that row** in FAISS with metadata
`collection` + `document_id` (Mongo `_id`).

**JSON rules today:** `pandas.read_json` must produce a **DataFrame** (typical
case: a **JSON array of objects**). Nested year files like `2026.json` with a
`months` object are **not** auto-flattened by this endpoint; use
`python -m scripts.ingest_awqaf` instead (or pre-flatten to an array file).

```bash
curl -F "file=@./data/uploads/sample_sales.xlsx" \
     -F "collection=sample_sales" \
     -F "replace=true" \
     http://localhost:8000/api/v1/ingest
```

### `POST /api/v1/analyze`

Run the full pipeline.

```bash
curl -s http://localhost:8000/api/v1/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What is the total amount by region?",
    "collection": "sample_sales"
  }' | jq
```

Response includes:
- `routing` — the routing decision (analytical / semantic / hybrid + matched keywords)
- `structured_data` — DB-computed rows (this is the source of truth)
- `vector_context` — top-k FAISS hits with source `collection#document_id`
- `insight` — Claude's Markdown analyst summary
- `chart` — base64 PNG + on-disk path + Plotly JSON spec

### `GET /api/v1/route?question=...`

Preview which path a question takes without executing it. Great for debugging.

### `GET /api/v1/search?q=...&top_k=5`

Raw FAISS search.

### `GET /api/v1/collections`

List available Mongo collections and MySQL tables.

### `GET /api/v1/health`

Liveness + index size.

---

## Routing examples

| Question | Route | Why |
| --- | --- | --- |
| "Total amount by region" | analytical | `total` + `by` |
| "Average quantity per category" | analytical | `average` + `per` |
| "Top 3 products by sales" | analytical | `top N` + `by` |
| "Why are eastern sales growing?" | semantic | `why` + no aggregation |
| "Total revenue by region and explain the trend" | hybrid | aggregation + `explain`/`trend` |

---

## Safety guarantees

- **No hallucinated numbers.** The LLM is shown only DB-computed rows; the
  system prompt forbids re-computation. Charts come from the same rows.
- **Read-only SQL.** `MySQLService.run_sql` rejects anything that isn't a
  single `SELECT`; the aggregation builder is fully parameterised.
- **Vector → source link.** Every FAISS vector carries the source Mongo
  collection and `document_id` so retrieved text is always traceable.
- **Graceful LLM fallback.** If `ANTHROPIC_API_KEY` is missing or Claude is
  unreachable, `/analyze` still returns the structured data + chart with a
  fallback summary instead of failing.

---

## Production notes

- Configuration is environment-driven via `pydantic-settings`; no hardcoded
  secrets.
- Use a managed MongoDB (e.g. Atlas) and a managed MySQL in staging /
  production. Set `APP_ENV=production` to disable hot-reload.
- For larger corpora, swap `IndexFlatIP` for `IndexHNSWFlat` or `IndexIVFPQ`
  in `vector_service.py` — the rest of the system doesn't need to change.
- Add Prometheus metrics by wrapping `analyst_orchestrator.answer`.
- Add an auth middleware (e.g. API key or OAuth) before exposing publicly.

---

## License

MIT
