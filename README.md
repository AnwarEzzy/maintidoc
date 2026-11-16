# MaintiDoc

A multimodal document assistant for maintenance technicians. Upload a PDF,
ask a question, and inspect the cited excerpts, figures and original pages.

**Milestone 1:** local multimodal RAG with a LangChain search agent and a
Streamlit interface. The application does not control equipment. Maintenance
decisions remain with authorized technicians.

## What works

- PDF text ingestion through LlamaIndex's PyMuPDF reader; figures and table
  candidates extracted with PyMuPDF and represented as LlamaIndex documents.
- BGE-M3 text embeddings and CLIP image embeddings in separate PGVector tables.
- Qwen3-VL captions with an automatic image/caption consistency filter;
  only retained captions enter the text index.
- LangChain selects passage, caption or CLIP search for the active PDF.
- A separate grounded generation call uses retrieved evidence and validates
  citation IDs. It can decline questions unsupported by that evidence.
- Streamlit PDF upload, resumable indexing, document selection, chat, source
  excerpts, image display, original-page previews and PDF download.

## Stack

| Component | Implementation |
| --- | --- |
| Python environment | Python 3.12, uv, `.venv` |
| Document/index layer | LlamaIndex, PyMuPDF |
| Text embeddings | BAAI/bge-m3, 1024 dimensions, TEI CPU in Docker |
| Visual embeddings | OpenCLIP ViT-B-32 / laion2b_s34b_b79k, 512 dimensions, PyTorch CPU |
| Vector storage | PostgreSQL 17 + pgvector in Docker |
| Text LLM | `qwen3:4b-instruct-2507-q4_K_M` through native Ollama |
| Vision LLM | `qwen3-vl:2b-instruct-q4_K_M` through native Ollama |
| Agent / client | LangChain + ChatOllama / Streamlit |

FastAPI, Angular, fine-tuning, MCP, intervention history and tickets are outside
this milestone. Streamlit directly calls the Python application functions.

## Run locally — PowerShell

Install Python/uv, Docker Desktop and a version of Ollama supporting the model
tags below. Run the commands from the repository directory.

```powershell
uv sync --locked
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` and set a non-empty `POSTGRES_PASSWORD`. Keep that file private.
Create the database and embedding containers:

```powershell
docker compose up -d
docker compose ps
docker compose logs --tail 20 embeddings
```

Wait until PostgreSQL is healthy and TEI is ready. Subsequent starts can be
performed manually using Docker Desktop's **maintidoc** group. Starting Docker
Desktop alone does not automatically start these containers.

Start Ollama and download both models once:

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull qwen3-vl:2b-instruct-q4_K_M
```

Launch the application:

```powershell
uv run streamlit run app.py
```

Open the local URL printed by Streamlit, normally **http://127.0.0.1:8501**.
Upload `examples/maintenance_demo.pdf` and click **Indexer ce PDF**. This small,
fictional manual is included for a reproducible demonstration; it is not a real
maintenance procedure. The first CLIP use downloads its weights.

Suggested questions:

- « Que signifie le voyant STATUS orange du MX-10 ? »
- « Quels libellés sont visibles sur le panneau de commande MX-10 ? »
- « Montre l'image du panneau de commande MX-10. »
- « Quelle est la capitale de l'Italie ? » — expected refusal.

The course PDF **Attention Is All You Need** can also be uploaded. It is not
bundled in the repository; the original paper is available on
[arXiv](https://arxiv.org/abs/1706.03762).

## Command-line use

Index a document without the web client:

```powershell
uv run python -X utf8 -m scripts.ingest_pdf examples/maintenance_demo.pdf
```

The report contains the internal `document` name, which includes a content hash.
Pass that exact name to scope the agent:

```powershell
uv run python -X utf8 -m scripts.agent "What is multi-head attention?" --document "attention is all you need.pdf"
```

The last example applies to the original paper indexed under its legacy name.
For new uploads, use the name in their indexing report. The individual scripts
used during the guided implementation remain available under `scripts/`.

## Validation

Offline tests exercise the real LangChain loop with a scripted model, citation
checks, document filters, ingestion resumption and the Streamlit UI:

```powershell
uv run python -X utf8 -m unittest discover -s tests -v
```

With services running and the maintenance demo indexed, run the live checks:

```powershell
uv run python -X utf8 -m scripts.evaluate
```

The report is saved to `data/processed/evaluation.json`. Checks cover expected
tools, source pages, selected keywords, refusal and document isolation. They do
not establish that every generated claim is correct.

## Documentation

- [Architecture and pipelines](docs/pipeline.md)
- [Understand LangChain step by step — French](docs/langchain-guide.md)
- [Technical note, requirements and limitations](docs/technical-note.md)
- [Demonstration checklist](docs/demo.md)
- [Qwen installation notes](docs/llm-setup.md)

Uploaded PDFs, generated artifacts, model caches and credentials are not
versioned. PostgreSQL and the TEI model cache persist in Docker volumes.
