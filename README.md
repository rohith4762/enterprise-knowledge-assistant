# Enterprise Knowledge Assistant — Hybrid RAG

[Project GitHub](https://github.com/rohith4762/enterprise-knowledge-assistant) · Author: Rohith Muppi

Document-grounded question answering over TXT, PDF, DOCX, XLSX, and PPTX files, using hybrid retrieval, cross-encoder reranking, and a FastAPI service.

## Problem and intended user

Employees and internal support teams need answers buried in policies, onboarding documents, spreadsheets, and presentations. I built this assistant to retrieve relevant evidence across those files and generate answers with source references. Example questions include “What should I do if my laptop is stolen?” and “How do I install the VPN client?”

## What I implemented

- Format-specific document loading with source metadata retained.
- Recursive splitting into 1,000-character chunks with 100-character overlap.
- OpenAI `text-embedding-3-large` embeddings and in-memory Qdrant retrieval.
- BM25 and dense retrieval, combined through weighted rank fusion.
- Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- GPT-4o-mini generation through LangChain LCEL.
- A FastAPI wrapper with lazy initialization, health reporting, and Docker configuration.

## Architecture and design decisions

| Stage | Implementation |
| --- | --- |
| Ingestion | Load supported files directly inside `doc_path`; retain loader metadata. |
| Chunking | Split into 1,000-character chunks, overlapping by 100 characters. |
| Retrieval | BM25 and Qdrant each request three chunks. Fuse rankings with weights 0.6 BM25 / 0.4 dense. |
| Reranking | Score fused question–chunk pairs with a cross-encoder; retain three chunks. |
| Generation | Pass question, selected content, and source paths to GPT-4o-mini at temperature 0.2. |
| Serving | `POST /ask` initializes the pipeline on the first request and returns the answer. |

BM25 helps with exact policy terms and identifiers; dense retrieval helps with paraphrases. The ensemble weights control rank fusion, rather than blending raw BM25 and cosine scores. The cross-encoder jointly scores each question and candidate chunk to reduce irrelevant context before generation.

The current LCEL chain calls retrieval and reranking separately for context and metadata. A shared retrieval pass remains a future improvement. Source paths are included in the prompt; structured citation validation and reliable page, sheet, or slide references are not implemented.

## Repository layout

| File | Purpose |
| --- | --- |
| `main.py` | FastAPI wrapper and RAG pipeline |
| `requirements.txt` | Supplied runtime dependencies; most are unpinned |
| `Dockerfile` | Python 3.12 container serving port 8000 |
| `.env.example` | Local configuration template |
| `notebook/EKA.ipynb` | Original exploratory notebook, with saved outputs cleared |

## Run locally

Use Python 3.12. Create a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
mkdir -p docs
```

Set `OPENAI_API_KEY` in `.env`, then place your documents directly inside `docs/`. Set `doc_path` if using another directory. Do not commit credentials or private documents.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Interactive API documentation: <http://localhost:8000/docs>.

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What should I do if my laptop is stolen?"}'
```

The first question loads documents, embeds the corpus, builds the in-memory index, and loads the reranker. This requires network access, OpenAI credentials, and a model download. Embedding and generation calls incur API charges. `/health` reports process status and whether initialization has completed; `status: ok` alone does not verify that a question can be answered.

Although the code reads `QDRANT_URL` and `QDRANT_API_KEY`, the current vector-store initialization uses `location=":memory:"`. Remote Qdrant credentials are unnecessary and do not enable remote storage.

## Run with Docker

```bash
docker build -t enterprise-knowledge-assistant .
docker run --rm -p 8000:8000 --env-file .env \
  -e doc_path=/app/docs \
  -v "$PWD/docs:/app/docs:ro" \
  enterprise-knowledge-assistant
```

Documents and `.env` files are excluded from the image. Mount documents at runtime as shown above.

## Run the notebook

Install JupyterLab separately and run the cells in order:

```bash
python -m pip install jupyterlab
jupyter lab notebook/EKA.ipynb
```

The notebook contains exploratory inspection and installation cells. Use the API instructions for the service entry point. The original source corpus is not included, so answers depend on the documents you provide.

## Baseline versus improved retrieval

The supplied project PDF reports a **synthetic demonstration benchmark** of 100 questions across three variants, totaling 300 requests. These are historical reported results, not measurements reproduced by this repository update.

| Metric | Dense baseline | Hybrid | Hybrid + demo reranker |
| --- | ---: | ---: | ---: |
| Completed requests | 100/100 | 100/100 | 100/100 |
| Recall@3 | 90.00% | 93.33% | 94.44% |
| Precision@3 | 31.48% | 33.33% | 33.70% |
| MRR | 0.7444 | 0.8630 | 0.9074 |

The reported dense-to-final improvement is 4.44 percentage points in Recall@3 and 0.1630 in MRR.

### Methodology and limits

- Seven short synthetic policies; 84 single-source paraphrases, six multi-source questions, and ten unanswerable questions.
- One repeat per variant. File-level retrieval metrics use the 90 answerable questions.
- The final variant was labeled a **demo reranker**. These results do not establish the performance of the actual cross-encoder in `main.py`.
- Timing and cost in the supplied report were simulated; they are not live API latency or billed usage. Answer correctness, citation support, and refusal quality were not established.
- The benchmark corpus, labels, evaluator, and raw results were not supplied. Exact reproduction requires publishing those artifacts, including duplicate-file handling and metric definitions.

The previous README's eight-question aggregate scores are not retained as verified results because the supporting evaluation artifacts are not included.

## Failure examples from the documented notebook outputs

These historical observations are separate from the synthetic benchmark. Notebook outputs have been cleared in the published source.

| Question or issue | Observed behavior | Change or remaining work |
| --- | --- | --- |
| “What should I do if my laptop is stolen?” | Hybrid retrieval included unrelated leave-policy content. | Reranking removed that candidate from the top three, retaining incident-response and password/MFA evidence. |
| “How do I reset or rotate my password?” | Unreranked candidates included leave-policy content and ranked incident-response content first. | Reranking placed two password-policy chunks first and removed leave-policy content from the top three. |
| “What are the standard working hours?” | Reranked results still included a leave-policy chunk alongside handbook chunks. | Remaining limitation: reranking does not guarantee every selected chunk supports the answer. |

## Current limitations and next steps

- Qdrant is in memory; every process restart rebuilds the index. Multiple server workers initialize separate indexes.
- Retrieval and reranking run twice per question for context and metadata.
- No explicit evidence threshold, abstention policy, or answer-faithfulness evaluator.
- No OCR, access control, document-update handling, or persistent indexing.
- Repeated chunks from one document can consume context slots.
- No token/cost telemetry, cloud deployment artifacts, or supplied service test suite. AWS ECS/Fargate deployment and previously mentioned passing-test counts are not verified by these files.
- Most dependencies are unpinned. A tested lockfile and reproducible benchmark artifacts remain future work.

Next priorities are a shared retrieval pass, passage-level relevance labels, evidence-support and unanswerable-query evaluation, and recorded live latency and token usage.
