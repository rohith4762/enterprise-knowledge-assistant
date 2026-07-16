# Hybrid RAG Pipeline with Cross-Encoder Reranking

A Retrieval-Augmented Generation (RAG) system that answers questions over internal
company documents (policies, guides, technical docs) using hybrid retrieval
(dense + sparse) and cross-encoder reranking, with source-attributed answers.

## Problem

Internal documentation is scattered across multiple formats (PDF, DOCX, XLSX, PPTX),
and employees need fast, accurate answers with traceability back to the source
document — not just a plausible-sounding response. This project builds a retrieval
pipeline designed to surface the *correct* source document reliably, then generates
an answer grounded in that source with citation metadata.

## Architecture

```
Documents (PDF/DOCX/XLSX/PPTX)
        │
        ▼
  Document Loader (format-specific: PyPDFLoader, Docx2txtLoader, etc.)
        │
        ▼
  Recursive Chunking (1000 chars, 100 overlap)
        │
        ├─────────────────────┐
        ▼                     ▼
  Dense Retrieval        Sparse Retrieval
  (Qdrant + OpenAI        (BM25)
   embeddings)
        │                     │
        └──────────┬──────────┘
                    ▼
          Ensemble Retriever
          (weighted: 0.6 BM25 / 0.4 dense)
                    │
                    ▼
          Cross-Encoder Reranker
          (ms-marco-MiniLM-L-6-v2)
                    │
                    ▼
          LLM Answer Generation (GPT-4o-mini)
          + source metadata citation
```

## Key Design Decisions

- **Hybrid retrieval (BM25 + dense) over dense-only:** BM25 catches exact keyword
  matches (document IDs, policy names) that embedding similarity can miss; dense
  retrieval catches semantic/paraphrased queries. Combining both via a weighted
  ensemble (0.6 BM25 / 0.4 dense) covers both cases.
- **Cross-encoder reranking on top of the ensemble:** the ensemble retriever
  returns documents in top 3 reliably, but not always in the *correct order*.
  A cross-encoder rescoring pass, which jointly encodes the query and each
  candidate document (rather than comparing pre-computed embeddings), corrects
  ranking errors the first-stage retrievers make.
- **Source metadata returned with every answer:** every generated answer includes
  the source document it was grounded in, so answers are traceable and verifiable
  rather than opaque.

## Evaluation

Evaluated on 8 hand-labeled questions against TechNova's internal document set
(HR policies, security guides, engineering docs, finance policy), checking whether
the correct source document was retrieved and how it was ranked.

| Metric                       | Ensemble (BM25 + Dense) | + Cross-Encoder Reranking |
|-------------------------------|:-----------------------:|:---------------------------:|
| Hit-rate@3 (correct doc present in top 3) | 100% (8/8)   | 100% (8/8)                  |
| Top-1 accuracy (correct doc ranked first) | 87.5% (7/8)  | **100% (8/8)**               |

**Finding:** The ensemble retriever reliably surfaced the correct document within
its top 3 results, but reranking was necessary to consistently rank it *first* —
correcting 1 of 8 cases where an unrelated document outranked the correct source.
Reranking also reduced irrelevant documents appearing alongside the correct one in
several queries' top-3 results.

### Example: Before vs. After Reranking

**Query:** "How do I reset or rotate my password?"

| Rank | Before Reranking | After Reranking |
|------|-------------------|-------------------|
| 1 | Incident_Response_Guide.docx ❌ | Password_MFA_Policy.pdf ✅ |
| 2 | Leave_Policy.docx (irrelevant) | Password_MFA_Policy.pdf ✅ |
| 3 | Incident_Response_Guide.docx | Incident_Response_Guide.docx |

## Example Output

**Query:** "What should I do if my laptop is stolen?"

> If your laptop is stolen, you should take the following steps:
> 1. Report it immediately — notify the Security team and IT as soon as possible.
> 2. Credential revocation — the Security team will revoke VPN and MFA credentials
>    tied to the device.
> 3. Remote wipe — IT will initiate a remote wipe of the device if it is enrolled
>    in device management.
>
> **Source:** Incident Response Guide (SEC-DOC-002)

## Tech Stack

- **Orchestration:** LangChain (LCEL chains)
- **Dense retrieval:** Qdrant + OpenAI `text-embedding-3-large`
- **Sparse retrieval:** BM25 (`rank_bm25`)
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (Sentence Transformers)
- **Generation:** GPT-4o-mini
- **Document loaders:** PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader,
  UnstructuredPowerPointLoader, TextLoader

## Known Limitations

- **In-memory vector store:** the Qdrant instance runs with `location=":memory:"`
  for this demo, so the index does not persist between runs. A production
  deployment would use a persistent or managed Qdrant instance.
- **Small evaluation set:** the 8-question evaluation set is sufficient to compare
  retrieval strategies directionally but is not large enough to draw statistically
  robust conclusions. A larger labeled set (30-50+ questions) would strengthen
  these results.
- **No answer-faithfulness scoring yet:** evaluation currently measures whether
  the *correct document* was retrieved, not whether the generated answer is fully
  faithful to that document's content. This is a planned next step.
- **OpenAI dependency:** embeddings and generation currently use OpenAI's API,
  which introduces cost and external dependency. A future iteration could swap
  in a local embedding model for cost/latency comparison.

## Next Steps

- [ ] Expand evaluation set to 30+ labeled questions
- [ ] Add answer-faithfulness / groundedness scoring
- [ ] Move vector store from in-memory to persistent Qdrant instance
- [ ] Wrap pipeline in a FastAPI service for interactive querying
- [ ] Compare OpenAI embeddings vs. a local/open-source embedding model

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with:
```
OPENAI_API_KEY=your_key_here
QDRANT_URL=your_qdrant_url        # optional, omit to use in-memory
QDRANT_API_KEY=your_qdrant_key    # optional
```

## Usage

Place source documents (PDF/DOCX/XLSX/PPTX/TXT) in a folder and point `doc_path`
to it, then run the notebook cells in order: load → chunk → embed → build
retrievers → query.

## Project Status

🟢 Active — evaluation and API wrapper planned as next iterations.
