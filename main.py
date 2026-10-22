import os
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

load_dotenv(override=True)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
doc_path = os.getenv("doc_path", "./docs")

app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    description="FastAPI wrapper around the uploaded Enterprise Knowledge Assistant RAG pipeline.",
    version="1.0.0",
)

rag_chains = None
docs_count = 0
splits_count = 0
init_lock = Lock()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


def load_documents(docspath):
    docs = []

    for file in os.listdir(docspath):
        path = os.path.join(docspath, file)
        if file.endswith(".txt"):
            loader = TextLoader(path)
            docs.extend(loader.load())
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(path)
            docs.extend(loader.load())
        elif file.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(path)
            docs.extend(loader.load())
        elif file.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(path)
            docs.extend(loader.load())
        else:
            continue

    return docs


def initialize_rag():
    global rag_chains, docs_count, splits_count

    if rag_chains is not None:
        return

    if not os.path.isdir(doc_path):
        raise RuntimeError(
            f"Document directory '{doc_path}' does not exist. "
            "Create it or set doc_path in .env."
        )

    docsx = load_documents(doc_path)

    if not docsx:
        raise RuntimeError(
            f"No supported documents found in '{doc_path}'. "
            "Add .txt, .pdf, .docx, .xlsx, or .pptx files."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    splits = splitter.split_documents(docsx)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    qdrant = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        location=":memory:",
        collection_name="my_collection",
    )

    bmretreiver = BM25Retriever.from_documents(splits)
    bmretreiver.k = 3

    dense_retriever = qdrant.as_retriever(search_kwargs={"k": 3})

    finalretriever = EnsembleRetriever(
        retrievers=[bmretreiver, dense_retriever],
        weights=[0.6, 0.4],
    )

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def reranking(question, result, top_k=3):
        texts = [doc.page_content for doc in result]
        pairs = [(question, text) for text in texts]
        scores = reranker.predict(pairs)
        scored_results = list(zip(result, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_results[:top_k]]

    openai = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant that answers questions based on the provided context.while answering provide the metadata as well",
        ),
        (
            "human",
            "{metadata}\n\n{context}\n\nQuestion: {question}\nAnswer:",
        ),
    ])

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def ch_metadata(docs):
        return "\n\n".join([
            f"Source: {doc.metadata.get('source')}" for doc in docs
        ])

    def retrieve_rank(question):
        result = finalretriever.invoke(question)
        return reranking(question, result, top_k=3)

    rag_chains = (
        {
            "context": RunnableLambda(retrieve_rank) | format_docs,
            "metadata": RunnableLambda(retrieve_rank) | ch_metadata,
            "question": RunnablePassthrough(),
        }
        | prompt
        | openai
        | StrOutputParser()
    )

    docs_count = len(docsx)
    splits_count = len(splits)


@app.get("/")
def root():
    return {
        "message": "Enterprise Knowledge Assistant API is running",
        "docs": "/docs",
        "endpoint": "POST /ask",
        "document_path": doc_path,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_initialized": rag_chains is not None,
        "document_path": doc_path,
        "documents_loaded": docs_count,
        "chunks_loaded": splits_count,
        "qdrant_mode": "in-memory",
    }


@app.post("/ask")
def ask(request: AskRequest):
    try:
        with init_lock:
            initialize_rag()

        answer = rag_chains.invoke(request.question)

        return {
            "question": request.question,
            "answer": answer,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
