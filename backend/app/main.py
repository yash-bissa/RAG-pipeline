import os
import re
import uuid
import json
import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


app = FastAPI(title="Local RAG Lab")


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")

COLLECTION = os.getenv("COLLECTION", "classroom_docs")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))


qdrant = AsyncQdrantClient(url=QDRANT_URL)


class Question(BaseModel):
    question: str
    top_k: int | None = None
    model: str | None = None


SYSTEM_PROMPT = """
You are a helpful RAG assistant.

Answer the user's question using only the provided context.

Answer in simple bullet points suitable for a 12-year-old.

If the answer is not present in the context, say:
"I don't know based on the provided documents."

Do not make up information.
"""

def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()

    if name.endswith(".pdf"):
        import io

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith((".txt", ".md", ".csv")):
        return data.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Use PDF, TXT, MD or CSV.")


def chunk_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            cut = text.rfind(" ", start, end)

            if cut > start:
                end = cut

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - CHUNK_OVERLAP)

    return chunks


def with_prefix(text: str, is_query: bool = False) -> str:
    if EMBED_MODEL == "nomic-embed-text":
        prefix = "search_query: " if is_query else "search_document: "
        return prefix + text

    return text


async def embed(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBED_MODEL,
                "input": [with_prefix(t) for t in texts],
            },
        )

        response.raise_for_status()

        return response.json()["embeddings"]


async def embed_query(text: str) -> list[float]:
    vectors = await embed([with_prefix(text, is_query=True)])
    return vectors[0]


async def ensure_collection(vector_size: int):
    collections = await qdrant.get_collections()

    names = [c.name for c in collections.collections]

    if COLLECTION not in names:
        await qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ollama": OLLAMA_URL,
        "qdrant": QDRANT_URL,
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
    }


@app.get("/api/models")
async def models():
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{OLLAMA_URL}/api/tags")

        response.raise_for_status()

        return response.json()


@app.post("/api/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        data = await file.read()

        try:
            text = extract_text(file.filename, data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        chunks = chunk_text(text)

        if not chunks:
            results.append(
                {
                    "filename": file.filename,
                    "chunks": 0,
                }
            )
            continue

        vectors = await embed(chunks)

        await ensure_collection(len(vectors[0]))

        points = []

        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{file.filename}:{i}",
                )
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "source": file.filename,
                        "chunk": i,
                        "text": chunk,
                    },
                )
            )

        await qdrant.upsert(
            collection_name=COLLECTION,
            points=points,
        )

        results.append(
            {
                "filename": file.filename,
                "chunks": len(chunks),
            }
        )

    return {"results": results}


@app.get("/api/documents")
async def documents():
    if not await qdrant.collection_exists(COLLECTION):
        return {"documents": []}

    result = await qdrant.scroll(
        collection_name=COLLECTION,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )

    points = result[0]

    counts = {}

    for point in points:
        source = point.payload.get("source", "unknown")

        counts[source] = counts.get(source, 0) + 1

    return {
        "documents": [
            {
                "source": source,
                "chunks": count,
            }
            for source, count in counts.items()
        ]
    }


@app.delete("/api/documents")
async def delete_all_documents():
    if await qdrant.collection_exists(COLLECTION):
        await qdrant.delete_collection(COLLECTION)

    return {"status": "deleted"}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    if not await qdrant.collection_exists(COLLECTION):
        return {"status": "deleted"}

    await qdrant.delete(
        collection_name=COLLECTION,
        points_selector={
            "filter": {
                "must": [
                    {
                        "key": "source",
                        "match": {
                            "value": filename
                        },
                    }
                ]
            }
        },
    )

    return {
        "status": "deleted",
        "filename": filename,
    }


@app.post("/api/ask")
async def ask(question: Question):
    if not await qdrant.collection_exists(COLLECTION):
        raise HTTPException(
            status_code=400,
            detail="No documents have been ingested yet.",
        )

    query_vector = await embed_query(question.question)

    top_k = question.top_k or TOP_K

    result = await qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    context_parts = []

    sources = []

    for i, point in enumerate(result.points, start=1):
        text = point.payload.get("text", "")
        source = point.payload.get("source", "")
        chunk = point.payload.get("chunk", 0)

        context_parts.append(f"[{i}] {text}")

        sources.append(
            {
                "id": i,
                "source": source,
                "chunk": chunk,
                "score": point.score,
            }
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Context:

{context}

Question:

{question.question}
"""

    model = question.model or CHAT_MODEL

    async def generate():
        yield json.dumps(
            {
                "type": "sources",
                "sources": sources,
            }
        ) + "\n"

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "stream": True,
                },
            ) as response:

                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    data = json.loads(line)

                    message = data.get("message", {})
                    content = message.get("content")

                    if content:
                        yield json.dumps(
                            {
                                "type": "token",
                                "content": content,
                            }
                        ) + "\n"

                    if data.get("done"):
                        break

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )