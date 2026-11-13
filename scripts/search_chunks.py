import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.engine import URL


EMBEDDING_DIMENSION = 1024
EMBEDDING_URL = "http://127.0.0.1:8080/embed"

parser = argparse.ArgumentParser(
    description="Search PDF passages using semantic similarity."
)
parser.add_argument(
    "question",
    help="Question used to search the document.",
)
parser.add_argument(
    "--top-k",
    type=int,
    default=3,
    help="Maximum number of passages to return (default: 3).",
)

args = parser.parse_args()
question = args.question.strip()

if not question:
    parser.error("The question cannot be empty.")

if args.top_k < 1:
    parser.error("--top-k must be at least 1.")

project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")

password = os.getenv("POSTGRES_PASSWORD")

if not password:
    parser.error("POSTGRES_PASSWORD is missing from the environment.")

payload = {
    "inputs": [question],
    "normalize": True,
    "truncate": False,
}

request = Request(
    url=EMBEDDING_URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urlopen(request, timeout=120) as response:
        vectors = json.load(response)
except HTTPError as error:
    details = error.read().decode("utf-8", errors="replace")
    parser.exit(1, f"TEI returned HTTP {error.code}:\n{details}\n")
except (URLError, TimeoutError) as error:
    parser.exit(1, f"Cannot reach the embedding service:\n{error}\n")

if not isinstance(vectors, list) or len(vectors) != 1:
    parser.error("Unexpected TEI response.")

question_vector = vectors[0]

if (
    not isinstance(question_vector, list)
    or len(question_vector) != EMBEDDING_DIMENSION
):
    parser.error("Expected a question embedding with 1024 dimensions.")

if not all(
    type(value) in (int, float) and math.isfinite(value)
    for value in question_vector
):
    parser.error("The question embedding contains invalid values.")

database_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("POSTGRES_USER", "maintidoc"),
    password=password,
    host="127.0.0.1",
    port=5433,
    database=os.getenv("POSTGRES_DB", "maintidoc"),
)

async_database_url = database_url.set(
    drivername="postgresql+asyncpg"
)

vector_store = PGVectorStore(
    connection_string=database_url.render_as_string(hide_password=False),
    async_connection_string=async_database_url.render_as_string(
        hide_password=False
    ),
    table_name="maintidoc_chunks",
    embed_dim=EMBEDDING_DIMENSION,
    use_jsonb=True,
    perform_setup=False,
    create_engine_kwargs={"connect_args": {"connect_timeout": 10}},
)

try:
    query = VectorStoreQuery(
        query_embedding=question_vector,
        similarity_top_k=args.top_k,
    )

    result = vector_store.query(query)

    print(f"Question: {question}")
    print(f"Passages found: {len(result.nodes or [])}")

    for rank, node in enumerate(result.nodes or [], start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Document: {node.metadata.get('document', 'Unknown')}")
        print(f"Page: {node.metadata.get('page', 'Unknown')}")
        print(f"Chunk: {node.metadata.get('chunk_index', 'Unknown')}")

        if result.similarities is not None:
            print(f"Similarity: {result.similarities[rank - 1]:.4f}")

        print(f"\n{node.text}")
finally:
    asyncio.run(vector_store.close())