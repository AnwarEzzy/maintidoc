import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.engine import URL


MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024

parser = argparse.ArgumentParser(
    description="Store existing chunk embeddings in PostgreSQL."
)
parser.add_argument(
    "embeddings_json",
    type=Path,
    help="Path to the JSON file containing chunks and embeddings.",
)

args = parser.parse_args()
input_path = args.embeddings_json.resolve()

if not input_path.is_file():
    parser.error(f"JSON file not found: {input_path}")

records = json.loads(input_path.read_text(encoding="utf-8"))

if not isinstance(records, list) or not records:
    parser.error("The input must contain a non-empty list of records.")

nodes = []

for position, record in enumerate(records, start=1):
    if record["embedding_model"] != MODEL_NAME:
        parser.error(f"Record {position}: unexpected embedding model.")

    vector = record["embedding"]

    if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION:
        parser.error(f"Record {position}: expected 1024 dimensions.")

    if not all(
        type(value) in (int, float) and math.isfinite(value)
        for value in vector
    ):
        parser.error(f"Record {position}: invalid embedding values.")

    if not isinstance(record["text"], str) or not record["text"].strip():
        parser.error(f"Record {position}: empty or invalid text.")

    identity = json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    node_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    nodes.append(
        TextNode(
            id_=node_id,
            text=record["text"],
            metadata={
                **record["metadata"],
                "chunk_index": record["chunk_index"],
                "embedding_model": record["embedding_model"],
            },
            embedding=vector,
        )
    )

node_ids = [node.node_id for node in nodes]

if len(set(node_ids)) != len(node_ids):
    parser.error("The input contains duplicate records.")

project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")

password = os.getenv("POSTGRES_PASSWORD")

if not password:
    parser.error("POSTGRES_PASSWORD is missing from the environment.")

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
    initialization_fail_on_error=True,
    create_engine_kwargs={"connect_args": {"connect_timeout": 10}},
)

try:
    existing_nodes = vector_store.get_nodes(node_ids=node_ids)
    existing_ids = {node.node_id for node in existing_nodes}

    new_nodes = [
        node for node in nodes
        if node.node_id not in existing_ids
    ]

    if new_nodes:
        vector_store.add(new_nodes)

    stored_nodes = vector_store.get_nodes(node_ids=node_ids)

    if len(stored_nodes) != len(nodes):
        parser.error("Verification failed: unexpected stored node count.")

    print(f"Chunks loaded: {len(nodes)}")
    print(f"Already stored: {len(existing_ids)}")
    print(f"New chunks stored: {len(new_nodes)}")
    print(f"Verified stored chunks: {len(stored_nodes)}")
    print("Table: public.data_maintidoc_chunks")
finally:
    asyncio.run(vector_store.close())