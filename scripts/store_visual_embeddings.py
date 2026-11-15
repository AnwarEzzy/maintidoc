import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core.schema import ImageNode
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.engine import URL


EMBEDDING_MODEL = "open_clip:ViT-B-32:laion2b_s34b_b79k"
EMBEDDING_DIMENSION = 512
PREPROCESSING = "white_background_square_pad_v1"

parser = argparse.ArgumentParser(
    description="Store CLIP visual embeddings in PostgreSQL."
)
parser.add_argument(
    "embeddings_json",
    type=Path,
    help="Path to the JSON file containing CLIP embeddings.",
)

args = parser.parse_args()
input_path = args.embeddings_json.resolve()

if not input_path.is_file():
    parser.error(f"JSON file not found: {input_path}")

records = json.loads(input_path.read_text(encoding="utf-8"))

if not isinstance(records, list) or not records:
    parser.error("Expected a non-empty list of visual records.")

project_root = Path(__file__).resolve().parents[1]
nodes = []

for position, record in enumerate(records, start=1):
    if record["embedding_model"] != EMBEDDING_MODEL:
        parser.error(f"Visual {position}: unexpected embedding model.")

    if record["preprocessing"] != PREPROCESSING:
        parser.error(f"Visual {position}: unexpected preprocessing.")

    vector = record["embedding"]

    if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION:
        parser.error(f"Visual {position}: expected 512 dimensions.")

    if not all(
        type(value) in (int, float) and math.isfinite(value)
        for value in vector
    ):
        parser.error(f"Visual {position}: invalid embedding values.")

    if not math.isclose(
        sum(value * value for value in vector),
        1.0,
        abs_tol=1e-4,
    ):
        parser.error(f"Visual {position}: embedding is not normalized.")

    image_path = (project_root / record["image_path"]).resolve()

    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")

    identity = json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    node_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    nodes.append(
        ImageNode(
            id_=node_id,
            image_path=str(image_path),
            image_mimetype="image/png",
            embedding=vector,
            metadata={
                **record["metadata"],
                "visual_id": record["visual_id"],
                "image_path": record["image_path"],
                "embedding_model": EMBEDDING_MODEL,
                "preprocessing": PREPROCESSING,
            },
        )
    )

node_ids = [node.node_id for node in nodes]

if len(set(node_ids)) != len(node_ids):
    parser.error("The input contains duplicate visual records.")

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
    table_name="maintidoc_visuals_clip",
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

    print(f"Visuals loaded: {len(nodes)}")
    print(f"Already stored: {len(existing_ids)}")
    print(f"New visuals stored: {len(new_nodes)}")
    print(f"Verified stored visuals: {len(stored_nodes)}")
    print("Table: public.data_maintidoc_visuals_clip")
finally:
    asyncio.run(vector_store.close())