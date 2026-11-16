import argparse
import asyncio
import os
from functools import lru_cache
from pathlib import Path

import open_clip
import torch
from dotenv import load_dotenv
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError

from scripts.embed_visuals import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    PREPROCESSING,
    PRETRAINED,
)


EMBEDDING_MODEL = f"open_clip:{MODEL_NAME}:{PRETRAINED}"


@lru_cache(maxsize=1)
def load_clip():
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device="cpu",
        precision="fp32",
    )
    model.eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    return model, tokenizer


def search_visuals(
    question: str,
    top_k: int = 3,
    document: str | None = None,
) -> VectorStoreQueryResult:
    question = question.strip()

    if not question:
        raise ValueError("The question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        raise ValueError(
            "POSTGRES_PASSWORD is missing from the environment."
        )

    model, tokenizer = load_clip()

    token_count = len(tokenizer.encode(question)) + 2

    if token_count > tokenizer.context_length:
        raise ValueError(
            "The question is too long for this CLIP model. "
            f"Use at most {tokenizer.context_length - 2} text tokens."
        )

    tokens = tokenizer([question])

    with torch.inference_mode():
        features = model.encode_text(
            tokens,
            normalize=True,
        )

    if tuple(features.shape) != (1, EMBEDDING_DIMENSION):
        raise ValueError("Unexpected question embedding shape.")

    if not torch.isfinite(features).all().item():
        raise ValueError("Invalid question embedding values.")

    question_vector = features[0].tolist()

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
        connection_string=database_url.render_as_string(
            hide_password=False
        ),
        async_connection_string=async_database_url.render_as_string(
            hide_password=False
        ),
        table_name="maintidoc_visuals_clip",
        embed_dim=EMBEDDING_DIMENSION,
        use_jsonb=True,
        perform_setup=False,
        create_engine_kwargs={
            "connect_args": {"connect_timeout": 10}
        },
    )

    try:
        conditions = [
            MetadataFilter(key="embedding_model", value=EMBEDDING_MODEL),
            MetadataFilter(key="preprocessing", value=PREPROCESSING),
        ]
        if document:
            conditions.append(MetadataFilter(key="document", value=document))

        query = VectorStoreQuery(
            query_embedding=question_vector,
            similarity_top_k=top_k,
            filters=MetadataFilters(filters=conditions),
        )

        return vector_store.query(query)
    finally:
        asyncio.run(vector_store.close())


def main():
    parser = argparse.ArgumentParser(
        description="Find PDF visuals using a CLIP text query."
    )
    parser.add_argument(
        "question",
        help="Short text describing the visual to find.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of results (default: 3).",
    )

    args = parser.parse_args()

    try:
        result = search_visuals(args.question, args.top_k)
    except (
        ValueError,
        RuntimeError,
        OSError,
        SQLAlchemyError,
    ) as error:
        parser.exit(1, f"Error: {error}\n")

    project_root = Path(__file__).resolve().parents[1]

    print(f"Question: {args.question}")
    print(f"Visuals found: {len(result.nodes or [])}")

    for rank, node in enumerate(result.nodes or [], start=1):
        metadata = node.metadata
        image_path = (
            project_root / metadata["image_path"]
        ).resolve()

        print(f"\n--- Result {rank} ---")
        print(f"Document: {metadata['document']}")
        print(f"Page: {metadata['page']}")
        print(f"Type: {metadata['content_type']}")
        print(f"Image: {image_path}")

        if result.similarities is not None:
            print(f"Similarity: {result.similarities[rank - 1]:.4f}")

        if "review_status" in metadata:
            print(f"Table review status: {metadata['review_status']}")


if __name__ == "__main__":
    main()
