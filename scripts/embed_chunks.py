import argparse
import json
import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
EMBEDDING_URL = "http://127.0.0.1:8080/embed"


parser = argparse.ArgumentParser(
    description="Generate embeddings for text chunks using local TEI."
)

parser.add_argument(
    "chunks_json",
    type=Path,
    help="Path to the JSON file containing text chunks.",
)

args = parser.parse_args()
input_path = args.chunks_json.resolve()

if not input_path.is_file():
    parser.error(f"JSON file not found: {input_path}")

chunk_records = json.loads(input_path.read_text(encoding="utf-8"))

if not isinstance(chunk_records, list) or not chunk_records:
    parser.error("The input must contain a non-empty list of chunks.")

embedded_records = []

for position, chunk in enumerate(chunk_records, start=1):
    text = chunk["text"]

    if not isinstance(text, str) or not text.strip():
        parser.error(f"Chunk {position} contains no valid text.")

    payload = {
        "inputs": [text],
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
        parser.exit(
            1,
            f"\nChunk {position}: TEI returned HTTP {error.code}.\n"
            f"{details}\n",
        )
    except (URLError, TimeoutError) as error:
        parser.exit(
            1,
            f"\nChunk {position}: cannot reach TEI.\n"
            f"Check that the embeddings container is ready.\n{error}\n",
        )

    if not isinstance(vectors, list) or len(vectors) != 1:
        parser.error(f"Chunk {position}: unexpected TEI response.")

    vector = vectors[0]

    if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION:
        parser.error(
            f"Chunk {position}: expected {EMBEDDING_DIMENSION} dimensions."
        )

    if not all(
        type(value) in (int, float) and math.isfinite(value)
        for value in vector
    ):
        parser.error(f"Chunk {position}: invalid numbers in embedding.")

    embedded_records.append(
        {
            "chunk_index": chunk["chunk_index"],
            "text": text,
            "metadata": chunk["metadata"],
            "embedding_model": MODEL_NAME,
            "embedding": vector,
        }
    )

    print(f"Embedded chunk {position}/{len(chunk_records)}", flush=True)

output_path = input_path.with_name(
    f"{input_path.stem}.embeddings.json"
)

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(
        embedded_records,
        output_file,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

print(f"\nChunks embedded: {len(embedded_records)}")
print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
print(f"Embeddings saved to: {output_path}")