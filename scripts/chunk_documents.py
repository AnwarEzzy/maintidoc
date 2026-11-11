import argparse
import json
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


parser = argparse.ArgumentParser(
    description="Split extracted PDF pages into searchable text chunks."
)

parser.add_argument(
    "pages_json",
    type=Path,
    help="Path to the JSON file containing extracted pages.",
)

args = parser.parse_args()
input_path = args.pages_json.resolve()

if not input_path.is_file():
    parser.error(f"JSON file not found: {input_path}")

page_records = json.loads(input_path.read_text(encoding="utf-8"))

documents = []

for record in page_records:
    if record["text"].strip():
        documents.append(
            Document(
                text=record["text"],
                metadata=record["metadata"],
            )
        )

splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=64,
)

nodes = splitter.get_nodes_from_documents(documents)

chunk_records = []

for index, node in enumerate(nodes, start=1):
    chunk_records.append(
        {
            "chunk_index": index,
            "text": node.text,
            "metadata": node.metadata,
        }
    )

output_path = input_path.with_name(f"{input_path.stem}.chunks.json")

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(chunk_records, output_file, ensure_ascii=False, indent=2)

print(f"Pages loaded: {len(page_records)}")
print(f"Chunks created: {len(chunk_records)}")
print(f"Chunks saved to: {output_path}")

if nodes:
    print("\nFirst chunk metadata:")
    print(nodes[0].metadata)

    print("\nFirst chunk preview:")
    print(nodes[0].text[:300])