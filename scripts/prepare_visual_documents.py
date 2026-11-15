import argparse
import hashlib
import json
from pathlib import Path

from llama_index.core.schema import Document, MediaResource


parser = argparse.ArgumentParser(
    description="Prepare LlamaIndex documents from extracted visuals."
)
parser.add_argument(
    "images_json",
    type=Path,
    help="Path to the extracted image manifest.",
)
parser.add_argument(
    "tables_json",
    type=Path,
    help="Path to the extracted table manifest.",
)

args = parser.parse_args()
project_root = Path(__file__).resolve().parents[1]

manifests = [
    (args.images_json, "figure", "image_index"),
    (args.tables_json, "table_preview", "table_index"),
]

documents = []
source_documents = set()

for manifest_path, content_type, index_key in manifests:
    manifest_path = manifest_path.resolve()

    if not manifest_path.is_file():
        parser.error(f"Manifest not found: {manifest_path}")

    records = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if not isinstance(records, list):
        parser.error(f"Expected a list in: {manifest_path}")

    for record in records:
        image_path = (
            project_root / record["image_path"]
        ).resolve()

        if not image_path.is_file():
            parser.error(f"Image not found: {image_path}")

        metadata = {
            **record["metadata"],
            "content_type": content_type,
            "visual_index": record[index_key],
            "image_path": record["image_path"],
        }

        source_documents.add(metadata["document"])

        image_hash = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()

        identity = (
            f"{content_type}|{metadata['document']}|"
            f"{metadata['page']}|{metadata['visual_index']}|"
            f"{image_hash}"
        )

        document_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        documents.append(
            Document(
                id_=document_id,
                image_resource=MediaResource(
                    path=str(image_path),
                    mimetype="image/png",
                ),
                metadata=metadata,
            )
        )

        print(
            f"Prepared: {content_type}, "
            f"page {metadata['page']}"
        )

if not documents:
    parser.error("No visual documents were prepared.")

if len(source_documents) != 1:
    parser.error("Both manifests must refer to the same PDF.")

pdf_name = next(iter(source_documents))
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = (
    output_dir / f"{Path(pdf_name).stem}.visual_documents.json"
)

serialized_documents = [
    json.loads(document.to_json())
    for document in documents
]

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(
        serialized_documents,
        output_file,
        ensure_ascii=False,
        indent=2,
    )

figure_count = sum(
    document.metadata["content_type"] == "figure"
    for document in documents
)

print(f"\nFigures prepared: {figure_count}")
print(f"Table previews prepared: {len(documents) - figure_count}")
print(f"Total visual documents: {len(documents)}")
print(f"Saved to: {output_path}")