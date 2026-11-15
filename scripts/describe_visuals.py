import argparse
import hashlib
import json
from pathlib import Path

from llama_index.core.schema import Document

from scripts import describe_image as vision


def save_progress(output_path: Path, records: list):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            records,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Save reviewable descriptions of PDF visuals."
    )
    parser.add_argument(
        "visual_documents_json",
        type=Path,
        help="Path to the LlamaIndex visual documents JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON path.",
    )

    args = parser.parse_args()
    input_path = args.visual_documents_json.resolve()

    if not input_path.is_file():
        parser.error(f"JSON file not found: {input_path}")

    records = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    if not isinstance(records, list) or not records:
        parser.error("Expected a non-empty list of documents.")

    documents = [
        Document.from_dict(record)
        for record in records
    ]

    input_ids = {document.id_ for document in documents}

    if len(input_ids) != len(documents):
        parser.error("The input contains duplicate document IDs.")

    output_path = (
        args.output.resolve()
        if args.output
        else input_path.with_name(
            f"{input_path.stem}.descriptions.json"
        )
    )

    if output_path == input_path:
        parser.error("The output must differ from the input.")

    existing_records = []

    if output_path.is_file():
        existing_records = json.loads(
            output_path.read_text(encoding="utf-8")
        )

        if not isinstance(existing_records, list):
            parser.error("The existing output must contain a list.")

    descriptions = {
        record["visual_id"]: record
        for record in existing_records
    }

    if len(descriptions) != len(existing_records):
        parser.error("The existing output contains duplicate IDs.")

    if not set(descriptions).issubset(input_ids):
        parser.error("The output contains unrelated visual documents.")

    generator_code_hash = hashlib.sha256(
        Path(vision.__file__).read_bytes()
    ).hexdigest()

    generated_count = 0

    for position, document in enumerate(documents, start=1):
        resource = document.image_resource

        if resource is None or resource.path is None:
            parser.error("A document has no local image resource.")

        image_path = resource.path

        if not image_path.is_file():
            parser.error(f"Image not found: {image_path}")

        image_hash = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()

        existing = descriptions.get(document.id_)

        if existing is not None:
            same_configuration = (
                existing.get("generation_model") == vision.MODEL_NAME
                and existing.get("generation_code_sha256")
                == generator_code_hash
                and existing.get("image_sha256") == image_hash
            )

            if not same_configuration:
                parser.error(
                    "Existing descriptions use different inputs or "
                    "generation code. Choose a new file with --output."
                )

            if not existing.get("description", "").strip():
                parser.error("An existing description is empty.")

            print(
                f"Already saved: {position}/{len(documents)}",
                flush=True,
            )
            continue

        print(
            f"Describing visual {position}/{len(documents)} "
            f"- page {document.metadata['page']}",
            flush=True,
        )

        try:
            description = vision.describe_image(image_path)
        except (ValueError, RuntimeError, OSError) as error:
            parser.exit(
                1,
                f"\nError: {error}\n"
                f"Completed descriptions: {len(descriptions)}.\n"
                "Run the same command again to resume.\n",
            )

        descriptions[document.id_] = {
            "visual_id": document.id_,
            "image_path": document.metadata["image_path"],
            "description": description,
            "description_status": "pending",
            "generation_model": vision.MODEL_NAME,
            "generation_code_sha256": generator_code_hash,
            "image_sha256": image_hash,
            "metadata": document.metadata,
        }

        ordered_records = [
            descriptions[item.id_]
            for item in documents
            if item.id_ in descriptions
        ]

        save_progress(output_path, ordered_records)
        generated_count += 1

        print("Saved with status: pending", flush=True)

    print(f"\nDescriptions available: {len(descriptions)}")
    print(f"New descriptions generated: {generated_count}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()