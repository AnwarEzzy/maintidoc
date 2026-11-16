import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare automatically accepted visual descriptions."
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    input_path = args.input_json.resolve()
    output_path = args.output_json.resolve()

    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    if input_path == output_path:
        parser.error("Input and output must be different files.")

    records = json.loads(input_path.read_text(encoding="utf-8"))

    if not isinstance(records, list):
        parser.error("The input must contain a list.")

    documents = []

    for record in records:
        if record.get("description_status") != "auto_accepted":
            continue

        description = record.get("description", "").strip()

        if not description:
            parser.error("An accepted record has an empty description.")

        metadata = {
            **record["metadata"],
            "visual_id": record["visual_id"],
            "image_path": record["image_path"],
            "source_type": "visual_description",
            "description_status": "auto_accepted",
            "generation_model": record["generation_model"],
        }

        documents.append(
            {
                "text": description,
                "metadata": metadata,
            }
        )

    if not documents:
        parser.error("No automatically accepted descriptions were found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Descriptions loaded: {len(records)}")
    print(f"Descriptions retained: {len(documents)}")
    print(f"Descriptions excluded: {len(records) - len(documents)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()