import argparse
import json
from pathlib import Path

from llama_index.readers.file import PyMuPDFReader


parser = argparse.ArgumentParser(
    description="Extract PDF text and page metadata into a JSON file."
)

parser.add_argument(
    "pdf_path",
    type=Path,
    help="Path to the PDF document.",
)

args = parser.parse_args()
pdf_path = args.pdf_path.resolve()

if not pdf_path.is_file():
    parser.error(f"PDF file not found: {pdf_path}")

reader = PyMuPDFReader()
pages = reader.load_data(file_path=pdf_path)

print(f"Document: {pdf_path.name}")
print(f"Pages extracted: {len(pages)}")

first_page = pages[0]

print("\nFirst page metadata:")
print(first_page.metadata)

print("\nFirst page text preview:")
print(first_page.text[:800])

records = []

for page in pages:
    records.append(
        {
            "text": page.text,
            "metadata": {
                "document": pdf_path.name,
                "page": int(page.metadata["source"]),
                "total_pages": len(pages),
            },
        }
    )

project_root = Path(__file__).resolve().parents[1]
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"{pdf_path.stem}.json"

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(records, output_file, ensure_ascii=False, indent=2)

print(f"\nJSON saved to: {output_path}")