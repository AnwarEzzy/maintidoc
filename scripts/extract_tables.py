import argparse
import json
from pathlib import Path

import pymupdf


parser = argparse.ArgumentParser(
    description="Extract PDF table candidates with visual previews."
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

project_root = Path(__file__).resolve().parents[1]
processed_dir = project_root / "data" / "processed"
tables_dir = processed_dir / pdf_path.stem / "tables"
tables_dir.mkdir(parents=True, exist_ok=True)

records = []

with pymupdf.open(pdf_path) as document:
    for page in document:
        finder = page.find_tables()

        for page_table_index, table in enumerate(
            finder.tables,
            start=1,
        ):
            cells = table.extract()

            image_name = (
                f"page_{page.number + 1:03d}"
                f"_table_{page_table_index:03d}.png"
            )
            image_path = tables_dir / image_name

            preview = page.get_pixmap(
                dpi=200,
                clip=pymupdf.Rect(table.bbox),
                alpha=False,
            )
            preview.save(str(image_path))

            records.append(
                {
                    "table_index": len(records) + 1,
                    "cells": cells,
                    "image_path": image_path.relative_to(
                        project_root
                    ).as_posix(),
                    "metadata": {
                        "document": pdf_path.name,
                        "page": page.number + 1,
                        "page_table_index": page_table_index,
                        "bbox": list(table.bbox),
                        "detected_rows": table.row_count,
                        "detected_columns": table.col_count,
                        "extraction_method": "pymupdf.find_tables",
                        "review_status": "pending",
                    },
                }
            )

            print(
                f"Table candidate: page {page.number + 1}, "
                f"table {page_table_index}"
            )
            print(
                f"Detected structure: "
                f"{table.row_count} rows x "
                f"{table.col_count} columns"
            )

output_path = processed_dir / f"{pdf_path.stem}.tables.json"

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(records, output_file, ensure_ascii=False, indent=2)

print(f"\nTable candidates extracted: {len(records)}")
print("Review status: pending")
print(f"Table data saved to: {output_path}")