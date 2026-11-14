import argparse
import json
from pathlib import Path

import pymupdf


parser = argparse.ArgumentParser(
    description="Extract embedded PDF images and their page metadata."
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
images_dir = processed_dir / pdf_path.stem / "images"
images_dir.mkdir(parents=True, exist_ok=True)

records = []

with pymupdf.open(pdf_path) as document:
    for page in document:
        seen_xrefs = set()

        for image_index, image_info in enumerate(
            page.get_images(full=True),
            start=1,
        ):
            xref = image_info[0]
            mask_xref = image_info[1]

            if xref in seen_xrefs:
                continue

            seen_xrefs.add(xref)

            pixmap = pymupdf.Pixmap(document, xref)

            if mask_xref:
                mask = pymupdf.Pixmap(document, mask_xref)
                pixmap = pymupdf.Pixmap(pixmap, mask)

            if (
                pixmap.colorspace is not None
                and pixmap.colorspace.n > 3
            ):
                pixmap = pymupdf.Pixmap(
                    pymupdf.csRGB,
                    pixmap,
                )

            image_name = (
                f"page_{page.number + 1:03d}"
                f"_image_{image_index:03d}.png"
            )
            image_path = images_dir / image_name
            pixmap.save(str(image_path))

            records.append(
                {
                    "image_index": len(records) + 1,
                    "image_path": image_path.relative_to(
                        project_root
                    ).as_posix(),
                    "metadata": {
                        "document": pdf_path.name,
                        "page": page.number + 1,
                        "xref": xref,
                        "width": pixmap.width,
                        "height": pixmap.height,
                        "has_alpha": bool(pixmap.alpha),
                    },
                }
            )

            print(
                f"Extracted: {image_name} "
                f"({pixmap.width}x{pixmap.height})"
            )

output_path = processed_dir / f"{pdf_path.stem}.images.json"

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(records, output_file, ensure_ascii=False, indent=2)

print(f"\nImages extracted: {len(records)}")
print(f"Image metadata saved to: {output_path}")