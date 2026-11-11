import argparse
from pathlib import Path

from llama_index.readers.file import PyMuPDFReader


parser = argparse.ArgumentParser(
    description="Extract text and page metadata from a PDF."
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