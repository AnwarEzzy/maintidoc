import sys
from pathlib import Path

from llama_index.readers.file import PyMuPDFReader


pdf_path = Path(sys.argv[1]).resolve()

reader = PyMuPDFReader()
pages = reader.load_data(file_path=pdf_path)

print(f"Document: {pdf_path.name}")
print(f"Pages extracted: {len(pages)}")

first_page = pages[0]

print("\nFirst page metadata:")
print(first_page.metadata)

print("\nFirst page text preview:")
print(first_page.text[:800])