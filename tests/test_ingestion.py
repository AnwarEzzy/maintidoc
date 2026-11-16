import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from scripts.ingest_pdf import ingest_pdf


def text_pdf(text="Maintenance reference. Inspect the filter."):
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 70), text)
        return document.tobytes()


class IngestionTests(unittest.TestCase):
    def test_image_only_pdf_is_rejected_before_any_service_call(self):
        with pymupdf.open() as document:
            document.new_page()
            content = document.tobytes()
        with self.assertRaisesRegex(ValueError, "OCR"):
            ingest_pdf(content, "scan.pdf")

    def test_resume_reuses_files_but_restores_database_and_separates_same_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            for name in ["extract_pdf", "chunk_documents", "embed_chunks", "store_embeddings",
                         "extract_images", "extract_tables"]:
                (root / "scripts" / f"{name}.py").write_text("# fixture", encoding="utf-8")

            def fake_execute(module, arguments):
                name = module.split(".")[-1]
                source = Path(arguments[0])
                directory = root / "data" / "processed"
                if name == "extract_pdf":
                    output = directory / f"{source.stem}.json"
                    data = [{"text": "maintenance", "metadata": {"document": source.name, "page": 1}}]
                elif name in {"extract_images", "extract_tables"}:
                    suffix = "images" if name == "extract_images" else "tables"
                    output, data = directory / f"{source.stem}.{suffix}.json", []
                elif name == "chunk_documents":
                    output, data = source.with_name(f"{source.stem}.chunks.json"), [{"text": "chunk"}]
                elif name == "embed_chunks":
                    output, data = source.with_name(f"{source.stem}.embeddings.json"), [{"embedding": [1]}]
                else:
                    return
                output.write_text(json.dumps(data), encoding="utf-8")

            with patch("scripts.ingest_pdf.ROOT", root), \
                 patch("scripts.ingest_pdf.CATALOG", root / "data" / "processed" / "library"), \
                 patch("scripts.ingest_pdf.execute", side_effect=fake_execute) as execute:
                first = ingest_pdf(text_pdf(), "manual.pdf")
                execute.reset_mock()
                # Same exact PDF bytes: no VLM/embedding recomputation.
                content = (root / first["pdf_path"]).read_bytes()
                second = ingest_pdf(content, "manual.pdf")
                self.assertEqual(second["status"], "ready")
                self.assertEqual([call.args[0] for call in execute.call_args_list], ["scripts.store_embeddings"])
                # Different PDF content under the same user filename stays isolated.
                other = ingest_pdf(text_pdf("A different maintenance reference."), "manual.pdf")
                self.assertNotEqual(first["document"], other["document"])


if __name__ == "__main__":
    unittest.main()
