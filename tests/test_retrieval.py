import io
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.search_chunks import search_chunks


class RetrievalTests(unittest.TestCase):
    def test_original_text_filter_handles_legacy_nodes_and_scopes_document(self):
        store = MagicMock()
        store.close = AsyncMock()
        response = io.BytesIO(json.dumps([[0.0] * 1024]).encode())
        with patch.dict(os.environ, {"POSTGRES_PASSWORD": "test-only"}), \
             patch("scripts.search_chunks.urlopen", return_value=response), \
             patch("scripts.search_chunks.PGVectorStore", return_value=store):
            search_chunks("attention", document="manual.pdf", text_only=True)
        query = store.query.call_args.args[0]
        original_text, document = query.filters.filters
        self.assertEqual(document.value, "manual.pdf")
        self.assertEqual(original_text.condition.value, "or")
        self.assertIsNone(original_text.filters[1].value)
        store.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
