import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    @patch("scripts.library.list_documents", return_value=[])
    def test_empty_library_offers_upload_and_disables_questions(self, _):
        app = AppTest.from_file(APP, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(app.button[0].disabled)

    @patch("scripts.library.source_pdf", return_value=None)
    @patch("scripts.library.list_documents", return_value=[
        {"document": "manual.pdf", "display_name": "Manual", "passages": 3, "descriptions": 1},
        {"document": "other.pdf", "display_name": "Other manual", "passages": 2, "descriptions": 0},
    ])
    @patch("scripts.agent.answer_with_agent")
    def test_chat_displays_sources_and_switching_pdf_clears_history(self, answer, *_):
        answer.return_value = {
            "answer": "Inspecter le filtre [S1].",
            "sources": [{"id": "S1", "document": "manual.pdf", "page": 1,
                         "source_type": "pdf_text", "text": "Inspect the filter.", "image_path": None}],
            "trace": [{"tool": "search_passages", "query": "filter", "results": 1}],
        }
        app = AppTest.from_file(APP, default_timeout=30).run()
        app.chat_input[0].set_value("Que signifie le voyant ?").run()
        self.assertFalse(app.exception)
        self.assertIn("[S1]", app.session_state["messages"][-1]["content"])
        self.assertEqual(answer.call_args.kwargs["document"], "manual.pdf")
        self.assertEqual(answer.call_args.kwargs["history"], [])
        app.selectbox[0].select("other.pdf").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["messages"], [])


if __name__ == "__main__":
    unittest.main()
