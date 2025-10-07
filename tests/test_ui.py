"""Render the Streamlit app headlessly against a temporary synthetic index."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "apps" / "ui" / "streamlit_app.py")


class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        env = mock.patch.dict(
            os.environ,
            {"RAW_DOCS_DIR": str(root / "raw"), "CHROMA_DB_DIR": str(root / "chroma"), "EMBEDDING_BACKEND": "hash",
             "LLM_PROVIDER": "stub", "OPS_API_URL": "http://127.0.0.1:9"},
        )
        env.start()
        self.addCleanup(env.stop)
        self.root = root

    def run_app(self) -> AppTest:
        app = AppTest.from_file(APP, default_timeout=30)
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_empty_index_shows_build_instructions(self):
        app = self.run_app()
        self.assertIn("No documents are indexed yet", app.warning[0].value)

    def test_question_shows_answer_and_sources(self):
        from ingestion.build_index import build_index
        from scripts.generate_pdfs import main as generate_pdfs

        generate_pdfs(self.root / "raw")
        build_index(self.root / "raw", self.root / "chroma", rebuild=True)
        app = self.run_app()
        self.assertEqual(len(app.warning), 0)
        app.chat_input[0].set_value("How do I re-send an invoice?").run()
        self.assertIn("[Finance_Processes_Guide.pdf p1]", app.chat_message[1].markdown[0].value)
        self.assertTrue(app.expander)

    def test_unreachable_connector_is_reported_and_buttons_lock(self):
        app = self.run_app()
        app.chat_input[0].set_value("Pause 'Chicken Burger' until 6pm").run()
        approve = next(b for b in app.button if b.label == "Approve")
        approve.click().run()
        self.assertIn("Could not reach the ops connector", app.error[0].value)
        self.assertTrue(all(b.disabled for b in app.button))


if __name__ == "__main__":
    unittest.main()
