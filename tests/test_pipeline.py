"""End to end: generate the synthetic PDFs, index them with the hash embedder, ask the graph."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.env = mock.patch.dict(
            os.environ,
            {
                "RAW_DOCS_DIR": str(root / "raw"),
                "CHROMA_DB_DIR": str(root / "chroma"),
                "EMBEDDING_BACKEND": "hash",
                "LLM_PROVIDER": "stub",
            },
        )
        cls.env.start()
        from ingestion.build_index import build_index
        from scripts.generate_pdfs import main as generate_pdfs

        generate_pdfs(root / "raw")
        cls.count = build_index(root / "raw", root / "chroma", rebuild=True)

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.tmp.cleanup()

    def test_index_holds_every_page(self):
        self.assertGreaterEqual(self.count, 6)

    def test_knowledge_question_retrieves_the_matching_guide(self):
        from agents.graph import invoke_graph

        result = invoke_graph("How do I re-send an invoice to a partner?", top_k=3)
        self.assertEqual(result["intent"], "knowledge")
        self.assertEqual(result["contexts"][0]["metadata"]["filename"], "Finance_Processes_Guide.pdf")
        self.assertIn("[Finance_Processes_Guide.pdf p1]", result["answer"])
        self.assertTrue(result["citation_shape_present"])
        self.assertEqual(result["log"][0], "Router → Knowledge")

    def test_rebuilding_does_not_duplicate_chunks(self):
        from ingestion.build_index import build_index

        root = Path(self.tmp.name)
        self.assertEqual(build_index(root / "raw", root / "chroma"), self.count)

    def test_index_built_with_another_backend_is_refused(self):
        from ingestion.vectorstore import HashingEmbedder, open_collection

        with self.assertRaisesRegex(RuntimeError, "--rebuild"):
            open_collection(HashingEmbedder(dimensions=64))


    def test_api_ask_returns_sources_and_action_proposals(self):
        from apps.api.main import app

        client = TestClient(app)
        self.assertGreaterEqual(client.get("/health").json()["indexed_chunks"], 6)
        body = client.post("/ask", json={"query": "When are payments issued?", "top_k": 2}).json()
        self.assertEqual(body["intent"], "knowledge")
        self.assertEqual(len(body["sources"]), 2)
        action = client.post("/ask", json={"query": "Pause 'Chicken Burger' until 6pm"}).json()
        self.assertEqual(action["proposed_action"]["item"], "Chicken Burger")
        self.assertEqual(action["sources"], [])
        self.assertEqual(client.post("/ask", json={"query": ""}).status_code, 422)



if __name__ == "__main__":
    unittest.main()
