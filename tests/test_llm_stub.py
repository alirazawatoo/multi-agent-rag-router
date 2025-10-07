import os
import unittest
from unittest import mock

from agents.llm import answer_with_citations, stub_answer

CONTEXTS = [
    {
        "document": "Payments are issued weekly on Fridays. Disputes must be raised within 7 days of receipt.",
        "metadata": {"filename": "Finance_Processes_Guide.pdf", "page": 1},
        "distance": 0.4,
    },
    {
        "document": "Use live chat for urgent issues. Finance tickets for payout concerns.",
        "metadata": {"filename": "System_Onboarding_and_Usage.pdf", "page": 1},
        "distance": 0.9,
    },
]


class StubAnswerTests(unittest.TestCase):
    def test_quotes_the_most_relevant_sentence_with_a_citation(self):
        answer = stub_answer("When must disputes be raised?", CONTEXTS)
        self.assertIn("Disputes must be raised within 7 days of receipt. [Finance_Processes_Guide.pdf p1]", answer)
        self.assertIn("[System_Onboarding_and_Usage.pdf p1]", answer)
        self.assertTrue(answer.startswith("[stub]"))

    def test_concise_style_limits_the_number_of_passages(self):
        many = CONTEXTS * 3
        self.assertEqual(stub_answer("payments", many, style="concise").count("\n"), 2)

    def test_no_contexts_explains_how_to_build_the_index(self):
        self.assertIn("ingestion.build_index", stub_answer("anything", []))

    def test_groq_without_key_never_calls_the_network(self):
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": ""}):
            self.assertTrue(answer_with_citations("q", CONTEXTS).startswith("[LLM disabled]"))

    def test_unknown_provider_is_rejected(self):
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            with self.assertRaises(ValueError):
                answer_with_citations("q", CONTEXTS)


if __name__ == "__main__":
    unittest.main()
