"""Answer composition: an offline extractive stub by default, Groq when explicitly configured.

``LLM_PROVIDER=stub`` (default) never makes a network call. It quotes the retrieved sentence
that shares the most words with the question from each passage and cites it as
``[filename pN]``, so the rest of the graph (critic, UI, API) behaves as it would with a model.
``LLM_PROVIDER=groq`` plus ``GROQ_API_KEY`` sends the question and retrieved text to Groq.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are an operations copilot for finance, menu, onboarding, and platform workflows. "
    "Use ONLY the provided context to answer. If the context is insufficient, say you are unsure and list what is missing. "
    "Write in clear, professional, natural English for an operations audience. Prefer concise but complete explanations. "
    "Formatting requirements:\n"
    "- Start with a one-sentence executive summary.\n"
    "- Then provide a numbered step-by-step procedure.\n"
    "- Add short ‘Notes’ for edge cases, validation, or policies if relevant.\n"
    "- Cite sources inline using [filename pX] exactly where claims are supported.\n"
    "Do NOT reveal chain-of-thought; return only the final answer."
)
STYLE_HINTS = {
    "concise": " Focus on brevity. Use at most 6–8 bullets in the procedure.",
    "detailed": " Provide rich, detailed steps and short rationale where helpful.",
}

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*;\s*")


def citation(context: Dict) -> str:
    meta = context.get("metadata", {})
    return f"[{meta.get('filename', meta.get('source', 'unknown'))} p{meta.get('page', '?')}]"


def format_context_block(contexts: List[Dict]) -> str:
    return "\n\n---\n\n".join(f"Source: {citation(c)[1:-1]}\n{c.get('document', '')}" for c in contexts)


def _best_sentence(query_words: set, text: str) -> str:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return ""
    return max(sentences, key=lambda s: len(query_words & set(_WORD_RE.findall(s.lower()))))


def stub_answer(query: str, contexts: List[Dict], style: str | None = None) -> str:
    if not contexts:
        return (
            "[stub] No indexed passages were retrieved. Build the index with "
            "`python -m ingestion.build_index` and ask again."
        )
    limit = 2 if (style or "").lower() == "concise" else 4
    query_words = set(_WORD_RE.findall(query.lower()))
    lines = []
    for context in contexts[:limit]:
        sentence = _best_sentence(query_words, context.get("document", ""))
        if sentence:
            lines.append(f"{len(lines) + 1}. {sentence} {citation(context)}")
    header = "[stub] No language model was called. The closest indexed passages say:"
    return "\n".join([header, *lines])


def _groq_answer(query: str, contexts: List[Dict], temperature: float, max_tokens: int, style: str | None) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "[LLM disabled] LLM_PROVIDER=groq needs GROQ_API_KEY in .env. Proceed with retrieved sources below."
    import groq

    client = groq.Groq(api_key=api_key)
    system_message = SYSTEM_PROMPT + STYLE_HINTS.get((style or "").lower(), "")
    prompt = f"Context:\n{format_context_block(contexts)}\n\nQuestion: {query}\nAnswer:"
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except groq.GroqError as exc:
        return f"[LLM error] {exc}. Proceed with retrieved sources below."
    return (resp.choices[0].message.content or "").strip()


def answer_with_citations(
    query: str,
    contexts: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 768,
    style: str | None = None,
) -> str:
    provider = os.environ.get("LLM_PROVIDER", "stub").strip().lower()
    if provider == "groq":
        return _groq_answer(query, contexts, temperature, max_tokens, style)
    if provider != "stub":
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; use 'stub' or 'groq'")
    return stub_answer(query, contexts, style)
