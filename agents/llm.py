from __future__ import annotations
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Optional import; only needed if GROQ_API_KEY is set
try:
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover
    Groq = None  # type: ignore

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

_client: Optional["Groq"] = None

def get_llm_client() -> Optional["Groq"]:
    global _client
    if _client is not None:
        return _client
    if GROQ_API_KEY and Groq is not None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

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

def format_context_block(contexts: List[Dict]) -> str:
    chunks = []
    for c in contexts:
        meta = c.get("metadata", {})
        src = meta.get("filename", meta.get("source", ""))
        page = meta.get("page", "?")
        text = c.get("document", "")
        chunks.append(f"Source: {src} p{page}\n{text}")
    return "\n\n---\n\n".join(chunks)


def answer_with_citations(
    query: str,
    contexts: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 768,
    style: str | None = None,
) -> str:
    client = get_llm_client()
    if client is None:
        return "[LLM disabled] Provide GROQ_API_KEY in .env. Proceed with retrieved sources below."

    context_block = format_context_block(contexts)
    prompt = f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"

    try:
        system_message = SYSTEM_PROMPT
        if style:
            if style.lower() == "concise":
                system_message += " Focus on brevity. Use at most 6–8 bullets in the procedure."
            elif style.lower() == "detailed":
                system_message += " Provide rich, detailed steps and short rationale where helpful."
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[LLM error] {e}. Proceed with retrieved sources below."


def stream_answer_with_citations(
    query: str,
    contexts: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 768,
    style: str | None = None,
):
    """Yield answer tokens progressively for a live-typing feel.

    This is a generator that yields string chunks. Caller can accumulate
    into a full response. If GROQ_API_KEY is not configured, yields a
    single fallback message.
    """
    client = get_llm_client()
    if client is None:
        yield "[LLM disabled] Provide GROQ_API_KEY in .env. Proceed with retrieved sources below."
        return

    context_block = format_context_block(contexts)
    prompt = f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"

    system_message = SYSTEM_PROMPT
    if style:
        if style.lower() == "concise":
            system_message += " Focus on brevity. Use at most 6–8 bullets in the procedure."
        elif style.lower() == "detailed":
            system_message += " Provide rich, detailed steps and short rationale where helpful."

    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content  # type: ignore[attr-defined]
            except Exception:
                delta = None
            if delta:
                yield delta
    except Exception as e:
        yield f"[LLM error] {e}. Proceed with retrieved sources below."
