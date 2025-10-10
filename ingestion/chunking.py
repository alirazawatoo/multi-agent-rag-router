from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import re
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
    """Extract text per page with basic cleanup. Returns a list of {page, text}."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"\s+", " ", text).strip()
        pages.append({"page": i + 1, "text": text})
    return pages


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Simple character-based splitter with overlap."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def chunk_pdf(pdf_path: Path, chunk_size: int = 900, overlap: int = 150) -> List[Dict]:
    """Yield chunk dicts with metadata for a single PDF."""
    results: List[Dict] = []
    pages = extract_text_from_pdf(pdf_path)
    for p in pages:
        page_no = p["page"]
        text = p["text"]
        if not text:
            continue
        chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, ch in enumerate(chunks):
            results.append(
                {
                    "id": f"{pdf_path.stem}-p{page_no}-c{idx}",
                    "text": ch,
                    "metadata": {
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "page": page_no,
                        "chunk": idx,
                        "title": pdf_path.stem.replace("_", " "),
                    },
                }
            )
    return results


def load_pdfs(source_dir: Path) -> List[Path]:
    return sorted([p for p in source_dir.glob("*.pdf") if p.is_file()])


if __name__ == "__main__":
    from pprint import pprint

    sample_dir = Path("data/raw")
    pdfs = load_pdfs(sample_dir)
    print(f"Found {len(pdfs)} PDFs in {sample_dir}")
    if pdfs:
        chunks = chunk_pdf(pdfs[0])
        print(f"First PDF: {pdfs[0].name}, chunks: {len(chunks)}")
        pprint(chunks[:2])
