"""
DietBite Pro - Knowledge Base Builder
- Reads PDFs from ./kb_docs
- Splits into chunks
- Builds a TF-IDF vector index (scikit-learn)
- Saves: kb_index.pkl
"""

import os
import re
import pickle
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "kb_docs"
OUT_PATH = BASE_DIR / "kb_index.pkl"

# Chunking settings
CHUNK_WORDS = 220
CHUNK_OVERLAP = 60


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_pdf_text(pdf_path: Path) -> List[Dict]:
    """
    Returns a list of dicts: [{"source": filename, "page": page_num, "text": "..."}]
    """
    results = []
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    for i in range(total_pages):
        try:
            page = reader.pages[i]
            txt = page.extract_text() or ""
            txt = clean_text(txt)
            if txt:
                results.append(
                    {
                        "source": pdf_path.name,
                        "page": i + 1,
                        "text": txt,
                    }
                )
        except Exception as e:
            print(f"[KB] Warning: failed to read {pdf_path.name} page {i+1}: {e}")

    return results


def chunk_text(page_text: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = page_text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        end = start + chunk_words
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def build_chunks(pages: List[Dict]) -> List[Dict]:
    """
    Converts per-page text into chunk objects.
    Each chunk has: source, page, text
    """
    all_chunks = []
    for p in pages:
        ctexts = chunk_text(p["text"], CHUNK_WORDS, CHUNK_OVERLAP)
        for c in ctexts:
            all_chunks.append(
                {
                    "source": p["source"],
                    "page": p["page"],
                    "text": c,
                }
            )
    return all_chunks


def main():
    print("Building DietBite Pro knowledge base...")

    if not DOCS_DIR.exists():
        print(f"[KB] ERROR: kb_docs folder not found at {DOCS_DIR}")
        print("[KB] Create kb_docs and put your PDFs there.")
        return

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    print(f"[KB] Found {len(pdfs)} PDF(s) in {DOCS_DIR}")

    if not pdfs:
        print("[KB] ERROR: No PDFs found. Add PDFs to kb_docs and re-run.")
        return

    all_pages = []
    for pdf in pdfs:
        print(f"[KB] Reading: {pdf.name}")
        pages = read_pdf_text(pdf)
        print(f"[KB]  - extracted text from {len(pages)} page(s)")
        all_pages.extend(pages)

    if not all_pages:
        print("[KB] ERROR: No extractable text found in PDFs.")
        print("[KB] If PDFs are scanned images, you'd need OCR (not included in this build).")
        return

    chunks = build_chunks(all_pages)
    print(f"[KB] Created {len(chunks)} chunk(s)")

    texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(texts)

    kb_index = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunks": chunks,
    }

    with open(OUT_PATH, "wb") as f:
        pickle.dump(kb_index, f)

    print(f"[KB] ✅ Saved index to: {OUT_PATH}")
    print("[KB] Done.")


if __name__ == "__main__":
    main()

