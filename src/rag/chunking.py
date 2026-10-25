"""Split filing sections into overlapping, paragraph-aligned chunks."""
from __future__ import annotations


def chunk_text(text: str, max_chars: int = 1500, overlap_paragraphs: int = 1) -> list[str]:
    """Greedily pack paragraphs into chunks of at most ~max_chars.

    The last `overlap_paragraphs` of each chunk are repeated at the start of
    the next so context isn't lost at boundaries. Oversized paragraphs are
    hard-split.
    """
    paras = []
    for p in (p.strip() for p in text.split("\n\n")):
        if not p:
            continue
        while len(p) > max_chars:
            paras.append(p[:max_chars])
            p = p[max_chars:]
        paras.append(p)

    chunks, current = [], []
    for p in paras:
        if current and sum(len(x) for x in current) + len(p) > max_chars:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            if sum(len(x) for x in current) + len(p) > max_chars:
                current = []
        current.append(p)
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if len(c) > 80]
