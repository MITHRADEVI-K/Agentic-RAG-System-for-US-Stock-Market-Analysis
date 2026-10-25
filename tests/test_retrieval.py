from src.rag.chunking import chunk_text
from src.rag.ingestion import extract_sections

FAKE_10K = """
TABLE OF CONTENTS
Item 1A. Risk Factors 12
Item 1B. Unresolved Staff Comments 30
Item 7. Management's Discussion and Analysis 40
Item 7A. Quantitative and Qualitative Disclosures 55

Item 1A. Risk Factors
""" + ("Our business depends on global supply chains that may be disrupted. " * 30) + """

Item 1B. Unresolved Staff Comments
None.

Item 7. Management's Discussion and Analysis of Financial Condition
""" + ("Revenue grew due to strong services demand in the fiscal year. " * 30) + """

Item 7A. Quantitative and Qualitative Disclosures About Market Risk
Interest rate risk.
"""


def test_extract_sections_skips_table_of_contents():
    sections = extract_sections(FAKE_10K, "10-K")
    assert set(sections) == {"Risk Factors", "MD&A"}
    assert "supply chains" in sections["Risk Factors"]
    assert "Unresolved" not in sections["Risk Factors"]
    assert "Revenue grew" in sections["MD&A"]


def test_chunks_respect_max_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(40))
    chunks = chunk_text(text, max_chars=1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 + 400 for c in chunks)


def test_chunks_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(10))
    chunks = chunk_text(text, max_chars=1000, overlap_paragraphs=1)
    last_para_of_first = chunks[0].split("\n\n")[-1]
    assert chunks[1].startswith(last_para_of_first)


def test_long_paragraph_is_split():
    chunks = chunk_text("y" * 5000, max_chars=1000)
    assert len(chunks) == 5
