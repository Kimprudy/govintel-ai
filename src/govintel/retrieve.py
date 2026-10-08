from govintel.embed import get_store


def search(query: str, k: int = 5) -> list:
    """Return the k most relevant chunks with their citations."""
    store = get_store()
    results = store.similarity_search_with_score(query, k=k)

    return [
        {
            "citation": doc.metadata["citation"],
            "heading": doc.metadata["heading"],
            "text": doc.page_content,
            "score": float(score),
        }
        for doc, score in results
    ]


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are the requirements for internal controls?"
    print(f"Query: {q}\n")
    for i, r in enumerate(search(q), 1):
        print(f"{i}. [{r['citation']}] {r['heading']}  (score {r['score']:.3f})")
        print(f"   {r['text'][:200]}...\n")


def search_deduped(query: str, k: int = 5, fetch: int = 20) -> list:
    """Retrieve wide, then keep only the best chunk per section.

    Long sections split into several chunks, and siblings crowd each other
    out of the top-k. Collapsing by citation gives k distinct sections.
    """
    store = get_store()
    results = store.similarity_search_with_score(query, k=fetch)

    best = {}
    for doc, score in results:
        cite = doc.metadata["citation"]
        if cite not in best or score < best[cite][1]:
            best[cite] = (doc, score)

    ranked = sorted(best.values(), key=lambda x: x[1])[:k]

    return [
        {
            "citation": doc.metadata["citation"],
            "heading": doc.metadata["heading"],
            "text": doc.page_content,
            "score": float(score),
        }
        for doc, score in ranked
    ]


import json as _json
from govintel.config import PROCESSED_DIR as _PROC

MAX_SECTION_CHARS = 12000
_SECTIONS = None


def _load_sections() -> dict:
    """Citation -> full section record. Loaded once, cached."""
    global _SECTIONS
    if _SECTIONS is None:
        _SECTIONS = {}
        with open(f"{_PROC}/title-2-sections.jsonl") as f:
            for line in f:
                s = _json.loads(line)
                _SECTIONS[s["citation"]] = s
    return _SECTIONS


def search_sections(query: str, k: int = 5, fetch: int = 25) -> list:
    """Find relevant sections by chunk, then return each section in full.

    Chunk-level search is good at finding *which* section is relevant, but the
    matched chunk often isn't the paragraph holding the answer. Returning the
    whole section keeps every paragraph and removes duplicate chunks.
    Oversized sections fall back to their matched chunks.
    """
    store = get_store()
    hits = store.similarity_search_with_score(query, k=fetch)

    order = []
    matched_chunks = {}
    for doc, score in hits:
        cite = doc.metadata["citation"]
        if cite not in matched_chunks:
            order.append((cite, float(score)))
            matched_chunks[cite] = []
        matched_chunks[cite].append(doc.page_content)

    sections = _load_sections()
    out = []
    for cite, score in order[:k]:
        rec = sections.get(cite)
        if rec and len(rec["text"]) <= MAX_SECTION_CHARS:
            text = f"{cite} — {rec['heading']}\n\n{rec['text']}"
            heading = rec["heading"]
        else:
            text = "\n\n[...]\n\n".join(matched_chunks[cite])
            heading = rec["heading"] if rec else ""
        out.append({
            "citation": cite,
            "heading": heading,
            "text": text,
            "score": score,
        })
    return out


from rank_bm25 import BM25Okapi as _BM25

_BM25_INDEX = None
_BM25_CITES = None


def _build_bm25():
    """Keyword index over full section text, built once."""
    global _BM25_INDEX, _BM25_CITES
    if _BM25_INDEX is None:
        sections = _load_sections()
        _BM25_CITES = list(sections.keys())
        corpus = [
            (sections[c]["heading"] + " " + sections[c]["text"]).lower().split()
            for c in _BM25_CITES
        ]
        _BM25_INDEX = _BM25(corpus)
    return _BM25_INDEX, _BM25_CITES


def search_hybrid(query: str, k: int = 5, fetch: int = 25, bm25_k: int = 10) -> list:
    """Vector search plus BM25 keyword search, merged by reciprocal rank fusion.

    Vector search finds semantically related sections but misses exact terms.
    BM25 catches literal wording. RRF merges the two rankings without needing
    the scores to be on the same scale.
    """
    store = get_store()
    vec_hits = store.similarity_search_with_score(query, k=fetch)

    vec_order = []
    for doc, _ in vec_hits:
        c = doc.metadata["citation"]
        if c not in vec_order:
            vec_order.append(c)

    index, cites = _build_bm25()
    scores = index.get_scores(query.lower().split())
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:bm25_k]
    bm_order = [cites[i] for i in top]

    # reciprocal rank fusion
    fused = {}
    for rank, c in enumerate(vec_order):
        fused[c] = fused.get(c, 0) + 1.0 / (60 + rank)
    for rank, c in enumerate(bm_order):
        fused[c] = fused.get(c, 0) + 1.0 / (60 + rank)

    best = sorted(fused, key=fused.get, reverse=True)[:k]

    sections = _load_sections()
    out = []
    for c in best:
        rec = sections.get(c)
        if not rec:
            continue
        text = rec["text"][:MAX_SECTION_CHARS]
        out.append({
            "citation": c,
            "heading": rec["heading"],
            "text": f"{c} — {rec['heading']}\n\n{text}",
            "score": fused[c],
        })
    return out
