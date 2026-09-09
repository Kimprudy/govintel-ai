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
