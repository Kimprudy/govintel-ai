import re
from langchain_anthropic import ChatAnthropic
from govintel.config import ANTHROPIC_API_KEY
from govintel.retrieve import search

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a federal regulatory analyst. You answer questions about the Code of Federal Regulations using ONLY the excerpts provided.

Rules, without exception:

1. Use only the provided excerpts. Never use outside knowledge, even if you are confident it is correct.
2. Every factual claim must carry an inline citation in square brackets, exactly matching a citation from the excerpts. Example: "The cost of alcoholic beverages is unallowable [2 CFR 200.423]."
3. If the excerpts do not contain enough information to answer, respond with exactly: INSUFFICIENT_CONTEXT followed by one sentence explaining what is missing. Do not guess, and do not partially answer.
4. Quote regulatory thresholds, dollar amounts, and deadlines exactly as written.
5. Be concise. Two to four sentences unless the question requires more."""


def build_context(chunks: list) -> str:
    return "\n\n---\n\n".join(
        f"[{c['citation']}] {c['heading']}\n{c['text']}" for c in chunks
    )


def extract_citations(text: str) -> set:
    """Pull citations out of the answer, normalizing away subsection suffixes.

    Matches [2 CFR 200.501] and [2 CFR 200.501(a)(2)], both -> "2 CFR 200.501".
    """
    matches = re.findall(r"\[(\d+)\s+CFR\s+([\d.]+?)(?:\([^\]]*\))?\]", text)
    return {f"{title} CFR {section.rstrip('.')}" for title, section in matches}


def verify_citations(answer: str, chunks: list) -> dict:
    """Check every cited section was actually retrieved. Catches hallucination."""
    cited = extract_citations(answer)
    available = {c["citation"] for c in chunks}
    unsupported = cited - available

    return {
        "cited": sorted(cited),
        "unsupported": sorted(unsupported),
        "uncited": len(cited) == 0,
        "valid": len(unsupported) == 0 and len(cited) > 0,
    }


def ask(question: str, k: int = 5) -> dict:
    chunks = search(question, k=k)

    llm = ChatAnthropic(
        model=MODEL,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=1000,
        temperature=0,
    )

    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", f"Excerpts:\n\n{build_context(chunks)}\n\nQuestion: {question}"),
    ])

    answer = response.content
    refused = answer.strip().startswith("INSUFFICIENT_CONTEXT")
    verification = verify_citations(answer, chunks)
    if refused:
        verification["valid"] = True
        verification["uncited"] = False

    return {
        "question": question,
        "answer": answer,
        "refused": refused,
        "retrieved": [c["citation"] for c in chunks],
        "verification": verification,
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Are alcoholic beverages an allowable cost?"
    r = ask(q)

    print(f"Q: {r['question']}\n")
    print(r["answer"])
    print(f"\nRetrieved: {', '.join(r['retrieved'])}")
    print(f"Cited: {', '.join(r['verification']['cited']) or 'none'}")
    print(f"Refused: {r['refused']}")
    if not r["verification"]["valid"]:
        print(f"⚠️  UNSUPPORTED CITATIONS: {r['verification']['unsupported']}")
