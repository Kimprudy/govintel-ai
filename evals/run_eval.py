import json
import sys
from govintel.answer import ask

GOLDEN = "evals/golden_set.jsonl"


def load_golden():
    with open(GOLDEN) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(k: int = 5):
    cases = load_golden()
    results = []

    for case in cases:
        print(f"[{case['id']}] {case['question'][:60]}...")
        r = ask(case["question"], k=k)

        expected = set(case["citations"])
        retrieved = set(r["retrieved"])
        cited = set(r["verification"]["cited"])
        should_refuse = case["type"] == "unanswerable"

        results.append({
            "id": case["id"],
            "type": case["type"],
            "question": case["question"],
            "expected": sorted(expected),
            "retrieved": sorted(retrieved),
            "cited": sorted(cited),
            "answer": r["answer"],
            "refused": r["refused"],
            # did retrieval surface the right section at all?
            "recall_hit": bool(expected & retrieved) if expected else None,
            # did the answer cite the right section?
            "citation_hit": bool(expected & cited) if expected else None,
            # citations that weren't retrieved = hallucination
            "unsupported": r["verification"]["unsupported"],
            "refusal_correct": r["refused"] == should_refuse,
        })

    return results


def report(results):
    answerable = [r for r in results if r["type"] == "answerable"]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if answerable:
        recall = sum(1 for r in answerable if r["recall_hit"]) / len(answerable)
        citation = sum(1 for r in answerable if r["citation_hit"]) / len(answerable)
        false_refusal = sum(1 for r in answerable if r["refused"]) / len(answerable)
        print(f"Recall@k (answerable):     {recall:.0%}  ({len(answerable)} cases)")
        print(f"Citation accuracy:         {citation:.0%}")
        print(f"False refusal rate:        {false_refusal:.0%}")

    if unanswerable:
        refusal = sum(1 for r in unanswerable if r["refused"]) / len(unanswerable)
        print(f"Refusal rate (unanswerable): {refusal:.0%}  ({len(unanswerable)} cases)")

    hallucinated = [r for r in results if r["unsupported"]]
    print(f"Hallucinated citations:    {len(hallucinated)}")

    print("\nFAILURES:")
    failed = False
    for r in results:
        problems = []
        if r["recall_hit"] is False:
            problems.append(f"MISSED {r['expected']}")
        if r["citation_hit"] is False:
            problems.append(f"cited {r['cited'] or 'nothing'}")
        if not r["refusal_correct"]:
            problems.append("wrong refusal behavior")
        if r["unsupported"]:
            problems.append(f"HALLUCINATED {r['unsupported']}")
        if problems:
            failed = True
            print(f"\n  [{r['id']}] {r['question']}")
            for p in problems:
                print(f"      - {p}")
            print(f"      got: {r['answer'][:150]}")

    if not failed:
        print("  none")

    with open("evals/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull output -> evals/results.json")


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    report(evaluate(k=k))
