import json
import sys
from govintel.config import PROCESSED_DIR


def show(citation_fragment: str):
    """Print the full text of any section matching the fragment."""
    path = f"{PROCESSED_DIR}/title-2-sections.jsonl"
    found = 0
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            if citation_fragment.lower() in s["citation"].lower():
                found += 1
                print("=" * 70)
                print(s["citation"], "|", s["heading"])
                print("=" * 70)
                print(s["text"])
                print()
    if not found:
        print(f"No section matching '{citation_fragment}'")


if __name__ == "__main__":
    show(" ".join(sys.argv[1:]) or "200.414")
