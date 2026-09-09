import json
import os
import re
from govintel.config import PROCESSED_DIR

MAX_CHARS = 1800
MIN_CHARS = 120


def clean_heading(heading: str) -> str:
    """Strip the leading section symbol/number and collapse whitespace."""
    h = re.sub(r"^§\s*[\d.]+\s*", "", heading)
    return re.sub(r"\s+", " ", h).strip()


def split_on_subsections(text: str, max_chars: int = MAX_CHARS) -> list:
    """Split at (a)/(b)/(1)/(2) paragraph boundaries, never mid-sentence."""
    if len(text) <= max_chars:
        return [text]

    # Break into paragraph units first
    parts = re.split(r"\n(?=\([a-z0-9]{1,3}\))", text)

    chunks = []
    current = ""
    for part in parts:
        if len(current) + len(part) + 1 <= max_chars:
            current = f"{current}\n{part}" if current else part
        else:
            if current.strip():
                chunks.append(current.strip())
            # A single paragraph longer than max_chars: hard-split it
            if len(part) > max_chars:
                for i in range(0, len(part), max_chars):
                    chunks.append(part[i:i + max_chars].strip())
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_section(section: dict) -> list:
    heading = clean_heading(section["heading"])
    prefix = f"{section['citation']} — {heading}\n\n"

    metadata = {k: v for k, v in section.items() if k != "text"}
    metadata["heading"] = heading

    pieces = split_on_subsections(section["text"])

    chunks = []
    for idx, piece in enumerate(pieces):
        if not piece.strip():
            continue
        chunks.append({
            **metadata,
            "chunk_index": idx,
            "chunk_id": f"{section['citation']}#{idx}",
            "text": prefix + piece.strip(),
        })
    return chunks


def chunk_title(title_number: int) -> str:
    in_path = os.path.join(PROCESSED_DIR, f"title-{title_number}-sections.jsonl")
    out_path = os.path.join(PROCESSED_DIR, f"title-{title_number}-chunks.jsonl")

    all_chunks = []
    with open(in_path) as f:
        for line in f:
            all_chunks.extend(chunk_section(json.loads(line)))

    with open(out_path, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    print(f"{len(all_chunks)} chunks -> {out_path}")
    return out_path


if __name__ == "__main__":
    chunk_title(2)
