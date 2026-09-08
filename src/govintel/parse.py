import json
import os
from lxml import etree
from govintel.config import RAW_DIR, PROCESSED_DIR


def parse_title(xml_path: str, title_number: int, snapshot_date: str) -> str:
    """Parse eCFR XML into one JSON object per section. Returns output path."""
    tree = etree.parse(xml_path)
    root = tree.getroot()

    sections = []

    for div8 in root.iter("DIV8"):
        if div8.get("TYPE") != "SECTION":
            continue

        section_num = div8.get("N", "")
        head_elem = div8.find("HEAD")
        heading = head_elem.text.strip() if head_elem is not None and head_elem.text else ""

        # Pull all text from <P> tags inside this section
        paragraphs = []
        for p in div8.findall(".//P"):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

        full_text = "\n".join(paragraphs)

        # Walk up the tree to capture the part/subpart hierarchy
        hierarchy = []
        parent = div8.getparent()
        while parent is not None:
            ptype = parent.get("TYPE")
            phead = parent.find("HEAD")
            if ptype and phead is not None and phead.text:
                hierarchy.append(f"{ptype.title()}: {phead.text.strip()}")
            parent = parent.getparent()
        hierarchy.reverse()

        part = section_num.split(".")[0] if "." in section_num else section_num

        sections.append({
            "citation": f"{title_number} CFR {section_num}",
            "title": title_number,
            "part": part,
            "section": section_num,
            "heading": heading,
            "hierarchy": hierarchy,
            "text": full_text,
            "snapshot_date": snapshot_date,
        })

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, f"title-{title_number}-sections.jsonl")

    with open(out_path, "w") as f:
        for s in sections:
            f.write(json.dumps(s) + "\n")

    print(f"Parsed {len(sections)} sections -> {out_path}")
    return out_path


if __name__ == "__main__":
    parse_title(
        xml_path=os.path.join(RAW_DIR, "title-2-2026-09-01.xml"),
        title_number=2,
        snapshot_date="2026-09-01",
    )
