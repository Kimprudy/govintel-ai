import os
import requests
from govintel.config import RAW_DIR

BASE_URL = "https://www.ecfr.gov/api/versioner/v1"


def download_title(title_number: int, date: str) -> str:
    """Download the full XML for one CFR title. Returns the saved file path."""
    url = f"{BASE_URL}/full/{date}/title-{title_number}.xml"
    print(f"Downloading {url}")

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"title-{title_number}-{date}.xml")

    with open(path, "wb") as f:
        f.write(response.content)

    size_mb = len(response.content) / 1_000_000
    print(f"Saved {path} ({size_mb:.1f} MB)")
    return path


if __name__ == "__main__":
    download_title(2, "2026-09-01")
