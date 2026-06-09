"""Update release metadata in CITATION.cff before the release tag is created.

Python Semantic Release runs this script as its ``build_command`` after it has
resolved ``NEW_VERSION`` and before it creates the release commit and tag. That
ordering keeps the tagged source archive aligned with the citation metadata.

The script deliberately removes the optional ``commit`` field. A release commit
cannot contain its own final SHA because the SHA is calculated from the committed
tree, including this file's contents. The Git tag is the stable pointer to the
exact release commit instead.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import re


CITATION_PATH = Path("CITATION.cff")


def _replace_or_append_scalar(text: str, key: str, value: str) -> str:
    """Replace a top-level scalar key in a CFF file, appending if missing."""

    pattern = re.compile(rf"^{re.escape(key)}:\s+.*$", flags=re.MULTILINE)
    replacement = f"{key}: {value}"

    if pattern.search(text):
        return pattern.sub(replacement, text)

    separator = "" if text.endswith("\n") else "\n"
    return f"{text}{separator}{replacement}\n"


def _remove_top_level_scalar(text: str, key: str) -> str:
    """Remove a top-level scalar key whose value is no longer source-stable."""

    pattern = re.compile(rf"^{re.escape(key)}:\s+.*\n?", flags=re.MULTILINE)
    return pattern.sub("", text)


def main() -> None:
    """Synchronize CFF version/date fields using PSR's release environment."""

    release_version = os.environ.get("NEW_VERSION", "").strip()
    if not release_version:
        raise SystemExit("NEW_VERSION is required from python-semantic-release")

    if not CITATION_PATH.exists():
        raise SystemExit("CITATION.cff not found; cannot sync release metadata")

    release_date = datetime.now(timezone.utc).date().isoformat()
    text = CITATION_PATH.read_text(encoding="utf-8")
    text = _remove_top_level_scalar(text, "commit")
    text = _replace_or_append_scalar(text, "version", release_version)
    text = _replace_or_append_scalar(text, "date-released", f"'{release_date}'")
    CITATION_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()