"""Copy FastAPI tutorial pages into data/, with their code examples inlined.

The FastAPI docs reference code with placeholders like
    {* ../../docs_src/handling_errors/tutorial001_py310.py hl[1] *}
Neither the embedding model nor Claude can follow those, so each one is
replaced with the real code in a ```python block.
"""

import re
from pathlib import Path

FASTAPI_REPO = Path.home() / "Documents" / "fastapi-src"
TUTORIAL_DIR = FASTAPI_REPO / "docs" / "en" / "docs" / "tutorial"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAGES = [
    "background-tasks",
    "body",
    "first-steps",
    "handling-errors",
    "path-params",
    "query-params",
    "response-model",
    "testing",
]
PLACEHOLDER = re.compile(r"\{\* \.\./\.\./(docs_src/\S+\.py)[^*]*\*\}")


def inline_code(markdown: str, repo: Path = FASTAPI_REPO) -> str:
    """Replace every code placeholder with the file it points to."""

    def replace(match: re.Match) -> str:
        code = (repo / match.group(1)).read_text(encoding="utf-8")
        return f"```python\n{code.rstrip()}\n```"

    return PLACEHOLDER.sub(replace, markdown)


if __name__ == "__main__":
    for page in PAGES:
        text = (TUTORIAL_DIR / f"{page}.md").read_text(encoding="utf-8")
        (DATA_DIR / f"{page}.md").write_text(inline_code(text), encoding="utf-8")
        print(f"{page}.md: {len(PLACEHOLDER.findall(text))} code blocks inlined")
