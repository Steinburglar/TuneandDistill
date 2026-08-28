"""Build the runnable Colab notebook from the vendored upstream tutorial.

    python docs/tutorial/build_notebook.py

Reads
    NequIP_Tutorial.ipynb   upstream, verbatim -- never edited by hand
    distill_section.md      our section, one cell per `## [markdown]`/`## [code]`

Writes
    NequIP_Distill_Tutorial.ipynb

Everything upstream is copied cell for cell; our cells are appended. Regenerate
after editing `distill_section.md` rather than editing the output notebook.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
UPSTREAM = HERE / "NequIP_Tutorial.ipynb"
SECTION = HERE / "distill_section.md"
OUT = HERE / "NequIP_Distill_Tutorial.ipynb"

# Anything before the first marker is repo-internal notes, not notebook content.
MARKER = re.compile(r"^## \[(markdown|code)\]\s*$", re.MULTILINE)
FENCE = re.compile(r"^```[a-z]*\n(.*)\n```\s*$", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->\s*", re.DOTALL)


def parse_section(text: str) -> list[dict]:
    """`## [markdown]` / `## [code]` headings -> notebook cells."""
    parts = MARKER.split(text)[1:]  # drop the preamble
    cells = []
    for kind, body in zip(parts[0::2], parts[1::2]):
        body = HTML_COMMENT.sub("", body).strip()
        if kind == "code":
            fenced = FENCE.match(body)
            if not fenced:
                raise ValueError(f"code cell is not a single fenced block:\n{body[:200]}")
            body = fenced.group(1)
        if not body:
            continue
        cells.append({"cell_type": kind, "source": body.splitlines(keepends=True)})
    return cells


def main() -> None:
    nb = json.loads(UPSTREAM.read_text())
    n_upstream = len(nb["cells"])
    # nbformat >= 4.5 requires a unique id per cell.
    needs_id = (nb["nbformat"], nb["nbformat_minor"]) >= (4, 5)

    for i, cell in enumerate(parse_section(SECTION.read_text())):
        new = {"cell_type": cell["cell_type"], "metadata": {}, "source": cell["source"]}
        if cell["cell_type"] == "code":
            new["execution_count"] = None
            new["outputs"] = []
        if needs_id:
            new["id"] = f"distill-{i:02d}"
        nb["cells"].append(new)

    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"{OUT.name}: {n_upstream} upstream cells + {len(nb['cells']) - n_upstream} ours")


if __name__ == "__main__":
    main()
