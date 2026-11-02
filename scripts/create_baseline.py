"""Export the split Python baseline files as Jupyter notebooks."""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [*sorted(ROOT.glob("task*/02_baseline.py")), ROOT / "03_make_submission.py"]


def make_notebook(path):
    cells = []
    kind = None
    lines = []

    def append_cell():
        if kind is None or not lines:
            return
        if kind == "markdown":
            text = "\n".join(line[2:] if line.startswith("# ") else "" for line in lines).strip()
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell("\n".join(lines).strip()))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            append_cell()
            kind = "markdown" if "[markdown]" in line else "code"
            lines = []
        else:
            lines.append(line)
    append_cell()
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.source_file = path.relative_to(ROOT).as_posix()
    notebook.metadata.execution_status = "not_run"
    target = path.with_suffix(".ipynb")
    nbformat.write(notebook, target)
    print("Created", target.relative_to(ROOT))


if __name__ == "__main__":
    for source in SOURCES:
        make_notebook(source)
