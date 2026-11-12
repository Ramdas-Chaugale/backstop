"""Fail if the committed notebook no longer matches its builder.

The notebook is generated. That only means something if nothing can edit the
committed file without the builder changing too, so CI compares the two.

Outputs are ignored on purpose. Re-executing produces different timings and
image bytes, and that is not drift. Only the source of each cell is compared.

    python notebooks/check_notebook.py
"""

from __future__ import annotations

import os
import sys

import nbformat as nbf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_notebook import OUT, build  # noqa: E402


def sources(nb) -> list[tuple[str, str]]:
    return [(c.cell_type, c.source.strip()) for c in nb.cells]


def main() -> int:
    if not os.path.exists(OUT):
        print(f"{os.path.basename(OUT)} is missing. Run build_notebook.py")
        return 1

    committed = sources(nbf.read(OUT, as_version=4))
    fresh = sources(build())

    if committed == fresh:
        print(f"{os.path.basename(OUT)} matches its builder ({len(fresh)} cells)")
        return 0

    print(f"{os.path.basename(OUT)} does not match build_notebook.py")
    if len(committed) != len(fresh):
        print(f"  committed has {len(committed)} cells, the builder makes {len(fresh)}")
    for i, (a, b) in enumerate(zip(committed, fresh)):
        if a != b:
            print(f"  first difference at cell {i} ({a[0]})")
            print(f"    committed: {a[1][:90]!r}")
            print(f"    builder  : {b[1][:90]!r}")
            break
    print("\n  Rebuild and re-execute:")
    print("    python notebooks/build_notebook.py")
    print("    jupyter nbconvert --execute --inplace notebooks/01_the_journey.ipynb")
    return 1


if __name__ == "__main__":
    sys.exit(main())
