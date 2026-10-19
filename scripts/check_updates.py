"""Pull a newer eCFR snapshot of Title 21 and report which sections changed.

Usage:
    python scripts/check_updates.py [--date YYYY-MM-DD] [--out eval/reports/changes.md]

The database is not modified; run ``cfr-rag reindex --since <old date>`` to apply.
"""

from __future__ import annotations

import sys

from cfr_rag.cli import main

if __name__ == "__main__":
    sys.exit(main(["check-updates", *sys.argv[1:]]))
