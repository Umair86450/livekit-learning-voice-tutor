from __future__ import annotations

import sys


def main() -> int:
    print(
        "Deprecated: build_rag_index.py is replaced by prepared-data + Qdrant flow.\n"
        "Use:\n"
        "1) python scripts/prepare_panaversity_rag_data.py\n"
        "2) start local Qdrant docker\n"
        "3) run agent with RAG_ENABLED=true",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

