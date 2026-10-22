"""CLI entry point for opt-in fast sparse-row building plus stamp dedup."""

from .stamp_dedup import install as install_stamp_dedup
from .sparse_rows_fast import install as install_fast_sparse_rows

install_stamp_dedup()
install_fast_sparse_rows()

from .__main__ import main

if __name__ == "__main__":
    main()
