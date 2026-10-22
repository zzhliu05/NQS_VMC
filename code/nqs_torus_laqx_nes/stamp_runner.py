"""Run the normal CLI with process-local stamp-table neighbor dedup enabled."""

import os
import resource

from .stamp_dedup import install

install()

from .__main__ import main

if __name__ == "__main__":
    try:
        main()
    finally:
        memory_path = os.environ.get("NQS_HOST_MEMORY_LOG")
        if memory_path:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            with open(memory_path, "w", encoding="utf-8") as stream:
                stream.write(f"max_rss_kib={usage.ru_maxrss}\n")
