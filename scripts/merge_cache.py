#!/usr/bin/env python
"""Merge response caches produced on different machines (e.g. cloud labels prefetched on the
5090 into the Jetson's cache). Keys are content hashes, so entries never conflict.

  python scripts/merge_cache.py results/cache_5090.sqlite results/cache.sqlite
"""
import sqlite3
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: merge_cache.py SRC.sqlite DST.sqlite")
    src, dst = sys.argv[1], sys.argv[2]
    con = sqlite3.connect(dst)
    con.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
    before = con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    con.execute("ATTACH DATABASE ? AS src", (src,))
    con.execute("INSERT OR IGNORE INTO cache (k, v) SELECT k, v FROM src.cache")
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    print(f"merged {after - before} new entries into {dst} ({after} total)")


if __name__ == "__main__":
    main()
