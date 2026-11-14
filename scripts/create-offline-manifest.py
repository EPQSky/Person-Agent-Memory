#!/usr/bin/env python3
"""Create deterministic metadata for an offline repository tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files = sorted(
        path for path in root.rglob('*') if path.is_file() and path.name != args.output.name
    )
    assets = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        assets.append(
            {
                'path': relative,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'size': path.stat().st_size,
            }
        )
    args.output.write_text(
        json.dumps({'schema_version': 1, 'assets': assets}, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
