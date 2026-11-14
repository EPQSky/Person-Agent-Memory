from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
BUILD = ROOT / 'scripts/build-offline-repository.sh'
MANIFEST = ROOT / 'scripts/create-offline-manifest.py'
VERIFY = ROOT / 'scripts/verify-offline-repository.sh'

def test_scripts_are_offline_and_have_help() -> None:
    result = subprocess.run([str(BUILD), '--help'], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert '--packages' in result.stdout and '--output' in result.stdout
    text = VERIFY.read_text(encoding='utf-8')
    assert '--network none' in text
    assert 'apt' in text and 'rpm' in text

def test_manifest_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'b').write_bytes(b'b')
    (root / 'a').write_bytes(b'a')
    out = tmp_path / 'manifest.json'
    for _ in range(2):
        subprocess.run([str(MANIFEST), '--root', str(root), '--output', str(out)], check=True)
        current = out.read_text(encoding='utf-8')
        if 'first' not in locals():
            first = current
        else:
            assert current == first
    data = json.loads(first)
    assert [item['path'] for item in data['assets']] == ['a', 'b']

def test_builder_rejects_empty_package_dir(tmp_path: Path) -> None:
    packages = tmp_path / 'packages'
    packages.mkdir()
    result = subprocess.run(
        [str(BUILD), '--packages', str(packages), '--output', str(tmp_path / 'out')],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert 'no deb packages' in result.stderr
