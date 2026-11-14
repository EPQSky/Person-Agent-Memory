from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / 'scripts/verify-release-matrix.sh'

def test_matrix_script_help_and_isolation_contract() -> None:
    result = subprocess.run([str(SCRIPT), '--help'], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert '--deb' in result.stdout and '--rpm' in result.stdout
    text = SCRIPT.read_text(encoding='utf-8')
    markers = (
        'ubuntu:22.04',
        'ubuntu:24.04',
        'rockylinux:9',
        'fedora:latest',
        '--network none',
        'XDG_STATE_HOME',
        'setup --uninstall',
    )
    for marker in markers:
        assert marker in text

def test_matrix_requires_artifacts(tmp_path: Path) -> None:
    result = subprocess.run(
        [str(SCRIPT), '--deb', str(tmp_path / 'a.deb'), '--rpm', str(tmp_path / 'a.rpm')],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert 'artifact' in result.stderr
