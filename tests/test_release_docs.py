from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
DOC = ROOT / 'docs/user/release.md'


def test_release_document_covers_package_and_script_channels() -> None:
    text = DOC.read_text(encoding='utf-8')
    for marker in (
        '发行包渠道',
        '脚本渠道',
        'sudo apt install',
        'sudo dnf install',
        'personal-agent-memory setup --uninstall',
        'SHA256SUMS',
        'ASSET-MANIFEST.json',
        'verify-release-matrix.sh',
        'build-packages-container.sh',
        'personal-agent-memory-<version>-offline-repositories.tar.gz',
        '--network none',
        'v0.3.1',
    ):
        assert marker in text


def test_documented_asset_names_match_package_builder() -> None:
    builder = (ROOT / 'scripts/build-packages.sh').read_text(encoding='utf-8')
    assert 'personal-agent-memory_${VERSION}_amd64.deb' in builder
    docs = (ROOT / 'docs/user/release.md').read_text(encoding='utf-8')
    assert 'personal-agent-memory-<version>-1.x86_64.rpm' in docs
