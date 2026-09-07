from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_native_verifier_uses_an_isolated_home_and_cleans_up(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]
    script = (repo_root / "scripts" / "verify-native-install.sh").read_text(encoding="utf-8")

    assert 'work_root="$(mktemp -d' in script
    assert 'export HOME="$work_root/home"' in script
    assert 'export CODEX_HOME="$HOME/.codex"' in script
    assert 'export XDG_CONFIG_HOME="$work_root/config"' in script
    assert 'export XDG_DATA_HOME="$work_root/data"' in script
    assert 'export XDG_BIN_HOME="$work_root/bin"' in script
    assert 'rm -rf "$work_root"' in script
    assert 'uv tool install' in script
    assert 'codex plugin marketplace add "$repo_root"' in script
    assert 'codex plugin add personal-agent-memory@personal-agent-memory' in script

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    isolated = tmp_path / "isolated"
    fake_mktemp = fake_bin / "mktemp"
    fake_mktemp.write_text(
        f"#!/bin/sh\nmkdir -p '{isolated}'\nprintf '%s\\n' '{isolated}'\n",
        encoding="utf-8",
    )
    fake_mktemp.chmod(0o755)
    for name in ("git", "node", "codex", "python3"):
        command = fake_bin / name
        command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        command.chmod(0o755)
    fake_uv = fake_bin / "uv"
    fake_uv.write_text("#!/bin/sh\nexit 19\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

    result = subprocess.run(
        [str(repo_root / "scripts" / "verify-native-install.sh")],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 19
    assert not isolated.exists()
