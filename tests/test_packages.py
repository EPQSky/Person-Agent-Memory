from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
BUILD = ROOT / "scripts/build-packages.sh"
METADATA = ROOT / "scripts/package-metadata.py"


def test_package_builder_help_and_scripts_are_file_only() -> None:
    result = subprocess.run([str(BUILD), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--payload" in result.stdout
    assert "--output" in result.stdout
    text = BUILD.read_text(encoding="utf-8")
    assert "systemctl --user" not in text
    assert "$HOME" not in text
    assert "HOME=" not in text
    for marker in ("postinst", "prerm", "postrm"):
        assert marker in text
    lifecycle = "\n".join(
        (ROOT / "packaging/deb" / name).read_text(encoding="utf-8")
        for name in ("postinst", "prerm", "postrm")
    )
    lifecycle += (ROOT / "packaging/rpm/personal-agent-memory.spec.in").read_text(encoding="utf-8")
    for marker in ("%post", "%preun", "%postun"):
        assert marker in lifecycle


def make_payload(tmp_path: Path, version: str = "0.2.1") -> tuple[Path, Path]:
    source = tmp_path / "source"
    payload = tmp_path / "payload"
    source.mkdir()
    (source / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    (source / "README.md").write_text("readme\n", encoding="utf-8")
    (source / "LICENSE").write_text("license\n", encoding="utf-8")
    plugin = payload / "opt/personal-agent-memory/plugins/personal-agent-memory/.codex-plugin"
    command = payload / "opt/personal-agent-memory/bin"
    plugin.mkdir(parents=True)
    command.mkdir(parents=True)
    (payload / "manifest.json").write_text(
        json.dumps(
            {
                "project_version": version,
                "plugin_version": version,
                "install_prefix": "/opt/personal-agent-memory",
                "entrypoint": "/opt/personal-agent-memory/bin/personal-agent-memory",
                "plugin_manifest": (
                    "/opt/personal-agent-memory/plugins/personal-agent-memory/"
                    ".codex-plugin/plugin.json"
                ),
            }
        ),
        encoding="utf-8",
    )
    (plugin / "plugin.json").write_text(json.dumps({"version": version}), encoding="utf-8")
    command_path = command / "personal-agent-memory"
    command_path.write_text("#!/bin/sh\n", encoding="utf-8")
    command_path.chmod(0o755)
    return source, payload


def test_package_metadata_requires_matching_project_payload_and_plugin_versions(
    tmp_path: Path,
) -> None:
    source, payload = make_payload(tmp_path)
    output = tmp_path / "package-metadata.json"
    result = subprocess.run(
        [
            str(METADATA),
            "--source",
            str(source),
            "--payload",
            str(payload),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["version"] == "0.2.1"
    assert metadata["payload_prefix"] == "/opt/personal-agent-memory"
    assert metadata["package_installation"] == "files-only-no-user-operations"


def test_package_metadata_rejects_version_drift(tmp_path: Path) -> None:
    source, payload = make_payload(tmp_path, version="0.2.1")
    drift_manifest = json.loads((payload / "manifest.json").read_text(encoding="utf-8"))
    drift_manifest.update({"project_version": "0.2.0", "plugin_version": "0.2.0"})
    (payload / "manifest.json").write_text(json.dumps(drift_manifest), encoding="utf-8")
    (
        payload
        / "opt/personal-agent-memory/plugins/personal-agent-memory/.codex-plugin/plugin.json"
    ).write_text(json.dumps({"version": "0.2.0"}), encoding="utf-8")
    result = subprocess.run(
        [
            str(METADATA),
            "--source",
            str(source),
            "--payload",
            str(payload),
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "version mismatch" in result.stderr


def test_wrapper_uses_only_payload_command() -> None:
    wrapper = (ROOT / "packaging/personal-agent-memory-wrapper").read_text(encoding="utf-8")
    assert 'exec /opt/personal-agent-memory/bin/personal-agent-memory "$@"' in wrapper
    assert "systemctl" not in wrapper


def test_package_builder_invokes_both_package_managers_with_fake_tools(tmp_path: Path) -> None:
    source, payload = make_payload(tmp_path)
    output = tmp_path / "out"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "dpkg-deb").write_text(
        '#!/bin/sh\nset -eu\nout=""\nfor arg do out=$arg; done\ntouch "$out"\n', encoding="utf-8"
    )
    (fakebin / "rpmbuild").write_text(
        """#!/bin/sh
set -eu
top=""
for arg do
  case "$arg" in
    _topdir*) top=${arg#_topdir } ;;
  esac
  shift
done
mkdir -p "$top/RPMS/x86_64"
touch "$top/RPMS/x86_64/personal-agent-memory-0.2.1.rpm"
""",
        encoding="utf-8",
    )
    for tool in ("dpkg-deb", "rpmbuild"):
        (fakebin / tool).chmod(0o755)
    env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}"}
    result = subprocess.run(
        [str(BUILD), "--source", str(source), "--payload", str(payload), "--output", str(output)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "personal-agent-memory_0.2.1_amd64.deb").exists()
    assert list(output.glob("*.rpm"))


def test_package_builder_rejects_output_overlap(tmp_path: Path) -> None:
    source, payload = make_payload(tmp_path)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    for tool in ("dpkg-deb", "rpmbuild"):
        path = fakebin / tool
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}"}
    result = subprocess.run(
        [str(BUILD), "--source", str(source), "--payload", str(payload), "--output", str(payload)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "must not overlap" in result.stderr


def test_package_e2e_harness_declares_full_acceptance_flow() -> None:
    harness = (ROOT / "scripts/verify-package-e2e.sh").read_text(encoding="utf-8")
    for marker in (
        "--network none",
        "setup",
        "health",
        "/api/v1/memories",
        "/api/v1/search",
        "setup --uninstall",
        "SKIP:",
    ):
        assert marker in harness
