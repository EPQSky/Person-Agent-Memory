from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/build-payload.sh"


def test_payload_builder_does_not_require_host_python() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert "command -v python3" not in script
    assert "python3 -c" not in script


def test_payload_builder_help_is_self_documenting() -> None:
    result = subprocess.run([str(SCRIPT), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--runtime-sha256" in result.stdout
    assert "--install-prefix" in result.stdout
    assert "--service-registration" in result.stdout
    assert "--base-image" in result.stdout


def test_payload_builder_rejects_mutable_base_image(tmp_path: Path) -> None:
    output = tmp_path / "payload"
    result = subprocess.run(
        [
            str(SCRIPT),
            "--output",
            str(output),
            "--base-image",
            "ubuntu:22.04",
            "--runtime-url",
            "https://example.invalid/python.tar.gz",
            "--runtime-sha256",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "immutable" in result.stderr
    assert not output.exists()


def test_payload_builder_preserves_existing_image_tag(tmp_path: Path) -> None:
    engine = tmp_path / "engine"
    log = tmp_path / "engine.log"
    engine.write_text(
        """#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$PAM_ENGINE_LOG"
if [ "$1" = image ] && [ "$2" = inspect ]; then
  [ "${PAM_FAKE_EXISTING_IMAGE:-}" = "$3" ] && exit 0
  exit 1
fi
if [ "$1" = run ]; then
  emit=false
  for arg in "$@"; do
    [ "$arg" = --emit ] && emit=true
    case "$arg" in
      type=bind,src=*,dst=/output) output=${arg#type=bind,src=}; output=${output%,dst=/output};;
      /*) [ "$arg" != /output ] && prefix=$arg;;
    esac
  done
  [ "$emit" = true ] || exit 0
  prefix=${prefix:-/opt/personal-agent-memory}
  mkdir -p "$output$prefix/runtime/bin" "$output$prefix/bin" \
    "$output$prefix/plugins/personal-agent-memory/.codex-plugin"
  printf '#!/bin/sh\nif [ "${1:-}" = - ]; then exec /usr/bin/python3 "$@"; fi\n' \
    > "$output$prefix/runtime/bin/python3"
  printf 'printf "0.3.0\\n"\n' >> "$output$prefix/runtime/bin/python3"
  chmod 755 "$output$prefix/runtime/bin/python3"
  printf '#!/bin/sh\n' > "$output$prefix/bin/personal-agent-memory"
  printf 'printf "0.3.0\\n"\n' >> "$output$prefix/bin/personal-agent-memory"
  chmod 755 "$output$prefix/bin/personal-agent-memory"
  printf '{"version":"0.3.0"}\n' \
    > "$output$prefix/plugins/personal-agent-memory/.codex-plugin/plugin.json"
  printf '{"project_version":"0.3.0","plugin_version":"0.3.0"}\n' \
    > "$output/manifest.json"
fi
""",
        encoding="utf-8",
    )
    engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
    output = tmp_path / "payload"
    env = {
        **os.environ,
        "PAM_CONTAINER_ENGINE": str(engine),
        "PAM_ENGINE_LOG": str(log),
        "PAM_FAKE_EXISTING_IMAGE": "protected-image",
    }
    result = subprocess.run(
        [
            str(SCRIPT),
            "--output",
            str(output),
            "--image",
            "protected-image",
            "--runtime-url",
            "https://example.invalid/python.tar.gz",
            "--runtime-sha256",
            "0" * 64,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    log_text = log.read_text(encoding="utf-8")
    assert "build" in log_text and "protected-image-payload-build-" in log_text
    assert "image rm protected-image\n" not in log_text


def test_payload_builder_rejects_nonempty_output_and_system_root(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "old-artifact").write_text("must not be mixed", encoding="utf-8")
    result = subprocess.run(
        [
            str(SCRIPT),
            "--output",
            str(existing),
            "--runtime-url",
            "https://example.invalid/python.tar.gz",
            "--runtime-sha256",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "empty" in result.stderr
    assert (existing / "old-artifact").read_text(encoding="utf-8") == "must not be mixed"

    root_result = subprocess.run(
        [
            str(SCRIPT),
            "--output",
            "/",
            "--runtime-url",
            "https://example.invalid/python.tar.gz",
            "--runtime-sha256",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert root_result.returncode == 2
    assert "dangerous" in root_result.stderr

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    symlink_result = subprocess.run(
        [
            str(SCRIPT),
            "--output",
            str(linked_parent / "payload"),
            "--runtime-url",
            "https://example.invalid/python.tar.gz",
            "--runtime-sha256",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert symlink_result.returncode == 2
    assert "symlink" in symlink_result.stderr


def test_payload_builder_rejects_comma_output_before_mount(tmp_path: Path) -> None:
    output = tmp_path / "payload,with-comma"
    result = subprocess.run(
        [
            str(SCRIPT),
            "--output",
            str(output),
            "--runtime-url",
            "https://example.invalid/python.tar.gz",
            "--runtime-sha256",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "comma" in result.stderr
    assert not output.exists()


def test_payload_builder_rejects_unsafe_prefix_before_engine(tmp_path: Path) -> None:
    output = tmp_path / "payload"
    result = subprocess.run(
        [
            str(SCRIPT),
            "--output",
            str(output),
            "--install-prefix",
            "/srv/a&b",
            "--runtime-url",
            "https://example.invalid/python.tar.gz",
            "--runtime-sha256",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "install prefix" in result.stderr
    assert not output.exists()


def test_payload_builder_rejects_unverified_runtime_before_engine(tmp_path: Path) -> None:
    engine = tmp_path / "engine"
    log = tmp_path / "engine.log"
    engine.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >>{log}\nexit 99\n", encoding="utf-8")
    engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
    output = tmp_path / "payload"
    env = {**os.environ, "PAM_CONTAINER_ENGINE": str(engine)}
    result = subprocess.run(
        [
            str(SCRIPT),
            "--output",
            str(output),
            "--runtime-url",
            "https://example.invalid/python.tar.gz",
            "--runtime-sha256",
            "0" * 63,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "64-character" in result.stderr
    assert not log.exists()
    assert not output.exists()


def test_payload_builder_emits_only_to_external_output(tmp_path: Path) -> None:
    engine = tmp_path / "engine"
    log = tmp_path / "engine.log"
    engine.write_text(
        """#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$PAM_ENGINE_LOG"
if [ "$1" = image ] && [ "$2" = inspect ]; then
  [ "${PAM_FAKE_EXISTING_IMAGE:-}" = "$3" ] && exit 0
  exit 1
fi
if [ "$1" = run ]; then
  emit=false
  for arg in "$@"; do
    [ "$arg" = --emit ] && emit=true
    case "$arg" in
      type=bind,src=*,dst=/output) output=${arg#type=bind,src=}; output=${output%,dst=/output};;
      /*) [ "$arg" != /output ] && prefix=$arg;;
    esac
  done
  if [ "$emit" != true ]; then exit 0; fi
  prefix=${prefix:-/opt/personal-agent-memory}
  mkdir -p "$output$prefix/runtime/bin" "$output$prefix/bin" \
    "$output$prefix/plugins/personal-agent-memory/.codex-plugin"
  printf '#!/bin/sh\nif [ "${1:-}" = - ]; then exec /usr/bin/python3 "$@"; fi\n' \
    > "$output$prefix/runtime/bin/python3"
  printf 'printf "0.3.0\\n"\n' >> "$output$prefix/runtime/bin/python3"
  chmod 755 "$output$prefix/runtime/bin/python3"
  printf '#!/bin/sh\n' > "$output$prefix/bin/personal-agent-memory"
  printf 'printf "0.3.0\\n"\n' >> "$output$prefix/bin/personal-agent-memory"
  chmod 755 "$output$prefix/bin/personal-agent-memory"
  printf '{"version":"0.3.0"}\n' \
    > "$output$prefix/plugins/personal-agent-memory/.codex-plugin/plugin.json"
  manifest_version=0.3.0
  [ "${PAM_FAKE_MISMATCH:-}" = 1 ] && manifest_version=0.2.0
  manifest_format='{"project":"personal-agent-memory",'
  manifest_format+='"project_version":"%s","plugin_version":"%s"}\n'
  printf "$manifest_format" "$manifest_version" "$manifest_version" \
    > "$output/manifest.json"
fi
""",
        encoding="utf-8",
    )
    engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
    output = tmp_path / "payload"
    env = {
        **os.environ,
        "PAM_CONTAINER_ENGINE": str(engine),
        "PAM_ENGINE_LOG": str(log),
    }
    result = subprocess.run(
        [
            str(SCRIPT),
            "--output",
            str(output),
            "--runtime-url",
            "https://example.invalid/python.tar.gz",
            "--runtime-sha256",
            "0" * 64,
            "--install-prefix",
            "/srv/personal-agent-memory",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "manifest.json").is_file()
    assert (output / "srv/personal-agent-memory/runtime/bin/python3").exists()
    assert "--network none" in log.read_text(encoding="utf-8")
    assert not (ROOT / "manifest.json").exists()

    mismatch_output = tmp_path / "mismatch-payload"
    mismatch_result = subprocess.run(
        [
            str(SCRIPT),
            "--output",
            str(mismatch_output),
            "--runtime-url",
            "https://example.invalid/python.tar.gz",
            "--runtime-sha256",
            "0" * 64,
        ],
        env={**env, "PAM_FAKE_MISMATCH": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert mismatch_result.returncode != 0
    assert "versions do not match" in mismatch_result.stderr
    assert not mismatch_output.exists()
