from __future__ import annotations

import hashlib
import json
import pathlib
import sys

prefix = pathlib.Path("/payload" + sys.argv[1])
plugin_manifest_path = pathlib.Path(
    "/src/plugins/personal-agent-memory/.codex-plugin/plugin.json"
)
manifest = json.loads(plugin_manifest_path.read_text())
version = manifest["version"]
manifest_out = {
    "schema_version": 1,
    "project": "personal-agent-memory",
    "project_version": version,
    "plugin_version": version,
    "runtime_version": sys.argv[2],
    "install_prefix": sys.argv[1],
    "service_registration": sys.argv[3],
    "entrypoint": sys.argv[1] + "/bin/personal-agent-memory",
    "plugin_manifest": sys.argv[1] + "/plugins/personal-agent-memory/.codex-plugin/plugin.json",
    "requires_python": ">=3.11,<3.14",
}
digest = hashlib.sha256()
for path in sorted((prefix / "plugins/personal-agent-memory").rglob("*")):
    if path.is_file():
        digest.update(str(path.relative_to(prefix)).encode())
        digest.update(path.read_bytes())
manifest_out["plugin_tree_sha256"] = digest.hexdigest()
pathlib.Path("/payload/manifest.json").write_text(
    json.dumps(manifest_out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
