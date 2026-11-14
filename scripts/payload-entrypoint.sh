#!/bin/sh
set -eu
PREFIX='@INSTALL_PREFIX@'
SITE_PACKAGES=''
for candidate in "${PREFIX}"/venv/lib/python*/site-packages; do
  if [ -d "$candidate" ]; then
    SITE_PACKAGES=$candidate
    break
  fi
done
[ -n "$SITE_PACKAGES" ] || { echo 'payload virtual environment is missing site-packages' >&2; exit 1; }
export PYTHONPATH="${SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PREFIX}/runtime/bin/python3" -m personal_agent_memory.cli "$@"
