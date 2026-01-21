#!/usr/bin/bash

set -eu

if [ "${DEVCONTAINER_MODE:-}" != "dev" ]; then
  exit 0
fi

mkdir -p /workspaces/.codex
ln -sfn /workspaces/.codex ~/.codex
