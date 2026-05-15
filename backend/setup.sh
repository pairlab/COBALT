#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"
DOCKER_ENV_FILE="${SCRIPT_DIR}/docker.env"
WITH_ISAACLAB=0

if [[ "${1:-}" == "--with-isaaclab" ]]; then
    WITH_ISAACLAB=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: ./setup.sh [--with-isaaclab]"
    exit 1
fi

set_compose_profile() {
    local profile="$1"
    local tmp_file
    tmp_file="$(mktemp)"

    if [[ -f "${DOCKER_ENV_FILE}" ]]; then
        grep -v '^COMPOSE_PROFILES=' "${DOCKER_ENV_FILE}" > "${tmp_file}" || true
    fi

    printf "\nCOMPOSE_PROFILES=%s\n" "${profile}" >> "${tmp_file}"
    mv "${tmp_file}" "${DOCKER_ENV_FILE}"
}

if [[ "${WITH_ISAACLAB}" -eq 1 ]]; then
    set_compose_profile "isaaclab"
    echo "Configured compose profile in docker.env: isaaclab"

    echo "Building IsaacLab base image..."
    echo "[X11]" > "${SRC_DIR}/isaaclab/docker/.container.cfg"
    echo "x11_forwarding_enabled = 0" >> "${SRC_DIR}/isaaclab/docker/.container.cfg"

    sudo "${SRC_DIR}/isaaclab/docker/container.py" start
    sudo "${SRC_DIR}/isaaclab/docker/container.py" stop

    echo "Building IsaacLab-enabled simulator image..."
    sudo docker build -t cobalt-simulators-isaaclab:latest -f "${SRC_DIR}/Dockerfile.isaaclab" "${SRC_DIR}"
else
    set_compose_profile "core"
    echo "Configured compose profile in docker.env: core"

    echo "Building core simulator image (no IsaacLab required)..."
    sudo docker build -t cobalt-simulators-core:latest -f "${SRC_DIR}/Dockerfile.core" "${SRC_DIR}"
fi