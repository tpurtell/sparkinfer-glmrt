#!/usr/bin/env bash
# Install the source-locked B12X wheel into an existing compatible venv.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path=${1:?Pass the destination venv path}
uv_binary=${UV_BIN:-uv}

(cd "${bundle_dir}" && sha256sum --check SHA256SUMS)
"${uv_binary}" pip install \
  --python "${venv_path}/bin/python" \
  --no-deps \
  --require-hashes \
  -r "${bundle_dir}/requirements-github.txt"
(cd / && "${venv_path}/bin/python" -c \
  'import pathlib, sys, b12x; location = pathlib.Path(b12x.__file__).resolve(); prefix = pathlib.Path(sys.prefix).resolve(); assert location.is_relative_to(prefix), (location, prefix); print("b12x_install=PASS")')
