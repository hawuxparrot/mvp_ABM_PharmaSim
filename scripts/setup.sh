#!/usr/bin/env bash
set -euo pipefail

# PharmaSim dev setup: uv env + CMake build (engine + nanobind extension)
# Prerequisites (install via your OS): git, cmake, ninja (or make), a C++ compiler, Python dev headers if needed.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || { echo "Install uv: https://docs.astral.sh/uv/"; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "Install cmake"; exit 1; }

echo "==> uv sync (install Python deps + create .venv)"
uv sync --all-groups

PY="$(uv run python -c "import sys; print(sys.executable)")"
echo "==> Using Python_EXECUTABLE: $PY"

BUILD_DIR="${BUILD_DIR:-build}"

echo "==> CMake configure: $BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR" \
  -G Ninja \
  -DPHARMASIM_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$PY" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "==> CMake build"
cmake --build "$BUILD_DIR"

if [[ -f "$BUILD_DIR/compile_commands.json" ]]; then
  ln -sf "$BUILD_DIR/compile_commands.json" "$ROOT/compile_commands.json"
  echo "==> Linked compile_commands.json -> $BUILD_DIR/compile_commands.json (for clangd)"
fi

echo "==> Done. Try: uv run pytest"