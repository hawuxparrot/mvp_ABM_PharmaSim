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

# If this build dir was configured on another machine/path, CMake will fail.
# Detect stale cache entries and clean only when necessary.
CACHE_FILE="$BUILD_DIR/CMakeCache.txt"
if [[ -f "$CACHE_FILE" ]]; then
  CACHE_SOURCE="$(sed -n 's|^CMAKE_HOME_DIRECTORY:INTERNAL=||p' "$CACHE_FILE" | head -n1)"
  CACHE_BUILD="$(sed -n 's|^CMAKE_CACHEFILE_DIR:INTERNAL=||p' "$CACHE_FILE" | head -n1)"
  BUILD_ABS="$(cd "$BUILD_DIR" && pwd)"

  if [[ -n "$CACHE_SOURCE" && "$CACHE_SOURCE" != "$ROOT" ]] || [[ -n "$CACHE_BUILD" && "$CACHE_BUILD" != "$BUILD_ABS" ]]; then
    echo "==> Detected stale CMake cache in $BUILD_DIR"
    echo "    cached source: ${CACHE_SOURCE:-<unset>}"
    echo "    current source: $ROOT"
    echo "    cached build:  ${CACHE_BUILD:-<unset>}"
    echo "    current build: $BUILD_ABS"
    echo "==> Removing incompatible build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
  fi
fi

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