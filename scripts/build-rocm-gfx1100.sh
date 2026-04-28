#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-rocm-gfx1100}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_TARGETS="${GPU_TARGETS:-gfx1100}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
GGML_CUDA_GRAPHS="${GGML_CUDA_GRAPHS:-ON}"
JOBS="${JOBS:-$(nproc)}"

if [[ -x "$ROCM_PATH/llvm/bin/amdclang" && -x "$ROCM_PATH/llvm/bin/amdclang++" ]]; then
    CC_BIN="$ROCM_PATH/llvm/bin/amdclang"
    CXX_BIN="$ROCM_PATH/llvm/bin/amdclang++"
elif [[ -x "$ROCM_PATH/llvm/bin/clang" && -x "$ROCM_PATH/llvm/bin/clang++" ]]; then
    CC_BIN="$ROCM_PATH/llvm/bin/clang"
    CXX_BIN="$ROCM_PATH/llvm/bin/clang++"
elif command -v clang >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1; then
    CC_BIN="$(command -v clang)"
    CXX_BIN="$(command -v clang++)"
else
    echo "No ROCm/Clang compiler found." >&2
    echo "Expected one of:" >&2
    echo "  $ROCM_PATH/llvm/bin/amdclang and $ROCM_PATH/llvm/bin/amdclang++" >&2
    echo "  $ROCM_PATH/llvm/bin/clang and $ROCM_PATH/llvm/bin/clang++" >&2
    echo "Or install clang in PATH." >&2
    exit 1
fi

echo "Configuring ROCm build"
echo "  repo:        $REPO_DIR"
echo "  build dir:   $BUILD_DIR"
echo "  ROCm path:   $ROCM_PATH"
echo "  GPU targets: $GPU_TARGETS"
echo "  C compiler:  $CC_BIN"
echo "  CXX compiler:$CXX_BIN"
echo "  CUDA graphs: $GGML_CUDA_GRAPHS"

cmake -S "$REPO_DIR" -B "$BUILD_DIR" -G Ninja \
    -DGGML_HIP=ON \
    -DGPU_TARGETS="$GPU_TARGETS" \
    -DGGML_CUDA_GRAPHS="$GGML_CUDA_GRAPHS" \
    -DCMAKE_C_COMPILER="$CC_BIN" \
    -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

cmake --build "$BUILD_DIR" -j "$JOBS"
