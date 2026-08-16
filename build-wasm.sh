#!/usr/bin/env bash
# Build fuzzgpu for WebAssembly
set -e

echo "Building fuzzgpu-wasm..."
cd crates/fuzzgpu-wasm

# Install wasm-pack if not present
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
fi

# Build
wasm-pack build --target web --release --out-dir ../../pkg

echo "Done! WASM package built to pkg/"
echo ""
echo "Usage in HTML:"
echo "  <script type=\"module\">"
echo "    import init, { levenshtein_distance, ratio, needleman_wunsch } from './pkg/fuzzgpu_wasm.js';"
echo "    await init();"
echo "    console.log(levenshtein_distance('kitten', 'sitting'));"
echo "    // Needleman-Wunsch scores are i64, exposed as JS BigInt:"
echo "    console.log(needleman_wunsch('AGTACGCA', 'TATGC', 2n, -1n, -2n));"
echo "  </script>"
