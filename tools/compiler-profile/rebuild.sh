#!/usr/bin/env bash
set -euo pipefail
umask 077

# Reproduce HeptaTorch's distinct native compiler capacity profile.
# The supplied legacy compiler is the pinned external bootstrap boundary.
# This script transports bytes; both compilation and ELF packing run in HIA.
if [[ $# != 3 ]]; then
  printf 'Usage: %s /absolute/pinned-legacy-compiler /absolute/generated-source-directory /new/output-directory\n' "$0" >&2
  exit 64
fi
legacy=$1
case "$legacy" in /*) ;; *) printf 'Legacy compiler path must be absolute.\n' >&2; exit 64;; esac
recipe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_dir=$2
case "$source_dir" in /*) ;; *) printf 'Generated source directory must be absolute.\n' >&2; exit 64;; esac
output=$3
case "$output" in /*) ;; *) printf 'Output path must be absolute.\n' >&2; exit 64;; esac
expected_legacy=87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54
printf '%s  %s\n' "$expected_legacy" "$legacy" | sha256sum --check --status
(cd "$source_dir" && sha256sum --check --status "$recipe_dir/source-pins.sha256")
mkdir -- "$output"

compile() {
  local compiler=$1 source=$2 target=$3
  env -i PATH= "$compiler" < "$source" > "$target.tmp"
  test -s "$target.tmp"
  mv -- "$target.tmp" "$target"
}

# This predecessor emits the expanded compiler as a flat HIA image. It need
# not execute that image or provide the larger workspace itself.
compile "$legacy" "$source_dir/compiler.hepta" "$output/A0.image"
compile "$legacy" "$source_dir/compiler-packer.hepta" "$output/packer"
chmod 700 "$output/packer"

pack() {
  local image=$1 target=$2 size byte shift
  size=$(wc -c < "$image")
  {
    printf 'HIAFLT01'
    for shift in 0 8 16 24 32 40 48 56; do
      byte=$(( (size >> shift) & 255 ))
      printf '%b' "\\$(printf '%03o' "$byte")"
    done
    cat -- "$image"
  } | env -i PATH= "$output/packer" > "$target.tmp"
  test -s "$target.tmp"
  chmod 700 "$target.tmp"
  mv -- "$target.tmp" "$target"
}

pack "$output/A0.image" "$output/compiler-0"
previous=$output/compiler-0
for generation in 1 2 3; do
  compile "$previous" "$source_dir/compiler.hepta" "$output/A$generation.image"
  pack "$output/A$generation.image" "$output/compiler-$generation"
  previous=$output/compiler-$generation
done
cmp -- "$output/A1.image" "$output/A2.image"
cmp -- "$output/A2.image" "$output/A3.image"
cmp -- "$output/A0.image" "$output/A1.image"
cmp -- "$output/compiler-0" "$output/compiler-1"
cmp -- "$output/compiler-1" "$output/compiler-2"
cmp -- "$output/compiler-2" "$output/compiler-3"

# Independently compile the unchanged packer through the successor too.
compile "$output/compiler-3" "$source_dir/compiler-packer.hepta" "$output/packer-rebuilt"
cmp -- "$output/packer" "$output/packer-rebuilt"
cp -- "$output/compiler-3" "$output/hia-native-compiler"
chmod 700 "$output/hia-native-compiler"
(cd "$source_dir" && sha256sum --check --status "$recipe_dir/source-pins.sha256")
printf '%s  %s\n' "$expected_legacy" "$legacy" | sha256sum --check --status
printf '%s  %s\n' '968a7db8bee752f5bba0ab6b2270090754a2f4fc8f7a72b0915c77b37e51cbf0' "$output/A3.image" | sha256sum --check --status
printf '%s  %s\n' 'd3495d2913bb3c926df7917fa467d9a40ed8a5c12d2db01f8e7232aa0d4b47a1' "$output/packer" | sha256sum --check --status
printf '%s  %s\n' 'c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6' "$output/hia-native-compiler" | sha256sum --check --status
sha256sum "$output/A0.image" "$output/A1.image" "$output/A2.image" "$output/A3.image" "$output/packer" "$output/hia-native-compiler"
