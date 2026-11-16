#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 3 ]]; then
  printf 'Usage: %s /absolute/expanded-compiler /absolute/generated-source-directory /new/output-directory\n' "$0" >&2
  exit 64
fi
predecessor=$1
source_dir=$2
output=$3
for path in "$predecessor" "$source_dir" "$output"; do
  case "$path" in /*) ;; *) printf 'All three paths must be absolute.\n' >&2; exit 64;; esac
done
recipe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
expected_predecessor=c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6
printf '%s  %s\n' "$expected_predecessor" "$predecessor" | sha256sum --check --status
(cd "$source_dir" && sha256sum --check --status "$recipe_dir/source-pins.sha256")
mkdir -- "$output"
compile() {
  local compiler=$1 source=$2 target=$3
  env -i PATH= "$compiler" < "$source" > "$target.tmp"
  test -s "$target.tmp"
  mv -- "$target.tmp" "$target"
}
compile "$predecessor" "$source_dir/compiler-packer.hepta" "$output/compiler-packer"
compile "$predecessor" "$source_dir/linker-packer.hepta" "$output/linker-packer"
chmod 700 "$output/compiler-packer" "$output/linker-packer"
pack() {
  local packer=$1 image=$2 target=$3 size byte shift
  size=$(wc -c < "$image")
  {
    printf 'HIAFLT01'
    for shift in 0 8 16 24 32 40 48 56; do
      byte=$(( (size >> shift) & 255 ))
      printf '%b' "\\$(printf '%03o' "$byte")"
    done
    cat -- "$image"
  } | env -i PATH= "$packer" > "$target.tmp"
  test -s "$target.tmp"
  chmod 700 "$target.tmp"
  mv -- "$target.tmp" "$target"
}
previous=$predecessor
for generation in 0 1 2; do
  compile "$previous" "$source_dir/module-compiler.hepta" "$output/M$generation.image"
  pack "$output/compiler-packer" "$output/M$generation.image" "$output/compiler-$generation"
  previous=$output/compiler-$generation
done
cmp -- "$output/M0.image" "$output/M1.image"
cmp -- "$output/M1.image" "$output/M2.image"
cmp -- "$output/compiler-0" "$output/compiler-1"
cmp -- "$output/compiler-1" "$output/compiler-2"
cp -- "$previous" "$output/module-compiler"
compile "$previous" "$source_dir/module-linker.hepta" "$output/module-linker.image"
compile "$predecessor" "$source_dir/module-linker.hepta" "$output/module-linker.predecessor.image"
cmp -- "$output/module-linker.image" "$output/module-linker.predecessor.image"
pack "$output/linker-packer" "$output/module-linker.image" "$output/module-linker"
for packer in compiler-packer linker-packer; do
  compile "$previous" "$source_dir/$packer.hepta" "$output/$packer.rebuilt"
  cmp -- "$output/$packer" "$output/$packer.rebuilt"
done
printf '%s  %s\n' 'e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59' "$output/module-compiler" | sha256sum --check --status
printf '%s  %s\n' '1ba40ffe025abfcc2edd8d69f1ad689817598571138c11806af7aa60047fc7b8' "$output/module-linker" | sha256sum --check --status
(cd "$source_dir" && sha256sum --check --status "$recipe_dir/source-pins.sha256")
printf '%s  %s\n' "$expected_predecessor" "$predecessor" | sha256sum --check --status
sha256sum "$output/module-compiler" "$output/module-linker" "$output/M2.image" "$output/module-linker.image" "$output/compiler-packer" "$output/linker-packer"
