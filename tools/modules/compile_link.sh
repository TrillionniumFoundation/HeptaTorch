#!/usr/bin/env bash
# Native HIA is responsible for source, signature, ownership, object and link
# admission. This script only owns files and transports explicit wire frames.
set -euo pipefail
umask 077
export LC_ALL=C
if [[ $# -lt 5 ]]; then
  printf 'Usage: %s /absolute/module-compiler /absolute/module-linker /new/output-directory entry source1.hepta [source2.hepta ...]\n' "$0" >&2
  exit 64
fi
compiler=$1
linker=$2
output=$3
entry=$4
shift 4
for path in "$compiler" "$linker" "$output"; do
  case "$path" in /*) ;; *) printf 'Compiler, linker and output paths must be absolute.\n' >&2; exit 64;; esac
done
if [[ ! $entry =~ ^[A-Za-z_][A-Za-z_0-9]{0,127}$ || $# -gt 64 ]]; then
  printf 'Entry must be a HIA identifier; at most 64 modules are admitted.\n' >&2
  exit 64
fi
printf '%s  %s\n' 'e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59' "$compiler" | sha256sum --check --status
printf '%s  %s\n' '1ba40ffe025abfcc2edd8d69f1ad689817598571138c11806af7aa60047fc7b8' "$linker" | sha256sum --check --status
mkdir -- "$output"
objects=()
snapshots=()
index=0
for source in "$@"; do
  target=$output/unit-$index.o
  snapshot=$output/unit-$index.hepta
  cp -- "$source" "$snapshot"
  snapshot_pin=$(sha256sum -- "$snapshot")
  env -i PATH= "$compiler" < "$snapshot" > "$target.tmp"
  printf '%s\n' "$snapshot_pin" | sha256sum --check --status
  cmp -- "$source" "$snapshot"
  test -s "$target.tmp"
  mv -- "$target.tmp" "$target"
  objects+=("$target")
  snapshots+=("$snapshot")
  index=$((index + 1))
done
word() {
  local value=$1 byte shift
  for shift in 0 8 16 24 32 40 48 56; do
    byte=$(( (value >> shift) & 255 ))
    printf '%b' "\\$(printf '%03o' "$byte")"
  done
}
{
  printf 'HIABLK01'
  word "${#objects[@]}"
  word "${#entry}"
  word 0
  printf '%s' "$entry"
  for object in "${objects[@]}"; do
    size=$(wc -c < "$object")
    word "$size"
    cat -- "$object"
  done
} > "$output/link-request.bin"
env -i PATH= "$linker" < "$output/link-request.bin" > "$output/application.tmp"
test -s "$output/application.tmp"
chmod 700 "$output/application.tmp"
index=0
for source in "$@"; do
  cmp -- "$source" "${snapshots[$index]}"
  index=$((index + 1))
done
printf '%s  %s\n' 'e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59' "$compiler" | sha256sum --check --status
printf '%s  %s\n' '1ba40ffe025abfcc2edd8d69f1ad689817598571138c11806af7aa60047fc7b8' "$linker" | sha256sum --check --status
mv -- "$output/application.tmp" "$output/application"
sha256sum -- "${snapshots[@]}" "${objects[@]}" "$output/link-request.bin" "$output/application"
