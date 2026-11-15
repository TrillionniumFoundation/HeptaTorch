#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
count=0
for request in "$root"/tests/fixtures/*.input.hex; do
    expected=${request%.input.hex}.output.hex
    test -f "$expected"
    env -i PATH= "$root/build/unhex" < "$request" > "$scratch/request"
    env -i PATH= "$root/build/unhex" < "$expected" > "$scratch/expected"
    env -i PATH= "$root/build/heptatorch" < "$scratch/request" > "$scratch/actual"
    if ! cmp -s "$scratch/expected" "$scratch/actual"; then
        printf 'FAIL %s\n' "${request##*/}" >&2
        exit 1
    fi
    count=$((count + 1))
done
test "$count" -gt 0
printf 'PASS: %s native packet regression fixtures\n' "$count"
