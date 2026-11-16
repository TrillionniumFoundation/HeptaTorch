#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test "$#" -le 1
executable=${1:-"$root/build/heptatorch"}
test -x "$executable"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
decode() {
    if test "$(wc -c < "$1")" -le 1048576; then
        env -i PATH= "$root/build/unhex" < "$1" > "$2"
    else
        # Transport large text fixtures in complete byte pairs. The native
        # decoder still validates every digit; its wrapper limits are unchanged.
        mkdir "$scratch/chunks"
        LC_ALL=C tr -d ' \t\r\n' < "$1" > "$scratch/compact.hex"
        split -b 524288 "$scratch/compact.hex" "$scratch/chunks/part-"
        : > "$2"
        for part in "$scratch"/chunks/part-*; do
            env -i PATH= "$root/build/unhex" < "$part" >> "$2"
        done
        rm -rf "$scratch/chunks" "$scratch/compact.hex"
    fi
}
count=0
for request in "$root"/tests/fixtures/*.input.hex; do
    expected=${request%.input.hex}.output.hex
    test -f "$expected"
    decode "$request" "$scratch/request"
    decode "$expected" "$scratch/expected"
    env -i PATH= "$executable" < "$scratch/request" > "$scratch/actual"
    if ! cmp -s "$scratch/expected" "$scratch/actual"; then
        printf 'FAIL %s\n' "${request##*/}" >&2
        exit 1
    fi
    count=$((count + 1))
done
test "$count" -gt 0
printf 'PASS: %s native packet regression fixtures\n' "$count"
