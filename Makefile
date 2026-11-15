SHELL := /bin/sh
HEPTA_COMPILER ?= hia-native-compiler
COMPILER_SHA256 := c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6

.PHONY: all test check-compiler clean

all: build/heptatorch build/unhex

check-compiler:
	@compiler=$$(command -v "$(HEPTA_COMPILER)") || { echo 'Set HEPTA_COMPILER to the pinned native HIA compiler; see docs/build.md.' >&2; exit 1; }; \
	printf '%s  %s\n' '$(COMPILER_SHA256)' "$$compiler" | sha256sum -c -

build:
	mkdir -p build

build/heptatorch: src/heptatorch.hepta | build check-compiler
	@set -eu; tmp=$$(mktemp build/.heptatorch.XXXXXX); trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	"$(HEPTA_COMPILER)" < "$<" > "$$tmp"; test -s "$$tmp"; chmod 755 "$$tmp"; mv "$$tmp" "$@"

build/unhex: tools/unhex.hepta | build check-compiler
	@set -eu; tmp=$$(mktemp build/.unhex.XXXXXX); trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	"$(HEPTA_COMPILER)" < "$<" > "$$tmp"; test -s "$$tmp"; chmod 755 "$$tmp"; mv "$$tmp" "$@"

test: all
	sh tests/run.sh

clean:
	rm -f build/heptatorch build/unhex build/.heptatorch.* build/.unhex.*
