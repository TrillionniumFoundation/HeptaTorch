SHELL := /bin/sh
HEPTA_COMPILER ?= hia-native-compiler
HEPTA_MODULE_COMPILER ?= hia-module-compiler
HEPTA_MODULE_LINKER ?= hia-module-linker
COMPILER_SHA256 := c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6
MODULE_COMPILER_SHA256 := e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59
MODULE_LINKER_SHA256 := 1ba40ffe025abfcc2edd8d69f1ad689817598571138c11806af7aa60047fc7b8

.PHONY: all monolithic test test-monolithic check-compiler check-module-tools acceptance-status full-acceptance clean

all: build/heptatorch build/unhex

monolithic: build/heptatorch-monolithic build/unhex

check-compiler:
	@compiler=$$(command -v "$(HEPTA_COMPILER)") || { echo 'Set HEPTA_COMPILER to the pinned native HIA compiler; see docs/build.md.' >&2; exit 1; }; \
	printf '%s  %s\n' '$(COMPILER_SHA256)' "$$compiler" | sha256sum -c -

check-module-tools:
	@set -eu; compiler=$$(command -v "$(HEPTA_MODULE_COMPILER)") || { echo 'Set HEPTA_MODULE_COMPILER to the pinned native module compiler; see docs/build.md.' >&2; exit 1; }; \
	linker=$$(command -v "$(HEPTA_MODULE_LINKER)") || { echo 'Set HEPTA_MODULE_LINKER to the pinned native module linker; see docs/build.md.' >&2; exit 1; }; \
	printf '%s  %s\n' '$(MODULE_COMPILER_SHA256)' "$$compiler" '$(MODULE_LINKER_SHA256)' "$$linker" | sha256sum -c -

build:
	mkdir -p build

build/heptatorch: src/heptatorch.hepta tools/modules/split_runtime.py tools/modules/compile_link.sh tools/modules/profile.json Makefile | build check-module-tools
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.modules.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	compiler=$$(realpath "$$(command -v "$(HEPTA_MODULE_COMPILER)")"); linker=$$(realpath "$$(command -v "$(HEPTA_MODULE_LINKER)")"); \
	python3 -I -S -B tools/modules/split_runtime.py --source "$<" --output "$$tmp/units"; \
	bash tools/modules/compile_link.sh "$$compiler" "$$linker" "$$tmp/native" entry "$$tmp/units/ht_runtime.hepta" "$$tmp/units/ht_kernel.hepta"; \
	chmod 755 "$$tmp/native/application"; mv "$$tmp/native/application" "$@"

build/heptatorch-monolithic: src/heptatorch.hepta Makefile | build check-compiler
	@set -eu; tmp=$$(mktemp build/.heptatorch.XXXXXX); trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	"$(HEPTA_COMPILER)" < "$<" > "$$tmp"; test -s "$$tmp"; chmod 755 "$$tmp"; mv "$$tmp" "$@"

build/unhex: tools/unhex.hepta | build check-compiler
	@set -eu; tmp=$$(mktemp build/.unhex.XXXXXX); trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	"$(HEPTA_COMPILER)" < "$<" > "$$tmp"; test -s "$$tmp"; chmod 755 "$$tmp"; mv "$$tmp" "$@"

test: all
	sh tests/run.sh

test-monolithic: monolithic
	sh tests/run.sh "$(CURDIR)/build/heptatorch-monolithic"

acceptance-status:
	python3 -I -S -B tools/check_acceptance.py

full-acceptance:
	python3 -I -S -B tools/check_acceptance.py --require-complete

clean:
	rm -f build/heptatorch build/heptatorch-monolithic build/unhex build/.heptatorch.* build/.unhex.*
