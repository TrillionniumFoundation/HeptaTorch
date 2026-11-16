SHELL := /bin/sh
HEPTA_COMPILER ?= hia-native-compiler
HEPTA_MODULE_COMPILER ?= hia-module-compiler
HEPTA_MODULE_LINKER ?= hia-module-linker
HEPTA_FLAT_LINKER ?= hia-flat-linker
HEPTA_SESSION_PACKER ?= hia-session-packer
HEPTA_SOURCE_ROOT ?= .toolchain/hepta
HEPTA_OWNED_COMPILER ?= owned-module-compiler
HEPTA_OWNED_LINKER ?= linker-abi2-w1m
HEPTA_OWNED_SESSION_BUILD ?= build/owned-session-tools
HEPTA_OWNED_SESSION_LINKER ?= $(HEPTA_OWNED_SESSION_BUILD)/production-native/owned-session-linker
COMPILER_SHA256 := c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6
MODULE_COMPILER_SHA256 := e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59
MODULE_LINKER_SHA256 := 1ba40ffe025abfcc2edd8d69f1ad689817598571138c11806af7aa60047fc7b8
FLAT_LINKER_SHA256 := 4c36e3409b397788664f1501896aff2719f667306795cb5fa95061d710fd6139
SESSION_PACKER_SHA256 := 22600cee8853b6a49dfbf66154ddcdc99bdbbc6fcfb2face2605ad7117b20a44
SESSION_INPUTS := src/heptatorch.hepta tools/session/build_runtime.py tools/session/entry.hepta tools/session/profile.json tools/modules/split_runtime.py Makefile
STORAGE_INPUTS := $(wildcard tools/storage_objects/*.py tools/storage_objects/*.authoring tools/storage_objects/*.hia.in) tools/storage_objects/source-lock.json Makefile
OWNED_TENSOR_INPUTS := src/heptatorch.hepta docs/functions.json tools/modules/split_runtime.py $(wildcard tools/owned_tensors/*.py tools/owned_tensors/*.hepta tools/owned_tensors/*.json) Makefile

.PHONY: all monolithic session session-monolithic storage-objects owned-tensors test test-monolithic test-session test-session-monolithic test-storage-objects test-owned-tensors check-compiler check-module-tools check-session-tools check-session-packer check-storage-tools acceptance-status full-acceptance clean

all: build/heptatorch build/unhex

monolithic: build/heptatorch-monolithic build/unhex

session: build/heptatorch-session

session-monolithic: build/heptatorch-session-monolithic

storage-objects: build/storage-native/build.json build/storage-interactive/build.json

owned-tensors: build/owned-tensors/build.json

build/owned-tensors/build.json: $(OWNED_TENSOR_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.owned-tensor-build.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/owned_tensors/build_runtime.py --repository "$(CURDIR)" --source-root "$(HEPTA_SOURCE_ROOT)" \
	  --compiler "$$(command -v "$(HEPTA_OWNED_COMPILER)")" --linker "$$(command -v "$(HEPTA_OWNED_SESSION_LINKER)")" --output "$$tmp/native"; \
	mkdir -p build/owned-tensors; cp -R "$$tmp/native/." build/owned-tensors/

test-owned-tensors: owned-tensors monolithic
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.owned-tensor-review.XXXXXX"); \
	python3 -I -S -B tools/owned_tensors/test_dynamic.py --service "$(CURDIR)/build/owned-tensors/owned-tensor-session" \
	  --build-report "$(CURDIR)/build/owned-tensors/build.json" --output "$$review/native"; \
	python3 -I -S -B tools/owned_tensors/test_typed_state.py --compiler "$$(command -v "$(HEPTA_OWNED_COMPILER)")" \
	  --linker "$$(command -v "$(HEPTA_OWNED_SESSION_LINKER)")" --build-directory "$(CURDIR)/build/owned-tensors" --output "$$review/typed"; \
	python3 -I -S -B tests/owned_tensor_review/candidate-03-review/executed-sources/review_dynamic.py \
	  --binary "$(CURDIR)/build/owned-tensors/owned-tensor-session" --canonical-binary "$(CURDIR)/build/heptatorch-monolithic" \
	  --transport-observer tools/owned_sessions/client.py --output "$$review/independent"; \
	python3 -I -S -B tests/check_owned_tensor_reports.py --native "$(CURDIR)/build/owned-tensors" \
	  --review "$$review" --output build/hosted-owned-tensor-native.json

check-storage-tools:
	@set -eu; compiler=$$(command -v "$(HEPTA_OWNED_COMPILER)"); linker=$$(command -v "$(HEPTA_OWNED_LINKER)"); session=$$(command -v "$(HEPTA_OWNED_SESSION_LINKER)"); \
	printf '%s  %s\n' '0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73' "$$compiler" \
	  '23eed229e329fb5b7e2116012f74f210d5de4acd2b4f8475e364890748da86ee' "$$linker" \
	  '55a2b0a762c8db574a65d0d0e6a18e3cedc668c01d20080a11c26cf1ccb601b3' "$$session" | sha256sum -c -

build/storage-native/build.json: $(STORAGE_INPUTS) | build check-storage-tools
	python3 -I -S -B tools/storage_objects/build.py --upstream "$(HEPTA_SOURCE_ROOT)" \
	  --compiler "$$(command -v "$(HEPTA_OWNED_COMPILER)")" --linker "$$(command -v "$(HEPTA_OWNED_LINKER)")" \
	  --output "$(CURDIR)/build/storage-native"

build/storage-interactive/build.json: build/storage-native/build.json tools/storage_objects/link_session.py | check-storage-tools
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.storage-link.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/storage_objects/link_session.py --modules "$(CURDIR)/build/storage-native" \
	  --linker "$$(command -v "$(HEPTA_OWNED_SESSION_LINKER)")" --output "$$tmp/native"; \
	mkdir -p build/storage-interactive; cp "$$tmp/native/"* build/storage-interactive/

test-storage-objects: storage-objects
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.storage-review.XXXXXX"); \
	python3 -I -S -B tools/owned_sessions/qualify.py --build "$(HEPTA_OWNED_SESSION_BUILD)" \
	  --compiler "$$(command -v "$(HEPTA_OWNED_COMPILER)")" --output "$$review/owned-session" \
	  build/storage-native/bytes.hepta build/storage-native/records.hepta build/storage-native/audit.hepta \
	  build/storage-native/ops.hepta build/storage-native/owners.hepta; \
	python3 -I -S -B tools/storage_objects/qualify.py --binary "$(CURDIR)/build/storage-native/storage-trace" --transport trace --output "$$review/trace"; \
	python3 -I -S -B tools/storage_objects/qualify.py --binary "$(CURDIR)/build/storage-interactive/storage-session" --transport session --output "$$review/interactive"; \
	python3 -I -S -B tools/storage_objects/qualify_guards.py --compiler "$$(command -v "$(HEPTA_OWNED_COMPILER)")" \
	  --linker "$$(command -v "$(HEPTA_OWNED_LINKER)")" --module-dir "$(CURDIR)/build/storage-native" --output "$$review/guards"; \
	python3 -I -S -B tests/owned_storage_independent.py --executable "$(CURDIR)/build/storage-interactive/storage-session" --output "$$review/model"; \
	python3 -I -S -B tests/check_storage_object_reports.py --session-build "$(HEPTA_OWNED_SESSION_BUILD)" \
	  --session-review "$$review/owned-session" --native "$(CURDIR)/build/storage-native" --interactive "$(CURDIR)/build/storage-interactive" \
	  --trace-review "$$review/trace" --interactive-review "$$review/interactive" --guard-review "$$review/guards" \
	  --model-review "$$review/model" --output build/hosted-owned-storage-gates.json

check-compiler:
	@compiler=$$(command -v "$(HEPTA_COMPILER)") || { echo 'Set HEPTA_COMPILER to the pinned native HIA compiler; see docs/build.md.' >&2; exit 1; }; \
	printf '%s  %s\n' '$(COMPILER_SHA256)' "$$compiler" | sha256sum -c -

check-module-tools:
	@set -eu; compiler=$$(command -v "$(HEPTA_MODULE_COMPILER)") || { echo 'Set HEPTA_MODULE_COMPILER to the pinned native module compiler; see docs/build.md.' >&2; exit 1; }; \
	linker=$$(command -v "$(HEPTA_MODULE_LINKER)") || { echo 'Set HEPTA_MODULE_LINKER to the pinned native module linker; see docs/build.md.' >&2; exit 1; }; \
	printf '%s  %s\n' '$(MODULE_COMPILER_SHA256)' "$$compiler" '$(MODULE_LINKER_SHA256)' "$$linker" | sha256sum -c -

check-session-packer:
	@packer=$$(command -v "$(HEPTA_SESSION_PACKER)") || { echo 'Set HEPTA_SESSION_PACKER to the pinned native session packer; see tools/session/README.md.' >&2; exit 1; }; \
	printf '%s  %s\n' '$(SESSION_PACKER_SHA256)' "$$packer" | sha256sum -c -

check-session-tools: check-session-packer
	@set -eu; compiler=$$(command -v "$(HEPTA_MODULE_COMPILER)") || { echo 'Set HEPTA_MODULE_COMPILER to the pinned native module compiler.' >&2; exit 1; }; \
	linker=$$(command -v "$(HEPTA_FLAT_LINKER)") || { echo 'Set HEPTA_FLAT_LINKER to the pinned native flat linker; see tools/session/README.md.' >&2; exit 1; }; \
	printf '%s  %s\n' '$(MODULE_COMPILER_SHA256)' "$$compiler" '$(FLAT_LINKER_SHA256)' "$$linker" | sha256sum -c -

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

build/heptatorch-session: $(SESSION_INPUTS) | build check-session-tools
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.session.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	compiler=$$(realpath "$$(command -v "$(HEPTA_MODULE_COMPILER)")"); linker=$$(realpath "$$(command -v "$(HEPTA_FLAT_LINKER)")"); packer=$$(realpath "$$(command -v "$(HEPTA_SESSION_PACKER)")"); \
	python3 -I -S -B tools/session/build_runtime.py --source "$(CURDIR)/src/heptatorch.hepta" --mode modular \
	  --module-compiler "$$compiler" --flat-linker "$$linker" --packer "$$packer" --output "$$tmp/native"; \
	cp "$$tmp/native/build.json" "$@-build.json"; chmod 755 "$$tmp/native/application"; mv "$$tmp/native/application" "$@"

build/heptatorch-session-monolithic: $(SESSION_INPUTS) | build check-compiler check-session-packer
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.session-monolithic.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	compiler=$$(realpath "$$(command -v "$(HEPTA_COMPILER)")"); packer=$$(realpath "$$(command -v "$(HEPTA_SESSION_PACKER)")"); \
	python3 -I -S -B tools/session/build_runtime.py --source "$(CURDIR)/src/heptatorch.hepta" --mode monolithic \
	  --compiler "$$compiler" --packer "$$packer" --output "$$tmp/native"; \
	cp "$$tmp/native/build.json" "$@-build.json"; chmod 755 "$$tmp/native/application"; mv "$$tmp/native/application" "$@"

build/unhex: tools/unhex.hepta | build check-compiler
	@set -eu; tmp=$$(mktemp build/.unhex.XXXXXX); trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	"$(HEPTA_COMPILER)" < "$<" > "$$tmp"; test -s "$$tmp"; chmod 755 "$$tmp"; mv "$$tmp" "$@"

test: all
	sh tests/run.sh

test-monolithic: monolithic
	sh tests/run.sh "$(CURDIR)/build/heptatorch-monolithic"

test-session: session
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.session-test.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/session/tests/independent.py --executable "$(CURDIR)/build/heptatorch-session" \
	  --fixtures "$(CURDIR)/tests/fixtures" --output "$$tmp/review"; cp "$$tmp/review/report.json" build/session-independent-modular.json

test-session-monolithic: session-monolithic
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.session-monolithic-test.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/session/tests/independent.py --executable "$(CURDIR)/build/heptatorch-session-monolithic" \
	  --fixtures "$(CURDIR)/tests/fixtures" --output "$$tmp/review"; cp "$$tmp/review/report.json" build/session-independent-monolithic.json

acceptance-status:
	python3 -I -S -B tools/check_acceptance.py

full-acceptance:
	python3 -I -S -B tools/check_acceptance.py --require-complete

clean:
	rm -f build/heptatorch build/heptatorch-monolithic build/heptatorch-session build/heptatorch-session-monolithic build/heptatorch-session-build.json build/heptatorch-session-monolithic-build.json build/session-independent-modular.json build/session-independent-monolithic.json build/unhex build/.heptatorch.* build/.unhex.*
