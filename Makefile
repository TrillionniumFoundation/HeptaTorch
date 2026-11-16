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
HEPTA_STACK_BUILD ?= build/stack-budget-tools
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
	mkdir -p build/owned-tensors; cp -R --remove-destination "$$tmp/native/." build/owned-tensors/

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

OWNED_TENSOR_VERSION_INPUTS := $(wildcard tools/owned_tensor_versions/*.py tools/owned_tensor_versions/*.hepta tools/owned_tensor_versions/*.json) Makefile

.PHONY: owned-tensor-versions test-owned-tensor-versions
owned-tensor-versions: build/owned-tensor-versions/build.json

build/owned-tensor-versions/build.json: build/owned-tensors/build.json $(OWNED_TENSOR_VERSION_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.tensor-versions.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/owned_tensor_versions/build.py --baseline "$(CURDIR)/build/owned-tensors" \
	  --compiler "$$(command -v "$(HEPTA_OWNED_COMPILER)")" \
	  --linker "$$(command -v "$(HEPTA_OWNED_SESSION_LINKER)")" --output "$$tmp/native"; \
	mkdir -p build/owned-tensor-versions; cp -a --remove-destination "$$tmp/native/." build/owned-tensor-versions/

test-owned-tensor-versions: owned-tensor-versions monolithic
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.tensor-versions-review.XXXXXX"); \
	python3 -I -S -B tools/owned_tensor_versions/qualify.py --native "$(CURDIR)/build/owned-tensor-versions" \
	  --compiler "$$(command -v "$(HEPTA_OWNED_COMPILER)")" \
	  --linker "$$(command -v "$(HEPTA_OWNED_SESSION_LINKER)")" \
	  --canonical-binary "$(CURDIR)/build/heptatorch-monolithic" --output "$$review/observed"; \
	cp "$$review/observed/report.json" build/hosted-owned-tensor-versions.json

STACK_TENSOR_INPUTS := $(wildcard tools/stack_budget/*) tests/stack_independent.py tests/check_stack_reports.py validation/stack-budget-package.sha256 Makefile

.PHONY: stack-tensors test-stack-tensors
stack-tensors: build/stack-tensors/build.json

build/stack-tensors/build.json: build/owned-tensor-versions/build.json $(STACK_TENSOR_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.stack-tensors.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	sha256sum -c validation/stack-budget-package.sha256; \
	python3 -I -S -B tools/stack_budget/compile_application.py \
	  --sources "$(HEPTA_STACK_BUILD)/sources" --native "$(HEPTA_STACK_BUILD)/native" \
	  --baseline-build "$(CURDIR)/build/owned-tensor-versions" --output "$$tmp/native" \
	  build/owned-tensor-versions/units/ht_kernel.hepta build/owned-tensor-versions/units/ht_runtime.hepta \
	  build/owned-tensor-versions/units/owned_tensor_session.hepta; \
	mkdir -p build/stack-tensors; cp -a --remove-destination "$$tmp/native/." build/stack-tensors/

test-stack-tensors: stack-tensors all
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.stack-review.XXXXXX"); \
	python3 -I -S -B tools/stack_budget/qualify.py --build "$(HEPTA_STACK_BUILD)" --repo "$(CURDIR)" \
	  --tensor-baseline "$(CURDIR)/build/owned-tensor-versions" --canonical-binary "$(CURDIR)/build/heptatorch" \
	  --service-sources tools/storage_objects/candidate-04/bytes.hepta tools/storage_objects/candidate-04/records.hepta \
	  tools/storage_objects/candidate-04/audit.hepta tools/storage_objects/candidate-04/ops.hepta tools/storage_objects/candidate-04/owners.hepta \
	  --output "$$review/native"; \
	python3 -I -S -B tests/stack_independent.py --native "$(HEPTA_STACK_BUILD)/native" --output "$$review/independent"; \
	python3 -I -S -B tests/check_stack_reports.py --build "$(HEPTA_STACK_BUILD)" \
	  --application-build "$(CURDIR)/build/stack-tensors" --review "$$review/native" \
	  --root-review "$$review/independent" --output build/hosted-stack-budget.json

TENSOR_OBJECT_INPUTS := $(wildcard tools/tensor_objects/*) tests/qualify_tensor_objects.py $(wildcard validation/tensor-object-*.json) Makefile

.PHONY: tensor-objects test-tensor-objects
tensor-objects: build/tensor-objects/abi2/build.json build/tensor-objects/abi3/build.json

build/tensor-objects/abi2/build.json: build/owned-tensors/build.json $(TENSOR_OBJECT_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.tensor-objects-abi2.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/tensor_objects/build.py --repository "$(CURDIR)" --source-root "$(HEPTA_SOURCE_ROOT)" \
	  --baseline "$(CURDIR)/build/owned-tensors" --compiler "$$(command -v "$(HEPTA_OWNED_COMPILER)")" \
	  --linker "$$(command -v "$(HEPTA_OWNED_SESSION_LINKER)")" --output "$$tmp/native"; \
	mkdir -p build/tensor-objects/abi2; cp -a --remove-destination "$$tmp/native/." build/tensor-objects/abi2/

build/tensor-objects/abi3/build.json: build/owned-tensors/build.json $(TENSOR_OBJECT_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.tensor-objects-abi3.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/tensor_objects/build.py --repository "$(CURDIR)" --source-root "$(HEPTA_SOURCE_ROOT)" \
	  --baseline "$(CURDIR)/build/owned-tensors" --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" \
	  --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" --stack-profile --output "$$tmp/native"; \
	mkdir -p build/tensor-objects/abi3; cp -a --remove-destination "$$tmp/native/." build/tensor-objects/abi3/

test-tensor-objects: tensor-objects
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.tensor-objects-review.XXXXXX"); \
	python3 -I -S -B tests/qualify_tensor_objects.py --abi2 "$(CURDIR)/build/tensor-objects/abi2" \
	  --abi3 "$(CURDIR)/build/tensor-objects/abi3" --output "$$review/observed"; \
	cp "$$review/observed/report.json" build/hosted-tensor-object-native.json

CPU_GENERATOR_INPUTS := $(wildcard tools/cpu_generators/* tools/cpu_generators/sources/*) validation/cpu-generator-package.sha256 Makefile

.PHONY: cpu-generators test-cpu-generators
cpu-generators: build/cpu-generators/rebuild.json

build/cpu-generators/rebuild.json: $(CPU_GENERATOR_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.cpu-generators.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	sha256sum -c validation/cpu-generator-package.sha256; \
	python3 -I -S -B tools/cpu_generators/rebuild.py --upstream "$(HEPTA_SOURCE_ROOT)" \
	  --stack-build "$(HEPTA_STACK_BUILD)" --rng-source tools/rng-cpu/rng.hepta \
	  --output "$$tmp/native"; \
	mkdir -p build/cpu-generators; cp -a --remove-destination "$$tmp/native/." build/cpu-generators/

test-cpu-generators: cpu-generators
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.cpu-generator-review.XXXXXX"); \
	sha256sum -c validation/cpu-generator-package.sha256; \
	python3 -B tools/cpu_generators/qualify.py --build "$(CURDIR)/build/cpu-generators" --output "$$review/native"; \
	python3 -B tools/cpu_generators/test_entry.py --build "$(CURDIR)/build/cpu-generators" \
	  --stack-build "$(HEPTA_STACK_BUILD)" --output "$$review/entry"; \
	python3 -I -S -B tests/check_cpu_generator_reports.py --build "$(CURDIR)/build/cpu-generators" \
	  --review "$$review" --output build/hosted-cpu-generator-native.json

HEPTA_TENSOR_OBJECT_BASELINE ?= build/tensor-objects/abi3
BACKWARD_RELEASE_INPUTS := $(wildcard tools/backward_release/* tools/backward_release/evidence/*) validation/backward-release-package.json tests/qualify_backward_release.py Makefile

.PHONY: backward-release test-backward-release
backward-release: build/backward-release/build.json

build/backward-release/build.json: $(HEPTA_TENSOR_OBJECT_BASELINE)/build.json $(BACKWARD_RELEASE_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.backward-release.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/backward_release/build.py --repository "$(CURDIR)" --source-root "$(HEPTA_SOURCE_ROOT)" \
	  --baseline "$(HEPTA_TENSOR_OBJECT_BASELINE)" --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" \
	  --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" --output "$$tmp/native"; \
	mkdir -p build/backward-release; cp -a --remove-destination "$$tmp/native/." build/backward-release/

test-backward-release: backward-release
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.backward-release-review.XXXXXX"); \
	python3 -I -S -B tests/qualify_backward_release.py --build "$(CURDIR)/build/backward-release" \
	  --baseline "$(HEPTA_TENSOR_OBJECT_BASELINE)" --stack-build "$(HEPTA_STACK_BUILD)" --output "$$review/observed"; \
	cp "$$review/observed/report.json" build/hosted-backward-release-native.json

BACKWARD_PARTIAL_INPUTS := $(wildcard tools/backward_partial/* tools/backward_partial/evidence/*) validation/backward-partial-package.json Makefile
GRADIENT_MATERIALIZATION_INPUTS := $(wildcard tools/gradient_materialization/* tools/gradient_materialization/evidence/*) validation/gradient-materialization-package.json tests/qualify_gradient_materialization.py Makefile

.PHONY: backward-partial gradient-materialization test-gradient-materialization
backward-partial: build/backward-partial/build.json
gradient-materialization: build/gradient-materialization/build.json

build/backward-partial/build.json: build/backward-release/build.json $(HEPTA_TENSOR_OBJECT_BASELINE)/build.json $(BACKWARD_PARTIAL_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.backward-partial.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/backward_partial/build.py --repository "$(CURDIR)" --source-root "$(HEPTA_SOURCE_ROOT)" \
	  --baseline "$(CURDIR)/build/backward-release" --tensor-objects "$(HEPTA_TENSOR_OBJECT_BASELINE)" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" --output "$$tmp/native"; \
	mkdir -p build/backward-partial; cp -a --remove-destination "$$tmp/native/." build/backward-partial/

build/gradient-materialization/build.json: build/backward-partial/build.json $(GRADIENT_MATERIALIZATION_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.gradient-materialization.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/gradient_materialization/build.py --repository "$(CURDIR)" --source-root "$(HEPTA_SOURCE_ROOT)" \
	  --baseline "$(CURDIR)/build/backward-partial" --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" \
	  --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" --output "$$tmp/native"; \
	mkdir -p build/gradient-materialization; cp -a --remove-destination "$$tmp/native/." build/gradient-materialization/

test-gradient-materialization: gradient-materialization
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.gradient-materialization-review.XXXXXX"); \
	python3 -I -S -B tests/qualify_gradient_materialization.py --build "$(CURDIR)/build/gradient-materialization" \
	  --partial-build "$(CURDIR)/build/backward-partial" --output "$$review/observed"; \
	cp "$$review/observed/report.json" build/hosted-gradient-materialization-native.json

TENSOR_RNG_INPUTS := $(wildcard tools/tensor_rng/* tools/tensor_rng/evidence/*) validation/tensor-rng-package.sha256 tools/rng-cpu/rng.hepta Makefile

.PHONY: tensor-rng
tensor-rng: build/tensor-rng/reproduction.json

build/tensor-rng/reproduction.json: $(HEPTA_TENSOR_OBJECT_BASELINE)/build.json $(TENSOR_RNG_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.tensor-rng.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/tensor_rng/rebuild.py --baseline "$(HEPTA_TENSOR_OBJECT_BASELINE)" \
	  --rng-source "$(CURDIR)/tools/rng-cpu/rng.hepta" --upstream "$(HEPTA_SOURCE_ROOT)" \
	  --stack-build "$(HEPTA_STACK_BUILD)" --output "$$tmp/native"; \
	mkdir -p build/tensor-rng; cp -a --remove-destination "$$tmp/native/." build/tensor-rng/

TENSOR_RNG_COMPOSITION_INPUTS := $(wildcard tools/tensor_rng_composition/* tools/tensor_rng_composition/evidence/*) validation/tensor-rng-composition-package.sha256 tests/qualify_tensor_rng_composition.py Makefile

.PHONY: tensor-rng-composition test-tensor-rng-composition
tensor-rng-composition: build/tensor-rng-composition/reproduction.json build/tensor-rng/reproduction.json

build/tensor-rng-composition/reproduction.json: build/gradient-materialization/build.json $(HEPTA_TENSOR_OBJECT_BASELINE)/build.json $(TENSOR_RNG_COMPOSITION_INPUTS) $(TENSOR_RNG_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.tensor-rng-composition.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/tensor_rng_composition/rebuild.py --tensor09 "$(HEPTA_TENSOR_OBJECT_BASELINE)" \
	  --partial-build "$(CURDIR)/build/gradient-materialization" --entry-authoring "$(CURDIR)/build/backward-partial/overlay-authoring.py" \
	  --rng-package "$(CURDIR)/tools/tensor_rng" --rng-source "$(CURDIR)/tools/rng-cpu/rng.hepta" \
	  --upstream "$(HEPTA_SOURCE_ROOT)" --stack-build "$(HEPTA_STACK_BUILD)" --output "$$tmp/native"; \
	mkdir -p build/tensor-rng-composition; cp -a --remove-destination "$$tmp/native/." build/tensor-rng-composition/

test-tensor-rng-composition: tensor-rng-composition
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.tensor-rng-composition-review.XXXXXX"); \
	python3 -I -S -B tests/qualify_tensor_rng_composition.py --build "$(CURDIR)/build/tensor-rng-composition" \
	  --tensor09 "$(HEPTA_TENSOR_OBJECT_BASELINE)" --release "$(CURDIR)/build/backward-release" \
	  --materialization "$(CURDIR)/build/gradient-materialization" --rng-build "$(CURDIR)/build/tensor-rng" \
	  --stack-build "$(HEPTA_STACK_BUILD)" --output "$$review/observed"; \
	cp "$$review/observed/report.json" build/hosted-tensor-rng-composition-native.json

GRADIENT_NONE_INPUTS := $(wildcard tools/gradient_none/* tools/gradient_none/evidence/*) validation/gradient-none-package.json tests/qualify_gradient_none.py Makefile
TENSOR_RNG_NULLABLE_INPUTS := $(wildcard tools/tensor_rng_nullable/* tools/tensor_rng_nullable/evidence/*) validation/tensor-rng-nullable-package.sha256 tests/qualify_gradient_none.py Makefile

.PHONY: gradient-none tensor-rng-nullable test-gradient-none
gradient-none: build/gradient-none/build.json
tensor-rng-nullable: build/tensor-rng-nullable/reproduction.json

build/gradient-none/build.json: build/gradient-materialization/build.json build/backward-partial/build.json $(HEPTA_TENSOR_OBJECT_BASELINE)/build.json $(GRADIENT_NONE_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.gradient-none.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/gradient_none/build.py --repository "$(CURDIR)" --source-root "$(HEPTA_SOURCE_ROOT)" \
	  --baseline "$(CURDIR)/build/gradient-materialization" --partial-baseline "$(CURDIR)/build/backward-partial" \
	  --tensor-objects "$(HEPTA_TENSOR_OBJECT_BASELINE)" --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" \
	  --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" --output "$$tmp/native"; \
	mkdir -p build/gradient-none; cp -a --remove-destination "$$tmp/native/." build/gradient-none/

build/tensor-rng-nullable/reproduction.json: build/gradient-none/build.json build/tensor-rng-composition/reproduction.json $(TENSOR_RNG_NULLABLE_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.tensor-rng-nullable.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/tensor_rng_nullable/rebuild.py --common-build "$(CURDIR)/build/tensor-rng-composition/native" \
	  --materialization-build "$(CURDIR)/build/gradient-materialization" --nullable-build "$(CURDIR)/build/gradient-none" \
	  --stack-build "$(HEPTA_STACK_BUILD)" --output "$$tmp/native"; \
	mkdir -p build/tensor-rng-nullable; cp -a --remove-destination "$$tmp/native/." build/tensor-rng-nullable/

test-gradient-none: tensor-rng-nullable
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.gradient-none-review.XXXXXX"); \
	python3 -I -S -B tests/qualify_gradient_none.py --nullable-build "$(CURDIR)/build/gradient-none" \
	  --build "$(CURDIR)/build/tensor-rng-nullable" --previous-common "$(CURDIR)/build/tensor-rng-composition/native" \
	  --materialization "$(CURDIR)/build/gradient-materialization" --stack-build "$(HEPTA_STACK_BUILD)" --output "$$review/observed"; \
	cp "$$review/observed/report.json" build/hosted-gradient-none-native.json

OWNED_OPTIMIZER_INPUTS := $(wildcard tools/owned_optimizers/* tools/owned_optimizers/evidence/* tests/optimizer_object_review/*) validation/owned-optimizer-package.json validation/optimizer-object-replay-package.json validation/accumulator-lifetime-replay-package.json tests/qualify_owned_optimizers.py Makefile

.PHONY: owned-optimizers test-owned-optimizers
owned-optimizers: build/owned-optimizers/reproduction.json

build/owned-optimizers/reproduction.json: build/tensor-rng-nullable/reproduction.json $(HEPTA_TENSOR_OBJECT_BASELINE)/build.json $(OWNED_OPTIMIZER_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.owned-optimizers.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/owned_optimizers/rebuild.py --baseline "$(CURDIR)/build/tensor-rng-nullable" \
	  --objects "$(HEPTA_TENSOR_OBJECT_BASELINE)" --common-authoring "$(CURDIR)/build/tensor-rng-composition/native/overlay-authoring.py" \
	  --nullable-authoring "$(CURDIR)/build/gradient-none/overlay-authoring.py" --partial-authoring "$(CURDIR)/build/backward-partial/overlay-authoring.py" \
	  --materialization-authoring "$(CURDIR)/build/gradient-materialization/overlay-authoring.py" \
	  --repository "$(CURDIR)" --source-root "$(HEPTA_SOURCE_ROOT)" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --output "$$tmp/native"; \
	mkdir -p build/owned-optimizers; cp -a --remove-destination "$$tmp/native/." build/owned-optimizers/

test-owned-optimizers: owned-optimizers
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.owned-optimizer-review.XXXXXX"); \
	python3 -I -S -B tests/qualify_owned_optimizers.py --build "$(CURDIR)/build/owned-optimizers" \
	  --nullable-build "$(CURDIR)/build/tensor-rng-nullable" --stack-build "$(HEPTA_STACK_BUILD)" --output "$$review/observed"; \
	cp "$$review/observed/report.json" build/hosted-owned-optimizers-native.json

OWNED_ADAM_INPUTS := $(wildcard tools/owned_adam/* tools/owned_adam/upstream/* tests/adam_object_review/*) validation/owned-adam-package.sha256 validation/adam-object-review-package.sha256 tests/qualify_owned_adam.py Makefile

.PHONY: owned-adam test-owned-adam
owned-adam: build/owned-adam/reproduction.json

build/owned-adam/reproduction.json: build/owned-optimizers/reproduction.json $(OWNED_ADAM_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.owned-adam.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/owned_adam/rebuild.py --baseline "$(CURDIR)/build/owned-optimizers" \
	  --repository "$(CURDIR)" --source-root "$(HEPTA_SOURCE_ROOT)" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --output "$$tmp/native"; \
	mkdir -p build/owned-adam; cp -a --remove-destination "$$tmp/native/." build/owned-adam/

test-owned-adam: owned-adam
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.owned-adam-review.XXXXXX"); \
	python3 -I -S -B tests/qualify_owned_adam.py --build "$(CURDIR)/build/owned-adam" \
	  --baseline "$(CURDIR)/build/owned-optimizers" --stack-build "$(HEPTA_STACK_BUILD)" --output "$$review/observed"; \
	cp "$$review/observed/report.json" build/hosted-owned-adam-native.json

CPU_DISTRIBUTION_INPUTS := $(wildcard tools/cpu_distributions/* tools/cpu_distributions/evidence/*) validation/cpu-distributions-package.sha256 Makefile
CPU_DISTRIBUTION_PARTIAL_INPUTS := $(wildcard tools/cpu_distributions_partial/* tools/cpu_distributions_partial/evidence/*) validation/cpu-distributions-partial-package.sha256 Makefile

.PHONY: cpu-distributions cpu-distributions-partial test-cpu-distributions
cpu-distributions: build/cpu-distributions/reproduction.json
cpu-distributions-partial: build/cpu-distributions-partial/reproduction.json

build/cpu-distributions/reproduction.json: build/tensor-rng-nullable/reproduction.json build/tensor-rng-composition/reproduction.json build/gradient-materialization/build.json $(CPU_DISTRIBUTION_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.cpu-distributions.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	sha256sum -c validation/cpu-distributions-package.sha256; \
	python3 -I -S -B tools/cpu_distributions/rebuild.py --baseline "$(CURDIR)/build/tensor-rng-nullable" \
	  --common-authoring "$(CURDIR)/build/tensor-rng-composition/native/overlay-authoring.py" \
	  --gradient-authoring "$(CURDIR)/build/gradient-materialization/overlay-authoring.py" \
	  --upstream "$(HEPTA_SOURCE_ROOT)" --stack-build "$(HEPTA_STACK_BUILD)" --output "$$tmp/native"; \
	mkdir -p build/cpu-distributions; cp -a --remove-destination "$$tmp/native/." build/cpu-distributions/

build/cpu-distributions-partial/reproduction.json: build/cpu-distributions/reproduction.json $(CPU_DISTRIBUTION_PARTIAL_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.cpu-distributions-partial.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	sha256sum -c validation/cpu-distributions-partial-package.sha256; \
	python3 -I -S -B tools/cpu_distributions_partial/rebuild.py --baseline "$(CURDIR)/build/cpu-distributions" \
	  --upstream "$(HEPTA_SOURCE_ROOT)" --stack-build "$(HEPTA_STACK_BUILD)" --output "$$tmp/native"; \
	mkdir -p build/cpu-distributions-partial; cp -a --remove-destination "$$tmp/native/." build/cpu-distributions-partial/

test-cpu-distributions: cpu-distributions-partial
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.cpu-distributions-review.XXXXXX"); \
	printf '%s\n' "CPU distribution review: $$review"; \
	sha256sum -c validation/cpu-distributions-package.sha256 validation/cpu-distributions-partial-package.sha256; \
	python3 -I -S -B tools/cpu_distributions/qualify.py --build "$(CURDIR)/build/cpu-distributions" \
	  --baseline "$(CURDIR)/build/tensor-rng-nullable" --stack-build "$(HEPTA_STACK_BUILD)" --output "$$review/atomic"; \
	python3 -I -S -B tools/cpu_distributions_partial/qualify.py --build "$(CURDIR)/build/cpu-distributions-partial" \
	  --baseline "$(CURDIR)/build/cpu-distributions" --atomic-tools "$(CURDIR)/tools/cpu_distributions" \
	  --stack-build "$(HEPTA_STACK_BUILD)" --output "$$review/partial"; \
	python3 -I -S -B tests/check_cpu_distribution_reports.py --atomic-build "$(CURDIR)/build/cpu-distributions" \
	  --build "$(CURDIR)/build/cpu-distributions-partial" --atomic-review "$$review/atomic" --review "$$review/partial" \
	  --output build/hosted-cpu-distributions-native.json

ADAM_DISTRIBUTION_INPUTS := $(wildcard tools/adam_distributions/* tools/adam_distributions/evidence/* tests/adam_distributions_review/* validation/adam-distributions/evidence/archive-parts/*) validation/adam-distributions/evidence/archive-parts.json tools/restore_frozen_archive.py validation/adam-distributions-package.sha256 validation/adam-distributions-review-package.sha256 tests/qualify_adam_distributions.py Makefile

.PHONY: adam-distributions test-adam-distributions
adam-distributions: build/adam-distributions/reproduction.json

build/adam-distributions/reproduction.json: build/owned-adam/reproduction.json build/cpu-distributions-partial/reproduction.json $(ADAM_DISTRIBUTION_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.adam-distributions.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/restore_frozen_archive.py --manifest validation/adam-distributions/evidence/archive-parts.json; \
	sha256sum -c validation/adam-distributions-package.sha256; \
	python3 -I -S -B tools/adam_distributions/rebuild.py --adam-build "$(CURDIR)/build/owned-adam" \
	  --distribution-build "$(CURDIR)/build/cpu-distributions-partial" --upstream "$(HEPTA_SOURCE_ROOT)" \
	  --stack-build "$(HEPTA_STACK_BUILD)" --output "$$tmp/native"; \
	mkdir -p build/adam-distributions; cp -a --remove-destination "$$tmp/native/." build/adam-distributions/

test-adam-distributions: adam-distributions
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.adam-distributions-review.XXXXXX"); \
	printf '%s\n' "Adam and distribution review: $$review"; \
	python3 -I -S -B tools/restore_frozen_archive.py --manifest validation/adam-distributions/evidence/archive-parts.json; \
	sha256sum -c validation/adam-distributions-package.sha256 validation/adam-distributions-review-package.sha256; \
	python3 -I -S -B tests/qualify_adam_distributions.py --build "$(CURDIR)/build/adam-distributions" \
	  --adam-build "$(CURDIR)/build/owned-adam" --atomic-build "$(CURDIR)/build/cpu-distributions" \
	  --distribution-build "$(CURDIR)/build/cpu-distributions-partial" --stack-build "$(HEPTA_STACK_BUILD)" \
	  --output "$$review/observed"; \
	cp "$$review/observed/report.json" build/hosted-adam-distributions-native.json

GRADIENT_OBJECT_INPUTS := $(wildcard tools/gradient_objects/* tools/gradient_objects/authoring/incoming/* tools/gradient_objects/authoring/public/* tools/gradient_objects/authoring/public/baseline-authoring/* tests/gradient_objects/* validation/gradient-objects/reference/*) tests/check_gradient_object_reports.py Makefile

.PHONY: gradient-objects test-gradient-objects
gradient-objects: build/gradient-objects/build.json

build/gradient-objects/build.json: build/owned-optimizers/reproduction.json $(GRADIENT_OBJECT_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.gradient-objects.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/gradient_objects/build.py --baseline "$(CURDIR)/build/owned-optimizers" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --output "$$tmp/native"; \
	mkdir -p build/gradient-objects; cp -a --remove-destination "$$tmp/native/." build/gradient-objects/

test-gradient-objects: owned-optimizers
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.gradient-object-review.XXXXXX"); \
	python3 -I -S -B tools/gradient_objects/qualify.py --repository "$(CURDIR)" \
	  --baseline "$(CURDIR)/build/owned-optimizers" --source-root "$(HEPTA_SOURCE_ROOT)" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --output "$$review/observed"; \
	python3 -I -S -B tests/check_gradient_object_reports.py --build "$$review/observed/native" \
	  --review "$$review/observed" --output build/hosted-gradient-objects-native.json; \
	mkdir -p build/gradient-objects; cp -a --remove-destination "$$review/observed/native/." build/gradient-objects/

GRADIENT_ADAM_DISTRIBUTION_INPUTS := $(wildcard tools/gradient_adam_distributions/* tests/gradient_adam_distributions_review/* tests/gradient_composition_review/* tests/adam_distributions_gradient_review/* validation/gradient-adam-distributions/reference/*) tests/qualify_gradient_adam_distributions.py Makefile

.PHONY: gradient-adam-distributions test-gradient-adam-distributions
gradient-adam-distributions: build/gradient-adam-distributions/build.json

build/gradient-adam-distributions/build.json: build/gradient-objects/build.json build/adam-distributions/reproduction.json $(GRADIENT_ADAM_DISTRIBUTION_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.gradient-adam-distributions.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/gradient_adam_distributions/build.py --gradient-build "$(CURDIR)/build/gradient-objects" \
	  --adam-distribution-build "$(CURDIR)/build/adam-distributions" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --output "$$tmp/native"; \
	mkdir -p build/gradient-adam-distributions; cp -a --remove-destination "$$tmp/native/." build/gradient-adam-distributions/

test-gradient-adam-distributions: gradient-objects adam-distributions
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.gradient-adam-distribution-review.XXXXXX"); \
	python3 -I -S -B tests/qualify_gradient_adam_distributions.py \
	  --gradient-build "$(CURDIR)/build/gradient-objects" --adam-distribution-build "$(CURDIR)/build/adam-distributions" \
	  --stack-build "$(HEPTA_STACK_BUILD)" --output "$$review/observed"; \
	cp "$$review/observed/report.json" build/hosted-gradient-adam-distributions-native.json; \
	mkdir -p build/gradient-adam-distributions; cp -a --remove-destination "$$review/observed/producer/native/." build/gradient-adam-distributions/


RETAINED_GRADIENT_INPUTS := $(wildcard tools/retained_gradients/* tests/retained_gradient_regression/* tests/retained_nonleaf_review/* tests/retained_resource_review/* tests/retained_extension_review/*) tests/qualify_retained_gradients.py Makefile

.PHONY: retained-gradients test-retained-gradients
retained-gradients: build/retained-gradients/build.json

build/retained-gradients/build.json: build/gradient-adam-distributions/build.json $(RETAINED_GRADIENT_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.retained-gradients.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/retained_gradients/build.py --parent-build "$(CURDIR)/build/gradient-adam-distributions" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --output "$$tmp/native"; \
	mkdir -p build/retained-gradients; cp -a --remove-destination "$$tmp/native/." build/retained-gradients/

test-retained-gradients: gradient-adam-distributions
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.retained-gradient-review.XXXXXX"); status=0; \
	python3 -I -S -B tests/qualify_retained_gradients.py --repo "$(CURDIR)" \
	  --parent-build "$(CURDIR)/build/gradient-adam-distributions" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --stack-build "$(HEPTA_STACK_BUILD)" --output "$$review/observed" || status=$$?; \
	if [ -f "$$review/observed/report.json" ]; then cp "$$review/observed/report.json" build/hosted-retained-gradients-native.json; fi; \
	if [ -f "$$review/observed/evidence-manifest.json" ]; then cp "$$review/observed/evidence-manifest.json" build/retained-gradients-ci-raw-manifest.json; fi; \
	if [ -f "$$review/observed/raw-evidence.tar.xz" ]; then cp "$$review/observed/raw-evidence.tar.xz" build/retained-gradients-ci-raw.tar.xz; fi; \
	test "$$status" -eq 0 || exit "$$status"; \
	test -f build/hosted-retained-gradients-native.json; \
	test -f build/retained-gradients-ci-raw-manifest.json; test -f build/retained-gradients-ci-raw.tar.xz; \
	mkdir -p build/retained-gradients; cp -a --remove-destination "$$review/observed/producer/native/." build/retained-gradients/


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
