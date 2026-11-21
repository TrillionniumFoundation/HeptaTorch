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
HEPTA_CURRENT_COMPILER ?= $(HEPTA_STACK_BUILD)/native/stack-module-compiler
HEPTA_ENTROPY_BUILD ?= build/current-runtime-entropy
HEPTA_CURRENT_LINKER ?= $(HEPTA_ENTROPY_BUILD)/generator-session-linker
HEPTA_TORCH_PYTHON ?= $(shell python3 -c 'import sys; print(sys.executable)')
COMPILER_SHA256 := c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6
MODULE_COMPILER_SHA256 := e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59
MODULE_LINKER_SHA256 := 1ba40ffe025abfcc2edd8d69f1ad689817598571138c11806af7aa60047fc7b8
FLAT_LINKER_SHA256 := 4c36e3409b397788664f1501896aff2719f667306795cb5fa95061d710fd6139
SESSION_PACKER_SHA256 := 22600cee8853b6a49dfbf66154ddcdc99bdbbc6fcfb2face2605ad7117b20a44
SESSION_INPUTS := src/heptatorch.hepta tools/session/build_runtime.py tools/session/entry.hepta tools/session/profile.json tools/modules/split_runtime.py Makefile
STORAGE_INPUTS := $(wildcard tools/storage_objects/*.py tools/storage_objects/*.authoring tools/storage_objects/*.hia.in) tools/storage_objects/source-lock.json Makefile
OWNED_TENSOR_INPUTS := src/heptatorch.hepta docs/functions.json tools/modules/split_runtime.py $(wildcard tools/owned_tensors/*.py tools/owned_tensors/*.hepta tools/owned_tensors/*.json) Makefile
CURRENT_RUNTIME_INPUTS := tools/current_runtime/build.py tools/stack_budget/inspect_stack.py \
  tools/current_runtime/source/manifest.json \
  tools/current_runtime/source/runtime.hepta \
  $(wildcard tools/current_runtime/source/units/*.hepta) Makefile

.PHONY: all current-runtime current-entropy-linker legacy monolithic session session-monolithic storage-objects owned-tensors test test-legacy test-monolithic test-session test-session-monolithic test-storage-objects test-owned-tensors check-compiler check-module-tools check-session-tools check-session-packer check-storage-tools acceptance-status full-acceptance clean

all: current-runtime

current-runtime: current-entropy-linker build/current-runtime/build.json build/current-runtime/heptatorch

# The complete fixed upstream authoring closure is required by the tool-only
# entropy linker build. The original application builder main is never called.
ENTROPY_AUTHORING_INPUTS := $(addprefix $(HEPTA_SOURCE_ROOT)/,\
  docs/HIA_SELFHOST_CORE_SUPPORT.json \
  hepta/__init__.py \
  hepta/artifact_publication.py \
  hepta/hia_agent_assembly.py \
  hepta/hia_capability_flow.py \
  hepta/hia_core_admission_native.py \
  hepta/hia_function_shape_native.py \
  hepta/hia_instruction_policy_native.py \
  hepta/hia_json_canonical_native.py \
  hepta/hia_json_decode.py \
  hepta/hia_json_decode_native.py \
  hepta/hia_json_preflight.py \
  hepta/hia_json_preflight_native.py \
  hepta/hia_machine_x64.py \
  hepta/hia_opcodes.py \
  hepta/hia_program.py \
  hepta/hia_program_shape_native.py \
  hepta/hia_program_tools.py \
  hepta/hia_schema.py \
  hepta/hia_sha256_native.py \
  tools/generate_hia_selfhost.py \
  tools/generate_hia_elf_linker.py \
  tools/hia_selfhost_chain_elf.py \
  tools/hia_selfhost_cli.py \
  tools/hia_selfhost_effects.py \
  tools/hia_selfhost_exports.py \
  tools/hia_selfhost_keywords.py \
  tools/hia_selfhost_limits.py \
  tools/hia_selfhost_multi_results.py \
  tools/hia_selfhost_names.py \
  tools/hia_selfhost_owned_elf.py \
  tools/hia_selfhost_returned_loans.py \
  tools/hia_selfhost_signed.py \
  tools/hia_selfhost_traps.py)
ENTROPY_TOOL_INPUTS := tools/current_runtime/entropy_linker/README.md \
  tools/current_runtime/entropy_linker/source-pins.sha256 \
  tools/current_runtime/entropy_linker/source/linker.hepta \
  tools/current_runtime/entropy_linker/source/linker-authoring.txt \
  tools/current_runtime/entropy_linker/source/startup-authoring.txt \
  tools/current_runtime/entropy_linker/build_tool.py \
  tools/current_runtime/entropy_linker/frozen_generator_build.py \
  $(HEPTA_STACK_BUILD)/native/stack-module-compiler \
  $(HEPTA_STACK_BUILD)/native/stack-linker-linker \
  $(HEPTA_STACK_BUILD)/sources/linker-session/authoring.txt \
  $(HEPTA_STACK_BUILD)/sources/linker-session/startup-authoring.txt \
  $(ENTROPY_AUTHORING_INPUTS) Makefile

current-entropy-linker: $(HEPTA_ENTROPY_BUILD)/build.json $(HEPTA_ENTROPY_BUILD)/generator-session-linker

# Keep the original successful stage: its receipt binds absolute process stream
# paths. Publish an exact copy rather than moving those original raw streams.
$(HEPTA_ENTROPY_BUILD)/build.json $(HEPTA_ENTROPY_BUILD)/generator-session-linker &: $(ENTROPY_TOOL_INPUTS) | build
	@set -eu; stage=$$(mktemp -d "$(CURDIR)/build/current-runtime-entropy-stage.XXXXXX"); \
	printf 'Entropy linker build staging: %s\n' "$$stage"; \
	sha256sum -c tools/current_runtime/entropy_linker/source-pins.sha256; \
	python3 -I -S -B tools/current_runtime/entropy_linker/build_tool.py \
	  --upstream "$(abspath $(HEPTA_SOURCE_ROOT))" --stack-build "$(abspath $(HEPTA_STACK_BUILD))" \
	  --output "$$stage/native"; \
	for name in linker.hepta linker-authoring.txt startup-authoring.txt; do \
	  cmp "$(CURDIR)/tools/current_runtime/entropy_linker/source/$$name" "$$stage/native/$$name"; \
	done; \
	if [ -e "$(HEPTA_ENTROPY_BUILD)" ] || [ -L "$(HEPTA_ENTROPY_BUILD)" ]; then \
	  python3 -I -S -B -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); assert p.is_dir() and not p.is_symlink(), "unrecognized entropy-tool path"; r=p/"build.json"; assert r.is_file() and not r.is_symlink(), "missing original entropy-tool receipt"; b=json.loads(r.read_bytes()); assert b.get("schema")=="heptatorch.generator-entropy-linker.tool-only-build.v1" and b.get("status")=="PASS_NATIVE_ENTROPY_LINKER_BUILD_ONLY", "unrecognized entropy-tool receipt"' "$(abspath $(HEPTA_ENTROPY_BUILD))"; \
	  mv "$(HEPTA_ENTROPY_BUILD)" "$$stage/previous"; \
	fi; \
	mkdir -p "$(dir $(HEPTA_ENTROPY_BUILD))"; \
	cp -a "$$stage/native" "$(HEPTA_ENTROPY_BUILD)"; \
	printf 'Entropy linker published; original raw stage retained: %s\n' "$$stage"

# GNU Make 4.3+ grouped targets rebuild both products when either is missing.
# The cold builder requires a fresh directory. Preserve every staging failure,
# and move a recognized previous build aside only after the new build succeeds.
build/current-runtime/build.json build/current-runtime/heptatorch &: $(CURRENT_RUNTIME_INPUTS) $(HEPTA_CURRENT_COMPILER) $(HEPTA_CURRENT_LINKER) $(HEPTA_ENTROPY_BUILD)/build.json | build
	@set -eu; stage=$$(mktemp -d "$(CURDIR)/build/current-runtime-stage.XXXXXX"); \
	printf 'Current runtime build staging: %s\n' "$$stage"; \
	python3 -I -S -B tools/current_runtime/build.py \
	  --manifest "$(CURDIR)/tools/current_runtime/source/manifest.json" \
	  --source-root "$(CURDIR)/tools/current_runtime/source" \
	  --compiler "$(abspath $(HEPTA_CURRENT_COMPILER))" --linker "$(abspath $(HEPTA_CURRENT_LINKER))" \
	  --output "$$stage/native"; \
	cmp "$(abspath $(HEPTA_ENTROPY_BUILD))/generator-session-linker" "$$stage/native/toolchain/stack-linker-session"; \
	if [ -e build/current-runtime ] || [ -L build/current-runtime ]; then \
	  python3 -I -S -B -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); assert p.is_dir() and not p.is_symlink(), "unrecognized current runtime path"; r=p/"build.json"; assert r.is_file() and not r.is_symlink(), "missing original current runtime receipt"; b=json.loads(r.read_bytes()); assert b.get("schema")=="heptatorch.complete-runtime-build.v1" and b.get("status")=="PASS_NATIVE_BUILD_ONLY" and b.get("native_basename")=="heptatorch", "unrecognized current runtime receipt"' "$(CURDIR)/build/current-runtime"; \
	  mv build/current-runtime "$$stage/previous"; \
	fi; \
	mv "$$stage/native" build/current-runtime; \
	printf 'Current runtime published: %s\n' "$(CURDIR)/build/current-runtime"

# Historical143 provenance is checked only when an explicit old-source target needs it.
# The source is restored from fixed history; frozen consumers and their gates stay unchanged.
.PHONY: legacy-source
legacy-source: tools/legacy_source143/manifest.json tools/legacy_source143/restore.py
	python3 -I -S -B tools/legacy_source143/restore.py --repository "$(CURDIR)"

src/heptatorch.hepta: | legacy-source

legacy: build/heptatorch build/unhex

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

test-storage-objects: storage-objects | legacy-source
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

test-stack-tensors: stack-tensors legacy
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

# Generator application and original 24/11/59 checks run in the required
# current-runtime Generator131 group. Exact retired sources remain in its fixed
# original26/qualifier packages; see their retirement.json location index.

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


NATIVE_DROPOUT_INPUTS := $(wildcard tools/native_dropout/* tests/native_dropout_review/* tests/native_dropout_review/*/* tests/native_dropout_identity9/* tests/dtype_resource_review/* tests/dtype_owner_accounting/* validation/native-dropout/reference/*) tests/qualify_native_dropout.py Makefile

.PHONY: native-dropout test-native-dropout
native-dropout: build/native-dropout/build.json

build/native-dropout/build.json: build/retained-gradients/build.json $(NATIVE_DROPOUT_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.native-dropout.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -I -S -B tools/native_dropout/build.py --parent "$(CURDIR)/build/retained-gradients" \
	  --repository "$(CURDIR)" --upstream "$(HEPTA_SOURCE_ROOT)" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --output "$$tmp/native"; \
	mkdir -p build/native-dropout; cp -a --remove-destination "$$tmp/native/." build/native-dropout/

test-native-dropout: retained-gradients
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.native-dropout-review.XXXXXX"); status=0; \
	python3 -I -S -B tests/qualify_native_dropout.py --repo "$(CURDIR)" \
	  --parent-build "$(CURDIR)/build/retained-gradients" --upstream "$(HEPTA_SOURCE_ROOT)" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --flat-compiler "$$(command -v "$(HEPTA_COMPILER)")" --stack-build "$(HEPTA_STACK_BUILD)" \
	  --output "$$review/observed" || status=$$?; \
	if [ -f "$$review/observed/report.json" ]; then cp "$$review/observed/report.json" build/hosted-native-dropout-native.json; fi; \
	if [ -f "$$review/observed/evidence-manifest.json" ]; then cp "$$review/observed/evidence-manifest.json" build/native-dropout-ci-raw-manifest.json; fi; \
	if [ -f "$$review/observed/raw-evidence.tar.xz" ]; then cp "$$review/observed/raw-evidence.tar.xz" build/native-dropout-ci-raw.tar.xz; fi; \
	test "$$status" -eq 0 || exit "$$status"; \
	test -f build/hosted-native-dropout-native.json; \
	test -f build/native-dropout-ci-raw-manifest.json; test -f build/native-dropout-ci-raw.tar.xz; \
	mkdir -p build/native-dropout; cp -a --remove-destination "$$review/observed/producer/native/." build/native-dropout/


INTEGER_TENSOR_INPUTS := $(wildcard tools/integer_tensors/* tests/integer_tensor_review/* tests/integer_tensor_review/*/* tests/int64_inplace_review/* tests/integer_resource_review/* validation/integer-tensors/reference/*) tests/qualify_integer_tensors.py Makefile

.PHONY: integer-tensors test-integer-tensors
integer-tensors: build/integer-tensors/build.json

build/integer-tensors/build.json: build/native-dropout/build.json $(INTEGER_TENSOR_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.integer-tensors.XXXXXX"); trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	python3 -S -B tools/integer_tensors/rebuild.py --parent "$(CURDIR)/build/native-dropout" \
	  --repository "$(CURDIR)" --upstream "$(HEPTA_SOURCE_ROOT)" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --output "$$tmp/native"; \
	mkdir -p build/integer-tensors; cp -a --remove-destination "$$tmp/native/." build/integer-tensors/

test-integer-tensors: native-dropout
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.integer-tensors-review.XXXXXX"); status=0; \
	python3 -I -S -B tests/qualify_integer_tensors.py --repo "$(CURDIR)" \
	  --parent-build "$(CURDIR)/build/native-dropout" --upstream "$(HEPTA_SOURCE_ROOT)" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --flat-compiler "$$(command -v "$(HEPTA_COMPILER)")" --stack-build "$(HEPTA_STACK_BUILD)" \
	  --output "$$review/observed" || status=$$?; \
	if [ -f "$$review/observed/report.json" ]; then cp "$$review/observed/report.json" build/hosted-integer-tensors-native.json; fi; \
	if [ -f "$$review/observed/evidence-manifest.json" ]; then cp "$$review/observed/evidence-manifest.json" build/integer-tensors-ci-raw-manifest.json; fi; \
	if [ -f "$$review/observed/raw-evidence.tar.xz" ]; then cp "$$review/observed/raw-evidence.tar.xz" build/integer-tensors-ci-raw.tar.xz; fi; \
	test "$$status" -eq 0 || exit "$$status"; \
	test -f build/hosted-integer-tensors-native.json; \
	test -f build/integer-tensors-ci-raw-manifest.json; test -f build/integer-tensors-ci-raw.tar.xz; \
	mkdir -p build/integer-tensors; cp -a --remove-destination "$$review/observed/producer10/native/." build/integer-tensors/


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

# Run every mandatory suite from the current qualification inventory, including separate Generator probes.
test: current-runtime
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/current-runtime-review.XXXXXX"); \
	build_sha=$$(sha256sum "$(CURDIR)/build/current-runtime/build.json" | cut -d ' ' -f 1); \
	"$(HEPTA_TORCH_PYTHON)" -I -S -B tests/f64_view/run.py \
	  --build-directory "$(CURDIR)/build/current-runtime" --expected-build-sha256 "$$build_sha" \
	  --torch-python "$(HEPTA_TORCH_PYTHON)" --output "$$review/f64-view"; \
	"$(HEPTA_TORCH_PYTHON)" -I -S -B tests/f64_view/qualify.py \
	  --build-directory "$(CURDIR)/build/current-runtime" --expected-build-sha256 "$$build_sha" \
	  --directory "$$review/f64-view" --output "$$review/f64-view/physical-qualification.json"; \
	"$(HEPTA_TORCH_PYTHON)" -I -S -B tests/qualify_current_runtime.py \
	  --build-directory "$(CURDIR)/build/current-runtime" --torch-python "$(HEPTA_TORCH_PYTHON)" \
	  --tool-build "$(abspath $(HEPTA_ENTROPY_BUILD))" --stack-build "$(abspath $(HEPTA_STACK_BUILD))" \
	  --f64-view-directory "$$review/f64-view" --output "$$review/results"

test-legacy: legacy
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


MIXED_INTEGER_AUTOGRAD_INPUTS := $(wildcard tools/mixed_integer_autograd/* tests/mixed_integer_autograd_review/* tests/mixed_integer_autograd_review/*/* tests/saved_public_lifetime_review/* tests/mixed_integer_owner_review/* validation/mixed-integer-autograd/reference/*) tests/qualify_mixed_integer_autograd.py Makefile

.PHONY: mixed-integer-autograd test-mixed-integer-autograd
mixed-integer-autograd: build/mixed-integer-autograd/build.json

build/mixed-integer-autograd/build.json: build/integer-tensors/build.json $(MIXED_INTEGER_AUTOGRAD_INPUTS) | build
	@set -eu; tmp=$$(mktemp -d "$(CURDIR)/build/.mixed-integer-autograd.XXXXXX"); \
	python3 -S -B tools/mixed_integer_autograd/rebuild.py --parent "$(CURDIR)/build/integer-tensors" \
	  --repository "$(CURDIR)" --upstream "$(HEPTA_SOURCE_ROOT)" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --output "$$tmp/native"; \
	mkdir -p build/mixed-integer-autograd; cp -a --remove-destination "$$tmp/native/." build/mixed-integer-autograd/

test-mixed-integer-autograd: integer-tensors
	@set -eu; review=$$(mktemp -d "$(CURDIR)/build/.mixed-integer-autograd-review.XXXXXX"); status=0; \
	python3 -I -S -B tests/qualify_mixed_integer_autograd.py --repo "$(CURDIR)" \
	  --parent-build "$(CURDIR)/build/integer-tensors" --upstream "$(HEPTA_SOURCE_ROOT)" \
	  --compiler "$(HEPTA_STACK_BUILD)/native/stack-module-compiler" --linker "$(HEPTA_STACK_BUILD)/native/stack-linker-session" \
	  --flat-compiler "$$(command -v "$(HEPTA_COMPILER)")" --stack-build "$(HEPTA_STACK_BUILD)" \
	  --output "$$review/observed" || status=$$?; \
	if [ -f "$$review/observed/report.json" ]; then cp "$$review/observed/report.json" build/hosted-mixed-integer-autograd-native.json; fi; \
	if [ -f "$$review/observed/evidence-manifest.json" ]; then cp "$$review/observed/evidence-manifest.json" build/mixed-integer-autograd-ci-raw-manifest.json; fi; \
	if [ -f "$$review/observed/raw-evidence.tar.xz" ]; then cp "$$review/observed/raw-evidence.tar.xz" build/mixed-integer-autograd-ci-raw.tar.xz; fi; \
	test "$$status" -eq 0 || exit "$$status"; \
	test -f build/hosted-mixed-integer-autograd-native.json; \
	test -f build/mixed-integer-autograd-ci-raw-manifest.json; test -f build/mixed-integer-autograd-ci-raw.tar.xz; \
	mkdir -p build/mixed-integer-autograd; cp -a --remove-destination "$$review/observed/producer5/native/." build/mixed-integer-autograd/

# Recovered 09b HIA recipe: every source/object/image is pinned independently.
TENSOR_TO_RECOVERED_BUILD ?= build/tensor-to-recovered
TENSOR_TO_RECOVERED_SOURCE_CHECK ?= build/tensor-to-recovered-source
TENSOR_TO_RECOVERED_SOURCE_SHA256 := 4f8f73f463f681b5eb29423d71a35a0b42d807fd92bf976bbf76c1970fbcee33
TENSOR_TO_RECOVERED_NATIVE_SHA256 := 53fb43b869858f020c0b936a65f00f3b14b97d9e37038a74c357e1da471fe083
TENSOR_TO_RECOVERED_INPUTS := $(wildcard tools/tensor_to_native_recovered/*.py tools/tensor_to_native_recovered/*.json tools/tensor_to_native_recovered/inputs/* tools/tensor_to_native_recovered/provenance/*)
DIFFERENTIABLE_VIEW_REFERENCE_BUILD ?= build/differentiable-view-reference14

.PHONY: tensor-to-recovered check-tensor-to-recovered-source test-tensor-to-recovered-tombstone test-tensor-to-recovered-live32 test-tensor-to-recovered-frozen35 test-tensor-to-recovered-copied18 test-differentiable-view-reference14

tensor-to-recovered: $(TENSOR_TO_RECOVERED_BUILD)/build.json

$(TENSOR_TO_RECOVERED_BUILD)/build.json: $(TENSOR_TO_RECOVERED_INPUTS) | build
	python3 -I -S -B tools/tensor_to_native_recovered/rebuild.py \
	  --compiler "$(abspath $(HEPTA_STACK_BUILD))/native/stack-module-compiler" \
	  --linker "$(abspath $(HEPTA_STACK_BUILD))/native/stack-linker-session" \
	  --output "$(abspath $(TENSOR_TO_RECOVERED_BUILD))"

check-tensor-to-recovered-source: | build
	python3 -S -B tools/tensor_to_native_recovered/self_check.py \
	  --output "$(abspath $(TENSOR_TO_RECOVERED_SOURCE_CHECK))"

test-tensor-to-recovered-tombstone: tensor-to-recovered
	python3 -I -S -B tests/tensor_to_frozen35_accounting/test_storage_tombstone_legacy_refusal.py \
	  --binary "$(TENSOR_TO_RECOVERED_BUILD)/tensor-to-copy-session" \
	  --source "$(TENSOR_TO_RECOVERED_BUILD)/runtime.hepta" \
	  --protocol tests/retained_nonleaf_review/protocol.py \
	  --output build/frozen35-tensor-to-node/storage-tombstone

test-tensor-to-recovered-live32: tensor-to-recovered
	@printf '%s  %s\n' '554671168c7a4a408dc1e1df89a19db15c3efcefcf4ae1a300af1b6dcae40160' tests/pytorch_tensor_rng_backward_oracle.py | sha256sum -c -
	python3 tests/pytorch_tensor_rng_backward_oracle.py \
	  --native "$(TENSOR_TO_RECOVERED_BUILD)/tensor-to-copy-session" \
	  --source "$(TENSOR_TO_RECOVERED_BUILD)/runtime.hepta" \
	  --expected-native-sha256 "$(TENSOR_TO_RECOVERED_NATIVE_SHA256)" \
	  --expected-source-sha256 "$(TENSOR_TO_RECOVERED_SOURCE_SHA256)" \
	  --repo . --output build/pytorch-tensor-to-recovered-backward.json > build/pytorch-tensor-to-recovered-backward.stdout

test-tensor-to-recovered-frozen35: tensor-to-recovered
	@printf '%s  %s\n' '149b41c455444edeb8d25d0ad98045e29683e780f37f7e210e7985242d359597' tests/tensor_object_oracle/pytorch_tensor_objects_oracle.py | sha256sum -c -
	python3 tests/tensor_to_frozen35_accounting/run_frozen35_accounted.py \
	  --legacy-directory tests/tensor_object_oracle \
	  --native-source "$(TENSOR_TO_RECOVERED_BUILD)/runtime.hepta" \
	  --observer-output build/frozen35-tensor-to-node/614 \
	  --observer-report build/pytorch-tensor-to-recovered-frozen35.json \
	  -- "$(TENSOR_TO_RECOVERED_BUILD)/tensor-to-copy-session" build/pytorch-tensor-to-recovered-frozen35-original.json \
	  --owned-reference tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py \
	  --session-reference tests/pytorch_session_oracle.py --reference tests/pytorch_oracle.py \
	  --lock compatibility/pytorch-oracle-lock.json --repo . \
	  --trace-directory build/frozen35-tensor-to-node-wire/614 \
	  --expected-native-sha256 "$(TENSOR_TO_RECOVERED_NATIVE_SHA256)"
	python3 -I -S -B tests/tensor_to_frozen35_accounting/check_frozen35_accounted.py \
	  --observer-report build/pytorch-tensor-to-recovered-frozen35.json \
	  --original-report build/pytorch-tensor-to-recovered-frozen35-original.json

test-tensor-to-recovered-copied18:
	@printf '%s  %s\n' 'b276b3239d254c52b7877919fe1401e95121014975fdce64f5adc40a264a523e' tests/tensor_to_frozen35_accounting/check_owner_snapshots_672.py | sha256sum -c -
	python3 -I -S -B tests/tensor_to_frozen35_accounting/check_owner_snapshots_672.py \
	  --observer tests/tensor_to_frozen35_accounting/owner_accounting_to.py \
	  --expected-profile tensor-to-copy-614-fixed35-F32-Saved \
	  --journals build/frozen35-tensor-to-node/614/owners \
	  --output build/frozen35-tensor-to-node/614/copied-controls

test-differentiable-view-reference14: $(DIFFERENTIABLE_VIEW_REFERENCE_BUILD)/report.json

$(DIFFERENTIABLE_VIEW_REFERENCE_BUILD)/report.json: tests/differentiable_view_native_review/prepare_reference.py $(wildcard validation/differentiable-view-reference14/*) | build
	python3 -I -S -B tests/differentiable_view_native_review/prepare_reference.py \
	  --output "$(abspath $(DIFFERENTIABLE_VIEW_REFERENCE_BUILD))"

# Distinct R2 parent and R3 differentiable view profiles; all units are rebuilt.
DIFFERENTIABLE_VIEW_R2_BUILD ?= build/differentiable-view-r2-parent
DIFFERENTIABLE_VIEW_BUILD ?= build/differentiable-view-recovered
DIFFERENTIABLE_VIEW_R2_SOURCE_SHA256 := 197bd52db4815f575ea3e9055d3f5556900314bd2f4fbb262e80cea76ea16ef2
DIFFERENTIABLE_VIEW_R2_NATIVE_SHA256 := 13cee2163e8f62081db35e831885714789ac5117fb80c030a1ccaf7194e0821c
DIFFERENTIABLE_VIEW_SOURCE_SHA256 := 3c2c0a125bcac6d47075bf58fc3f78c0e7b9a0b1370757981b99dcd3793b21c6
DIFFERENTIABLE_VIEW_NATIVE_SHA256 := 7841b19051960b6fa13cb98841dbb8e4d6ead97f03e2ad65af415e1a35b03221
DIFFERENTIABLE_VIEW_OBSERVER_SHA256 := 68d0f4657c6cc7062d98cbf033565b16c7b51f3381c6490247e42512c5baf4a2
DIFFERENTIABLE_VIEW_FUNCTIONS := 627
DIFFERENTIABLE_VIEW_UNITS := 13
DIFFERENTIABLE_VIEW_R2_INPUTS := $(wildcard tools/differentiable_view_recovered_v2/*.py tools/differentiable_view_recovered_v2/*.json tools/differentiable_view_recovered_v2/*.hepta tools/differentiable_view_recovered_v2/provenance/*) tools/owned_optimizers/compact_functions.py
DIFFERENTIABLE_VIEW_INPUTS := $(wildcard tools/differentiable_view_recovered_r3/*.py tools/differentiable_view_recovered_r3/*.json tools/differentiable_view_recovered_r3/*.hepta tools/differentiable_view_recovered_r3/*.md tools/differentiable_view_recovered_r3/provenance/* tools/differentiable_view_recovered_r3/generated/* tools/differentiable_view_recovered_r3/generated/units/*.hepta) tools/owned_optimizers/compact_functions.py tools/owned_tensors/authoring.py tools/owned_tensors/profile.json

.PHONY: differentiable-view-r2-parent differentiable-view-recovered test-differentiable-view-native14

differentiable-view-r2-parent: $(DIFFERENTIABLE_VIEW_R2_BUILD)/build.json

$(DIFFERENTIABLE_VIEW_R2_BUILD)/build.json: $(DIFFERENTIABLE_VIEW_R2_INPUTS) $(TENSOR_TO_RECOVERED_BUILD)/build.json | build
	python3 -B tools/differentiable_view_recovered_v2/build.py \
	  --previous "$(abspath $(TENSOR_TO_RECOVERED_BUILD))" \
	  --repository "$(CURDIR)" --upstream "$(abspath $(HEPTA_SOURCE_ROOT))" \
	  --compactor "$(CURDIR)/tools/owned_optimizers/compact_functions.py" \
	  --compiler "$(abspath $(HEPTA_STACK_BUILD))/native/stack-module-compiler" \
	  --linker "$(abspath $(HEPTA_STACK_BUILD))/native/stack-linker-session" \
	  --output "$(abspath $(DIFFERENTIABLE_VIEW_R2_BUILD))"
	@printf '%s  %s\n' "$(DIFFERENTIABLE_VIEW_R2_SOURCE_SHA256)" "$(DIFFERENTIABLE_VIEW_R2_BUILD)/runtime.hepta" \
	  "$(DIFFERENTIABLE_VIEW_R2_NATIVE_SHA256)" "$(DIFFERENTIABLE_VIEW_R2_BUILD)/tensor-differentiable-view-session" | sha256sum -c -

differentiable-view-recovered: $(DIFFERENTIABLE_VIEW_BUILD)/build.json

$(DIFFERENTIABLE_VIEW_BUILD)/build.json: $(DIFFERENTIABLE_VIEW_INPUTS) $(DIFFERENTIABLE_VIEW_R2_BUILD)/build.json | build
	python3 -B tools/differentiable_view_recovered_r3/build.py \
	  --previous "$(abspath $(DIFFERENTIABLE_VIEW_R2_BUILD))" \
	  --repository "$(CURDIR)" --upstream "$(abspath $(HEPTA_SOURCE_ROOT))" \
	  --compactor "$(CURDIR)/tools/owned_optimizers/compact_functions.py" \
	  --compiler "$(abspath $(HEPTA_STACK_BUILD))/native/stack-module-compiler" \
	  --linker "$(abspath $(HEPTA_STACK_BUILD))/native/stack-linker-session" \
	  --output "$(abspath $(DIFFERENTIABLE_VIEW_BUILD))"
	@printf '%s  %s\n' "$(DIFFERENTIABLE_VIEW_SOURCE_SHA256)" "$(DIFFERENTIABLE_VIEW_BUILD)/runtime.hepta" \
	  "$(DIFFERENTIABLE_VIEW_NATIVE_SHA256)" "$(DIFFERENTIABLE_VIEW_BUILD)/tensor-differentiable-view-session" | sha256sum -c -

test-differentiable-view-native14: differentiable-view-recovered $(DIFFERENTIABLE_VIEW_REFERENCE_BUILD)/report.json
	@printf '%s  %s\n' 'deb9f6fafc3f91b2749bdfac030e19cb0ccdb832fa238af91ed9feb396e62eb8' tests/differentiable_view_input4_review/replay_direct_view14_profile.py | sha256sum -c -
	python3 -I -S -B tests/differentiable_view_input4_review/replay_direct_view14_profile.py \
	  --binary "$(DIFFERENTIABLE_VIEW_BUILD)/tensor-differentiable-view-session" \
	  --source "$(DIFFERENTIABLE_VIEW_BUILD)/runtime.hepta" \
	  --build-report "$(DIFFERENTIABLE_VIEW_BUILD)/build.json" \
	  --protocol tests/differentiable_view_native_review/protocol.py \
	  --observer tests/differentiable_view_input4_review/native_observer_direct03.py \
	  --reference "$(DIFFERENTIABLE_VIEW_REFERENCE_BUILD)/payload/original/actual.json" \
	  --probe "$(DIFFERENTIABLE_VIEW_REFERENCE_BUILD)/payload/original/inputs/differentiable-view14-probe-1ca74384.py" \
	  --expected-source-sha256 "$(DIFFERENTIABLE_VIEW_SOURCE_SHA256)" \
	  --expected-native-sha256 "$(DIFFERENTIABLE_VIEW_NATIVE_SHA256)" \
	  --expected-observer-sha256 "$(DIFFERENTIABLE_VIEW_OBSERVER_SHA256)" \
	  --expected-functions "$(DIFFERENTIABLE_VIEW_FUNCTIONS)" --expected-native-units "$(DIFFERENTIABLE_VIEW_UNITS)" \
	  --output build/differentiable-view-native14

# Original four immutable AsStrided regression suites recovered byte-for-byte.
TENSOR_TO_ORIGINAL16_SUITES := copy4 reduction4 lanes2 branches6
TENSOR_TO_ORIGINAL16_BUILD ?= build/tensor-to-recovered-bits
TENSOR_TO_ORIGINAL16_OBSERVER_SHA256 := 42b03aaccb5fc7df828ef9aa9953b15a8ff1d911fe4d644d990e60f0348571fc

.PHONY: test-tensor-to-recovered-original16 $(addprefix test-tensor-to-recovered-bits-,$(TENSOR_TO_ORIGINAL16_SUITES))
test-tensor-to-recovered-original16: $(addprefix test-tensor-to-recovered-bits-,$(TENSOR_TO_ORIGINAL16_SUITES))

$(addprefix test-tensor-to-recovered-bits-,$(TENSOR_TO_ORIGINAL16_SUITES)): test-tensor-to-recovered-bits-%: tensor-to-recovered
	python3 -I -S -B tests/tensor_to_original16_review/replay_as_strided_$*.py \
	  --binary "$(TENSOR_TO_RECOVERED_BUILD)/tensor-to-copy-session" \
	  --source "$(TENSOR_TO_RECOVERED_BUILD)/runtime.hepta" \
	  --build-report "$(TENSOR_TO_RECOVERED_BUILD)/build.json" \
	  --protocol tests/retained_nonleaf_review/protocol.py \
	  --observer tests/tensor_to_original16_review/native_observer09.py \
	  --reference validation/tensor-to-original16-recovery/reference/as-strided-$*.json \
	  --expected-source-sha256 "$(TENSOR_TO_RECOVERED_SOURCE_SHA256)" \
	  --expected-native-sha256 "$(TENSOR_TO_RECOVERED_NATIVE_SHA256)" \
	  --expected-observer-sha256 "$(TENSOR_TO_ORIGINAL16_OBSERVER_SHA256)" \
	  --output "$(TENSOR_TO_ORIGINAL16_BUILD)/$*"

# Restore all original input4 bytes and modes; this does not rerun Torch.
DIFFERENTIABLE_VIEW_INPUT4_REFERENCE_BUILD ?= build/differentiable-view-input4-reference
.PHONY: test-differentiable-view-input4-reference
test-differentiable-view-input4-reference: $(DIFFERENTIABLE_VIEW_INPUT4_REFERENCE_BUILD)/report.json

$(DIFFERENTIABLE_VIEW_INPUT4_REFERENCE_BUILD)/report.json: tests/differentiable_view_input4_review/prepare_reference.py $(wildcard validation/differentiable-view-input4-reference/*) | build
	python3 -I -S -B tests/differentiable_view_input4_review/prepare_reference.py \
	  --output "$(abspath $(DIFFERENTIABLE_VIEW_INPUT4_REFERENCE_BUILD))"

.PHONY: test-differentiable-view-native-input4
test-differentiable-view-native-input4: differentiable-view-recovered $(DIFFERENTIABLE_VIEW_INPUT4_REFERENCE_BUILD)/report.json
	@printf '%s  %s\n' '5b8809eb9e405bf89a3586741d64210483fe3fbc5bc323e2cc16cb712ce4de66' tests/differentiable_view_input4_review/replay_input_boundaries4.py | sha256sum -c -
	python3 -I -S -B tests/differentiable_view_input4_review/replay_input_boundaries4.py \
	  --binary "$(DIFFERENTIABLE_VIEW_BUILD)/tensor-differentiable-view-session" \
	  --source "$(DIFFERENTIABLE_VIEW_BUILD)/runtime.hepta" \
	  --build-report "$(DIFFERENTIABLE_VIEW_BUILD)/build.json" \
	  --protocol tests/differentiable_view_native_review/protocol.py \
	  --observer tests/differentiable_view_input4_review/native_observer_direct03.py \
	  --reference "$(DIFFERENTIABLE_VIEW_INPUT4_REFERENCE_BUILD)/payload/actual/actual.json" \
	  --probe "$(DIFFERENTIABLE_VIEW_INPUT4_REFERENCE_BUILD)/payload/source/pytorch_differentiable_view_input_boundaries_probe.py" \
	  --support-reader tests/differentiable_view_native_review/replay_direct_view14.py \
	  --expected-source-sha256 "$(DIFFERENTIABLE_VIEW_SOURCE_SHA256)" \
	  --expected-native-sha256 "$(DIFFERENTIABLE_VIEW_NATIVE_SHA256)" \
	  --expected-observer-sha256 "$(DIFFERENTIABLE_VIEW_OBSERVER_SHA256)" \
	  --expected-reference-sha256 f4abe6a750ecc791b3a7bc7aa507b4c77eb0a2fcccf45b0c6a828db0e01490d3 \
	  --expected-functions "$(DIFFERENTIABLE_VIEW_FUNCTIONS)" --expected-native-units "$(DIFFERENTIABLE_VIEW_UNITS)" \
	  --output build/differentiable-view-native-input4

# Three finite R3 preserve-resource controls; retain all raw evidence on failures.
.PHONY: test-differentiable-view-preserve-resources
test-differentiable-view-preserve-resources: differentiable-view-recovered $(DIFFERENTIABLE_VIEW_INPUT4_REFERENCE_BUILD)/report.json
	@printf '%s  %s\n' 'edb879dff167e41fb72b37f6d14d2ad7cfd7f48eef87fade1c96146cee7308bc' tests/differentiable_view_preserve_resource_review/test_preserve_resources.py | sha256sum -c -
	python3 -S -B tests/differentiable_view_preserve_resource_review/test_preserve_resources.py \
	  --native-build "$(abspath $(DIFFERENTIABLE_VIEW_BUILD))" \
	  --profile tools/differentiable_view_recovered_r3/profile.json \
	  --protocol tests/differentiable_view_preserve_resource_review/protocol.py \
	  --reference "$(abspath $(DIFFERENTIABLE_VIEW_INPUT4_REFERENCE_BUILD))/payload/actual/actual.json" \
	  --output "$(abspath build/differentiable-view-preserve-resources)"

# R4 concrete view/reshape and Scalar07b: fixed fresh units, explicit success stamps.
DIFFERENTIABLE_VIEW_R4_SHAPE_BUILD ?= build/differentiable-view-r4-shape
DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD ?= build/differentiable-view-r4-scalar07b
DIFFERENTIABLE_VIEW_R4_VIEW26_REFERENCE ?= build/view-reshape-r4-reference26
DIFFERENTIABLE_VIEW_R4_SCALAR32_REFERENCE ?= build/view-reshape-r4-reference32
DIFFERENTIABLE_VIEW_R4_REPLAY_ROOT ?= build/view-reshape-r4-native
DIFFERENTIABLE_VIEW_R4_SOURCE_SHA256 := c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a
DIFFERENTIABLE_VIEW_R4_NATIVE_SHA256 := 0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74
DIFFERENTIABLE_VIEW_R4_OBSERVER_SHA256 := 59019b5c4aeae0d7e04b362354bd9ca55917d2196791d3d18c4908a05fdbba8b
DIFFERENTIABLE_VIEW_R4_BUILD_REVIEW := $(wildcard tools/differentiable_view_r4_build_review/*.py tools/differentiable_view_r4_build_review/*.json)
DIFFERENTIABLE_VIEW_R4_SHAPE_INPUTS := $(wildcard tools/differentiable_view_shape_r4/*.py tools/differentiable_view_shape_r4/*.md tools/differentiable_view_shape_r4/generated/* tools/differentiable_view_shape_r4/generated/units/*.hepta)
DIFFERENTIABLE_VIEW_R4_SCALAR_INPUTS := $(wildcard tools/differentiable_view_scalar_r4/*.py tools/differentiable_view_scalar_r4/*.md tools/differentiable_view_scalar_r4/generated/* tools/differentiable_view_scalar_r4/generated/units/*.hepta)

.PHONY: differentiable-view-r4-shape differentiable-view-r4-scalar07b test-differentiable-view-r4-references test-differentiable-view-r4-native76 test-differentiable-view-r4-native26 test-differentiable-view-r4-scalar32 test-differentiable-view-r4-native14 test-differentiable-view-r4-input4

differentiable-view-r4-shape: $(DIFFERENTIABLE_VIEW_R4_SHAPE_BUILD)/success.json

$(DIFFERENTIABLE_VIEW_R4_SHAPE_BUILD)/success.json: $(DIFFERENTIABLE_VIEW_R4_BUILD_REVIEW) $(DIFFERENTIABLE_VIEW_R4_SHAPE_INPUTS) $(DIFFERENTIABLE_VIEW_BUILD)/build.json | build
	python3 -I -S -B tools/differentiable_view_r4_build_review/build.py \
	  --stage shape --previous "$(abspath $(DIFFERENTIABLE_VIEW_BUILD))" \
	  --upstream "$(abspath $(HEPTA_SOURCE_ROOT))" \
	  --compiler "$(abspath $(HEPTA_STACK_BUILD))/native/stack-module-compiler" \
	  --linker "$(abspath $(HEPTA_STACK_BUILD))/native/stack-linker-session" \
	  --output "$(abspath $(DIFFERENTIABLE_VIEW_R4_SHAPE_BUILD))"

differentiable-view-r4-scalar07b: $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD)/success.json

$(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD)/success.json: $(DIFFERENTIABLE_VIEW_R4_BUILD_REVIEW) $(DIFFERENTIABLE_VIEW_R4_SCALAR_INPUTS) $(DIFFERENTIABLE_VIEW_R4_SHAPE_BUILD)/success.json | build
	python3 -I -S -B tools/differentiable_view_r4_build_review/build.py \
	  --stage scalar07b --previous "$(abspath $(DIFFERENTIABLE_VIEW_R4_SHAPE_BUILD))/native" \
	  --upstream "$(abspath $(HEPTA_SOURCE_ROOT))" \
	  --compiler "$(abspath $(HEPTA_STACK_BUILD))/native/stack-module-compiler" \
	  --linker "$(abspath $(HEPTA_STACK_BUILD))/native/stack-linker-session" \
	  --output "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))"

test-differentiable-view-r4-references: $(DIFFERENTIABLE_VIEW_R4_VIEW26_REFERENCE)/.complete $(DIFFERENTIABLE_VIEW_R4_SCALAR32_REFERENCE)/.complete

$(DIFFERENTIABLE_VIEW_R4_VIEW26_REFERENCE)/.complete: tests/differentiable_view_shape_reference_review/prepare_reference.py tests/differentiable_view_shape_reference_review/verify_packet.py $(wildcard validation/differentiable-view-shape-reference26/*) | build
	python3 -I -S -B tests/differentiable_view_shape_reference_review/prepare_reference.py \
	  --suite view26 --output "$(abspath $(DIFFERENTIABLE_VIEW_R4_VIEW26_REFERENCE))"
	touch "$@"

$(DIFFERENTIABLE_VIEW_R4_SCALAR32_REFERENCE)/.complete: tests/differentiable_view_shape_reference_review/prepare_reference.py $(wildcard validation/wrapped-f64-reference32/*) | build
	python3 -I -S -B tests/differentiable_view_shape_reference_review/prepare_reference.py \
	  --suite scalar32 --output "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR32_REFERENCE))"
	touch "$@"

test-differentiable-view-r4-native76: test-differentiable-view-r4-native26 test-differentiable-view-r4-scalar32 test-differentiable-view-r4-native14 test-differentiable-view-r4-input4

test-differentiable-view-r4-native26: differentiable-view-r4-scalar07b $(DIFFERENTIABLE_VIEW_R4_VIEW26_REFERENCE)/.complete
	@printf '%s  %s\n' 'b37c58e597e3c180736133ee4ac7826b76dd9b0ac41d7b902b3057eabc7297a6' tests/differentiable_view_shape_review/replay_shape26_epoch_r1.py | sha256sum -c -
	python3 -B tests/differentiable_view_shape_review/replay_shape26_epoch_r1.py \
	  --binary "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))/native/tensor-shape-scalar-session" \
	  --source "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))/native/runtime.hepta" \
	  --build-report "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))/native/build.json" \
	  --protocol "tests/differentiable_view_native_review/protocol.py" \
	  --observer "tests/differentiable_view_shape_review/native_observer_shape02.py" \
	  --reference "$(abspath $(DIFFERENTIABLE_VIEW_R4_VIEW26_REFERENCE))/actual.json" \
	  --probe "$(abspath $(DIFFERENTIABLE_VIEW_R4_VIEW26_REFERENCE))/probe.py" \
	  --expected-source-sha256 "c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a" \
	  --expected-native-sha256 "0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74" \
	  --expected-observer-sha256 "59019b5c4aeae0d7e04b362354bd9ca55917d2196791d3d18c4908a05fdbba8b" \
	  --expected-functions "658" \
	  --expected-native-units "13" \
	  --output "$(abspath $(DIFFERENTIABLE_VIEW_R4_REPLAY_ROOT))/view26" \
	  --support-reader "tests/differentiable_view_native_review/replay_direct_view14.py" \
	  --expected-reference-sha256 "fecf1956790f78a9b1238f535c1c7cce9f1237b211c389ac0736d88687bbb9c9" \
	  --identity-adapter "tests/differentiable_view_shape_review/identity_epochs.py" \
	  --parent-reader "tests/differentiable_view_shape_review/replay_shape26_r3.py"

test-differentiable-view-r4-scalar32: differentiable-view-r4-scalar07b $(DIFFERENTIABLE_VIEW_R4_SCALAR32_REFERENCE)/.complete
	@printf '%s  %s\n' '7386703b3deb01d6ab809447a1e6ae83d3c87ab1b4264d9e54641dbf99b93ad0' tests/differentiable_view_shape_review/replay_wrapped_scalar32_r1.py | sha256sum -c -
	python3 -B tests/differentiable_view_shape_review/replay_wrapped_scalar32_r1.py \
	  --binary "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))/native/tensor-shape-scalar-session" \
	  --source "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))/native/runtime.hepta" \
	  --build-report "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))/native/build.json" \
	  --protocol "tests/differentiable_view_native_review/protocol.py" \
	  --observer "tests/differentiable_view_shape_review/native_observer_shape02.py" \
	  --reference "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR32_REFERENCE))/actual.json" \
	  --probe "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR32_REFERENCE))/probe.py" \
	  --expected-source-sha256 "c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a" \
	  --expected-native-sha256 "0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74" \
	  --expected-observer-sha256 "59019b5c4aeae0d7e04b362354bd9ca55917d2196791d3d18c4908a05fdbba8b" \
	  --expected-functions "658" \
	  --expected-native-units "13" \
	  --output "$(abspath $(DIFFERENTIABLE_VIEW_R4_REPLAY_ROOT))/scalar32" \
	  --support-reader "tests/differentiable_view_native_review/replay_direct_view14.py"

test-differentiable-view-r4-native14: differentiable-view-r4-scalar07b $(DIFFERENTIABLE_VIEW_REFERENCE_BUILD)/report.json
	@printf '%s  %s\n' 'deb9f6fafc3f91b2749bdfac030e19cb0ccdb832fa238af91ed9feb396e62eb8' tests/differentiable_view_input4_review/replay_direct_view14_profile.py | sha256sum -c -
	python3 -B tests/differentiable_view_input4_review/replay_direct_view14_profile.py \
	  --binary "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))/native/tensor-shape-scalar-session" \
	  --source "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))/native/runtime.hepta" \
	  --build-report "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))/native/build.json" \
	  --protocol "tests/differentiable_view_native_review/protocol.py" \
	  --observer "tests/differentiable_view_shape_review/native_observer_shape02.py" \
	  --reference "$(abspath $(DIFFERENTIABLE_VIEW_REFERENCE_BUILD))/payload/original/actual.json" \
	  --probe "$(abspath $(DIFFERENTIABLE_VIEW_REFERENCE_BUILD))/payload/original/inputs/differentiable-view14-probe-1ca74384.py" \
	  --expected-source-sha256 "c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a" \
	  --expected-native-sha256 "0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74" \
	  --expected-observer-sha256 "59019b5c4aeae0d7e04b362354bd9ca55917d2196791d3d18c4908a05fdbba8b" \
	  --expected-functions "658" \
	  --expected-native-units "13" \
	  --output "$(abspath $(DIFFERENTIABLE_VIEW_R4_REPLAY_ROOT))/view14"

test-differentiable-view-r4-input4: differentiable-view-r4-scalar07b $(DIFFERENTIABLE_VIEW_INPUT4_REFERENCE_BUILD)/report.json
	@printf '%s  %s\n' '5b8809eb9e405bf89a3586741d64210483fe3fbc5bc323e2cc16cb712ce4de66' tests/differentiable_view_input4_review/replay_input_boundaries4.py | sha256sum -c -
	python3 -B tests/differentiable_view_input4_review/replay_input_boundaries4.py \
	  --binary "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))/native/tensor-shape-scalar-session" \
	  --source "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))/native/runtime.hepta" \
	  --build-report "$(abspath $(DIFFERENTIABLE_VIEW_R4_SCALAR_BUILD))/native/build.json" \
	  --protocol "tests/differentiable_view_native_review/protocol.py" \
	  --observer "tests/differentiable_view_shape_review/native_observer_shape02.py" \
	  --reference "$(abspath $(DIFFERENTIABLE_VIEW_INPUT4_REFERENCE_BUILD))/payload/actual/actual.json" \
	  --probe "$(abspath $(DIFFERENTIABLE_VIEW_INPUT4_REFERENCE_BUILD))/payload/source/pytorch_differentiable_view_input_boundaries_probe.py" \
	  --expected-source-sha256 "c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a" \
	  --expected-native-sha256 "0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74" \
	  --expected-observer-sha256 "59019b5c4aeae0d7e04b362354bd9ca55917d2196791d3d18c4908a05fdbba8b" \
	  --expected-functions "658" \
	  --expected-native-units "13" \
	  --output "$(abspath $(DIFFERENTIABLE_VIEW_R4_REPLAY_ROOT))/input4" \
	  --support-reader "tests/differentiable_view_native_review/replay_direct_view14.py" \
	  --expected-reference-sha256 "f4abe6a750ecc791b3a7bc7aa507b4c77eb0a2fcccf45b0c6a828db0e01490d3"

