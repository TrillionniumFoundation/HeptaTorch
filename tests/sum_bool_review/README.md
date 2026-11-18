# Explicit Bool sum validation

This suite qualifies command 115 with an explicit Bool destination for Bool and
I64 inputs on the bool700 runtime. It retains the original Integral24 and five
dtype controls without edits.

## Reference

reference.py executes 27 literal input recipes through the fixed official
PyTorch 2.15.0+cpu wheel. It reuses the unchanged provenance checks in
tests/pytorch_oracle.py: Python/platform, backend/version, wheel build commit,
installed distribution, PEP 610 origin, wheel SHA-256 and original pip report.
The wheel build is d0f2af8b66b20da4baa50644c0cd3f887a2eae2e; the existing lock
records its source equivalence with release
31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457.

Run from the repository containing the original install report:

~~~sh
/path/to/pinned/python -B /path/to/sum_bool_review/reference.py \
  --repository /absolute/repository \
  --output /new/reference27
~~~

The three files reference.py, cases.py and common.py must be together.
The repository must contain the original tests/pytorch_oracle.py,
compatibility/pytorch-oracle-lock.json, compatibility/upstream-lock.json and
build/pytorch-install.json. The --prepare-only option validates the local
source/input plan without importing Torch or starting a native executable.

The reference records actual output values and geometry, physical and logical
Storage bytes, Versions, gradient/leaf/view state, input mutation, Tensor and
Storage weak lifetimes, exceptions, warnings and complete 5,056-byte CPU RNG
states. All initial inputs and dimensions are literals. Cases include integer
cancellation, high-bit values, rank zero, None and empty dimension lists,
multiple/negative axes, duplicate/out-of-range axes, holes, overlap, zero
strides, empty reduced dimensions and empty outputs.

## Native comparison

First build the current runtime with tools/current_runtime/build.py. The
native runner verifies the source, all 14 freshly compiled units, the linker
request, all 15 actual compiler/linker process results and complete EOF records.
The fixed 700-function source is
b83a8346b162e12c5ed77e0287d626619a18f84bf7a1ab02a2514d30ae4295b4; its executable
is c6afeedc14ba8cc408a62d6fbef0abb5b78dff0039b8099f3449d813e290a90e.

~~~sh
python3 -B tests/sum_bool_review/native.py \
  --repository /absolute/repository \
  --binary /absolute/build/heptatorch \
  --source /absolute/build/runtime.hepta \
  --build-report /absolute/build/build.json \
  --reference /absolute/reference27/report.json \
  --expected-source-sha256 b83a8346b162e12c5ed77e0287d626619a18f84bf7a1ab02a2514d30ae4295b4 \
  --expected-native-sha256 c6afeedc14ba8cc408a62d6fbef0abb5b78dff0039b8099f3449d813e290a90e \
  --expected-build-sha256 ACTUAL_BUILD_REPORT_SHA256 \
  --expected-reference-sha256 ACTUAL_REFERENCE_REPORT_SHA256 \
  --output /new/native27
~~~

The two report hashes identify the actual independently observed reference and
the current cold build. Both reports are bound before a child starts. The
native comparison requires only Python's standard library, Linux child
observation support and the freshly built HIA ELF.

Typed input Storage uses command 97. Input views use the current command 111
entry and the exact packet layout of the original Integral24 suite's inherited
Case.view. Native error normalization preserves that suite's distinction:
an out-of-range dimension IndexError maps to cause 2, while the exact duplicate
dimension RuntimeError maps to cause 7.

common.py recovers the frozen Integral owner and byte observer from the
committed reviewed intent. Every original byte is hash checked. The exact,
reversible adaptation changes only the owner format tags and 700-function
profile. owner_adapter.py adds a separate Bool-success branch: dtype 1,
one byte per output element, fresh Tensor/Storage/Version, no gradient, Saved
value or input/view edge. The original eight-byte I64 predicate is retained.

Every native frame retains its request, response, complete owner before and
after, original 84-word ownership checks, RNG state and process cleanup. The
27 cases compare all reference marks in order. Missing observations and
differences fail the run.

The additional controls have separate denominators:

| Group | Controls |
| --- | --- |
| Explicit dtype contract | Two supported Bool destinations; three inherited scope refusals; F32-to-Bool and wrapped-F64-to-Bool refusals |
| Packet and capability | Ten invalid/reserved/payload/stale-capability commands with full owner rollback |
| Allocation | Actual output Storage growth failure, complete owner/mappings/RNG rollback, restoration of the test child's original memory limit, and successful execution of the same command |

The allocation control permits the ordinary same-size transaction copy while
denying pool growth. It verifies that a same-capacity query succeeds under the
limit and that result descriptor pages and handle slots already exist. It
changes only the resource limit of the child created by this test and restores
the original limit in a finally block.

report.json reports PASS only after all 27 reference cases, seven dtype
controls and both boundary groups pass. Full raw evidence remains under each
case directory, including raw/transport.json with both EOFs. Preserve the
whole output directory as the run artifact.

This is the explicit Bool-destination extension. The separate inherited
Integral24, Sum32 and original76 suites still require execution on this same
image; none of their older results transfers automatically. Full PyTorch
schema and system gate completion remains governed by full-acceptance.json.

## Observed qualification

The 2026-11-18 local run compared the cold-built image above with the actual
27-case fixed-wheel reference, whose report SHA-256 is
21dd75aeb86414818f7aa381f8b866d78b16f9bb3f5738e9a60d27ac3d15dd28.
The native report SHA-256 is
e85a1d1ed6cecf2ccec7665475deb119b42f382280fc6fcaf124b4b3f34d6bd2.
It records 27/27 reference cases, 7/7 dtype controls, all ten entry/capability
controls, and the actual allocation-failure/recovery control passing. The
36 native children completed 6,123 frames with exit 0 and both output EOFs;
all bound inputs remained unchanged. Of the reference cases, 23 accept the
requested sum and four observe the expected invalid-axis exception.

The first run remains a recorded FAIL with report SHA-256
74d25ad3cc6b075dec7ab6286d9733a877af3e8220514d50cd9ead744ca1cd16.
Its new runner selected the older command 69 helper for input views and
incorrectly grouped out-of-range IndexError with duplicate-axis errors.
The subsequent run corrected those two test interfaces using the original
Integral24 contracts. Every tensor, ownership, lifetime and rollback assertion
was retained; HIA source, native executable, reference probe and input recipes
were unchanged. The original failed report and exact runner sources are
retained with the execution evidence.
