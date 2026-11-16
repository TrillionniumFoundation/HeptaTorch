# Typed native dropout: evidence and limits

This package binds one native revision: HIA source `6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3` and ELF `830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51`. Its parent is the frozen retained446 profile. The new runtime contains 523 core functions in nine core modules, plus the unchanged RNG and owner modules. All Tensor computation and object ownership execute in HIA.

The first portable cold run reproduced all eleven source/object units and the final executable, then passed its thirteen named top-level steps. The inherited sixteen-suite coordinator is one of those steps. `reports/portable-cold-11.json` is the original complete report; `qualification.json` explains the scope without changing any subordinate report. The fixed source/native identities are also required by `tools/native_dropout/profile.json`.

## What is represented

Bool masks have one-byte Storage and real dtype metadata. Clone, detach, views, public Saved values, Weak/Hold records and unpack preserve dtype and extent. Legacy F32-only operations reject bool before four-byte data access. Gradient assignment checks dtype as well as shape. Actual mask mutation changes its shared Version and triggers Saved checking; the Saved output holds the real mask Tensor identity.

The optional-training and empty branches follow fixed actual Torch observations. `None` and `True` have the same nonempty forward mask, but different derivative scaling. `False` creates a clone and bool ones without consuming random state, including the observed invalid-probability inputs. Empty native dropout returns the same input Tensor and an empty F32 mask. For a differentiable leaf, changing that Tensor's current history exposes the original captured AccumulateGrad edge and produces the observed interior error. Empty nonleaf inputs and forward-without-grad followed by enabling grad retain their distinct observed behavior.

The strong cycle observed in the invalid empty-leaf history path is preserved explicitly; it is not hidden by falsely freeing a live Tensor. `detach_` changes the same Tensor's history, keeps Storage, Version and existing gradient identities, and can break the cycle. True views reject `detach_`, including bool and empty views. The bool logical-not operation performs actual one-byte logical negation and one Version increment, including empty tensors.

Existing Optimizer membership remains valid when the same previously accepted leaf acquires history. Constructor eligibility is not rerun as a lifetime invariant. Private Tensor roots, incarnation, dtype, shape, state keys and reference counts remain checked. Private zero-grad works on the existing member without changing public leaf-only APIs. The first SGD momentum follows the actual effective gradient: maximize creates a unary result, weight decay creates a pointwise result, and only then is the momentum cloned. Plain SGD preserves the original gradient-clone path. No reference stride or Version was changed to accommodate an implementation.

## Read the reports separately

| Evidence | Admission and purpose |
| --- | --- |
| Actual native-dropout 28 | Every fixed case passes. Values, dtype, extent, layout, aliasing, Saved errors, gradient outcomes and complete RNG states are compared. Private scalar Saved attributes are not represented by this interface. |
| Original native-dropout 8 | Additive replay of original distribution rows 167 through 174. The immutable 183-row parent report retains its old classifications. |
| Existing Optimizer 6 | All six fixed recipes pass, including zero-grad and steps after same-Tensor history changes. |
| SGD effective-layout 6 | All six independent fixed recipes pass, including plain and maximize branches, empty and transposed assigned gradients. |
| Retained 12 | Only exact `QUALIFIED_EXPOSED_SCOPE`, twelve executed/twelve qualified/zero failures, unchanged source and eleven original provenance checks are accepted. |
| Native controls | Typed storage lifetime/migration, all 256 byte inputs for logical-not lowering, 101 explicit F32/dtype refusals, complete owner/RNG rollback and real OS allocation failure/recovery. |
| Inherited 16 | Original numerical assertions, tolerances, partial rows, unrepresented rows and state-lifetime diagnostics stay unchanged. |
| Independent actual identity 9 | A separate parent-authored replay validates the complete nine recipes, including all five weak-lifetime variants. This producer's eight mathematical/resource controls do not replace it. |
| Independent detach 26 and native owner gates | Separate peer package validates all original operations and witnesses, real typed-owner barriers and output-capacity transactions. |
| Independent 640-byte accounting | Must be admitted separately; it is not inferred from the producer's internal audit. |

These suites overlap. Their frame, assertion and value counts are not a unique upstream test count or API coverage denominator. The global denominator remains 2590, the fully closed schema count remains zero, and full-project closure remains on hold across all seventeen system dimensions.

## Original failures are retained

The archives preserve exact original inputs, reports and source/native identities. The following distinctions matter:

| Revision or test | Original result | Correction or subsequent evidence |
| --- | --- | --- |
| Candidate01/02 | Host source/callee preparation refused before native compilation | Exact source target and real HIA helper resolved; no checker limit changed |
| Candidate03 | Native compilation finished, but source custody rejected a source edit during the build | New candidate with frozen author inputs; old receipt remains rejected |
| Candidate04 | A bool Hold entered the old F32 reader | Candidate05 adds real dtype admission, including empty Holds |
| Candidate04/05 contract 28 | Twelve empty identity cases explicitly unrepresented | Later independent same-Tensor/history implementation and fixed replay |
| Initial slice control | Missing explicit drop of a published gradient token | Host recipe correction, original FAIL retained |
| Admission controls | A write prefix changed its own Saved Version, then a cross-dtype case expected the wrong error priority | Separate host corrections with unchanged production and original failures |
| Initial allocation control | Setup exhausted address-space before the target allocation | Explicit 32MiB setup and snapshot placement; actual target OOM and recovery run |
| Initial history controls | Opcode2 performed Sub, while the intended actual recipe used Mul | Original Sub resource report retained; exactly five opcode literals corrected in a separate Mul control |
| Candidate07 | Existing Optimizer audit and zero wrongly reapplied leaf/retained constructor eligibility | Candidate10 separates membership invariants from construction; all other private-root and shape checks remain |
| Candidate08/09 | Wrong host alias refused before native emission | Exact real helper naming fixed in a new candidate |
| Candidate10 actual6 | Five pass, one true SGD empty momentum stride failure | Candidate11 implements effective-gradient layout |
| Candidate10 new maximize6 | Four pass, two true empty layout failures | Same fixed actual comparison passes on Candidate11 |
| Final11 development coordinator | Aggregate FAIL because it did not recognize retained runner's existing success label; all child exits were zero | Additive strict admission checks the exact twelve cases, reference, status and provenance; original FAIL is unchanged |

The inherited06 preparation ENOSPC and all original coordinator/test output are also preserved. Revisions are not relabeled. A host-side correction is not presented as a native bug fix; a native bug is not dismissed as a host issue.

## Custody and rebuild

`tools/native_dropout/package-manifest.json` freezes the 99 exact execution inputs and original receipts that were present before the first portable cold run. The original cold report binds that manifest. Supplementary complete raw archives, this README and the additive qualification are covered by the outer `handoff-manifest.json`; they do not alter the already executed inputs.

Each compressed archive has a member manifest containing length, SHA-256 and mode. The archive was read back and every member verified. `.tar.xz` packaging is additionally checked against the same uncompressed tar stream as its original gzip staging copy. No raw source, request, response, owner dump or old failure is regenerated from a summary. Historical absolute paths in original receipts are development references, not a promise that those paths remain available; archive member paths and repository-relative package paths provide the portable copy.

The producer does not vendor or alter the upstream compiler. Use the documented pinned bootstrap and a real retained446 build. `tools/native_dropout/build.py` emits and compiles the actual HIA source, then requires byte identity of all units and the final image. `qualify.py` runs the named suites in a new output directory. The independent packages keep their own manifests and must be source-bound by the final aggregate.

This is a CPU binary32/bool profile with explicit rank, size, request and work budgets. F64/I64 kernels, generic promotion, arbitrary Python wrapper identities, warning parity, the complete Saved introspection API, every optimizer signature and complete PyTorch remain open. The source layout reserves room for later dtype work without claiming those kernels already exist.
