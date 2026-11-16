# Native default backward and Saved release

This component adds command57 to the frozen numerical Tensor object runtime.
It implements successful default backward release and explicit `retain_graph`
while preserving the existing frame transaction. Its arithmetic, forward
operators, matrix snapshot policy, public object interfaces and legacy backward
behavior on an unspent graph remain those of the checked candidate09 source.
It does not claim full PyTorch compatibility or closure of all 2,590 schemas.

## Public interface and internal state

The existing outer `HTOWNR01` / `HTOWNS01` framing and inner `HTORCH03` /
`HTORSP03` packet formats remain unchanged. A command record has eight u64
words; command57 uses these words:

| Word | Meaning |
| --- | --- |
| 0 | 57 |
| 1 | Public output Tensor ID, or a prior same-frame Tensor result reference |
| 2 | Public explicit seed Tensor ID, or zero for an implicit scalar seed |
| 3 | 0 releases Saved roles after successful application; 1 retains them |
| 4–7 | Reserved unused words, following the existing record contract |

Flags outside 0/1 fail with status7. An explicit seed must have exactly the
output shape and may have strides. The omitted seed requires a one-element
output. Existing typed-handle, shape and gradient requirements are unchanged.
Success returns the usual result record with scalar value1 and no payload.
Command57's value is not a Tensor result reference.

Status10 means that a required AD Saved Tensor role was released by an earlier
default backward. A later retained call cannot restore that role. Legacy
commands7/29 also return10 when they reach an already spent node. The released
role is checked before Saved version mismatch status6 on that node. Other
statuses retain the previous meanings.

The internal owned numerical state is now tagged `HTSTATE4`; frozen candidate09
uses `HTSTATE3`. Both accept a live empty HIA buffer, but reject the other's
nonempty state with status7 before cloning or mutation. This guards the new
invariants: work storage can be absent, and node offset328 marks released Saved
roles. The wire contains no owner pointer. The public wire version does not
identify the private owned state layout.

## Lifetimes and transaction semantics

Default backward releases private Saved Storage/Version obligations after each
successfully applied node. It also releases saved index payloads, including empty
indices where a Saved role still exists. Scatter/index-add need an index role only
when the source requires gradients; their self-only route may repeat after its
unused index allocation is freed. Cat/stack parent lists remain graph topology.
Conv/pool configuration metadata remains until graph collection.

After a successful non-retained pass, work buffers and nonleaf persistent gradient
buffers are freed. Leaf accumulated gradients, graph metadata and leaf
AccumulateGrad references remain. Every backward entry ensures work storage
after graph preflight and before any gradient write. Thus a graph whose actual
derivatives require no Saved Tensor can repeat after release, including
add/clone/view/transpose paths. Public Saved objects and their independent
logical layouts are unaffected by private AD Saved release.

**Command57 keeps whole-frame failure rollback.** A failure restores prior
gradients, Saved obligations, allocations, versions and preceding commands.
Actual PyTorch can release already completed downstream nodes before an upstream
node fails, and a retry can therefore report a released-Saved error instead of
the original version mismatch. The recorded true-Torch case demonstrates this
remaining difference. This component does not assert equivalent failure side
effects; a separate successor interface is being implemented for that behavior.

## Build and identity

Use the already built frozen Tensor object candidate09 as `--baseline`. The
builder verifies its complete core, generated authoring, owned module and profile
pins, then exactly re-emits the three baseline control functions before applying
the lifecycle changes. It does not maintain another copy of the numerical core.

```sh
python3 -I -S -B tools/backward_release/build.py \
  --repository . --source-root "$HIA_SOURCE_ROOT" \
  --baseline build/tensor-objects-stack \
  --compiler "$HEPTA_STACK_MODULE_COMPILER" \
  --linker "$HEPTA_STACK_SESSION_LINKER" \
  --output build/backward-release
```

The selected tools must match the pins in `profile.json`. This profile uses the
existing ABI3 native module compiler and session linker with its guarded 8 MiB
stack. It does not increase compiler, module, instruction or Tensor capacity
limits. The output is five independently compiled HIA modules and one native
interactive executable.

| Identity | SHA256 |
| --- | --- |
| Frozen candidate09 core | `4e2cc7f7612c826ec35280432be527cf98075af7d0f2c35a560df67e35c3904e` |
| This component's core | `e9e325374153f64744fbddfd2ccee00e570585cc42e4ad239361ce035ad4d2c2` |
| Native session | `f146b71b5614b7d1022ef41875f5fbfd857e80028d33b2d634ecd255163f320b` |

There are 242 numerical/lifecycle functions, 2,850 blocks and 11,098 instructions,
plus the three unchanged owned entry functions. Of the 237 candidate09 functions,
232 remain byte-identical. `backward_seed`, `mark_graph` and `object_command`
change; `init_state` and `state_header` differ only in the internal format tag.
Five functions are added. `source-map.json` and `candidate09-to-release.diff`
record every body. The build emits `complete-overlay.hepta` and binds producer,
input, module snapshot, compiler, linker and executable bytes before and after.

## Reproducible qualification

The native fixed-contract suite and the independent controls can be run without
Torch. Their Python only drives the wire and calculates independent expected
results; no Python numeric implementation is used by the executable.

```sh
python3 -I -S -B tools/backward_release/test_release.py \
  --executable build/backward-release/tensor-release-session \
  --protocol tools/tensor_objects/protocol.py --output build/release-native
python3 -I -S -B tools/backward_release/native_release_controls.py \
  --binary build/backward-release/tensor-release-session \
  --expected-sha256 f146b71b5614b7d1022ef41875f5fbfd857e80028d33b2d634ecd255163f320b \
  --source build/backward-release/runtime.hepta \
  --expected-source-sha256 e9e325374153f64744fbddfd2ccee00e570585cc42e4ad239361ce035ad4d2c2 \
  --transport-observer tools/owned_sessions/client.py \
  --output build/release-independent
```

Actual executions on this identity passed 10 native groups / 144 frames and
14 independent controls / 207 frames. The independent replay of the first20
operator cases from the actual36-case pinned Torch report passed 353 frames /
1,363 exact fields. A separate replay of the remaining16 took183 frames and
matched15 cases, with the one declared partial-failure difference described
above and zero unexpected failures. That replay compares142 exact fields and
15 approximate tanh values at its recorded tolerances. The total is35 matching
cases plus one explicit difference; it is not36-case PyTorch parity.
Their exact reports and source identities are retained in `evidence/`.
The original unsuccessful allocation-test setup and its revised fragmentation
setup remain separately labelled; no executable changed to address that setup.

### Legal cross-format owner handoff

This qualification links the complete old and new implementations under
reversible symbol prefixes plus one checked HIA controller. The controller moves
the unique owner into actual `old_step` or `new_step` calls. It never makes a
nonempty owner from a pointer or copies an owner descriptor.

```sh
python3 -I -S -B tools/backward_release/build_state_handoff.py \
  --old build/tensor-objects-stack --new build/backward-release \
  --compiler "$HEPTA_STACK_MODULE_COMPILER" \
  --linker "$HEPTA_STACK_SESSION_LINKER" --output build/state-handoff
python3 -I -S -B tools/backward_release/test_state_handoff.py \
  --executable build/state-handoff/state-handoff-session \
  --build-receipt build/state-handoff/report.json \
  --protocol tools/tensor_objects/protocol.py --output build/state-handoff-test
```

Both directions passed in28 interactive frames: status7 leaves the address,
length and all595,232 compared state bytes unchanged; the original step resumes
and accumulates the expected gradients. Each implementation's close actually
unmaps the other format's owner while the process is still alive. The returned
live empty owner can initialize the other implementation. `/proc` is used only
for read-only observation. The controller has a separate qualification-only
wire and executable identity and is not the production Tensor interface.

All rank, dtype, operator, allocation, descriptor and graph limits inherited
from Tensor object candidate09 remain explicit. This work adds neither
higher-order differentiation nor `create_graph`, hooks or a general PyTorch
frontend. Logical protocol reference counts remain distinct from PyTorch's
internal intrusive pointer counts.
