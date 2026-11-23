# Opt-in public value dtypes for bounded CPU masked_select
               
This follows #67 exact I64 commit c08694a893c76da9c898df4e018b864ec0b1f124,
tree ea4a8114ab9cdb688b3f081bf0f697d58d78578b. The existing F32/I64 roles
remain. The only source change is in ms_tensor: it admits public Bool,
F64, F16 and BF16 values with byte widths 1, 8, 2 and 2 while requiring
the mask to be Bool. Private tag3 and all other unknown dtypes remain
unadmitted. A selected width-1 value is checked to be canonical 0 or 1;
the value is then stored unchanged, not used as a selection condition.
F32/I64/F64/F16/BF16 values are copied as raw words without casts.

The previous shape, cap, owner, allocation, cursor and transaction rules
remain: rank1–4 value, mask shape equal or right-aligned one-way only
(including scalar Bool), ordinary owned zero-offset and no-grad CPU,
input N <= 262,144, fresh result rank1 [K], element stride1 and Version0.
The writable data span is exactly K times byte width. This does not
enable out, symmetric shape expansion, views as positive inputs,
autograd, a new default, or whole-runtime compatibility.

Two distinct cold builds bound exact combined source
25a495092c57ade642c4d3648a16bc42ded060a88473c06b60d22080b99ee73f
to the same native ELF 23ee4cc96d5f000c48dd128fbf65088a1b58b87dd72cd22add6cf0209144ad74.
Locked CPU PyTorch 2.15.0 at threads1 matches all 73 frozen cases;
1,966,544 raw words across four dtypes preserve signed qNaN and sNaN
payloads. The all-select Bool fixture retains nine false bytes.
Eight theoretical out-of-scope inputs are expected to be refused locally,
regardless of whether PyTorch selects zero or two. No allocator OOM
claim.

The exact native predecessor refused all 73 wide/half/Bool-value
fixtures with status7 across 913 no-Torch frames. The corrected
candidate's 1,961-frame campaign admitted all 65 in-scope fixtures,
1,966,536 raw words; eight true overcap requests returned status3
after full original source and mask readback. Both runs independently
validated five large F64 inputs from the original functional F64 add
path, with no use of cat. Original typed factories supplied the smaller
widths. A separate independent reader checks wire and storage buffers,
fresh Storage/Version, signed payloads, and source/result lifetimes.

The initial campaign and two isolated supplements proved 85 op130/weak
rollback snapshots in total (65 weak and 20 non-weak or scope guards).
Two requested negative constructors could not create their fixtures:
F64 view opcode111 returned status7, and F16 grad-mark opcode90 returned
status5. These captures are saved separately as setup-only limitations,
not mislabeled as op130 evidence. A genuine live Bool byte-offset view
was independently proven and refused by op130, and previously frozen
F32/I64 owner-view/refusal evidence is part of the exact retention set.

The staged old F64-value response at I64 campaign frame685 is deliberately
changed from status7 to 0; frame684's non-Bool I64 mask remains byte-exact
status7. Both old/new prefixes and current F64 output identity and raw
four zeros are captured independently. The 3,090 remaining frames are
exact wire comparisons, including previous overlapping witness prefixes,
not the full hosted default gate.
