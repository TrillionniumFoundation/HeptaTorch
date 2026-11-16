# Mixed F32 / int64 autograd revision 10 handoff

This handoff keeps the exact frozen producer package and independent Saved
lifetime package, followed by additive evidence. The producer source manifest
remains `PREPARED_NOT_RUN`: its source bytes are immutable. The separate actual
cold report records five completed gates; it does not rewrite that source file.

The actual cold execution rebuilt every one of eleven modules and linked a new
native image. Source `5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49`
and ELF `69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6`
were reproduced exactly. The five gates are cold construction, fixed mixed AD36,
eight inherited integer gates, twelve legacy523 gates (including sixteen
inherited suites), and the original independent public Saved lifetime14 reader.

All reference values, raw bits, versions, identity relations, weak lifetime,
RNG and owner rollback assertions retain their original meaning. The data sets
overlap and their case counts are not added into API coverage. The old explicit
partial classifications remain visible. The independent owner/resource/entry/
typed package is separate; new-profile live32 and accounted35 remain separate
pending evidence. The whole-system conclusion remains HOLD.

## Rebuild and replay

The command in `tools/mixed_integer_autograd/README.md` is the complete normal
checkout CLI. It requires a real pinned integer07 parent build, the repository's
pinned compiler sources and real native compilers. It never reuses a mixed image.
All paths are explicit. The root-independent lifetime reader is included under
its own unchanged namespace and exact manifest.

## Preserve and restore evidence

`archive-index.json` lists eight complete archives with exact member inventories.
The historical revision08 zero-grad cleanup failure and revision09 view-base
lifetime failure remain separate from later passes. The independent lifetime
package also preserves its first diagnostic serialization failure, original09
13/14 result and original10 14/14 result with both native images and all units.
The cold10 archive contains compiler calls, source, all eleven objects, native
image, complete subprocess stdout/stderr/commands and every replay raw archive.
Its nested evidence archives retain their own complete manifests and RESTORE
records; extracting a top-level archive does not silently discard them.

Run the following after replacing MANIFEST with the supplied exact handoff SHA:

```sh
python -I -S -B validation/mixed-integer-autograd-evidence/verify_evidence.py \
  --expected-manifest-sha256 MANIFEST
```

To restore one original archive without overwriting any existing path, add:

```sh
  --extract-evidence candidate10-portable-cold --output NEW_EMPTY_PATH
```

The verifier checks every file's size, SHA and mode, every historical archive
member, and exact single-stream XZ EOF. Verification runs no native image and no
Torch; it is byte custody, not a new numerical qualification.
