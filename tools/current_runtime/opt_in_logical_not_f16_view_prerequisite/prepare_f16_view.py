"""Reversible source projection for a standalone F16 op111 constructor."""
from pathlib import Path
from hashlib import sha256
import argparse, importlib.util, json, re

HERE = Path(__file__).resolve().parent
PARENT = "5c62c11900d20782c58b18b948b4074385e816d1936da2d7a03710715dba4f48"
UNIT_PARENT = "bc94220020d7b7c63f9e79abaf46a79cf735c4d01e09e5c9f892f45958226408"
UNIT = "runtime800_11"
COMPOSABLE = "logical_out_permuted_f32_bool_i64_f64_subspan"
BUILDER_SHA = "9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d"
NAME = b"public_f16_whole_dense_view"
HELPER = (HERE / "f16_view.hepta").read_bytes()
REPL = {
    b"to_command": (
        b"  v85:bool = bool.or(rejected_f64,v84)\n",
        b"  rejected_f16:bool = bool.and(v82,other_f64_command)\n"
        b"  rejected_half:bool = bool.or(rejected_f16,v83)\n"
        b"  v85:bool = bool.or(rejected_f64,rejected_half)\n",
    ),
    b"to_strided_view": (
        b"  branch direct_f64 f64_constructor(v120,v121,v122,v123,v124,v125,v126) guarded_constructor(v120,v121,v122,v123,v124,v125,v126)\n",
        b"  branch direct_f64 f64_constructor(v120,v121,v122,v123,v124,v125,v126) half_constructor_select(v120,v121,v122,v123,v124,v125,v126,constructor_dtype)\n"
        b"block half_constructor_select(hr:readbuf,hs:mutbuf,hh:u64,hp:u64,he:u64,hi:u64,hd:u64,hdt:u64)\n"
        b'  f16_tag:u64 = const() {"value":5}\n'
        b"  direct_f16:bool = compare.eq(hdt,f16_tag)\n"
        b"  branch direct_f16 f16_constructor(hr,hs,hh,hp,he,hi,hd) guarded_constructor(hr,hs,hh,hp,he,hi,hd)\n"
        b"block f16_constructor(xr:readbuf,xs:mutbuf,xh:u64,xp:u64,xe:u64,xi:u64,xd:u64)\n"
        b'  xv:u64 = call(xr,xs,xh,xp,xe,xi,xd) {"function":"public_f16_whole_dense_view"}\n'
        b"  jump constructed(xr,xs,xi,xd,xv)\n",
    ),
}
IMPORTS = (
    b'fn is_dense(v0:mutbuf,v1:u64) -> (u64) effects(read,write,trap) external()\n'
    b'fn so_version_live(v0:mutbuf,v1:u64) -> (u64) effects(read,write,trap) external()\n'
)

def sha(b): return sha256(b).hexdigest()

def swap(raw, old, new):
    assert raw.count(old) == 1, old
    return raw.replace(old, new, 1)

def project(raw, unit=False):
    original = raw
    pairs = []
    for fn, (before, after) in REPL.items():
        old = re.search(rb"^fn " + fn + rb"\([^\n]+\n.*?^end\n", raw, re.M | re.S)[0]
        new = swap(old, before, after)
        if fn == b"to_strided_view":
            new = HELPER + new
        raw = swap(raw, old, new)
        pairs.append((new, old))
    if unit:
        old_line = raw.splitlines(keepends=True)[1]
        assert old_line.startswith(b"exports(") and old_line.endswith(b")\n")
        assert NAME not in old_line
        line = old_line[:-2] + b"," + NAME + b")\n" + IMPORTS
        raw = swap(raw, old_line, line)
        pairs.append((line, old_line))
    reverse = raw
    for new, old in reversed(pairs):
        reverse = swap(reverse, new, old)
    assert reverse == original
    return raw

def controls(builder, raw):
    signatures = builder.parse_source(raw)["signatures"]
    def nargs(p): return len(p.split(b",")) if p else 0
    for fn, fn_body in builder.parse_source(raw)["definitions"].items():
        if fn.encode() not in {*REPL.keys(), NAME}: continue
        labels = dict((m[1], nargs(m[2])) for m in re.finditer(rb"^block (\w+)\(([^)]*)\)$", fn_body, re.M))
        for line in fn_body.splitlines():
            if line.startswith((b"  jump ", b"  branch ")):
                for target, args in re.findall(rb"(\w+)\(([^)]*)\)", line):
                    assert labels[target] == nargs(args), (fn, line)
            c = re.search(rb' = call\(([^)]*)\) \{"function":"(\w+)"\}', line)
            if c:
                parameter_types = signatures[c[2].decode()][0]
                assert len(parameter_types) == nargs(c[1]), (fn, line)

def load_builder(tool):
    assert sha(tool.read_bytes()) == BUILDER_SHA
    spec = importlib.util.spec_from_file_location("pinned_builder", tool)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    return b

def source_set(src, build, exact=None):
    m = json.loads((src / "manifest.json").read_bytes())
    runtime = (src / m["runtime"]["path"]).read_bytes()
    assert sha(runtime) == m["runtime"]["sha256"]
    old = build.parse_source(runtime)
    data = {Path(ent["path"]): (src / ent["path"]).read_bytes() for ent in m["units"]}
    for ent in m["units"]:
        assert sha(data[Path(ent["path"])]) == ent["sha256"]
    unitpath = Path("units") / (UNIT + ".hepta")
    assert sha(data[unitpath]) == UNIT_PARENT
    if exact is None:
        assert sha(runtime) == PARENT
    else:
        exactold, _, original_units = exact
        assert old["signatures"] == exactold["signatures"]
        changed = {n for n, body in old["definitions"].items() if body != exactold["definitions"][n]}
        assert changed == {COMPOSABLE}
        assert original_units.keys() == data.keys()
        changed_units = {p for p in data if data[p] != original_units[p]}
        assert changed_units == {Path("units/runtime800_12.hepta")}
    save_original_units = dict(data)
    projection = project(runtime)
    aggregate = build.parse_source(projection)
    controls(build, projection)
    assert set(aggregate["definitions"]) - set(old["definitions"]) == {NAME.decode()}
    assert set(n for n in old["definitions"] if old["definitions"][n] != aggregate["definitions"][n]) == {n.decode() for n in REPL}
    assert all(aggregate["signatures"][n] == sig for n, sig in old["signatures"].items())
    m.pop("expected_native_sha256", None)
    m["name"] = "opt_in_public_f16_whole_dense_view"
    m["runtime"].update(sha256=sha(projection), **{k: aggregate[k] for k in ("functions", "blocks", "instructions")})
    data[unitpath] = project(data[unitpath], unit=True)
    for ent in m["units"]:
        if ent["name"] == UNIT:
            ent["sha256"] = sha(data[unitpath])
            ent.pop("expected_object_sha256", None)
    data[Path(m["runtime"]["path"])] = projection
    checks = build.check_sources(m, src, lambda path, digest: verified(data[path.relative_to(src)], digest))[2]
    return (old, (m, data, checks), save_original_units)

def verified(raw, digest):
    assert sha(raw) == digest
    return raw

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("builder", type=Path)
    p.add_argument("--compose", type=Path)
    p.add_argument("--output", type=Path, help="only once sole physical lane authorizes source writes")
    a = p.parse_args()
    build = load_builder(a.builder)
    exact = source_set(a.source, build)
    m, data, checks = (source_set(a.compose, build, exact) if a.compose else exact)[1]
    if a.output:
        a.output.mkdir()
        (a.output / "units").mkdir()
        for fp, contents in data.items():
            (a.output / fp).write_bytes(contents)
        (a.output / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, sort_keys=True))

if __name__ == "__main__":
    main()
