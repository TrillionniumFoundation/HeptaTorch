#!/usr/bin/env python3
"""Separate exact #63 Generator131 replay. Original numerical and raw predicates unchanged."""
from argparse import ArgumentParser
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import types

BASE = "aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c"
BASE_ELF = "083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5"
BASE_MANIFEST = "a5cf09f191db4887b4452001e0b72e75fffb2147f2476544cc3be49d8ffa2c85"
JOINT = "f4c43f037a3fb046425f7634215ec382aa5f8d6b0e7abe126ef64236adb5753a"
ELF = "5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72"
MANIFEST = "9f24d9903a9ffa96cd6121cbc7130cefe8536f2c3e5a425e6d669f72242de1f0"
BUILD_SOURCE = "9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d"
QUALIFICATION = "f4be6d41c29659c3132f1f64014ad9d8ebcf612db3a9cb556ee5ddd178a540b6"
DUAL_ADMIT = "51b32b1835fe8a294f1f42b98ca3b87b10a5096dec08d34840ac7acab61f672d"
TOOL_REPORT = "074847348cb44cc8d03c09a0855ea571c2fb32891bbdeb757e17586c8a1692a1"
PREPARE = "d31ac9360ba8e91118bdf6543aacd07de9a0033f1d950681ff89adc70cde8bd2"
BINDING = "ca188c5c0a81ab086b658a4e2b86d15be38e8a89d41e81d940453f9ee994cece"
PACKAGE = "98907362668fc194df87fddaba21c96bb3eaf52a3a203416cf8e364662f5e157"
FIXTURES = "eb282d3d61386201d84f6264be31a47a1a73ab4842cf837ad964499a6e89868a"
READER = "aa62ed395d9b7744f9779e859d9f930f28d7072bc841f4de9c880fd62dfadec8"
PIN = {
    "admission": "e22a70f794e20fe6194d3a0b3807f53429896d0d6bef60742f4468eb6d8afe53",
    "run": "28d6f75c0fd966afc391ded54b5eade953f4450b6bd99e4c63373143cb30217f",
    "observe": "75316a0ad636a9b5e1f684cf8da1c9adaae40d5d52efd366bab6e7e3581dbd2f",
    "processes": "13baca50b2df820cf173ccfbea6b36ae795e01c2be369e9ebfbc91f042b64988",
    "controls": "b6d306bcaf2d59ab93ef547d90066a6dbcbdf7019ddfa6c100771278292fed8c",
}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path, pin=None):
    need(path.resolve() == path and path.is_file(), f"unaltered ordinary input required: {path}")
    data = path.read_bytes()
    need(pin is None or sha(data) == pin, f"wrong source identity: {path}")
    return data


def load(path, raw, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, path, "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


def identity_edit(raw, pairs):
    new = raw
    for before, after in pairs:
        b, a = before.encode(), after.encode()
        need(new.count(b) == 1 and a not in new, "single exact identity anchor required")
        new = new.replace(b, a, 1)
    back = new
    for before, after in reversed(pairs):
        need(back.count(after.encode()) == 1, "single exact inverse anchor required")
        back = back.replace(after.encode(), before.encode(), 1)
    need(back == raw, "whole reader/result identity inverse required")
    return new


def dump(path, data):
    with path.open("x") as f:
        f.write(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def avail_memory():
    return int(next(l for l in Path("/proc/meminfo").read_text().splitlines()
                    if l.startswith("MemAvailable:")).split()[1]) * 1024


def physical_budget(a):
    need(shutil.disk_usage("/tmp").free >= 2_500_000_000, "/tmp initial free <2.5GB")
    need(shutil.disk_usage(a.repo).free >= 1_200_000_000, "Workspace free <1.2GB")
    need(avail_memory() >= 6_000_000_000, "MemAvailable initially <6GB")
    need(a.output.is_relative_to(Path("/tmp")) and a.projected.is_relative_to(Path("/tmp")),
         "large outputs restricted to /tmp")


class Proof:
    def __init__(self, a):
        self.a = a
        self.original = {n: read(a.package / (n+".py"), pin) for n, pin in PIN.items()}
        self.read_path = a.repo / "tests/current_runtime_generator131.py"
        self.raw_reader = read(self.read_path, READER)
        self.admission_raw = identity_edit(self.original["admission"],
                              [(BASE, JOINT), (BASE_ELF, ELF), (BASE_MANIFEST, MANIFEST)])
        self.reader_raw = identity_edit(self.raw_reader, [(BASE, JOINT), (BASE_ELF, ELF)])
        self.build_sha = sha(read(a.build/"build.json"))
        self.tool_sha = sha(read(a.tool/"build.json", TOOL_REPORT))
        read(a.package/"package-manifest.json", PACKAGE)
        read(a.repo/"tests/current_runtime_generator_fixtures.py", FIXTURES)
        read(a.repo/"tests/current_runtime_generator_binding.py", BINDING)
        read(a.repo/"tools/current_runtime/qualification.py", QUALIFICATION)
        read(a.repo/"tools/current_runtime/build.py", BUILD_SOURCE)
        self.offline = load(a.repo/"tools/current_runtime/opt_in_admission/admit.py",
                        read(a.repo/"tools/current_runtime/opt_in_admission/admit.py", DUAL_ADMIT),
                        "fixed_exact_joint_offline")
        self.baseline = a.repo/"tools/current_runtime/source"
        self.identity = self.offline.admit("dual", self.baseline/"manifest.json", a.build, {"dual_scoped":a.scoped})
        need(self.identity["native_image_sha256"] == ELF and self.identity["manifest_sha256"] == MANIFEST
             and self.identity["source_sha256"] == JOINT, "exact joint input profile")
        self.prepare = load(a.prepare, read(a.prepare, PREPARE), "fixed_prepare_joint")
        self.source = self.prepare.prepare(self.baseline, a.logical, a.abs, a.projected)
        need(self.source["composition_order_independent"] is True and self.source["all_exact_inverses"] is True
             and self.source["source_sha256"] == JOINT and self.source["manifest_sha256"] == MANIFEST,
             "joint canonical inverse must be complete")
        need(read(a.projected/"runtime.hepta", JOINT) == read(a.build/"runtime.hepta", JOINT)
             and read(a.projected/"manifest.json", MANIFEST) == read(a.build/"manifest.json", MANIFEST),
             "actual executable build uses independently rederived exact source")
        self.bindings = []

    def binding_adapter(self, binding):
        a = self.a
        canonical = binding.bind_current
        def bind_current(repo, candidate, build, build_sha, inputs):
            need(repo == a.repo and candidate == a.candidate and build == a.build
                 and build_sha == self.build_sha, "different joint build or predecessor arguments")
            try:
                canonical(repo, candidate, build, build_sha, inputs)
            except ValueError as error:
                need(str(error) == "build used another source manifest", "legacy admission failed differently: "+str(error))
            else:
                raise ValueError("unchanged canonical reader unexpectedly admitted candidate")
            parent, parent_proof = binding.parent_source_files(repo, inputs)
            for filename, raw in parent.items():
                need(inputs.bind(candidate/filename, sha(raw)) == raw, "original source-only predecessor changed")
            path = repo/"tools/current_runtime/build.py"
            builder = load(path, inputs.bind(path, BUILD_SOURCE), "fixed_joint_source_builder")
            manifest_raw = inputs.bind(a.projected/"manifest.json", MANIFEST)
            manifest = json.loads(manifest_raw, object_pairs_hook=builder.unique_object)
            runtime, units, summary = builder.check_sources(manifest, a.projected, inputs.bind)
            need(summary["source_sha256"] == JOINT and summary["functions"] == summary["complete_core_bodies_matched"] == 800
                 and summary["units_planned"] == 15, "exact complete candidate source census")
            need(runtime == inputs.bind(build/"runtime.hepta", JOINT)
                 and manifest_raw == inputs.bind(build/"manifest.json", MANIFEST), "actual source and manifest copies changed")
            for entry, source, _ in units:
                need(source == inputs.bind(build/(entry["name"]+".hepta"), entry["sha256"]),
                     "actual candidate partition changed")
            repeated = self.offline.admit("dual", self.baseline/"manifest.json", build, {"dual_scoped":a.scoped})
            need(repeated == self.identity, "actual offline dual admission changed")
            self.bindings.append("build used another source manifest")
            return runtime, units, summary, manifest_raw, json.loads(inputs.bind(build/"build.json",build_sha)), parent_proof
        binding.bind_current = bind_current
        return binding

    @contextmanager
    def package(self, package, inputs=None):
        need(Path(package) == self.a.package, "unexpected portable package")
        previous = {n:sys.modules.get(n) for n in ("admission","observe","processes","controls")}
        modules = {}
        try:
            for n in ("admission","observe","processes","controls","run"):
                path = package/(n+".py")
                source = self.original[n]
                if inputs is not None:
                    need(inputs.bind(path, PIN[n]) == source, "canonical portable file changed")
                mod = load(path, self.admission_raw if n == "admission" else source,
                           n if n != "run" else "joint_generator131_live")
                modules[n] = mod
                if n != "run":
                    sys.modules[n] = mod
                if n == "admission":
                    original_loader = mod.loaded_module
                    def candidate_loader(path, raw, name):
                        entity = original_loader(path,raw,name)
                        if name == "generator131_physical800_binding":
                            need(path == self.a.repo/"tests/current_runtime_generator_binding.py" and sha(raw) == BINDING,
                                 "wrong target of candidate-only binding")
                            return self.binding_adapter(entity)
                        return entity
                    mod.loaded_module = candidate_loader
            yield modules
        finally:
            for name, old in previous.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old

    def identities(self):
        return dict(original_portable_package_sha256=PACKAGE, original_sources=PIN,
                    candidate_admission_sha256=sha(self.admission_raw),
                    separate_reader_identity_sha256=sha(self.reader_raw),
                    original_reader_sha256=READER, wrapper_sha256=sha(read(Path(__file__).resolve())),
                    original_numerical_predicates_edited=False, original_canonical_gate_modified=False,
                    dual_build_sha256=self.build_sha, candidate_source_sha256=JOINT, candidate_elf_sha256=ELF,
                    dual_source_inverse=self.source, exact_local_entropy_build_sha256=self.tool_sha)

    def run(self):
        a = self.a
        with self.package(a.package) as m:
            run = m["run"]
            run.SOURCE, run.IMAGE = JOINT, ELF
            code = run.main(["--repository",str(a.repo),"--candidate",str(a.candidate),
                 "--build-directory",str(a.build),"--tool-build",str(a.tool),"--stack-build",str(a.stack),
                 "--original-package-root",str(a.original),"--expected-native-sha256",ELF,
                 "--expected-build-sha256",self.build_sha,"--expected-tool-build-sha256",self.tool_sha,
                 "--output",str(a.output),"--group","all"])
        raw = read(a.output/"report.json")
        report = json.loads(raw)
        dump(a.evidence, dict(self.identities(), phase="native", observer_exit=code,
             report_sha256=sha(raw), canonical_denials=self.bindings, status=report["status"],
             groups={k:v["status"] for k,v in report["groups"].items()}))
        return code

    def readback(self):
        a = self.a
        raw = read(a.output/"report.json")
        report = json.loads(raw)
        retained = json.loads(read(a.native_receipt))
        need(retained["status"] == "PASS_CURRENT800_GENERATOR131" and retained["report_sha256"] == sha(raw)
             and retained["observer_exit"] == 0 and self.identities() == {
                 k: retained[k] for k in self.identities()},
             "must read back the exact admitted fully executed candidate")
        # The original independent reader remains byte-bound. Its global package
        # loader is pointed at the exact same reversible in-memory identity profile.
        reader = load(self.read_path, self.reader_raw, "joint_generator131_independent_reader")
        reader.readers = self.package
        reader.SOURCE, reader.IMAGE = JOINT, ELF
        with self.package(a.package) as m:
            inputs = m["admission"].Inputs()
        values = reader.qualify_generator131(report, a.repo, a.build, self.build_sha, a.output,
                    a.package, a.candidate, a.original, a.tool, a.stack, inputs)
        need(all(inputs.unchanged().values()) and values["cases"] == 131 and len(self.bindings) == 1,
             "original independent readback did not fully verify")
        entry = report["groups"]["entry59"]
        probe = entry["actual_current15_probe_sha256"]
        reordered = entry["actual_current15_reordered_image_sha256"]
        need(len({probe,reordered,BASE_ELF,ELF}) == 4, "genuine distinct jointly-linked probe and reversed image")
        dump(a.evidence, dict(self.identities(), phase="readback", status="PASS_EXACT_JOINT_GENERATOR131",
             report_sha256=sha(raw), original_observer_receipt_sha256=sha(read(a.native_receipt)),
             canonical_denials=self.bindings,
             candidate_probe_elf_sha256=probe, candidate_reordered_elf_sha256=reordered,
             checked_input_count=len(inputs.rows), values=values, candidate_full_suite_qualification=False))
        return 0


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("prepare","execute","readback"))
    for key in ("repo","package","candidate","tool","stack","original","prepare","logical","abs","build",
                "scoped","projected","output","evidence","native-receipt"):
        p.add_argument("--"+key,type=Path,required=key!="native-receipt")
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize, "isolated assertions/no bytecode required")
    for key in vars(a):
        value=getattr(a,key)
        if isinstance(value,Path):
            need(not value.is_symlink(), "symlink path denied")
            setattr(a,key,value.absolute().resolve())
    if a.phase != "readback":
        physical_budget(a)
    else:
        need(avail_memory() >= 5_000_000_000,"MemAvailable <5GB before independent readback")
    sys.path.insert(0,str(a.repo/"tests"))
    proof = Proof(a)
    if a.phase=="prepare":
        with proof.package(a.package) as m:
            inp = m["admission"].Inputs()
            verified = load(a.repo/"tests/current_runtime_generator_fixtures.py",
                  inp.bind(a.repo/"tests/current_runtime_generator_fixtures.py",FIXTURES),"canonical_fixture_verify")
            verified.verify_qualifier(a.package,inp)
            m["run"].package(inp)
            args = types.SimpleNamespace(repository=a.repo,candidate=a.candidate,build_directory=a.build,tool_build=a.tool,
                 stack_build=a.stack,original_package_root=a.original,expected_native_sha256=ELF,
                 expected_build_sha256=proof.build_sha,expected_tool_build_sha256=proof.tool_sha)
            out = m["admission"].admit(args,inp)
            need(out["status"]=="ADMITTED_NOT_RUN" and all(inp.unchanged().values()), "pure joint preparation failed")
        dump(a.evidence,dict(proof.identities(),status="JOINT_GENERATOR_PROFILE_ADMITTED_NOT_RUN", admission=out,
                            canonical_denials=proof.bindings,native_executed=False))
        return 0
    if a.phase=="execute":
        return proof.run()
    return proof.readback()


if __name__=="__main__":
    raise SystemExit(main())
