"""Check the standalone bounded scope crosswalk; never admit canonical contracts."""
import argparse
import json
from pathlib import Path
import re

ROOT = "https://github.com/TrillionniumFoundation/HeptaTorch"
ROUTES = {"where": 129, "masked_select": 130, "logical_and": 131,
          "logical_or": 133, "logical_xor": 134, "nonzero": 135}
EVIDENCE = {
    "66": {"where"}, "67": {"masked_select"},
    "68": {"logical_and", "logical_or", "logical_xor"}, "69": {"nonzero"},
    "70": set(ROUTES) - {"where"}, "71": set(ROUTES),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def unique(items):
        out = {}
        for key, val in items:
            require(key not in out, f"duplicate key {key}")
            out[key] = val
        return out
    with path.open("rb") as source:
        blob = source.read(2_000_001)
    require(len(blob) <= 2_000_000, "oversize JSON")
    return json.loads(blob, object_pairs_hook=unique,
                      parse_constant=lambda s: require(False, f"invalid number {s}"))


def validate(data):
    require(data["schema"] == "heptatorch.opt-in-ordinary-coverage.v1", "wrong schema")
    acct = data["accounting"]
    require((acct["native_total"], acct["derivative_total"], acct["subsystems_total"]) == (2590, 688, 17),
            "wrong frozen counts")
    require(acct["overall"] == "HOLD" and
            all(type(acct[k]) is int and acct[k] == 0
                for k in ("qualified_native", "qualified_derivative", "closed_subsystems")),
            "crosswalk is not complete-contract credit")
    require(set(data["operators"]) == set(ROUTES) and set(data["evidence"]) == set(EVIDENCE),
            "unexpected route or evidence")
    c = data["canonical"]
    outer = c["outer_evidence"]
    require(c["canonical_execution_evidence"] is None and
            outer["independent_full_inner_admission"] is False, "outer provenance grants no admission")
    q, u = outer["qualification"], outer["custody"]
    require(q["status"] == "PASS_CURRENT_RUNTIME_FOCUSED" and q["failures"] == 0
            and u["status"] == "PASS" and u["case_admissions_executed_here"] == 0,
            "wrong reported outer status")
    require((u["reported_required_groups"], u["reported_required_cases"]) == (29, 13851)
            and u["includes_generator131"] is True, "wrong original finite roster")
    require(outer["artifact_id"] == 14713706649
            and (q["bytes"], q["sha256"]) ==
            (76650, "a92cf94efaa4745168d4ab82239b1af60b4819942e6f89c82a794777004a9902")
            and (u["bytes"], u["sha256"]) ==
            (103458, "0ea8a3a22f7d4c2541590f18f98b3631d51603c70aa79ef8330714bc676172a9"),
            "wrong outer receipt identities")
    inner = c["inner_receipts"]
    require([(x["path"], x["bytes"], x["sha256"]) for x in inner] == [
        ("current-runtime/build/build.json", 22757, "c1b4cb1c483705d935b453f3b9ab1f82b20b59ea658048bc5750cc33bf9b0d33"),
        ("current-runtime/build/manifest.json", 4416, "348ed8e949256f985cdca0d9c612dc646ae738beee6013c27ab405d2df3e67c8"),
        ("current-runtime/review/generator131/report.json", 12970068, "5ce818551600d18b45ab1af1e1c8b625844c251020238603bc9d3e24cf6e0c36"),
    ], "wrong inner receipt identities")
    for op, spec in data["operators"].items():
        require(spec["route"] == ROUTES[op] and spec["full_declaration_qualified"] is False, "wrong route/credit")
        require(set(spec["evidence"]) == {int(pr) for pr, ops in EVIDENCE.items() if op in ops},
                "wrong evidence-to-route relation")
        require(spec["frozen_native"] not in spec["related_native_not_covered"], "same declaration listed twice")
        require(spec["frozen_derivative"] in (None, spec["frozen_native"]), "unexpected derivative declaration")
    for pr, record in data["evidence"].items():
        require(set(record["ops"]) == EVIDENCE[pr], f"wrong # {pr} scope")
        require(re.fullmatch(r"[0-9a-f]{40}", record["head"]), f"bad # {pr} head")
        for key in ("source_sha256", "elf_sha256"):
            require(re.fullmatch(r"[0-9a-f]{64}", record[key]), f"bad # {pr} {key}")
        require(re.fullmatch(r"tools/current_runtime/opt_in_[a-z0-9_]+", record["root"]),
                "bad pinned evidence path")
        require(all(re.fullmatch(r"[a-z-]+\.json", record[k]) for k in ("manifest", "receipt")),
                "bad receipt filename")
        require(all(type(x) is int and x > 0 for x in record["last_positive"].values()), "bad positive counts")
    require(data["older_combined_nonzero_pin"] == "2b5b2fcd8b7937d5441317da1becf9cbef360a1b"
            and data["evidence"]["69"]["head"] != data["older_combined_nonzero_pin"],
            "must distinguish latest and frozen older #69")
    r = data["evidence"]["71"]
    apps = r["separate_cpu_app_evidence"]
    require(r["last_positive"] == {"chains":50, "combined_outputs":171,
                                  "isolated_outputs":171, "values":1362}, "keep coexistence distinct")
    require(re.fullmatch(r"[0-9a-f]{40}", r["tree"]) and
            len(apps) == 4 and [x["id"] for x in apps] ==
            ["two_layer_inference", "one_scalar_loss_sgd_step", "fixed_manual_demo", "two_layer_sgd_step"], "wrong app scopes")
    infer, sgd, demo, train = apps
    require(tuple(infer[k] for k in ("cases","stages","raw_f32_words")) == (3,15,80),
            "wrong two-layer scope")
    require(tuple(sgd[k] for k in ("cases","raw_f32_words","backward_calls","sgd_calls",
                                   "loss_before","loss_after","version_delta")) == (1,26,1,1,"8","2.25",1)
            and sgd["commit"] == "72be4108b176fa92873a726c50d2ddf5aea6ba46", "wrong scalar-loss scope")
    require(demo["includes_two_layer_sgd"] is False, "manual demo scope expands")
    require(tuple(train[k] for k in ("cases", "raw_f32_words", "frames", "loss_before", "loss_after"))
            == (1,54,265,"8","12625/4096") and train["commit"] == r["head"], "wrong two-layer SGD scope")
    for app in apps:
        require(re.fullmatch(r"[0-9a-f]{40}", app["commit"]) and
                re.fullmatch(r"tools/current_runtime/opt_in_[a-z0-9_]+", app["root"]) and
                re.fullmatch(r"(?:[a-z0-9-]+\.json|README\.md)", app["receipt"]), "invalid app pin")
    return data


def validate_inventories(data, root):
    a = data["accounting"]
    n = read(root / a["native_inventory"])
    der = read(root / a["derivative_inventory"])
    ledger = read(root / a["catalog"])
    require(ledger["upstream_commit"] == a["upstream_commit"] and ledger["status"] == "HOLD", "wrong ledger")
    require(len(n["schemas"]) == n["count"] == a["native_total"] and
            len(der["derivatives"]) == der["count"] == a["derivative_total"] and
            len(ledger["gates"]) == a["subsystems_total"], "inventories not at crosswalk scope")
    require(all(x["status"] != "qualified" for x in n["schemas"] + der["derivatives"]) and
            all(x["status"] != "CLOSED" for x in ledger["gates"]), "canonical accounting changed")
    native = {x["schema"]: x["status"] for x in n["schemas"]}
    derivatives = {x["name"]: x["status"] for x in der["derivatives"]}
    for s in data["operators"].values():
        relevant = [s["frozen_native"], *s["related_native_not_covered"]]
        require(all(native.get(k) == "unimplemented" for k in relevant), "native declaration drift")
        dec = s["frozen_derivative"]
        require(dec is None or derivatives.get(dec) == "unimplemented", "derivative declaration drift")


def render(d):
    def title(x):
        return x.replace("_", " ")
    c, a = d["canonical"], d["accounting"]
    main = f"{ROOT}/tree/{c['commit']}"
    out = ["# Bounded opt-in CPU coverage", "",
           "This crosswalk tracks published evidence for six functional operator routes, without "
           "qualifying a full PyTorch declaration or changing which runtime is default.", "",
           f"Canonical [{c['commit'][:7]}]({main}) is typed-ReLU775. Its "
           f"[manifest]({main}/{c['manifest']}) is separate from all rows below. The tracked finite "
           f"ledger requires {c['required_finite_groups']} groups / {c['required_finite_cases']:,} cases. "
           f"The [exact hosted run]({ROOT}/actions/runs/{c['metadata_only_hosted_run']}) reports success, "
           "but this crosswalk did not independently admit the inner original reports.", "",
           "Published-format outer receipts from artifact "
           f"`{c['outer_evidence']['artifact_id']}`: "
           f"qualification `{c['outer_evidence']['qualification']['sha256']}`, "
           f"byte/mode custody `{c['outer_evidence']['custody']['sha256']}`. "
           "They report 29 groups / 13,851 cases including Generator131; the custody reader "
           "records zero independent case admissions. Canonical execution evidence remains null in this crosswalk.", "",
           "Bounded inner receipt readback (no new case admissions): " +
           "; ".join(f"`{x['path']}` `{x['sha256']}` ({x['scope']})" for x in c["inner_receipts"]) + ".", "",
           f"[Full acceptance]({main}/{a['catalog']}) stays **{a['overall']}**: "
           f"0/{a['native_total']:,} complete native declarations, "
           f"0/{a['derivative_total']} derivative declarations and "
           f"0/{a['subsystems_total']} complete subsystem contracts.", "",
           "| Route | Frozen declaration | Evidence PRs | Out-of-scope overloads or derivative |",
           "| --- | --- | --- | --- |"]
    for op, s in sorted(d["operators"].items(), key=lambda it: it[1]["route"]):
        linked = ", ".join(f"[#{x}]({ROOT}/pull/{x})" for x in s["evidence"])
        gaps = ", ".join(f"`{n.split('(')[0]}`" for n in s["related_native_not_covered"])
        if s["frozen_derivative"]:
            gaps += "; frozen derivative unqualified"
        out.append(f"| {s['route']} | `{s['frozen_native']}` | {linked} | {gaps} |")
    out += ["", "## Exact evidence slices", "",
            "Counts describe the named latest slice, not additive totals or the entire operator contract.", ""]
    for pr, s in sorted(d["evidence"].items(), key=lambda it: int(it[0])):
        b = f"{ROOT}/blob/{s['head']}/{s['root']}"
        nums = ", ".join(f"{x:,} {title(k)}" for k, x in sorted(s["last_positive"].items()))
        out += [f"### #{pr} ({s['head'][:7]}): {s['profile']}", "",
                s["detail"], "",
                f"**Measured:** {nums}; {s['last_positive_scope']}. "
                f"[Manifest]({b}/{s['manifest']}) · [Receipt]({b}/{s['receipt']}).", "",
                f"**Control boundary:** {s['controls']}", ""]
    apps = d["evidence"]["71"]["separate_cpu_app_evidence"]
    out += ["## Separate #71 CPU application evidence", "",
            "These are separate exact-image experiments, not additional operator-coexistence counts.", ""]
    a, b, demo, train = apps
    href = lambda x: f"{ROOT}/blob/{x['commit']}/{x['root']}/{x['receipt']}"
    out += [f"- **Two-layer F32 inference:** {a['cases']} cases, {a['stages']} stages, "
            f"{a['raw_f32_words']} exact words. {a['scope']} [Receipt]({href(a)}).",
            f"- **One owned scalar-loss SGD step:** {b['raw_f32_words']} exact words, "
            f"loss {b['loss_before']} → {b['loss_after']}, weight Version +{b['version_delta']}. "
            f"{b['scope']} [Receipt]({href(b)}).",
            f"- **Manual demo:** {demo['scope']} [Instructions]({href(demo)}) · [Live receipt]({ROOT}/blob/{demo['commit']}/{demo['root']}/{demo['live_receipt']}).",
            f"- **One two-layer F32 SGD fixture:** {train['raw_f32_words']} exact words, {train['frames']} frames, "
            f"loss {train['loss_before']} → {train['loss_after']}. {train['scope']} [Receipt]({href(train)}).", "",
            "No application evidence here qualifies any full forward or gradient contract, default path, "
            "speed advantage or canonical acceptance.", "",
            "## Pins and exclusions", "",
            f"#70 and #71 pin earlier #69 `{d['older_combined_nonzero_pin'][:7]}`; "
            "newer #69 rank-two numeric, rank-three, and rank-four evidence does not roll forward into them.",
            "Held mixed logical and held Cat work are not credited. No crosswalk entry affects default "
            "routing, existing hosted CI, independent registered verifiers, or the complete-project gate.", ""]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--canonical-root", type=Path, help="optional locally supplied frozen main checkout")
    p.add_argument("--check-document", type=Path, help="require exact rendered Markdown")
    args = p.parse_args()
    d = validate(read(args.manifest))
    if args.canonical_root:
        validate_inventories(d, args.canonical_root)
    doc = render(d)
    if args.check_document:
        require(args.check_document.read_text(encoding="utf-8") == doc, "Markdown drift")
        print("PASS_OPT_IN_CROSSWALK_ONLY")
    else:
        print(doc, end="")


if __name__ == "__main__":
    main()
