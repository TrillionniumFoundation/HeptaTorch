#!/usr/bin/env python3
"""Create an explicit new fixed35 Saved-T profile without changing its parents."""
import argparse,ast,hashlib,json,re,shutil
from pathlib import Path

SOURCE='5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49'
IMAGE='69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6'
PROFILE='mixed-integer-556-fixed35-F32-Saved'
GENERAL='aa8a9c2faf0fccabe0acb42d3f21e316c10c5c8b8626baa323bd8f186bead7ea'
PARENT='ca011c95e1d19c40ecd365d0ba41b05cc818e5ba3d9adb8defd33c80be40e09a'

def sha(b):return hashlib.sha256(b).hexdigest()

OVERLAY=r'''_tree = ast.parse(_legacy_raw, filename=str(LEGACY))
_selected = ("sha", "canonical", "need", "word", "identity", "semantic_snapshot",
             "verify_snapshot", "verify_transition", "Journal", "install")
_nodes = {node.name: node for node in _tree.body
          if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _selected}
assert set(_nodes) == set(_selected)
SOURCE_PINS = next(ast.literal_eval(node.value) for node in _tree.body
    if isinstance(node, ast.Assign) and len(node.targets) == 1
    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "SOURCE_PINS")
_proof = {"legacy_source_sha256": LEGACY_SHA, "structural_source_sha256": STRUCTURAL_SHA,
          "full_640_snapshot_no_projection": True, "functions": {}}
for _name in _selected:
    _original = _nodes[_name]
    _text = ast.get_source_segment(_legacy_raw.decode(), _original)
    _edits = []
    def _edit(label, before, after, count=1):
        global _text
        assert _text.count(before) == count, (label, _text.count(before), count)
        _text = _text.replace(before, after)
        _edits.append({"label":label,"before":before,"after":after,"count":count})
    if _name == "verify_snapshot":
        _edit("width", "len(node) == 72", "len(node) == 80")
        _edit("actual-original-T-pair-counter", "    gradients = set()\n",
              "    gradients = set()\n    saved_T_refs = _saved_tensor_references(snapshot)\n")
        _edit("dead-node-no-Saved-root", "(200, 208, 240, 88, 312, 384, 392, 400)",
              "(200, 208, 240, 88, 312, 384, 392, 400, 600)")
        _edit("rowless-no-owned-T-root", "(88, 312, 384, 392, 400)", "(88, 312, 384, 392, 400, 600)")
        _edit("reference-sum-from-pairs", "graph_refs[index] + word(node, 312) + grad_refs[index]",
              "graph_refs[index] + word(node, 312) + grad_refs[index] + saved_T_refs[index]", 2)
        _edit("exact600-from-pairs", "        need(word(node, 88) == leaf_refs[index],",
              "        need(word(node, 600) == saved_T_refs[index], \"fixed35-original-Saved-T-reference-count\", node=index)\n"
              "        need(word(node, 88) == leaf_refs[index],")
        _edit("reachable-original-T-roots",
              '        pending.extend(parent for parent, _ in snapshot["edges"][str(index)] if parent)\n',
              '        pending.extend(parent for parent, _ in snapshot["edges"][str(index)] if parent)\n'
              '        pending.extend(_saved_tensor_targets(snapshot, index))\n')
    if _name == "verify_transition":
        _edit("reset-original-T-root-closure",
              '                pending.extend(p for p, _ in before["edges"][str(index)] if p)\n',
              '                pending.extend(p for p, _ in before["edges"][str(index)] if p)\n'
              '                pending.extend(_saved_tensor_targets(before, index))\n')
    _reverse = _text
    for _change in reversed(_edits):
        assert _reverse.count(_change["after"]) == _change["count"]
        _reverse = _reverse.replace(_change["after"], _change["before"])
    _dump = lambda n: ast.dump(n, include_attributes=False)
    assert _dump(ast.parse(_reverse).body[0]) == _dump(_original)
    _node = ast.parse(_text).body[0]
    if _name == "verify_snapshot": _node.name = "_fixed35_snapshot"
    if _name == "verify_transition": _node.name = "_fixed35_transition"
    _proof["functions"][_name] = {
        "original_ast_sha256": _sha(_dump(_original).encode()),
        "executed_ast_sha256": _sha(_dump(_node).encode()),
        "width_predicates_changed": int(_name == "verify_snapshot"),
        "name_binding_changed": _node.name != _original.name,
        "inverse_entire_function_AST_equals_original": True,
        "explicit_role_edits": _edits}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[_node], type_ignores=[])),
                 str(LEGACY) + "::<explicit-640-Saved-binding>", "exec"), globals())

def _saved_tensor_targets(snapshot, index):
    values = snapshot["binary_saved_roots"].get(str(index), [0,0,0,0])
    need(len(values) == 4 and all(type(v) is int and 0 <= v < 1 << 64 for v in values),
         "fixed35-original-Saved-pair-shape")
    return [values[slot] for slot in (0,2) if values[slot]]

def _saved_tensor_references(snapshot):
    references = Counter()
    for index in snapshot["binary_saved_roots"]:
        references.update(_saved_tensor_targets(snapshot,index))
    return references

'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('parent-directory','general-owner','source','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    root=out/'tests/mixed_frozen35_accounting';root.mkdir(parents=True)
    general=a.general_owner.read_bytes();assert sha(general)==GENERAL
    structural=general.replace(b'integer-tensor-556-general-owner10',b'integer-556-fixed35-F32-Saved')
    assert structural!=general and structural.replace(b'integer-556-fixed35-F32-Saved',b'integer-tensor-556-general-owner10')==general
    # Keep the full general source unchanged beside this independently named profile.
    parentgeneral=out/'tests/mixed_integer_owner_review/owner640_mixed10.py';parentgeneral.parent.mkdir(parents=True)
    parentgeneral.write_bytes(general);(root/'owner640_saved_fixed35.py').write_bytes(structural)
    (root/'legacy_owner_rules.py').write_bytes((a.parent_directory/'legacy_owner_rules.py').read_bytes())
    for n in ('storage-tombstone-correction.json',):
        shutil.copyfile(a.parent_directory/n,root/n)
    adaptation={'parent_general_source_sha256':GENERAL,'new_profile_source_sha256':sha(structural),
        'only_change':'PROFILE name literal; all source/native pins and all checker function bodies unchanged',
        'whole_inverse_exact':True}
    (root/'structural-profile-adaptation.json').write_text(json.dumps(adaptation,indent=2)+'\n')
    changes={}
    def derive(name,fn):
        original=(a.parent_directory/name).read_bytes();text=original.decode();edits=[]
        def replace(label,before,after,count=1):
            nonlocal text
            assert text.count(before)==count,(name,label,text.count(before),count)
            text=text.replace(before,after);edits.append({'label':label,'before':before,'after':after,'count':count})
        fn(replace,text)
        inverse=text
        for e in reversed(edits):
            assert inverse.count(e['after'])==e['count'],(name,e['label'])
            inverse=inverse.replace(e['after'],e['before'])
        assert inverse.encode()==original
        ast.parse(text)
        target=name.replace('owner_accounting_int64.py','owner_accounting_mixed.py')
        (root/target).write_text(text)
        changes[target]={'original_sha256':sha(original),'current_sha256':sha(text.encode()),'exact_inverse':True,'edits':edits}
        return text

    def owner(replace,text):
        assert sha(text.encode())==PARENT
        begin=text.index('_tree = ast.parse');end=text.index('# These are source-derived')
        replace('exact-legacy-roles-overlay',text[begin:end],OVERLAY)
        replace('general-dependency','HERE.parent / "dtype_owner_accounting" / "owner640.py"',
                'HERE.parent / "mixed_integer_owner_review" / "owner640_mixed10.py"')
        replace('fixed-structural','HERE / "owner640_fixed35.py"','HERE / "owner640_saved_fixed35.py"')
        replace('general-source-pin','6ea9e8b33164089034a2a71a03f4850e1ec18ed1a09155d7841f700ef311c833',GENERAL)
        replace('structural-source-pin','d40fb59a74edfaf87113e7eac1e13587c7e8d33aa6ca1edea36a7e9da7bbde89',sha(structural))
        replace('all-core-count','len(bodies) == 544','len(bodies) == 556')
        replace('limited-role-domain','''        need(not any(word(node, off) for off in range(576, 640, 8)),
             "fixed35-unrequested-dtype-history-role", node=int(key))''','''        need(word(node,576) == 0 and not any(word(node,off) for off in (608,616,624,632)),
             "fixed35-unrequested-dtype-history-role", node=int(key))
        role = word(node,584)
        need(role == 0 or role > 65535, "fixed35-unrequested-mask-or-I64-view-role", node=int(key))
        if role == 0:
            need(word(node,592) == 0, "fixed35-unowned-Saved-relative-address", node=int(key))''')
        replace('doc-old-global-width-only','The only changed legacy predicate is descriptor width 72 -> 80.',
                'Legacy deltas are explicit width and original-Saved-T reference/closure equations; all old numeric, transition and refusal checks remain.')
        replace('doc-roles','NativeDropout99,\ndetach_101, retained hooks and owned Optimizers are outside that domain',
                'NativeDropout99,\ndetach_101, retained hooks and owned Optimizers are outside that domain; ordinary F32 mul/div really hold original Saved T pairs')
    accounting=derive('owner_accounting_int64.py',owner)
    runtime=a.source.read_bytes();assert sha(runtime)==SOURCE
    bodies={m[1].decode():m[0] for m in re.finditer(rb'^fn (\w+)\([^\n]* entry\([^\n]*\)\n.*?^end\n?',runtime,re.M|re.S)}
    assert len(bodies)==556
    oldconfig=json.loads((a.parent_directory/'profile.json').read_bytes())
    targets=list(oldconfig['capture_callers'])+['binary_saved_bind','binary_saved_retain_one','binary_saved_release',
        'binary_saved_get','binary_saved_command','tensor_row_release','gradient_detach_inplace']
    config={'schema':'heptatorch.frozen35.mixed556-F32-Saved-domain-profile.v1','source_sha256':SOURCE,'native_sha256':IMAGE,
        'descriptor_bytes':640,'structural_observer_sha256':sha(structural),'legacy_owner_rules_sha256':oldconfig['legacy_owner_rules_sha256'],
        'function_pins':{n:sha(b) for n,b in bodies.items()},
        'capture_callers':{n:sorted(k for k,b in bodies.items() if ('"function":"'+n+'"').encode() in b) for n in targets},
        'structural_claim':'Full actual80word nodes, complete newSaved original-T Hold32B/600/208 accounting. The unchanged legacy transition ledger gains only explicit root-closure for this real role. No fabricated snapshot, no residual allocation count.',
        'fixed35_domain':'Only original35 commands1..56 and dtypeF32. Ordinary mul/div may own private32B original-T Saved Hold584/592 and incoming600 references. All99/101 captured-history/mask roles and I64 view roots are absent. GeneralmixedI64 independent4/17 is a separate qualification.',
        'original_structural_observer_sha256':GENERAL,'parent_profile_sha256':sha((a.parent_directory/'profile.json').read_bytes()),
        'parent_source_sha256':oldconfig['source_sha256'],'parent_native_sha256':oldconfig['native_sha256'],
        'parent_observer_sha256':PARENT,'program_descriptor_counts_unchanged':True,
        'new_private_Saved_Hold_allocates_no_Tensor_descriptor':True}
    (root/'profile.json').write_text(json.dumps(config,indent=2)+'\n')
    def launcher(replace,text):
        replace('new-import','import owner_accounting_int64 as observer','import owner_accounting_mixed as observer')
        replace('contract-text','New99/101 roles explicitly absent; exact stats/reset high-water, no inequality fallback.',
                'Ordinary F32mul/div Saved-original-T Hold/600/208 equations are explicit; 99/101 roles absent. Exactstats/reset high-water, no inequality fallback.')
        replace('scope-text','separately pinned523/640 owner','separately pinned556/640 owner')
        replace('scope-Saved','General99/101 ownership roles separately qualified; they are explicitly absent in this fixed35 command domain.',
                'New ordinaryF32mul/div original-T Saved roles are independently checked on every full snapshot. General99/101 ownership roles remain outside this domain.')
    launch=derive('run_frozen35_accounted.py',launcher)
    def controls(replace,text):
        replace('observer-pin',PARENT,sha(accounting.encode()))
        replace('profile','integer-tensor-544-fixed35-F32',PROFILE)
    derive('check_owner_snapshots.py',controls)
    def tombstone(replace,text):
        replace('new-import','import owner_accounting_int64 as observer','import owner_accounting_mixed as observer')
        replace('profile','integer-tensor-544-fixed35-F32',PROFILE)
    # This file may bind profile through native pin instead of a name; exact
    # adaptation is made only for actual literals present in the parent source.
    tomb=(a.parent_directory/'test_storage_tombstone.py').read_text()
    def tombstone_actual(replace,text):
        if 'import owner_accounting_int64 as observer' in text:
            replace('new-import','import owner_accounting_int64 as observer','import owner_accounting_mixed as observer')
        for label,before,after in [('profile','integer-tensor-544-fixed35-F32',PROFILE),('source',oldconfig['source_sha256'],SOURCE),('image',oldconfig['native_sha256'],IMAGE),('observer',PARENT,sha(accounting.encode()))]:
            if before in text:replace(label,before,after,text.count(before))
    derive('test_storage_tombstone.py',tombstone_actual)
    def admission(replace,text):
        for label,before,after in [('source',oldconfig['source_sha256'],SOURCE),('image',oldconfig['native_sha256'],IMAGE),('profile','integer-tensor-544-fixed35-F32',PROFILE),('observer',PARENT,sha(accounting.encode())),('launcher','b95543285c1d6fb66aa95890bb517d7317941455048bfec04f802f788f4ef2ad',sha(launch.encode())),('structural',oldconfig['structural_observer_sha256'],sha(structural)),('general',oldconfig['original_structural_observer_sha256'],GENERAL),('profileJSON',sha((a.parent_directory/'profile.json').read_bytes()),sha((root/'profile.json').read_bytes()))]:
            replace(label,before,after,text.count(before))
        anchor=' assert inverse["functions"]["verify_snapshot"]["width_predicates_changed"]==1\n'
        insert=anchor+''' assert [e["label"] for e in inverse["functions"]["verify_snapshot"]["explicit_role_edits"]]==["width","actual-original-T-pair-counter","dead-node-no-Saved-root","rowless-no-owned-T-root","reference-sum-from-pairs","exact600-from-pairs","reachable-original-T-roots"]
 assert [e["label"] for e in inverse["functions"]["verify_transition"]["explicit_role_edits"]]==["reset-original-T-root-closure"]
 assert all(not e["explicit_role_edits"] for n,e in inverse["functions"].items() if n not in ("verify_snapshot","verify_transition"))
'''
        replace('strict-new-source-equations',anchor,insert)
    derive('check_frozen35_accounted.py',admission)
    receipt={'status':'PREPARED_NOT_RUN','parent_accounting_sha256':PARENT,'new_accounting_sha256':sha(accounting.encode()),
        'source_sha256':SOURCE,'native_sha256':IMAGE,'profile':PROFILE,'changes':changes,
        'producer_or_original_oracles_changed':False,'new_native_or_Torch_executions':0}
    (out/'source-adaptation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'executed-builder.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in receipt.items() if k!='changes'}))

if __name__=='__main__':main()
