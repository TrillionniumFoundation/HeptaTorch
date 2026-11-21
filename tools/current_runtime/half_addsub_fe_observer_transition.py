"""Exact1006→1004 observer inverse for two original same-frame FE additions.

The immutable1004 request recipes and all their numerical predicates remain.
This successor appends the two missed original factory97/FE-add requests and
requires their genuine reference results, complete ownership and rollback.
No native source changes and no prior execution qualification is transferred.
"""
import hashlib
import importlib.util
import json
from pathlib import Path


DELTA='275adaec09dadc1d436eb7cc851239844790c2566452a38a6907eea15fd116b3'
PATHS=('tests/half_addsub/cases.py','tests/half_addsub/reference.py','tests/half_addsub/run.py')
SOURCE='b4b2996978dce56c9375ce97b2d2638472cbb154a4c0a7c5de822234e31a9c68'
MANIFEST='100f60ce946ac809949d79faf4eca8e8e16d5f4aeacfd907d228edc1ea0c1fd2'
OLD_CASES='553e1a15e63bc66a60faf1276e827410165c9ed8ff0b05d6fb013421930cc6d8'
CASES='07fee181f5b4ea4eff68ccb98c15fbfaf1b81e4b39f0573b21489681443aeea8'
LABELS=('same-frame-half-FE-binary-5','same-frame-half-FE-binary-6')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def restore_file(raw,record):
    require(len(raw)==record['source_bytes'] and sha(raw)==record['source_sha256'],
        'complete current observer identity')
    parent_parts=[];source_at=parent_at=0
    for edit in record['edits']:
        require(set(edit)=={'parent_start','parent_end','before','after'},'exact observer edit fields')
        start,end=edit['parent_start'],edit['parent_end'];before,after=edit['before'].encode(),edit['after'].encode()
        require(parent_at<=start<=end<=record['parent_bytes'] and end-start==len(before)
            and before!=after,'ordered complete parent byte intervals')
        unchanged=start-parent_at
        parent_parts.append(raw[source_at:source_at+unchanged]);source_at+=unchanged
        require(raw[source_at:source_at+len(after)]==after,'exact complete successor edit bytes')
        parent_parts.append(before);source_at+=len(after);parent_at=end
    parent_parts.append(raw[source_at:]);parent=b''.join(parent_parts)
    require(len(parent)==record['parent_bytes'] and sha(parent)==record['parent_sha256'],
        'every original1004 observer byte reconstructed')
    current_parts=[];at=0
    for edit in record['edits']:
        start,end=edit['parent_start'],edit['parent_end']
        require(parent[start:end]==edit['before'].encode(),'exact forward parent edit')
        current_parts.extend((parent[at:start],edit['after'].encode()));at=end
    current_parts.append(parent[at:])
    require(b''.join(current_parts)==raw,'complete forward observer round trip')
    return parent


def restore_observers(bind):
    root=Path(__file__).resolve().parents[2];bind(Path(__file__).resolve())
    delta_raw=bind(Path(__file__).with_name('half_addsub_fe_observer_delta.json'))
    require(sha(delta_raw)==DELTA,'immutable complete FE observer delta')
    delta=json.loads(delta_raw)
    require(delta['schema']=='heptatorch.half-addsub-fe1006.observer-delta.v1'
        and delta['parent_commit']=='3fe56570dca9b755f0cceeb6fa5df78ffa531581'
        and set(delta['files'])==set(PATHS),'exact three successor observer files')
    source = bind(root/'tools/current_runtime/source/runtime.hepta')
    wide_path = Path(__file__).with_name('wide_neg_source_transition.py')
    bind(wide_path)
    spec = importlib.util.spec_from_file_location('fe_wide_neg_source_inverse', wide_path)
    wide = importlib.util.module_from_spec(spec); spec.loader.exec_module(wide)
    parent, wide_proof = wide.restore_source(source, bind)
    manifest = bind(root/'tools/current_runtime/source/manifest.json')
    require(sha(parent)==SOURCE and wide.PARENT_SOURCE==SOURCE
        and sha(manifest)=='32d826cfaaaaca910c7e2b209d977dd705a314f64267cfd0808f01f08d3865d1',
        'exact767 native source inverts to the complete765 FE predecessor')
    parents={path:restore_file(bind(root/path),delta['files'][path]) for path in PATHS}
    return parents,dict(schema='heptatorch.half-addsub-fe1006.observer-inverse.v1',
        delta_sha256=DELTA,files={path:{key:value for key,value in delta['files'][path].items() if key!='edits'} for path in PATHS},
        parent_cases=1004,cases=1006,parent_case_sha256=OLD_CASES,case_sha256=CASES,
        original_fe_successor_labels=list(LABELS),all1004_original_recipes_retained=True,
        all1004_original_numerical_error_resource_and_lifetime_predicates_retained=True,
        exact_original_two_command_FE_packets_checked=True,
        real_reference_bits_complete_owner_and_rollback_required=True,
        source_sha256=wide.SOURCE, parent_source_sha256=SOURCE, native_source_unchanged=False,
        native_wide_neg_inverse=wide_proof, original_FE_native_bodies_unchanged=True,
        manifest_sha256=sha(manifest), parent_manifest_sha256=MANIFEST,
        whole_source_inverse_exact=True,whole_round_trip_exact=True,runtime_qualification_transferred=False)
