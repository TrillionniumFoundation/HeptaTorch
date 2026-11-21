"""Exact complete native771 to frozen native767 source inverse."""
import hashlib
import json
from pathlib import Path
import re

SOURCE='f534c77069abc15ba15879d44492aacbc5f3709f6a861a0f0b9e0e365232fca1'
PARENT_SOURCE='e0c8e0a3c3ee16d11cad8af6be1b52bd56295581e74bdcc65131f71a57540025'
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def definitions(raw):
    rows=list(re.finditer(rb'^fn ([a-zA-Z0-9_]+)\([^\n]*\n.*?^end\n',raw,re.M|re.S))
    result={m.group(1).decode():m.group(0) for m in rows}
    require(len(rows)==len(result),'unique complete HIA function bodies')
    return result


def restore_source(source,bind):
    bind(Path(__file__).resolve())
    require(sha(source)==SOURCE,'exact newly built771 source')
    raw=bind(Path(__file__).with_name('half_div_source_delta.json'))
    delta=json.loads(raw)
    require(delta['source_sha256']==SOURCE and delta['parent_source_sha256']==PARENT_SOURCE
        and delta['functions']==771 and delta['parent_functions']==767
        and delta['changed_functions']==['binary'] and delta['unchanged_parent_functions']==766
        and delta['appended_functions']==['half_div_f32','half_div_f64','half_div_element','half_div_tensor'],
        'exact four-function division extension plan')
    current=definitions(source);require(len(current)==771,'all771 HIA definitions')
    suffix=b''.join(current[name] for name in delta['appended_functions'])
    require(len(suffix)==delta['appended_bytes'] and source.endswith(suffix)
        and all(sha(current[name])==delta['appended_bodies'][name] for name in delta['appended_functions']),
        'all and only the appended division bodies')
    previous_binary=delta['original_binary'].encode();binary=current['binary']
    require(sha(binary)==delta['new_binary_sha256'] and sha(previous_binary)==delta['original_binary_sha256']
        and binary.count(delta['new_route'].encode())==1
        and binary.replace(delta['new_route'].encode(),delta['original_route'].encode(),1)==previous_binary,
        'one operation4 route change, every preexisting route byte restored')
    previous=source[:-len(suffix)].replace(binary,previous_binary,1)
    require(sha(previous)==PARENT_SOURCE and len(definitions(previous))==767,'every original767 byte restored')
    require(previous.replace(previous_binary,binary,1)+suffix==source,'complete767-to771 forward round trip')
    return previous,dict(schema='heptatorch.half-div.source-inverse.v1',source_sha256=SOURCE,
        parent_source_sha256=PARENT_SOURCE,functions=771,parent_functions=767,
        changed_functions=['binary'],unchanged_parent_functions=766,appended_functions=delta['appended_functions'],
        actual_binary_sha256=sha(binary),parent_binary_sha256=sha(previous_binary),
        new_route_sha256=sha(delta['new_route'].encode()),
        original_route_sha256=sha(delta['original_route'].encode()),
        non_half_routes_unchanged=True,non_division_routes_unchanged=True,
        only_new_half_binary_operation=4,
        whole_source_inverse_exact=True,whole_round_trip_exact=True,prior_native769_evidence_used=False,
        native_executed=False,Torch_executed=False,host_expected_values_computed=False)
