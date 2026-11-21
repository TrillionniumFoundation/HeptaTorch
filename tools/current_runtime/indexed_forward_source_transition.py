"""Exact inverse of the two no-grad indexed forward call-site corrections."""
import hashlib
from pathlib import Path
import re
SOURCE='ee214f05b56401ea17a5f55e792aaccf73447974ba0b7e0416990439bff6a8ff'
PARENT='db88eaad9fb69010ae4f6925f4e8c54edce815b9f46507bc0c7c2536a4cb2c0b'
PINS={'indexed_gather_forward_source': 'ad58e35a69de71e4d68727ef12fd30e245aa2f4455058421171c4e8b22676ce4', 'indexed_add_forward_inputs': '53d8e8f8f2c46da36acd9f408491a681a665cf356b1622be4a7960d57b362c79', 'indexed_add_forward_location': '8b30b70dacca94cbfab4c085c281783c692371e76768dd1482a933933a81d5cb'}

def restore_source(raw,bind):
    bind(Path(__file__).resolve())
    sha=lambda value:hashlib.sha256(value).hexdigest()
    def require(condition,message):
        if not condition:raise ValueError(message)
    header=b'hia-asm.v1 hepta.hia.program.v8 heptatorch_runtime\n'
    def parse(value):
        matches=list(re.finditer(rb'^fn (\w+)\([^\n]*\n.*?^end\n',value,re.M|re.S))
        names=[match.group(1).decode() for match in matches]
        bodies={name:match.group() for name,match in zip(names,matches)}
        require(len(names)==len(bodies),'unique original function names')
        require(value==header+b''.join(bodies[name] for name in names),'exact original source envelope and complete ordered functions')
        return names,bodies
    require(sha(raw)==SOURCE,'exact frozen indexed forward source')
    names,definitions=parse(raw)
    require(len(definitions)==781 and names[-3:]==list(PINS),'three forward helpers appended in exact order')
    restored=raw
    for name,pin in PINS.items():
        body=definitions[name];require(sha(body)==pin and restored.count(body)==1,'complete exact added helper '+name)
        restored=restored.replace(body,b'')
    for after,before in [
      (b'call(v889,v903,v899,v890) {"function":"indexed_gather_forward_source"}',b'call(v889,v903,v899) {"function":"gather_source"}'),
      (b'call(v371,v387,v372,v373,v377) {"function":"indexed_add_forward_inputs"}',b'call(v371,v387) {"function":"indexed_add_forward"}')]:
        require(restored.count(after)==1,'one exact indexed forward call site')
        restored=restored.replace(after,before)
    require(sha(restored)==PARENT,'exact whole778 parent source restored')
    parent_names,parent=parse(restored)
    require(parent_names==names[:-3] and len(parent)==778,'every original function remains in its original order')
    changed=[name for name in parent_names if parent[name]!=definitions[name]]
    require(changed==['gather','indexed_add'],'only the two declared forward call sites differ')
    require(sum(parent[name]==definitions[name] for name in parent_names)==776,'all776 other complete original bodies retained')
    rebuilt=restored
    for name in changed:rebuilt=rebuilt.replace(parent[name],definitions[name],1)
    rebuilt+=b''.join(definitions[name] for name in PINS)
    require(rebuilt==raw,'complete byte-for-byte forward reconstruction and round trip')
    return restored,dict(source_sha256=SOURCE,parent_source_sha256=PARENT,functions=781,parent_functions=778,
        unchanged_parent_functions=776,changed_functions=['gather','indexed_add'],appended_functions=list(PINS),
        whole_source_inverse_exact=True,whole_round_trip_exact=True,function_order_retained=True,
        source_envelope_retained=True,original_requires_grad_dispatch_preserved=True,
        original_backward_functions_unchanged=True)
