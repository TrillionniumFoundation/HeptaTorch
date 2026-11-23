"""Experimental CPU contiguous F32 abs source transform for the exact #63 joint image."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

MANIFEST = '9f24d9903a9ffa96cd6121cbc7130cefe8536f2c3e5a425e6d669f72242de1f0'
ELF = '5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72'
PINS = {'elf': 'f9408443b278cd4c5d2131845ae31b3ad0e03469ca5e6e3c69c031e31694fe46', 'runtime800_03': 'b76b0cb3db499e5de1b3cee8d39b2ae90843285834a4191a39cc633b28827229', 'runtime800_05': 'c211f22bd91352eb8d34ae3bcf579f158076d1be9c741fd0b7c12f4ae4c41b08', 'runtime800_06': '83ad2859dfe9005c22e6c54e6a95913d2ae169e85ceb7a1a73b32993391f936f'}
OUTPUT_MANIFEST = '11f5061fc49a6c9b1bd949553f5e57cd11dd98477e08e634eeb15cc655b7f747'
TAG = 201
CHANGES = {'back_unary':'03','partial_apply_node_gradient':'03','record_body':'05',
           'unary_value':'06','half_neg_previous_unary':'06'}

def need(ok,msg):
    if not ok:
        raise ValueError(msg)

def sha(raw):
    return sha256(raw).hexdigest()

def body(raw,name):
    matches = re.findall(rb'^fn '+name.encode()+rb'\([^\n]*\n.*?^end\n',raw,re.M|re.S)
    need(len(matches)==1,'missing/nonunique body: '+name)
    return matches[0]

def replace(raw,before,after):
    need(raw.count(before)==1,'nonunique transform anchor')
    return raw.replace(before,after,1)

def edit(raw,name):
    src=raw.decode()
    if name=='unary_value':
        src=replace(src,'fn unary_value(v0:u64,v1:u64)','fn unary_value(o:u64,x:u64)')
        src=replace(src,'entry(b0)\n','''entry(abs_route)
block abs_route(o:u64,x:u64)
  external_abs:u64 = const() {"value":7}
  is_abs:bool = compare.eq(o,external_abs)
  branch is_abs abs_value(x) b0(o,x)
block abs_value(a:u64)
  magnitude:u64 = call(a) {"function":"absf"}
  ret(magnitude)
''')
    elif name=='half_neg_previous_unary':
        src=replace(src,'branch v5 b1(v0) b2(v0,v1,v2)','''branch v5 b1(v0) abs_admit(v0,v1,v2)
block abs_admit(s:mutbuf,o:u64,x:u64)
  external_abs:u64 = const() {"value":7}
  is_abs:bool = compare.eq(o,external_abs)
  branch is_abs abs_layout(s,o,x) b2(s,o,x)
block abs_layout(sl:mutbuf,ol:u64,xl:u64)
  contiguous:u64 = call(sl,xl) {"function":"is_contiguous"}
  zero_l:u64 = const() {"value":0}
  bad_layout:bool = compare.eq(contiguous,zero_l)
  branch bad_layout b5(sl) b7(sl,ol,xl)''')
        src=replace(src,'  v103:u64 = call(v86,v90,v87,v88,v102)',f'''  wire_abs:u64 = const() {{"value":7}}
  graph_abs:u64 = const() {{"value":{TAG}}}
  abs:bool = compare.eq(v87,wire_abs)
  tag:u64 = select(abs,graph_abs,v87)
  v103:u64 = call(v86,v90,tag,v88,v102)''')
    elif name=='record_body':
        src=replace(src,'  v378:u64 = select(v376,v377,v363)',f'''  abs_tag:u64 = const() {{"value":{TAG}}}
  saves_abs:bool = compare.eq(v357,abs_tag)
  save_operand:bool = bool.or(v376,saves_abs)
  v378:u64 = select(save_operand,v377,v363)''')
    elif name=='partial_apply_node_gradient':
        src=replace(src,'  branch v48 b6(v31,v32,v34) b8(v31,v32,v34)',f'''  abs_tag:u64 = const() {{"value":{TAG}}}
  is_abs:bool = compare.eq(v34,abs_tag)
  unary:bool = bool.or(v48,is_abs)
  branch unary b6(v31,v32,v34) b8(v31,v32,v34)''')
    elif name=='back_unary':
        src=replace(src,'  branch v43 b4(v21,v22,v23,v24,v25,v26,v27,v41) b8(v21,v22,v23,v24,v25,v26,v27,v41)',f'''  branch v43 b4(v21,v22,v23,v24,v25,v26,v27,v41) abs_route(v21,v22,v23,v24,v25,v26,v27,v41)
block abs_route(s:mutbuf,y:u64,p:u64,x:u64,t:u64,n:u64,i:u64,g:u64)
  abs_tag:u64 = const() {{"value":{TAG}}}
  abs:bool = compare.eq(t,abs_tag)
  branch abs abs_backward(s,y,p,x,t,n,i,g) b8(s,y,p,x,t,n,i,g)
block abs_backward(sa:mutbuf,ya:u64,pa:u64,xa:u64,ta:u64,na:u64,ia:u64,ga:u64)
  raw:u64 = call(sa,xa,ia) {{"function":"tget"}}
  zero:u64 = const() {{"value":0}}
  neg:u64 = call(raw,zero) {{"function":"ltf"}}
  pos:u64 = call(zero,raw) {{"function":"ltf"}}
  is_neg:bool = compare.ne(neg,zero)
  is_pos:bool = compare.ne(pos,zero)
  plus_one:u64 = const() {{"value":1065353216}}
  minus_one:u64 = const() {{"value":3212836864}}
  p_sign:u64 = select(is_pos,plus_one,zero)
  sign:u64 = select(is_neg,minus_one,p_sign)
  derivative:u64 = call(ga,sign) {{"function":"mulf"}}
  jump b8(sa,ya,pa,xa,ta,na,ia,derivative)''')
    else:
        raise ValueError(name)
    return src.encode()

def derive(base,dest):
    pins=PINS
    original_manifest=(base/'manifest.json').read_bytes()
    need(sha(original_manifest)==MANIFEST,'not the exact joint manifest')
    m=json.loads(original_manifest)
    original=(base/'runtime.hepta').read_bytes()
    need(sha(original)==m['runtime']['sha256'] and m['expected_native_sha256']==ELF,'wrong prior source')
    need(f'"value":{TAG}}}'.encode() not in original,'already-used graph tag literal')
    changes={n:(body(original,n),edit(body(original,n),n)) for n in CHANGES}
    def transform(raw,names):
        final=raw
        for n in names:
            final=replace(final,*changes[n])
        inverse=final
        for n in reversed(names):
            inverse=replace(inverse,*changes[n][::-1])
        need(inverse==raw,'non-exact source inverse')
        return final
    sources={}
    final=transform(original,list(CHANGES))
    for u in m['units']:
        p=u['path']
        prior=(base/p).read_bytes()
        need(sha(prior)==u['sha256'],'wrong input partition: '+p)
        names=[n for n,v in CHANGES.items() if u['name']=='runtime800_'+v]
        if names:
            new=transform(prior,names)
            u['sha256']=sha(new)
            u['expected_object_sha256']=pins[u['name']]
        else:
            new=prior
        sources[p]=new
    m['expected_native_sha256']=pins['elf']
    m['name']='current800_opt_in_contiguous_f32_abs'
    m['runtime'].update(sha256=sha(final),instructions=final.count(b' = '),blocks=len(re.findall(rb'^block ',final,re.M)))
    manifest_raw=(json.dumps(m,indent=2,sort_keys=True)+'\n').encode()
    need(sha(manifest_raw)==OUTPUT_MANIFEST,'candidate manifest differs')
    dest.mkdir()
    (dest/'units').mkdir()
    for p,raw in sources.items():
        (dest/p).write_bytes(raw)
    (dest/'runtime.hepta').write_bytes(final)
    (dest/'manifest.json').write_bytes(manifest_raw)
    return {n:{'prior_sha256':sha(a),'candidate_sha256':sha(b),'partition':CHANGES[n]} for n,(a,b) in changes.items()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('base',type=Path)
    p.add_argument('dest',type=Path)
    a=p.parse_args()
    need(not a.dest.resolve().is_relative_to(a.base.resolve()),'scratch output must be separate')
    print(json.dumps(derive(a.base,a.dest),indent=2,sort_keys=True))
if __name__=='__main__':
    main()
