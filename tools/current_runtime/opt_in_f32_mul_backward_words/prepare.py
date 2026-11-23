"""Stack a CPU F32 same-shape tag3 backward word profile on exact #64 6bacbeb."""
import argparse,json,re
from hashlib import sha256
from pathlib import Path
BASE_MANIFEST='e4ab4fe1b39550dd22f45c82fb2a9f16ed2a97f904a7042b132d312ee9f28185'
BASE_SOURCE='06dd36061f90ccd185afc6124322fad0cf1368e75c61f10c84ba741ac56a56c1'
BASE_ELF='3e427044155ae8787621a926425a1ab0911611c69cd2f3da7f08417dfa9d7774'
PINS={'elf': '9074399826154287c2a97bd7172d6d86562eb5dfe1f5ab641ac33d6a03fb59de', 'runtime800_02': '99263aee01fdddaf6267536ee2dd6571f258986b1c5de740d44399cff1516345'}
OUTPUT_MANIFEST='1fd4916426f8a0d9cbd8598bcdd333ac14ae4f1c5f7fb73ac0b934e812f2a2f6'
NAME='scalar_previous_back_binary'
UNIT='runtime800_02'
def sha(b):return sha256(b).hexdigest()
def need(v,msg):
    if not v:raise ValueError(msg)
def bound(p,d):
    need(p.is_file() and not p.is_symlink(),'invalid physical input')
    b=p.read_bytes();need(sha(b)==d,'input digest mismatch '+p.name);return b
def body(b):
    a=re.findall(rb'^fn '+NAME.encode()+rb'\([^\n]*\n.*?^end\n',b,re.M|re.S)
    need(len(a)==1,'wrong exact body census');return a[0]
def replace(b,a,c):
    need(b.count(a)==1 and c not in b,'wrong exact splice')
    return b.replace(a,c,1)
def multiply(src,old,grad,peer,out,state,node,left,right,key):
    a=lambda name:key+'_'+name
    def v(name,typ,expr):return f'  {a(name)}:{typ} = {expr}\n'
    def const(name,value):return v(name,'u64','const() {"value":'+str(value)+'}')
    assign=lambda name,typ,op,*args:v(name,typ,f'{op}({",".join(args)})')
    call=lambda name,fn,*args:v(name,'u64','call('+','.join(args)+') {"function":"'+fn+'"}')
    o=''
    for k,n in (('mag_mask',0x7fffffff),('inf',0x7f800000),('zero',0),('quiet',0x400000),('indef',0xffc00000)):
        o+=const(k,n)
    for k,p in (('g',grad),('p',peer)):
        o+=assign(k+'mag','u64','bit.and',p,a('mag_mask'))
        o+=assign(k+'nan','bool','compare.gt',a(k+'mag'),a('inf'))
        o+=assign(k+'inf','bool','compare.eq',a(k+'mag'),a('inf'))
        o+=assign(k+'zero','bool','compare.eq',a(k+'mag'),a('zero'))
    # Never assign dual-NaN precedence. Both original operands and product remain unchanged outside the profile.
    o+=assign('one_nan','bool','compare.ne',a('gnan'),a('pnan'))
    o+=assign('gp_invalid','bool','bool.and',a('ginf'),a('pzero'))
    o+=assign('pg_invalid','bool','bool.and',a('pinf'),a('gzero'))
    o+=assign('invalid','bool','bool.or',a('gp_invalid'),a('pg_invalid'))
    o+=assign('exception','bool','bool.or',a('one_nan'),a('invalid'))
    # Preserve the original raw call for ordinary, out-of-scope, and dual-NaN cases.
    o+=call('legacy','mulf',grad,peer)
    for k,p in (('n',node),('l',left),('r',right)):
        o+=call(k+'type','tensor_dtype',state,p)
        o+=assign(k+'f32','bool','compare.eq',a(k+'type'),a('zero'))
        o+=call(k+'contig','is_contiguous',state,p)
        o+=assign(k+'dense','bool','compare.ne',a(k+'contig'),a('zero'))
    o+=call('left_same','gradient_direct',state,node,left)
    o+=call('right_same','gradient_direct',state,node,right)
    o+=assign('left_match','bool','compare.ne',a('left_same'),a('zero'))
    o+=assign('right_match','bool','compare.ne',a('right_same'),a('zero'))
    group=['exception','nf32','lf32','rf32','ndense','ldense','rdense','left_match','right_match']
    for i,p in enumerate(group[1:]):o+=assign(f'guard{i}','bool','bool.and',a(group[0] if i==0 else f'guard{i-1}'),a(p))
    o+=assign('nan','u64','select',a('gnan'),grad,peer)
    o+=assign('qnan','u64','bit.or',a('nan'),a('quiet'))
    o+=assign('reference','u64','select',a('one_nan'),a('qnan'),a('indef'))
    o+=f'  {out}:u64 = select({a("guard"+str(len(group)-2))},{a("reference")},{a("legacy")})\n'
    return replace(src,old,o)
def derive(base,scratch):
    pins=PINS
    m=json.loads(bound(base/'manifest.json',BASE_MANIFEST))
    rt=bound(base/'runtime.hepta',BASE_SOURCE)
    need(m['expected_native_sha256']==BASE_ELF and not scratch.resolve().is_relative_to(base.resolve()),'wrong predecessor/output')
    old=body(rt);b=old.decode()
    b=multiply(b,'  v122:u64 = call(v105,v121) {"function":"mulf"}\n','v105','v121','v122','v93','v94','v96','v97','left')
    b=multiply(b,'  v301:u64 = call(v284,v300) {"function":"mulf"}\n','v284','v300','v301','v272','v273','v275','v276','right')
    new=b.encode()
    def splice(raw):
        x=replace(raw,old,new)
        need(replace(x,new,old)==raw,'failed byte-exact inverse')
        return x
    final=splice(rt); data={}
    for u in m['units']:
        path=u['path']; raw=bound(base/path,u['sha256'])
        if u['name']==UNIT:
            raw=splice(raw);u['sha256']=sha(raw)
            u['expected_object_sha256']=pins[UNIT]
        data[path]=raw
    m['expected_native_sha256']=pins['elf']
    m['name']='current800_opt_in_dense_f32_mul_backward_words'
    m['runtime'].update(sha256=sha(final),instructions=final.count(b' = '),blocks=len(re.findall(rb'^block ',final,re.M)))
    mr=(json.dumps(m,indent=2,sort_keys=True)+'\n').encode()
    need(sha(mr)==OUTPUT_MANIFEST,'unexpected output manifest')
    scratch.mkdir(); (scratch/'units').mkdir()
    for n,b in data.items():(scratch/n).write_bytes(b)
    (scratch/'runtime.hepta').write_bytes(final);(scratch/'manifest.json').write_bytes(mr)
    return {NAME:{'prior_sha256':sha(old),'candidate_sha256':sha(new),'partition':UNIT},'source_sha256':sha(final),'manifest_sha256':sha(mr)}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    print(json.dumps(derive(a.base,a.output),indent=2,sort_keys=True))
