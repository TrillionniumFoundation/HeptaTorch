"""Reversible opt-in edit of only one function in exact predecessor #65 f9807c9."""
import hashlib,json,re,sys
from pathlib import Path

BASE='e43b13aa0d4ad14d297f03b82212e73f0f13b4377dc3067a4e12fed7deaba1f8'
MAN='0eb5ecb407c8f2bbd273448084b3f4ac752ca86b3be117ff221d5697b49d3f53'
UNIT='1dcd29b900d2036fba7ad538680ecc8edab576436a675617779db9e7c0ca5dd2'
sha=lambda b:hashlib.sha256(b).hexdigest()
def replace(text,a,b):
    assert text.count(a)==1,a
    return text.replace(a,b,1)
def body(raw,name):
    m=re.findall(rb'^fn '+name.encode()+rb'\([^\n]*\n.*?^end\n',raw,re.M|re.S)
    assert len(m)==1
    return m[0]
def transform(raw):
    diffs=[]
    def add(name,changes):
        old=body(raw,name); txt=old.decode()
        for a,b in changes: txt=replace(txt,a,b)
        diffs.append((old,txt.encode()))
    add('logical_out_apply',[
       ('  ls_input_bool:bool = compare.eq(lo_admit_idtype,lo_admit_one)\n',''),
       ('  ls_self_bool:bool = bool.and(ls_same_tensor,ls_input_bool)\n',''),
       ('bool.or(ls_self_bool,ls_disjoint)','bool.or(ls_same_tensor,ls_disjoint)'),
    ])
    out=raw
    for before,after in diffs:out=replace(out,before,after)
    back=out
    for before,after in diffs:back=replace(back,after,before)
    assert back==raw
    return out

def main(src,dst,pins=None):
    orig=(src/'manifest.json').read_bytes();assert sha(orig)==MAN
    m=json.loads(orig)
    raw=(src/'runtime.hepta').read_bytes();assert sha(raw)==BASE
    final=transform(raw)
    dst.mkdir();(dst/'units').mkdir()
    (dst/'runtime.hepta').write_bytes(final)
    m['name']='opt_in_joint_logical_not_numeric_self'
    m.pop('expected_native_sha256')
    m['runtime'].update(sha256=sha(final),instructions=final.count(b' = '))
    for u in m['units']:
        p=u['path'];old=(src/p).read_bytes();assert sha(old)==u['sha256']
        if u['name']=='runtime800_12':
            assert sha(old)==UNIT
            new=transform(old);u['sha256']=sha(new);u.pop('expected_object_sha256')
            if pins is not None:
                r=json.loads(pins.read_bytes());assert r['status']=='PASS_NATIVE_BUILD_ONLY' and r['source_sha256']==sha(final)
                u['expected_object_sha256']=next(v['object_sha256'] for v in r['units'] if v['name']==u['name'])
                m['expected_native_sha256']=r['native_sha256']
            for n in ['logical_out_apply','logical_out_command']:
                assert body(new,n)==body(final,n)
        else:new=old
        (dst/p).write_bytes(new)
    (dst/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    print(sha(final))
if __name__=='__main__': main(*map(Path,sys.argv[1:]))
