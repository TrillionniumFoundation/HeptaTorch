"""Reversible opt-in edit of only one function in exact predecessor #65 ed155be."""
import hashlib,json,re,sys
from pathlib import Path

BASE='f29e15f1efddd726cff1739dfc24e349566f96c24a2a3bdeb47c716f8e0b4b80'
MAN='4d34cdcfa6cb557280fc2edbbb9dd167038accc6d53061327c125e3a57c8c0c4'
UNIT='2982726d67268a8ee435b84e54f864caa5e0f295a096ff1706b7ccad2fe81880'
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
       ('  lo_admit_test5:bool = compare.eq(lo_admit_odtype,lo_admit_one)',
        '''  ln_otype_ceiling:bool = compare.le(lo_admit_odtype,lo_admit_maxdtype)
  ln_otype_public:bool = compare.ne(lo_admit_odtype,lo_admit_private3)
  lo_admit_test5:bool = bool.and(ln_otype_ceiling,ln_otype_public)'''),
       ('lo_admit_idesc,lo_admit_odesc,lo_admit_idtype,lo_admit_istore',
        'lo_admit_idesc,lo_admit_odesc,lo_admit_idtype,lo_admit_odtype,lo_admit_istore'),
       ('lo_descriptors_idtype:u64,lo_descriptors_istore:u64',
        'lo_descriptors_idtype:u64,lo_descriptors_odtype:u64,lo_descriptors_istore:u64'),
       ('  lo_descriptors_olayout:u64 = call(lo_descriptors_s,lo_descriptors_odesc,lo_descriptors_out_node,lo_descriptors_ostore,lo_descriptors_over,lo_descriptors_one)',
        '''  ln_owidth:u64 = call(lo_descriptors_odtype) {"function":"dtype_width"}
  lo_descriptors_olayout:u64 = call(lo_descriptors_s,lo_descriptors_odesc,lo_descriptors_out_node,lo_descriptors_ostore,lo_descriptors_over,ln_owidth)'''),
       ('  lo_store_discard1:u64 = call(lo_store_s,lo_store_at,lo_store_stride,lo_store_byte) {"function":"typed_store"}',
        '''  ln_otype:u64 = call(lo_store_s,lo_store_out_node) {"function":"tensor_dtype"}
  ln_width:u64 = call(ln_otype) {"function":"dtype_width"}
  ln_zero:u64 = const() {"value":0}
  ln_is_zero:bool = compare.eq(lo_store_byte,ln_zero)
  ln_tag_f32:u64 = const() {"value":0}
  ln_tag_f64:u64 = const() {"value":4}
  ln_tag_f16:u64 = const() {"value":5}
  ln_tag_bf16:u64 = const() {"value":6}
  ln_one_f32:u64 = const() {"value":1065353216}
  ln_one_f64:u64 = const() {"value":4607182418800017408}
  ln_one_f16:u64 = const() {"value":15360}
  ln_one_bf16:u64 = const() {"value":16256}
  ln_f32:bool = compare.eq(ln_otype,ln_tag_f32)
  ln_f64:bool = compare.eq(ln_otype,ln_tag_f64)
  ln_f16:bool = compare.eq(ln_otype,ln_tag_f16)
  ln_bf16:bool = compare.eq(ln_otype,ln_tag_bf16)
  ln_first:u64 = select(ln_f32,ln_one_f32,lo_store_stride)
  ln_second:u64 = select(ln_f64,ln_one_f64,ln_first)
  ln_third:u64 = select(ln_f16,ln_one_f16,ln_second)
  ln_one:u64 = select(ln_bf16,ln_one_bf16,ln_third)
  ln_encoded:u64 = select(ln_is_zero,ln_zero,ln_one)
  lo_store_discard1:u64 = call(lo_store_s,lo_store_at,ln_width,ln_encoded) {"function":"typed_store"}'''),
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
    m['name']='opt_in_joint_logical_not_numeric_out'
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
