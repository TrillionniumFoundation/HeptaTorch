"""Reversible opt-in edit of only two functions in exact predecessor #65."""
import hashlib,json,re,sys
from pathlib import Path

BASE='fa0437a53f80d0bb8c2f846a32efe30988b7a33f558fac5d829d4d9701f4c9f5'
MAN='248c54b02029338dea96790e567f338eed4dd8e896148e7377b800b779ca04c0'
UNIT='ab6194de2350442ed7545c1e20c57518ac5bbc27355dafa698619b04b74e7271'
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
       ('lo_admit_out_handle:u64) ->', 'lo_admit_out_handle:u64,lo_admit_same_handle:bool) ->'),
       ('lo_admit_out_handle:u64)\n','lo_admit_out_handle:u64,lo_admit_same_handle:bool)\n'),
       ('  lo_admit_test3:bool = compare.le', '''  ls_same_node:bool = compare.eq(lo_admit_in_node,lo_admit_out_node)
  ls_same_storage:bool = compare.eq(lo_admit_istore,lo_admit_ostore)
  ls_same_version:bool = compare.eq(lo_admit_iver,lo_admit_over)
  ls_input_bool:bool = compare.eq(lo_admit_idtype,lo_admit_one)
  ls_all_owners_same:bool = bool.and(ls_same_storage,ls_same_version)
  ls_same_owner_node:bool = bool.and(ls_all_owners_same,ls_same_node)
  ls_same_tensor:bool = bool.and(ls_same_owner_node,lo_admit_same_handle)
  ls_self_bool:bool = bool.and(ls_same_tensor,ls_input_bool)
  ls_distinct_handle:bool = bool.not(lo_admit_same_handle)
  ls_disjoint:bool = bool.and(ls_distinct_handle,lo_admit_gate2)
  ls_admitted_owners:bool = bool.or(ls_self_bool,ls_disjoint)
  lo_admit_test3:bool = compare.le'''),
       ('bool.and(lo_admit_gate2,lo_admit_test3)','bool.and(ls_admitted_owners,lo_admit_test3)'),
    ])
    add('logical_out_command',[
       ('lo_resolve_out_node,lo_resolve_output_word) reply','lo_resolve_out_node,lo_resolve_input_word,lo_resolve_output_word) reply'),
       ('lo_execute_out_node:u64,lo_execute_output_word:u64)','lo_execute_out_node:u64,lo_execute_input_word:u64,lo_execute_output_word:u64)'),
       ('  lo_execute_result:u64 = call(lo_execute_s,lo_execute_in_node,lo_execute_out_node,lo_execute_existing_handle)',
        '''  ls_in_handle:u64 = call(lo_execute_s,lo_execute_input_word) {"function":"resolve_handle"}
  ls_same_handle:bool = compare.eq(ls_in_handle,lo_execute_existing_handle)
  lo_execute_result:u64 = call(lo_execute_s,lo_execute_in_node,lo_execute_out_node,lo_execute_existing_handle,ls_same_handle)'''),
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
    m['name']='opt_in_joint_logical_not_self_bool'
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
