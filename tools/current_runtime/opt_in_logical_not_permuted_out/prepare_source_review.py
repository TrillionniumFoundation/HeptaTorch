"""Static reversible source draft; no compile, native, or Torch execution."""
import hashlib
from pathlib import Path
import re

D = Path(__file__).resolve().parent
PENDING = 'bd811a3b484002467c8f54c67c138f6d8701fc378a389b67422d61db5ca5783b'
PREDECESSOR = 'e43b13aa0d4ad14d297f03b82212e73f0f13b4377dc3067a4e12fed7deaba1f8'
HELPER = (D / 'logical_out_permuted_f32.hepta').read_bytes()
PENDING_DECL = b'  ls_admitted_owners:bool = bool.or(ls_same_tensor,ls_disjoint)\n'
BEFORE_DECL = b'  ls_admitted_owners:bool = bool.or(ls_self_bool,ls_disjoint)\n'
BOOL_TEST = b'  ls_input_bool:bool = compare.eq(lo_admit_idtype,lo_admit_one)\n'
NAMES = br'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n'
OLD_TAIL = (b'  branch lo_descriptors_gate1 shape(lo_descriptors_s,lo_descriptors_in_node,lo_descriptors_out_node,'
            b'lo_descriptors_out_handle,lo_descriptors_idtype,lo_descriptors_icount,lo_descriptors_z,lo_descriptors_one) refuse(lo_descriptors_s)\n')
NEW_TAIL = b'''  branch lo_descriptors_gate1 shape(lo_descriptors_s,lo_descriptors_in_node,lo_descriptors_out_node,lo_descriptors_out_handle,lo_descriptors_idtype,lo_descriptors_icount,lo_descriptors_z,lo_descriptors_one) permute_check(lo_descriptors_s,lo_descriptors_in_node,lo_descriptors_out_node,lo_descriptors_out_handle,lo_descriptors_odesc,lo_descriptors_idtype,lo_descriptors_odtype,lo_descriptors_ostore,lo_descriptors_over,lo_descriptors_icount,lo_descriptors_z,lo_descriptors_one,lo_descriptors_test0,lo_descriptors_disjoint)
block permute_check(pc_s:mutbuf,pc_in:u64,pc_out:u64,pc_handle:u64,pc_desc:u64,pc_idtype:u64,pc_odtype:u64,pc_store:u64,pc_ver:u64,pc_count:u64,pc_z:u64,pc_one:u64,pc_input_ok:bool,pc_disjoint:bool)
  pc_gate:bool = bool.and(pc_input_ok,pc_disjoint)
  branch pc_gate permute_layout(pc_s,pc_in,pc_out,pc_handle,pc_desc,pc_idtype,pc_odtype,pc_store,pc_ver,pc_count,pc_z,pc_one) refuse(pc_s)
block permute_layout(pl_s:mutbuf,pl_in:u64,pl_out:u64,pl_handle:u64,pl_desc:u64,pl_idtype:u64,pl_odtype:u64,pl_store:u64,pl_ver:u64,pl_count:u64,pl_z:u64,pl_one:u64)
  pl_layout:u64 = call(pl_s,pl_out,pl_desc,pl_store,pl_ver,pl_odtype) {"function":"logical_out_permuted_f32"}
  pl_ok:bool = compare.ne(pl_layout,pl_z)
  branch pl_ok shape(pl_s,pl_in,pl_out,pl_handle,pl_idtype,pl_count,pl_z,pl_one) refuse(pl_s)
'''

def sha(b):return hashlib.sha256(b).hexdigest()

def replace(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b,1)

def transition(raw):
    rec=[m for m in re.finditer(NAMES,raw,re.M|re.S) if m[1]==b'logical_out_apply']
    assert len(rec)==1
    old=rec[0][0]
    new=replace(old,b'lo_admit_z,lo_admit_one) refuse(lo_admit_s)',b'lo_admit_z,lo_admit_one,ls_disjoint) refuse(lo_admit_s)')
    new=replace(new,b'lo_descriptors_z:u64,lo_descriptors_one:u64)\n',b'lo_descriptors_z:u64,lo_descriptors_one:u64,lo_descriptors_disjoint:bool)\n')
    new=replace(new,OLD_TAIL,NEW_TAIL)
    attached=HELPER+new
    output=replace(raw,old,attached)
    assert replace(output,attached,old)==raw
    return output,old,new

def pending_undo(raw):
    old_test=b'  ls_same_version:bool = compare.eq(lo_admit_iver,lo_admit_over)\n'
    old_disjoint=b'  ls_distinct_handle:bool = bool.not(lo_admit_same_handle)\n'
    return replace(replace(replace(raw,PENDING_DECL,BEFORE_DECL),old_test,old_test+BOOL_TEST),old_disjoint,b'  ls_self_bool:bool = bool.and(ls_same_tensor,ls_input_bool)\n'+old_disjoint)

def static_arities(raw,names):
    sig={m[1]:len(m[2].split(b',')) if m[2] else 0
         for m in re.finditer(rb'^fn (\w+)\((.*?)\)',raw,re.M)}
    for m in re.finditer(NAMES,raw,re.M|re.S):
        if m[1] not in names:continue
        body=m[0]
        blocks=dict((r[1],len(r[2].split(b',')) if r[2] else 0)
                    for r in re.finditer(rb'^block (\w+)\((.*?)\)',body,re.M))
        for line in body.splitlines():
            c=re.search(rb' = call\(([^)]*)\) \{"function":"(\w+)"\}',line)
            if c:assert c[2] in sig and len(c[1].split(b','))==sig[c[2]],line
            if not line.startswith((b'  jump ',b'  branch ')):continue
            for b in re.finditer(rb'(\w+)\(([^)]*)\)',line):
                assert b[1] in blocks and (len(b[2].split(b',')) if b[2] else 0)==blocks[b[1]],line

def main():
    src=Path('/tmp/heptatorch-logical-numeric-self/pinned-source')
    raw=(src/'runtime.hepta').read_bytes()
    unit=(src/'units/runtime800_12.hepta').read_bytes()
    assert sha(raw)==PENDING and sha(pending_undo(raw))==PREDECESSOR
    out,old,new=transition(raw)
    ou,uold,unew=transition(unit)
    assert (old,new)==(uold,unew)
    assert pending_undo(out)==transition(pending_undo(raw))[0]
    assert pending_undo(ou)==transition(pending_undo(unit))[0]
    prev=dict((m[1],m[0]) for m in re.finditer(NAMES,raw,re.M|re.S))
    nxt=dict((m[1],m[0]) for m in re.finditer(NAMES,out,re.M|re.S))
    assert set(nxt)-set(prev)=={b'logical_out_permuted_f32'} and set(prev)<set(nxt)
    assert {n for n in prev if prev[n]!=nxt[n]}=={b'logical_out_apply'}
    assert [n for n in nxt if n in prev]==list(prev)
    static_arities(out,{b'logical_out_apply',b'logical_out_permuted_f32'})
    print('exact f9807c9 predecessor =',sha(pending_undo(raw)))
    print('exact numeric-self pending =',sha(raw))
    print('composed monolith =',sha(out),'functions =',len(nxt))
    print('pending partition runtime800_12 =',sha(unit))
    print('composed partition runtime800_12 =',sha(ou))
    print('all original bodies unchanged except logical_out_apply; exact reverse and both patch orders agree')
if __name__=='__main__':main()
