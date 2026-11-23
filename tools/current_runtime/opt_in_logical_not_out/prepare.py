"""Opt-in only source draft. Does not invoke or produce an admitted build."""
import hashlib,json,re,sys
from pathlib import Path

BASERT='f4c43f037a3fb046425f7634215ec382aa5f8d6b0e7abe126ef64236adb5753a'
BASEMAN='9f24d9903a9ffa96cd6121cbc7130cefe8536f2c3e5a425e6d669f72242de1f0'
OP=128
EXT={
 'predicate_bits':'u64,u64,u64',
 'incoming_same_shape':'mutbuf,u64,u64',
 'is_contiguous':'mutbuf,u64',
 'so_version_live':'mutbuf,u64',
 'resolve_handle':'mutbuf,u64',
}

def hash_(b): return hashlib.sha256(b).hexdigest()
def exactly(raw,old,new):
    assert raw.count(old)==1
    return raw.replace(old,new,1)
def signature(n,args):
    return ','.join(f'{n}{i}:{typ}' for i,typ in enumerate(args.split(',')))
def read_field(reg,state,base,off,fn='getw'):
    if off==0: return f'  {reg}:u64 = call({state},{base}) {{"function":"{fn}"}}\n'
    return f'  {reg}_off:u64 = const() {{"value":{off}}}\n  {reg}_at:u64 = int.add.wrap({base},{reg}_off)\n  {reg}:u64 = call({state},{reg}_at) {{"function":"{fn}"}}\n'
def preds(pairs, all_=True):
    out=''; prior=None
    for i,(cond,a,b) in enumerate(pairs):
        out+=f'  test{i}:bool = compare.{cond}({a},{b})\n'
        if prior is None:
            prior='test0'
        else:
            out+=f'  gate{i}:bool = bool.{"and" if all_ else "or"}({prior},test{i})\n'
            prior=f'gate{i}'
    return out,prior
def guard(pairs,yes,no):
    regs,last=preds(pairs)
    return regs + f'  branch {last} {yes} {no}\n'
def constants(**data):
    return ''.join(f'  {k}:u64 = const() {{"value":{v}}}\n' for k,v in data.items())
def args(s):
    return ','.join(f'{x}:{ "mutbuf" if x=="s" or x=="out" else "readbuf" if x=="req" else "u64"}' for x in s.split())

layout_args='s desc node storage version width'
layout = f'fn logical_out_layout({args(layout_args)}) -> (u64) effects(read,write,trap) entry(layout)\nblock layout({args(layout_args)})\n'
layout+=constants(z=0,one=1,ranklimit=4,countlimit=262144,tag=2**63)
for field,off in [('rank',0),('count',8),('start',16),('requires',32),('edge',40),('offset',232),('viewflag',664)]:
    layout+=read_field(field,'s','desc',off)
layout+='  live:u64 = call(s,storage) {"function":"so_storage_live"}\n  v_live:u64 = call(s,version) {"function":"so_version_live"}\n'
layout+=guard([('le','rank','ranklimit'),('le','count','countlimit'),('eq','requires','z'),('eq','edge','z'),('eq','offset','z'),('eq','viewflag','z'),('ne','live','z'),('ne','v_live','z')],'extent(s,node,width,count,start,live,tag,z)','bad(z)')
layout+=f'block extent({args("s node width count start live tag z")})\n'
layout+=read_field('base','s','live',16,'so_uget')+read_field('size','s','live',24,'so_uget')
layout+='  expected_start:u64 = int.add.wrap(tag,base)\n  needed:u64 = int.mul.wrap(count,width)\n  contig:u64 = call(s,node) {"function":"is_contiguous"}\n'
layout+=guard([('ne','contig','z'),('eq','start','expected_start'),('le','needed','size')],'ok()','bad(z)')
layout+='block ok()\n  success:u64 = const() {"value":1}\n  ret(success)\nblock bad(fail_value:u64)\n  ret(fail_value)\nend\n'

apply_args='s in_node out_node out_handle'
apply=f'fn logical_out_apply({args(apply_args)}) -> (u64) effects(read,write,trap) entry(admit)\nblock admit({args(apply_args)})\n'
apply+=constants(z=0,one=1,maxdtype=6,private3=3)+'  idesc:u64 = call(s,in_node) {"function":"dptr"}\n  odesc:u64 = call(s,out_node) {"function":"dptr"}\n  idtype:u64 = call(s,in_node) {"function":"tensor_dtype"}\n  odtype:u64 = call(s,out_node) {"function":"tensor_dtype"}\n'
for field,base,off in [('istore','idesc',216),('ostore','odesc',216),('iver','idesc',224),('over','odesc',224),('icount','idesc',8),('ocount','odesc',8)]:
    apply+=read_field(field,'s',base,off)
apply+=guard([('ne','in_node','out_node'),('ne','istore','ostore'),('ne','iver','over'),('le','idtype','maxdtype'),('ne','idtype','private3'),('eq','odtype','one'),('eq','icount','ocount')],'descriptors(s,in_node,out_node,out_handle,idesc,odesc,idtype,istore,ostore,iver,over,icount,z,one)','refuse(s)')
desc_args='s in_node out_node out_handle idesc odesc idtype istore ostore iver over icount z one'
apply+=f'block descriptors({args(desc_args)})\n  iwidth:u64 = call(idtype) {{"function":"dtype_width"}}\n  ilayout:u64 = call(s,idesc,in_node,istore,iver,iwidth) {{"function":"logical_out_layout"}}\n  olayout:u64 = call(s,odesc,out_node,ostore,over,one) {{"function":"logical_out_layout"}}\n'
apply+=guard([('ne','ilayout','z'),('ne','olayout','z')],'shape(s,in_node,out_node,out_handle,idtype,icount,z,one)','refuse(s)')
shape_args='s in_node out_node out_handle idtype icount z one'
apply+=f'block shape({args(shape_args)})\n  same:u64 = call(s,in_node,out_node) {{"function":"incoming_same_shape"}}\n  shape_ok:bool = compare.ne(same,z)\n  branch shape_ok version(s,in_node,out_node,out_handle,idtype,icount,z,one) refuse(s)\n'
apply+=f'block version({args(shape_args)})\n  bumpable:u64 = call(s,out_node,one) {{"function":"version_can_bump"}}\n  can:bool = compare.ne(bumpable,z)\n  branch can loop(s,in_node,out_node,out_handle,idtype,icount,z) failed(z)\n'
loop_args='s in_node out_node out_handle idtype icount index'
apply+=f'block loop({args(loop_args)})\n  remaining:bool = compare.lt(index,icount)\n  branch remaining load(s,in_node,out_node,out_handle,idtype,icount,index) commit(s,out_node,out_handle)\n'
apply+=f'block load({args(loop_args)})\n  raw:u64 = call(s,in_node,index) {{"function":"tget"}}\n  zero:u64 = const() {{"value":0}}\n  status:u64 = call(s,zero) {{"function":"getw"}}\n  loaded:bool = compare.eq(status,zero)\n  branch loaded store(s,in_node,out_node,out_handle,idtype,icount,index,raw) failed(zero)\n'
apply+=f'block store({args(loop_args+" raw")})\n  truth_op:u64 = const() {{"value":127}}\n  stride:u64 = const() {{"value":1}}\n  byte:u64 = call(raw,idtype,truth_op) {{"function":"predicate_bits"}}\n  at:u64 = call(s,out_node,index) {{"function":"taddr"}}\n  _ = call(s,at,stride,byte) {{"function":"typed_store"}}\n  next_index:u64 = int.add.wrap(index,stride)\n  jump loop(s,in_node,out_node,out_handle,idtype,icount,next_index)\n'
apply+='block commit(s:mutbuf,out_node:u64,out_handle:u64)\n  advance:u64 = call(s,out_node) {"function":"bump_version"}\n  z:u64 = const() {"value":0}\n  token:u64 = select(SECURITY_GUARD,z,out_handle)\n  ret(token)\n'.replace('  token:', '  bump_failed:bool = compare.eq(advance,z)\n  token:').replace('SECURITY_GUARD','bump_failed')
apply+='block refuse(s:mutbuf)\n  unsupported:u64 = const() {"value":7}\n  _ = call(s,unsupported) {"function":"terror"}\n  z:u64 = const() {"value":0}\n  ret(z)\nblock failed(none:u64)\n  ret(none)\nend\n'

command=f'fn logical_out_command({args("req out s at op out_at")}) -> (u64) effects(read,write,trap) entry(wire)\nblock wire({args("req out s at op out_at")})\n'
for word,offset in [('input_word',8),('output_word',16),*[(f'pad{i}',i*8) for i in range(3,8)]]:
    command+=read_field(word,'req','at',offset,'getr')
command+=constants(z=0)+guard([('eq',f'pad{i}','z') for i in range(3,8)],'resolve(out,s,op,out_at,input_word,output_word,z)','bad_wire(out,s,op,out_at,z)')
command+=f'block resolve({args("out s op out_at input_word output_word z")})\n  in_node:u64 = call(s,input_word) {{"function":"public_node"}}\n  out_node:u64 = call(s,output_word) {{"function":"public_node"}}\n'
command+=guard([('ne','in_node','z'),('ne','out_node','z')],'execute(out,s,op,out_at,in_node,out_node,output_word)','reply(out,s,op,out_at,z)')
command+=f'block execute({args("out s op out_at in_node out_node output_word")})\n  existing_handle:u64 = call(s,output_word) {{"function":"resolve_handle"}}\n  result:u64 = call(s,in_node,out_node,existing_handle) {{"function":"logical_out_apply"}}\n  jump reply(out,s,op,out_at,result)\n'
command+=f'block bad_wire({args("out s op out_at z")})\n  bad_opcode:u64 = const() {{"value":7}}\n  _ = call(s,bad_opcode) {{"function":"terror"}}\n  jump reply(out,s,op,out_at,z)\n'
command+=f'block reply({args("out s op out_at result")})\n'+constants(zero=0,word=8,payload=16,length=24,total=32)+'  status:u64 = call(s,zero) {"function":"getw"}\n  failed:bool = compare.ne(status,zero)\n  value:u64 = select(failed,zero,result)\n  status_at:u64 = int.add.wrap(out_at,word)\n  value_at:u64 = int.add.wrap(out_at,payload)\n  length_at:u64 = int.add.wrap(out_at,length)\n  _ = call(out,out_at,op) {"function":"putw"}\n  _ = call(out,status_at,status) {"function":"putw"}\n  _ = call(out,value_at,value) {"function":"putw"}\n  _ = call(out,length_at,zero) {"function":"putw"}\n  done:u64 = int.add.wrap(out_at,total)\n  ret(done)\nend\n'

def normalize_ssa(text):
    ncall=0
    output=[]
    for line in text.splitlines(keepends=True):
        if line.startswith('  _ = call'):
            ncall+=1; line=line.replace('_ = call',f'discard{ncall}:u64 = call',1)
        if line.startswith('fn '):
            head, body=line.split('(',1)
            param, tail=body.split(')',1)
            start=re.search(r'entry\((\w+)\)',tail)[1]
            line=head+'('+re.sub(r'\b(\w+):',lambda m:'lo_'+start+'_'+m[0],param)+')'+tail
        if line.startswith('block '):
            name=line.split('(')[0].split()[1]
            names={}
        if line.startswith('block ') or line.startswith('  '):
            source, sep, attributes=line.partition(' {"')
            for register in re.findall(r'\b(\w+):(?:u64|bool|readbuf|mutbuf)\b',source):
                names[register]='lo_'+name+'_'+register
            pre,par,body=source.partition('(') if line.startswith('block ') else ('','','')
            if par:
                source=pre+par+re.sub(r'\b\w+\b',lambda t:names.get(t[0],t[0]),body)
            else:
                # These helpers use unique block labels except names such as 'loop';
                # rename operands and definitions, never jump/branch targets.
                source=re.sub(r'\b\w+\b',lambda t:names.get(t[0],t[0]) if source[t.end():t.end()+1]!='(' else t[0],source)
            line=source+sep+attributes
        output.append(line)
    return ''.join(output)

def patch(raw,unit):
    helpers=normalize_ssa(layout+apply+command).encode()
    entry_old=b'  branch typed_logical_not_entry_admitted b13(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139) b14(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139)\n'
    entry_new=(f'  logical_out_op:u64 = const() {{"value":{OP}}}\n'
        '  logical_out_selected:bool = compare.eq(v139,logical_out_op)\n'
        '  branch logical_out_selected logical_out_dispatch(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139) logical_keep_dispatch(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139)\n'
        'block logical_keep_dispatch(v119:readbuf,v120:mutbuf,v121:mutbuf,v122:mutbuf,v124:u64,v125:u64,v126:u64,v127:u64,v128:u64,v139:u64):\n')
    # v139 and predicate condition are live across the new split; give the new block fresh parameter names.
    entry_new=entry_new.split('block logical_keep_dispatch')[0].replace('logical_keep_dispatch(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139)','logical_keep_dispatch(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139,typed_logical_not_entry_admitted)')
    entry_new+='block logical_keep_dispatch(kr:readbuf,ko:mutbuf,ks:mutbuf,kw:mutbuf,kn:u64,ke:u64,ki:u64,kp:u64,kflag:u64,kop:u64,ka:bool)\n  branch ka b13(kr,ko,ks,kw,kn,ke,ki,kp,kflag,kop) b14(kr,ko,ks,kw,kn,ke,ki,kp,kflag,kop)\n'
    entry_new+='block logical_out_dispatch(lr:readbuf,lo:mutbuf,ls:mutbuf,lw:mutbuf,ln:u64,le:u64,li:u64,lp:u64,lflag:u64,lop:u64)\n  header:u64 = const() {"value":32}\n  recsize:u64 = const() {"value":64}\n  recdelta:u64 = int.mul.wrap(li,recsize)\n  request_offset:u64 = int.add.wrap(header,recdelta)\n  written:u64 = call(lr,lo,ls,request_offset,lop,lp) {"function":"logical_out_command"}\n  jump b45(lr,lo,ls,lw,ln,le,li,lp,lflag,lop,written)\n'
    entry_new=entry_new.encode()
    raw=exactly(raw,entry_old,entry_new)
    anchor=b'fn predicate_command('
    raw=exactly(raw,anchor,helpers+anchor)
    declarations=b''
    if unit:
        for name,types in EXT.items():
            if not re.search(rb'^fn '+name.encode()+rb'\(',raw,re.M):
                declarations+=f'fn {name}({signature("p",types)}) -> (u64) effects(read,write,trap) external()\n'.encode()
        raw=exactly(raw,helpers,declarations+helpers)
    undone=exactly(exactly(raw,declarations+helpers,b''),entry_new,entry_old)
    if unit:
        raw=exactly(raw,b'exports(predicate_command,',b'exports(logical_out_layout,logical_out_apply,logical_out_command,predicate_command,')
        assert exactly(raw,b'exports(logical_out_layout,logical_out_apply,logical_out_command,predicate_command,',b'exports(predicate_command,')==exactly(undone,entry_old,entry_new).replace(b'fn predicate_command(',declarations+helpers+b'fn predicate_command(',1)
    return raw,undone

def main(source,dest):
    mb=(source/'manifest.json').read_bytes()
    assert hash_(mb)==BASEMAN
    m=json.loads(mb)
    orig=(source/'runtime.hepta').read_bytes()
    assert hash_(orig)==BASERT
    new,reverse=patch(orig,False)
    assert reverse==orig
    dest.mkdir(); (dest/'units').mkdir()
    (dest/'runtime.hepta').write_bytes(new)
    plan=dict(status='SOURCE_PREPARED_NOT_BUILT',base_manifest_sha256=BASEMAN,base_source_sha256=BASERT,opcode=OP,production_disabled=True,proposed_source_sha256=hash_(new),units=[])
    for unit in m['units']:
        path=unit['path']; before=(source/path).read_bytes()
        assert hash_(before)==unit['sha256']
        if unit['name']=='runtime800_12':
            after,reversed_=patch(before,True)
            assert reversed_==before
            for fn in ('logical_out_layout','logical_out_apply','logical_out_command','entry'):
                pattern=rb'^fn '+fn.encode()+rb'\([^\n]*\n.*?^end\n'
                assert re.search(pattern,new,re.M|re.S)[0]==re.search(pattern,after,re.M|re.S)[0]
        else: after=before
        plan['units'].append(dict(path=path,source_sha256=hash_(after),source_changed=before!=after,expected_object_sha256=None if before!=after else unit['expected_object_sha256']))
        (dest/path).write_bytes(after)
    # Deliberately not an executable build manifest: new image/object pins do not exist.
    (dest/'source-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(plan['proposed_source_sha256'],next(u for u in plan['units'] if u['source_changed']))
if __name__=='__main__': main(*map(Path,sys.argv[1:]))
