"""Exact source-only projection: scoped I64 value dtype for masked_select."""
import re,hashlib,sys,json,importlib.util
from pathlib import Path
src,checker,dst,manifest=map(Path,sys.argv[1:])
sha=lambda x:hashlib.sha256(x).hexdigest()
def repl(v,a,c):
 assert v.count(a)==1,a
 return v.replace(a,c,1)
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checker',checker); B=importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
raw=manifest.read_bytes(); assert sha(raw)=='c7340a61bb2bb2456589828e686404e06eee475e8aef2c8702f730a385c155d1'
m=json.loads(raw,object_pairs_hook=B.unique_object);r,units,_=B.check_sources(m,src,B.read_file)
assert sha(r)=='7fea1364d10781f1b8f8c8d7f62973e6247d8d708b8af80a831e503e953b9960'
unit=next(u for u in m['units'] if u['name']=='runtime775_11');up=unit['path']
rows={u['path']:v for u,v,_ in units}; ou=rows[up]
assert sha(ou)=='cb68dae178ec76344bb72f3c2d328a8a72c05d88d8ed052ee6bf19f17250f5a7'
O=B.parse_source(r);old=O['definitions']['ms_tensor'];core=old.decode()
core=repl(core,'  xt_ok:bool = compare.eq(xt,z)\n','''  i64_tag:u64 = const() {"value":2}
  xt_f32:bool = compare.eq(xt,z)
  xt_i64:bool = compare.eq(xt,i64_tag)
  xt_ok:bool = bool.or(xt_f32,xt_i64)
  value_width:u64 = select(xt_i64,e8,four)
''')
core=repl(core,'shape(s,x,m,xd,md,e8)','shape(s,x,m,xd,md,e8,value_width)')
core=repl(core,'''  ow4:u64 = const() {"value":4}
''','')
core=repl(core,'call(os,ox,ow4)','call(os,ox,owidth)')
core=repl(core,'  out:u64 = call(as,a1,ap0,a0,a1,a0) {"function":"typed_descriptor"}', '''  value_tag:u64 = call(as,ax) {"function":"tensor_dtype"}
  out:u64 = call(as,a1,ap0,value_tag,a1,a0) {"function":"typed_descriptor"}''')
core=repl(core,'inspect(as,ax,am,an,ak,out,amode)','inspect(as,ax,am,an,ak,out,amode,value_tag)')
core=repl(core,'vo:u64,vmode:u64)','vo:u64,vmode:u64,vtag:u64)')
core=repl(core,'call(vs,vo,vk) {"function":"ms_output"}','call(vs,vo,vk,vtag,vwidth) {"function":"ms_output"}')
core=repl(core,'''  t4:u64 = const() {"value":4}
''','')
core=repl(core,'call(ts,target,t4,traw)','call(ts,target,twidth,traw)')
mode={'shape':'swidth','expand_mask':'uwidth','verify_shape':'vgwidth','cap':'cwidth','owner':'owidth','count':'cwidth0','bit':'bwidth',
'bit_load':'bbwidth','allocate':'awidth','inspect':'vwidth','ready':'rwidth','copy':'pwidth','choose':'hwidth',
'choose_load':'hcwidth','selected':'jwidth','room':'mwidth','read_value':'dwidth','store':'twidth'}
block=None;lines=[]
for l in core.splitlines(keepends=True):
 if q:=re.match(r'^block (\w+)\(',l):
  block=q[1]
  if block in mode:l=l.replace(')',','+mode[block]+':u64)',1)
 elif block in mode and re.match(r'^  (branch|jump) ',l):
  for dest in mode:l=re.sub(r'\b'+dest+r'\(([^)]*)\)',lambda q:dest+'('+q[1]+','+mode[block]+')',l)
 lines.append(l)
core=''.join(lines)
preout=O['definitions']['ms_output'];out=preout.decode()
out=repl(out,'fn ms_output(s:mutbuf,o:u64,k:u64)', 'fn ms_output(s:mutbuf,o:u64,k:u64,want_tag:u64,byte_width:u64)')
out=repl(out,'block check(s:mutbuf,o:u64,k:u64)','block check(s:mutbuf,o:u64,k:u64,want_tag:u64,byte_width:u64)')
out=repl(out,'''  four:u64 = const() {"value":4}
''','')
out=repl(out,'call(s,o,four)','call(s,o,byte_width)')
out=repl(out,'meta(s,o,k)','meta(s,o,k,want_tag,byte_width)')
out=repl(out,'block meta(ms:mutbuf,mo:u64,mk:u64)','block meta(ms:mutbuf,mo:u64,mk:u64,mtag:u64,mwidth:u64)')
out=repl(out,'''  m4:u64 = const() {"value":4}
''','')
out=repl(out,'int.mul.wrap(mk,m4)','int.mul.wrap(mk,mwidth)')
out=repl(out,'  isf:bool = compare.eq(dt,m0)','  tag_ok:bool = compare.eq(dt,mtag)')
out=repl(out,'bool.and(isf,rank)','bool.and(tag_ok,rank)')
pairs=[(old,core.encode()),(preout,out.encode())]
def rewrite(v,inverse=False):
 for a,b in pairs:
  if inverse:a,b=b,a
  v=repl(v,a,b)
 return v
R=rewrite(r);U=rewrite(ou);assert rewrite(R,True)==r and rewrite(U,True)==ou
N=B.parse_source(R)
assert set(N['definitions'])==set(O['definitions'])
assert {k for k in O['definitions'] if O['definitions'][k]!=N['definitions'][k]}=={'ms_tensor','ms_output'}
assert {k for k in O['signatures'] if O['signatures'][k]!=N['signatures'][k]}=={'ms_output'}
assert R.count(b'"function":"ms_output"')==1
assert R.count(b'"value":130')==1
assert b'call(mrr,mro,mrs,mrat,mrop,mrp)' in N['definitions']['entry']
assert 'call(ts,target,twidth,traw)' in core and 'call(ds,dx,di) {"function":"tget"}' in core
assert 'call(as,a1,ap0,value_tag,a1,a0)' in core
assert 'compare.eq(element_stride,m1)' in out and 'int.mul.wrap(mk,mwidth)' in out
for key in ('ms_tensor','ms_output'):
 f=N['definitions'][key].decode()
 args=lambda s:s.split(',') if s else []
 sig=re.match(r'^fn \w+\(([^\n]*)\) -> .* entry\((\w+)\)\nblock (\w+)\(([^\n]+)\)\n',f)
 assert sig and sig[1]==sig[4] and sig[2]==sig[3]
 vs=re.findall(r'\b(\w+):(?:u64|bool|readbuf|mutbuf|usize)\b',f.split('\n',1)[1])
 assert len(vs)==len(set(vs)),(key,[x for x in vs if vs.count(x)>1])
 targets={mm[1]:len(args(mm[2])) for mm in re.finditer(r'^block (\w+)\(([^)]*)\)',f,re.M)}
 for t in re.findall(r'^  (?:branch|jump) ([^\n]+)',f,re.M):
  for g in re.finditer(r'(\w+)\(([^)]*)\)',t):assert targets[g[1]]==len(args(g[2])),(key,g[0])
 assert '_ = call(' not in f
m.pop('expected_native_sha256');unit.pop('expected_object_sha256')
unit['sha256']=sha(U); m['name']='opt_in_main_f32_i64_masked_select_one_way_bool_mask'
rp=m['runtime']['path'];m['runtime'].update(sha256=sha(R),**{k:N[k] for k in ('functions','blocks','instructions')})
rows.update({up:U,rp:R})
dst.mkdir();(dst/'units').mkdir()
for k,v in rows.items():(dst/k).write_bytes(v)
(dst/'manifest.json').write_bytes(B.json_bytes(m));B.check_sources(m,dst,B.read_file)
print(json.dumps(dict(parent='695c770273ef7a1260272d54c14610dbb0e282cf',tree='48c5bdcc1aa7c8eba6983140e8f4514f558d9519',
 runtime=sha(R),unit11=sha(U),manifest=sha(B.json_bytes(m)),inverse=True,static_split=True),sort_keys=True))
