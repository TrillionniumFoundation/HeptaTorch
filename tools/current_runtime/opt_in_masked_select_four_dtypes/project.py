"""Pure projection of four remaining public value dtypes into exact #67 masked_select."""
import hashlib,json,sys,importlib.util,re
from pathlib import Path
p,mp,check,dst=map(Path,sys.argv[1:])
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha(check.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checker',check);B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
assert sha(mp.read_bytes())=='6119fea38b648e70b3b8fcea78632c39e8336a6c9899a20786b2a27d59b189e4'
m=json.loads(mp.read_bytes(),object_pairs_hook=B.unique_object);r,u,_=B.check_sources(m,p,B.read_file)
assert sha(r)=='c5198ea3ddbbfbc3aeff244997dfa05660dba1c98d05edcf42d18c899b40768f'
files={x['path']:v for x,v,_ in u}
unit=next(x for x in m['units'] if x['name']=='runtime775_11'); path=unit['path']; u0=files[path]
assert sha(u0)=='403927005c5db896efaaa16216653f420cb5b0ba367224c3d2b7cc0fdcd32825'
A=b'''  xt_ok:bool = bool.or(xt_f32,xt_i64)
  value_width:u64 = select(xt_i64,e8,four)
'''
BOTH=b'''  f16_tag:u64 = const() {"value":5}
  bf16_tag:u64 = const() {"value":6}
  xt_bool:bool = compare.eq(xt,one)
  xt_f64:bool = compare.eq(xt,four)
  xt_f16:bool = compare.eq(xt,f16_tag)
  xt_bf16:bool = compare.eq(xt,bf16_tag)
  reduced_ok:bool = bool.or(xt_f16,xt_bf16)
  wide_ok:bool = bool.or(xt_i64,xt_f64)
  small_ok:bool = bool.or(xt_bool,reduced_ok)
  word_ok:bool = bool.or(xt_f32,wide_ok)
  xt_ok:bool = bool.or(word_ok,small_ok)
  value_wide:u64 = select(wide_ok,e8,four)
  value_red:u64 = select(reduced_ok,i64_tag,value_wide)
  value_width:u64 = select(xt_bool,one,value_red)
'''
C=b'''  branch rd_ok store(ds,dx,dm,dn,dk,do,dj,dnext,raw,dmode,dwidth) already_failed(ds)
'''
D=b'''  d_one:u64 = const() {"value":1}
  bool_value:bool = compare.eq(dwidth,d_one)
  need_bool:bool = bool.and(rd_ok,bool_value)
  branch need_bool selected_bool(ds,dx,dm,dn,dk,do,dj,dnext,raw,dmode,dwidth) selected_word(ds,dx,dm,dn,dk,do,dj,dnext,raw,dmode,dwidth,rd_ok)
block selected_word(ws1:mutbuf,wx1:u64,wm1:u64,wn1:u64,wk1:u64,wo1:u64,wj1:u64,wix1:u64,raw1:u64,mode1:u64,width1:u64,read1:bool)
  branch read1 store(ws1,wx1,wm1,wn1,wk1,wo1,wj1,wix1,raw1,mode1,width1) already_failed(ws1)
block selected_bool(vs1:mutbuf,vx1:u64,vm1:u64,vn1:u64,vk1:u64,vo1:u64,vj1:u64,vix1:u64,raw2:u64,mode2:u64,width2:u64)
  v_one:u64 = const() {"value":1}
  value_bit_ok:bool = compare.le(raw2,v_one)
  branch value_bit_ok store(vs1,vx1,vm1,vn1,vk1,vo1,vj1,vix1,raw2,mode2,width2) invalid(vs1)
'''
def apply(v,pairs):
 for a,b in pairs:
  assert v.count(a)==1 and v.count(b)==(1 if a in b else 0)
  v=v.replace(a,b,1)
 return v
R=apply(r,((A,BOTH),(C,D)));U=apply(u0,((A,BOTH),(C,D)))
assert apply(R,((BOTH,A),(D,C)))==r and apply(U,((BOTH,A),(D,C)))==u0
old=B.parse_source(r); new=B.parse_source(R)
assert old['signatures']==new['signatures'] and set(old['definitions'])==set(new['definitions'])
assert [f for f in old['definitions'] if old['definitions'][f]!=new['definitions'][f]]==['ms_tensor']
f=new['definitions']['ms_tensor'].decode(); args=lambda x:x.split(',') if x else []
sig=re.match(r'^fn \w+\(([^\n]*)\) -> .* entry\((\w+)\)\nblock (\w+)\(([^\n]+)\)\n',f)
assert sig and sig[1]==sig[4] and sig[2]==sig[3]
ids=re.findall(r'\b(\w+):(?:u64|bool|readbuf|mutbuf|usize)\b',f.split('\n',1)[1]);assert len(ids)==len(set(ids))
targets={mm[1]:len(args(mm[2])) for mm in re.finditer(r'^block (\w+)\(([^)]*)\)',f,re.M)}
for t in re.findall(r'^  (?:branch|jump) ([^\n]+)',f,re.M):
 for g in re.finditer(r'(\w+)\(([^)]*)\)',t):assert targets[g[1]]==len(args(g[2])),g[0]
assert '_ = call(' not in f and f.count('{"function":"ms_bindex"}')==2
assert f.count('{"function":"typed_store"}')==1 and 'call(ts,target,twidth,traw)' in f
assert f.count('{"function":"typed_descriptor"}')==1 and 'call(as,a1,ap0,a0,a1,value_tag)' in f
assert 'call(vs,vo,vk,vtag,vwidth)' in f and 'call(os,ox,owidth)' in f and 'call(os,om,ow1)' in f
assert 'mt_ok:bool = compare.eq(mt,one)' in f and 'call(ds,dx,di) {"function":"tget"}' in f
assert b'call(mrr,mro,mrs,mrat,mrop,mrp)' in new['definitions']['entry']
assert B.parse_source(U,unit=True)['definitions']['ms_tensor']==new['definitions']['ms_tensor']
m.pop('expected_native_sha256');unit.pop('expected_object_sha256')
unit['sha256']=sha(U);m['name']='opt_in_main_public_value_masked_select_one_way_bool_mask'
rp=m['runtime']['path'];m['runtime'].update(sha256=sha(R),**{k:new[k] for k in ('functions','blocks','instructions')})
files.update({rp:R,path:U});dst.mkdir();(dst/'units').mkdir()
for k,v in files.items():(dst/k).write_bytes(v)
(dst/'manifest.json').write_bytes(B.json_bytes(m));B.check_sources(m,dst,B.read_file)
print(json.dumps(dict(head='c08694a893c76da9c898df4e018b864ec0b1f124',tree='ea4a8114ab9cdb688b3f081bf0f697d58d78578b',
 runtime=sha(R),unit11=sha(U),manifest=sha(B.json_bytes(m)),inverse=True,static_split=True,actual_native=False),sort_keys=True))
