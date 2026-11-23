"""Static current800 projection only. No runtime or build invocation."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
base,add,bfile,author,out=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(bfile.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
assert sha(author.read_bytes())=='27ca2a8a6da366214cd4d611f0c1aeffff993f325db87b158ca67f2916f96cfa'
sp=importlib.util.spec_from_file_location('pinned_check',bfile);B=importlib.util.module_from_spec(sp);sp.loader.exec_module(B)
def load(path,manifest,combined,unit_tag,unit_hash):
 rawm=(path/'manifest.json').read_bytes();assert sha(rawm)==manifest
 m=json.loads(rawm,object_pairs_hook=B.unique_object)
 ra,ls,_=B.check_sources(m,path,B.read_file);assert sha(ra)==combined
 u={e['name']:(e,a,b) for e,a,b in ls};assert sha(u[unit_tag][1])==unit_hash
 return m,ra,B.parse_source(ra),u
base_manifest='7a87caa6e0c8def3578708fb2d727f53bb689597e9661dcc591cd972a945cb02'
donor_manifest='e0c5ecefb81c93007edbef5e0c5e17f71a22ab840a82cbfe5ca505b1751123dd'
m,r,M,units=load(base,base_manifest,'43999e8e9afa40b78f24007697623d5a297308643bff160392fec49d388fe26b','runtime800_12','f0da1fd432a2a0c80d01873c8dc4cbb8b7f2932a1c5bf7f0dc76bb7634b6d2e2')
_,donor,D,du=load(add,donor_manifest,'d64f15921b169ed48199356989aa1f31dc7937fed1f772e27e0e291b9717cc1f','runtime775_11','87e4f20e717ef3290329dd4aa7460a5b7184072369add9538139c2a4ecaa5dc0')
assert sha((add/'projection.json').read_bytes())=='000a600b2a48df8bd8cf78285a50c08cfb1d1119031f9ce8fe7d46503fc3bb8c'
p=json.loads((add/'projection.json').read_bytes())
u,orig_u,U=units['runtime800_12']; baseline=M['definitions']
extra={n:v for n,v in D['definitions'].items() if n in p['exact_adopted_operator_definitions']}
assert len(extra)==39 and not extra.keys()&baseline.keys()
assert {k:sha(b) for k,b in extra.items()}==p['exact_adopted_operator_definitions']
dunit=du['runtime775_11'][2]
assert all(dunit['definitions'][n]==val and n in dunit['exports'] for n,val in extra.items())
extra_e=[n for n in dunit['exports'] if n in extra]
direct=set.union(*(D['calls'][n] for n in extra))-extra.keys()
assert len(direct)==21
providers={n:(e['name'],parts['signatures'][n],parts['definitions'][n]) for e,raw,parts in units.values() for n in parts['exports']}
assert all(M['definitions'][n]==D['definitions'][n] and M['signatures'][n]==D['signatures'][n]==providers[n][1] for n in direct)
closure=set(extra)
while (nxt:=closure|set.union(*(D['calls'].get(n,set()) for n in closure)))!=closure:closure=nxt
closure-=extra.keys()
assert len(closure)==81
assert {n for n in closure if M['definitions'][n]!=D['definitions'][n]}=={'so_tensor_live'}
extra_i={}
for n in sorted(direct):
 if n in U['signatures']:
  assert U['signatures'][n]==M['signatures'][n]
 elif providers[n][0]!='runtime800_12':
  h=providers[n][2].splitlines()[0]
  assert b' entry(' in h
  extra_i[n]=h.split(b' entry(',1)[0]+b' external()\n'
refs=('wkr','wko','wks','wkw','wkn','wke','wki','wkp','wkflag','wkop')
ts=('readbuf','mutbuf','mutbuf','mutbuf')+('u64',)*6
call=lambda n,ls:f'{n}('+','.join(ls)+')'
params=lambda p,tt:tuple(p+str(i) for i in range(len(tt)))
def label(n,vals,types):
 return 'block '+call(n,[f'{v}:{t}' for v,t in zip(vals,types,strict=True)])
gen=[
 '  co_header:u64 = const() {"value":32}','  co_width:u64 = const() {"value":64}',
 '  co_delta:u64 = int.mul.wrap(wki,co_width)','  co_at:u64 = int.add.wrap(co_header,co_delta)',
 '  co_c130:u64 = const() {"value":130}','  co_c131:u64 = const() {"value":131}',
 '  co_c133:u64 = const() {"value":133}','  co_c134:u64 = const() {"value":134}',
 '  co_c135:u64 = const() {"value":135}','  co_ms:bool = compare.eq(wkop,co_c130)',
 '  co_and:bool = compare.eq(wkop,co_c131)','  co_or:bool = compare.eq(wkop,co_c133)',
 '  co_xor:bool = compare.eq(wkop,co_c134)','  co_nz:bool = compare.eq(wkop,co_c135)',
 '  co_or_xor:bool = bool.or(co_or,co_xor)','  co_la:bool = bool.or(co_and,co_or_xor)',
 f'  branch co_ms {call("co_run_ms",refs+("co_at",))} {call("co_check_la",refs+("co_at","wka","co_la","co_nz"))}']
tail=ts+('u64','bool','bool','bool')
la=params('co_la_arg',tail)
gen.extend([label('co_check_la',la,tail),
 f'  branch {la[12]} {call("co_run_la",la[:11])} {call("co_check_nz",la[:12]+(la[13],))}'])
nt=ts+('u64','bool','bool')
n=params('co_nz_arg',nt)
gen.extend([label('co_check_nz',n,nt),f'  branch {n[12]} {call("co_run_nz",n[:11])} {call("co_original",n[:10]+(n[11],))}'])
ot=ts+('bool',);o=params('co_old_arg',ot)
gen.extend([label('co_original',o,ot),f'  branch {o[10]} {call("b13",o[:10])} {call("b14",o[:10])}'])
for pref in ('ms','la','nz'):
 z=params('co_'+pref+'_run',ts+('u64',));ret='co_'+pref+'_written'
 gen.extend([label('co_run_'+pref,z,ts+('u64',)),
 f'  {ret}:u64 = '+call('call',[z[i] for i in (0,1,2,10,9,7)])+' '+json.dumps({'function':pref+'_command'},separators=(',',':')),
 '  jump '+call('b45',z[:10]+(ret,))])
dsp=('\n'.join(gen)+'\n').encode()
old_branch=('  branch wka '+call('b13',refs)+' '+call('b14',refs)+'\n').encode()
old_entry=baseline['entry']
assert old_branch in old_entry.split(b'block where_keep',1)[1].split(b'block where_dispatch',1)[0]
def repl(s,a,b):
 assert s.count(a)==1,a[:130]
 return s.replace(a,b,1)
entry=repl(old_entry,old_branch,dsp)
anchor=next(v for n,v in baseline.items() if n!='entry')
ua=U['definitions'][U['exports'][0]];added=b''.join(extra.values())
exp0=b'exports('+','.join(U['exports']).encode()+b')\n'
exp1=b'exports('+','.join([*extra_e,*U['exports']]).encode()+b')\n'
newimports=b''.join(extra_i[n] for n in sorted(extra_i))
R=repl(repl(r,old_entry,entry),anchor,added+anchor)
Q=repl(repl(repl(orig_u,old_entry,entry),ua,added+ua),exp0,exp1+newimports)
assert repl(repl(R,added+anchor,anchor),entry,old_entry)==r
assert repl(repl(repl(Q,exp1+newimports,exp0),added+ua,ua),entry,old_entry)==orig_u
N=B.parse_source(R);S=B.parse_source(Q,unit=True)
assert set(N['definitions'])==set(baseline)|set(extra)
assert {n for n in baseline if N['definitions'][n]!=baseline[n]}=={'entry'}
assert all(N['definitions'][n]==S['definitions'][n]==v for n,v in extra.items())
assert M['definitions']['so_tensor_live']==N['definitions']['so_tensor_live']!=D['definitions']['so_tensor_live']
assert all(N['signatures'][n]==M['signatures'][n] for n in baseline)
# Lexical formal bindings and exact branch type checks on added labels and each adopted function.
author=author.read_text()
section=lambda n:author.split('def '+n+'(',1)[1].split('\ndef ',1)[0]
size=0; keywords=set()
for l in section('keyword').splitlines():
 if a:=re.search(r'if size == (\d+)',l): size=int(a[1])
 if a:=re.search(r'choose\((.*?), (\d+), result\)',l):
  w={k:int(v) for k,v in re.findall(r'([abcd]) == (\d+)',a[1])}
  keywords.add(b''.join(w.get(k,0).to_bytes(8,'little') for k in 'abcd')[:size].decode())
excl=set()
for kind in ('reserved_function','export_loader_hook'):
 for a in re.finditer(r'match\(s, begin, size, (\d+), (\d+), (\d+), (\d+), (\d+)\)',section(kind)):
  n,*ws=map(int,a.groups())
  excl.add(b''.join(w.to_bytes(8,'little') for w in ws)[:n].decode())
assert len(keywords)>=91 and not set(extra)&excl
def parse_params(s):return dict(q.split(':') for q in s.split(',')) if s else {}
def verify(body,n,allblock=False):
 s=body.decode()
 h=re.match(r'^fn (\w+)\(([^\n]*)\) -> .* entry\((\w+)\)\nblock (\w+)\(([^\n]*)\)\n',s)
 assert h and h[2]==h[5] and h[3]==h[4],n
 labels={}; vars_f={}; branch_checks=0; current=None
 for l in s.splitlines():
  if a:=re.fullmatch(r'block (\w+)\(([^)]*)\)',l):
   current=a[1]; assert current not in labels
   labels[current]=parse_params(a[2]);vars_f[current]=labels[current].copy()
  if a:=re.match(r'  (\w+):(\w+) = ',l):
   assert a[1] not in vars_f[current]
   vars_f[current][a[1]]=a[2]
 relevant={k for k in labels if allblock or k.startswith('co_')}
 typed_all=[a for k in vars_f.values() for a in k]
 fresh=[a for a in typed_all if allblock or a.startswith('co_')]
 assert len(fresh)==len(set(fresh)) and not (set(fresh)|relevant|({n} if allblock else set()))&keywords, (n,(set(fresh)|relevant|{n})&keywords)
 assert max(map(len,fresh+list(relevant)))<=128
 for l in s.splitlines():
  if a:=re.fullmatch(r'block (\w+)\([^)]*\)',l):current=a[1]
  if current not in relevant and not (n=='entry' and current=='where_keep' and ('co_ms ' in l or 'call(' in l)):continue
  if l.startswith(('  jump ','  branch ')):
   for dest,raw in re.findall(r'(\w+)\(([^)]*)\)',l):
    actual=raw.split(',') if raw else []
    assert [vars_f[current][a] for a in actual]==list(labels[dest].values()),(n,l)
    branch_checks+=1
  if a:=re.fullmatch(r'  (\w+):(\w+) = call\(([^)]*)\) \{"function":"(\w+)"\}',l):
   actual=a[3].split(',') if a[3] else []
   sig=N['signatures'][a[4]]
   assert [vars_f[current][w].encode() for w in actual]==list(sig[0]) and [a[2].encode()]==list(sig[1]),(n,l)
  assert '_ = call(' not in l
 return branch_checks
tested=[*extra,'entry']
typed=sum(verify(N['definitions'][n],n,n!='entry') for n in tested)
effects=0
for n in tested:
 own=set(N['signatures'][n][2])
 for dst in N['calls'][n]:
  assert set(N['signatures'][dst][2])<=own
  effects+=1
m['name']='opt_in_current800_where_and_ordinary_coexistence'
m.pop('expected_native_sha256',None)
u.pop('expected_object_sha256',None)
u['sha256']=sha(Q)
rp=m['runtime']['path'];m['runtime'].update(sha256=sha(R),**{k:N[k] for k in ('functions','blocks','instructions')})
source={e['path']:a for e,a,_ in units.values()}
source[u['path']]=Q;source[rp]=R
def bind(f,d):
 b=source[str(f.relative_to('/source-only-current800'))]
 assert sha(b)==d
 return b
_,_,summary=B.check_sources(m,Path('/source-only-current800'),bind)
stats={'source_bytes':len(Q),'functions':len(S['signatures']),'blocks':S['blocks'],'instructions':S['instructions'],'max_block_args':max(len(z.split(b',')) if z else 0 for z in re.findall(rb'^block \w+\(([^)]*)\)',Q,re.M))}
out.mkdir(exist_ok=False)
for path,raw in source.items():
 f=out/path;f.parent.mkdir(exist_ok=True,parents=True);f.write_bytes(raw)
manifest=B.json_bytes(m);(out/'manifest.json').write_bytes(manifest)
result=dict(base_manifest=base_manifest,donor_manifest=donor_manifest,base_git='7583ea54756f6c76f722b95b2e569f417d584fc4',donor_git='b1e1fd20de94fa12068ce1e33da2bc7c6f1f8026',runtime=sha(R),unit12=sha(Q),manifest=sha(manifest),source_check='PASS',static_branch_edges=typed,effect_pairs=effects,keyword_count=len(keywords),unit12_stats=stats,limits=B.LIMITS,original_so_tensor_live=sha(M['definitions']['so_tensor_live']),bytes_inverse=True,unchanged_units=len(units)-1,added_definition_hashes={n:sha(v) for n,v in extra.items()},new_imports=sorted(extra_i),routes={130:'ms_command',131:'la_command',133:'la_command',134:'la_command',135:'nz_command'},existing_routes_126_129_preserved=True,other_opcodes_original_fallback=True)
(out/'projection.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('runtime','unit12','manifest','unit12_stats','new_imports','static_branch_edges','effect_pairs')}))
