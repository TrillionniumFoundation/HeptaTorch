"""Compose pinned operator definitions and audit source only; never launch native tools."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
base,masked,logic,nz,bfile,author,out=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(bfile.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
assert sha(author.read_bytes())=='27ca2a8a6da366214cd4d611f0c1aeffff993f325db87b158ca67f2916f96cfa'
spec=importlib.util.spec_from_file_location('pinned_source_check',bfile)
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
pins=[
('main','e9444902df9878060ab389a70ee957c1b7f4a887','0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43','8095595a1859fb2973f02bb567d56b4bc960c5b4666c8914f8a53079adfe7db9'),
('ms','270fde9f8b597c1444fd711026121fef4845b36d','877bd9b77c4cf57cd67549c9e9a7ff6a5f4fb87c829587dab77e46347c1c2b3c','0f01d8ef3d66325746c56a9661a2813157315bee7a99d1a9de0ee02b658e06b5'),
('la','dc4c9fd5efb9526241ea21ac83a4fa5b9196570f','a9a5d93a97709069d57109aa073775c6748f53a26aa3b7eb51b92b55f02658bc','35ffb91aee54acf9a0682eecca624bc1f786cba8790897e8ca384bdc56b8f913'),
('nz','2b5b2fcd8b7937d5441317da1becf9cbef360a1b','8420d6599d484e2b6844576a0f60207d6b318cb21bda269f0d60bf135bac10db','8d151d5b8b8fe40e04bafdfeb3e373bd54b1a43156438898a6a9ffd778d08ce3')]
def load(d,pin):
 m=json.loads((d/'manifest.json').read_bytes(),object_pairs_hook=B.unique_object)
 raw,us,_=B.check_sources(m,d,B.read_file)
 units={e['name']:(e,r,p) for e,r,p in us}
 z=units['runtime775_11'][1]
 assert [sha(raw),sha(z)]==list(pin[2:])
 return m,raw,B.parse_source(raw),units
results=[load(d,p) for d,p in zip((base,masked,logic,nz),pins,strict=True)]
m,r,M,units=results[0]
u,orig_u,U=units['runtime775_11']
baseline=M['definitions']; extra={}; extra_e={}; extra_i={}
for (pref,*_),(_,_,P,other) in zip(pins[1:],results[1:],strict=True):
 for k in set(other)-{'runtime775_11'}:
  assert other[k][1]==units[k][1]
 assert set(other)==set(units)
 assert {n for n in baseline if P['definitions'][n]!=baseline[n]}=={'entry'}
 new={n:v for n,v in P['definitions'].items() if n not in baseline}
 assert new and all(n.startswith(pref+'_') for n in new) and not new.keys()&extra.keys()
 split=other['runtime775_11'][2]
 assert all(split['definitions'][n]==v for n,v in new.items())
 extra.update(new);extra_e.update((n,None) for n in split['exports'] if n in new)
 assert set(split['exports'])-set(U['exports'])==new.keys()
 for n,sig in split['imports'].items():
  if n in U['imports']: assert U['imports'][n]==sig
  else:
   line=re.search(rb'^fn '+n.encode()+rb'\([^\n]*external\(\)\n',other['runtime775_11'][1],re.M)[0]
   assert n not in extra_i or extra_i[n]==line
   extra_i[n]=line
 assert set(U['imports'])<=set(split['imports'])
 assert ('{"function":"'+pref+'_command"}').encode() in P['definitions']['entry']
refs=('v119','v120','v121','v122','v124','v125','v126','v127','v128','v139')
ts=('readbuf','mutbuf','mutbuf','mutbuf')+('u64',)*6
call=lambda n,ls:f'{n}('+','.join(ls)+')'
params=lambda p,tt:tuple(p+str(i) for i in range(len(tt)))
def label(n,vals,types):
 return 'block '+call(n,[f'{v}:{t}' for v,t in zip(vals,types,strict=True)])
gen=[
 '  co_c130:u64 = const() {"value":130}','  co_c131:u64 = const() {"value":131}',
 '  co_c133:u64 = const() {"value":133}','  co_c134:u64 = const() {"value":134}',
 '  co_c135:u64 = const() {"value":135}','  co_ms:bool = compare.eq(v139,co_c130)',
 '  co_and:bool = compare.eq(v139,co_c131)','  co_or:bool = compare.eq(v139,co_c133)',
 '  co_xor:bool = compare.eq(v139,co_c134)','  co_nz:bool = compare.eq(v139,co_c135)',
 '  co_or_xor:bool = bool.or(co_or,co_xor)','  co_la:bool = bool.or(co_and,co_or_xor)',
 f'  branch co_ms {call("co_run_ms",refs+("v138",))} {call("co_check_la",refs+("v138","v144","co_la","co_nz"))}']
tail=ts+('u64','bool','bool','bool')
la=params('co_la_arg',tail)
gen.extend([label('co_check_la',la,tail),
 f'  branch {la[12]} {call("co_run_la",la[:11])} {call("co_check_nz",la[:12]+(la[13],))}'])
nt=ts+('u64','bool','bool')
n=params('co_nz_arg',nt)
gen.extend([label('co_check_nz',n,nt), f'  branch {n[12]} {call("co_run_nz",n[:11])} {call("co_original",n[:10]+(n[11],))}'])
ot=ts+('bool',);o=params('co_old_arg',ot)
gen.extend([label('co_original',o,ot),f'  branch {o[10]} {call("b13",o[:10])} {call("b14",o[:10])}'])
for pref in ('ms','la','nz'):
 z=params('co_'+pref+'_run',ts+('u64',)); ret='co_'+pref+'_written'
 gen.extend([label('co_run_'+pref,z,ts+('u64',)),
  f'  {ret}:u64 = '+call('call',[z[i] for i in (0,1,2,10,9,7)])+' '+json.dumps({'function':pref+'_command'},separators=(',',':')),
  '  jump '+call('b45',z[:10]+(ret,))])
dsp=('\n'.join(gen)+'\n').encode()
old_branch=('  branch v144 '+call('b13',refs)+' '+call('b14',refs)+'\n').encode()
old_entry=baseline['entry']
def repl(data,a,b):
 assert data.count(a)==1,a[:180]
 return data.replace(a,b,1)
entry=repl(old_entry,old_branch,dsp)
anchor=next(v for n,v in baseline.items() if n!='entry')
ua=U['definitions'][U['exports'][0]]
added=b''.join(extra.values())
R=repl(repl(r,old_entry,entry),anchor,added+anchor)
exp0=b'exports('+','.join(U['exports']).encode()+b')\n'
exp1=b'exports('+','.join([*extra_e,*U['exports']]).encode()+b')\n'
newline_imports=b''.join(extra_i[n] for n in sorted(extra_i))
Q=repl(repl(repl(orig_u,old_entry,entry),ua,added+ua),exp0,exp1+newline_imports)
assert repl(repl(R,added+anchor,anchor),entry,old_entry)==r
assert repl(repl(repl(Q,exp1+newline_imports,exp0),added+ua,ua),entry,old_entry)==orig_u
N=B.parse_source(R); S=B.parse_source(Q,unit=True)
assert set(N['definitions'])==set(baseline)|set(extra)
assert {n for n in baseline if N['definitions'][n]!=baseline[n]}=={'entry'}
assert all(N['definitions'][n]==S['definitions'][n]==v for n,v in extra.items())
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
  if current not in relevant and not (n=='entry' and current=='b12' and 'co_ms ' in l):continue
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
m['name']='opt_in_coexistence_masked_logical_nonzero'
del m['expected_native_sha256']
u.pop('expected_object_sha256')
u['sha256']=sha(Q)
rp=m['runtime']['path'];m['runtime'].update(sha256=sha(R),**{k:N[k] for k in ('functions','blocks','instructions')})
src={e['path']:b for e,b,p in units.values()}
src[u['path']]=Q;src[rp]=R
# Exercise the exact source checker through an in-memory read-only binder.
root=Path('/source-only-coexistence')
def bind(f,d):
 b=src[str(f.relative_to(root))]
 assert sha(b)==d
 return b
_,_,summary=B.check_sources(m,root,bind)
stats={'source_bytes':len(Q),'functions':len(S['signatures']),'blocks':S['blocks'],'instructions':S['instructions'],'max_block_args':max(len(z.split(b',')) if z else 0 for z in re.findall(rb'^block \w+\(([^)]*)\)',Q,re.M))}
out.mkdir(exist_ok=False)
for path,raw in src.items():
 dest=out/path;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(raw)
manifest=B.json_bytes(m);(out/'manifest.json').write_bytes(manifest)
result=dict(pins=pins,runtime=sha(R),unit11=sha(Q),manifest=sha(manifest),source_checker_hash=sha(bfile.read_bytes()),source_check='PASS',unit11_stats=stats,unit_limits=B.LIMITS,byte_exact_inverse=True,old_other_units_identical=len(units)-1,exact_adopted_operator_definitions={k:sha(v) for k,v in extra.items()},imports=sorted(extra_i),opcode_routes={130:'ms_command',131:'la_command',133:'la_command',134:'la_command',135:'nz_command'},unlisted_ops_original=True,response_position='old v127 argument; old b45 preserved',lexical_keywords=len(keywords),branch_type_edges_checked=typed,effect_pairs_checked=effects)
(out/'projection.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('runtime','unit11','manifest','unit11_stats','imports','branch_type_edges_checked')}))
