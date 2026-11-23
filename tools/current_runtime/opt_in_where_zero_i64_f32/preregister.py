"""Frozen stdlib-only true rank-0 I64 + dimensional F32 preregistration."""
import hashlib,itertools,json,math,sys,zlib
from pathlib import Path
D=Path(__file__).parent
P=Path('/tmp/hepta-where-i64-f32-prereg')
sys.path.insert(0,str(D))
from read_old import Owner,frames,req,resp,sha,q
def bits(n):
    """Integer distance and ties to even; avoid any floating intermediate."""
    assert -(1<<63)<=n<1<<63
    a=abs(n)
    if a==0:return 0
    step=1<<max(a.bit_length()-24,0)
    sig,remain=divmod(a,step)
    if min((remain,sig%2,sig),(step-remain,(sig+1)%2,sig+1))[2]!=sig:sig+=1
    best=sig*step; exp=best.bit_length()-1
    return ((n<0)<<31)|((exp+127)<<23)|(((best<<23)>>exp)&0x7fffff)
assert bits((1<<62)+(1<<38)+1)==0x5e800001
assert bits(-(1<<63))==0xdf000000 and bits((1<<63)-1)==0x5f000000
WIDTH={0:4,1:1,2:8,4:8,5:2,6:2}
F=(0,0x80000000,0x00000001,0x80000001,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x7f812345,0xff812345,0x007fffff,0x00800000)
def tensor(tag,sh,words):
    assert len(words)==math.prod(sh)
    w=WIDTH[tag]; xx=[x&((1<<(w*8))-1) for x in words]
    return [tag,sh,xx]
def layout(shs):
    r=max(map(len,shs));padded=[[1]*(r-len(s))+s for s in shs];d=[]
    for a in zip(*padded,strict=True):
        v=set(a)-{1}
        if len(v)>1:return None
        d.append(next(iter(v),1))
    return d
def load(a,c):
    _,sh,w=a;p=0
    for s,k in zip(sh,c[len(c)-len(sh):],strict=True):p=p*s+(0 if s==1 else k)
    return w[p]
def calc(m,x,y):
    sh=layout([j[1] for j in (m,x,y)])
    if sh is None:return None
    stride=[math.prod(max(1,d) for d in sh[i+1:]) for i in range(len(sh))]
    out=[]
    for c in itertools.product(*(range(d) for d in sh)):
        picked=x if load(m,c) else y
        w=load(picked,c)
        if picked[0]==2:w=bits(w-(1<<64) if w>>63 else w)
        assert picked[0] in (0,2)
        out.append(w)
    return [sh,stride,out]
def case(name,m,x,y):
    result=calc(m,x,y);assert result and result[2]
    assert len(x[1])!=0 if x[0]==0 else x[1]==[]
    assert len(y[1])!=0 if y[0]==0 else y[1]==[]
    return dict(label=name,args=[m,x,y],expected=result)
ex=(P/'expectation.json').read_bytes()
assert sha(ex)=='f141705d3a987ec01e08a031bec72a384706f0fd1ec2e59c0ba5167141db7aba'
prior=json.loads(ex)
values=prior['positives'][0]['x']['words'];traps=prior['direct_double_rounding_traps']
assert len(values)==216 and len(traps)==20
assert all(bits(a['i64'])==a['direct_f32'] for a in traps)
assert {(t['i64']%(1<<64)) for t in traps}<=set(values)
def ordered(flat,scalar,rev):return (flat,scalar) if rev else (scalar,flat)
rows=[];neg=[]
for rev in (0,1):
    for i,v in enumerate(values):
        # Mask adds an independent outer axis; both sources contribute nonempty raw words.
        vfloat=tensor(0,[1,len(F)],F);scalar=tensor(2,[],[v])
        rows.append(case(f'ties-r{rev}-{i}',tensor(1,[2,1,1],[1-rev,rev]),*ordered(vfloat,scalar,rev)))
    for flag in (0,1):
        rows.append(case(f'scalar-mask-{flag}-r{rev}',tensor(1,[],[flag]),*ordered(tensor(0,[1],[0xff812345]),tensor(2,[],[1<<63]),rev)))
    for m,l,s,lab in (
      (tensor(1,[],[1]),tensor(2,[1],[1]),tensor(0,[],[F[8]]),'reverse-role'),
      (tensor(1,[],[1]),tensor(2,[],[1]),tensor(0,[],[F[8]]),'both-zero'),
      (tensor(1,[0,1,1],[]),tensor(0,[1,1,4],F[:4]),tensor(2,[],[1]),'empty-mask'),
      (tensor(1,[1,1],[1]),tensor(0,[2,0],[]),tensor(2,[],[1]),'empty-value'),
      (tensor(1,[2],[1,0]),tensor(0,[3],[F[8],F[7],F[10]]),tensor(2,[],[1]),'incompatible'),
      (tensor(1,[],[1]),tensor(0,[1],[F[8]]),tensor(5,[],[0xfc39]),'exclude-f16-scalar'),
      (tensor(1,[],[1]),tensor(0,[1],[F[8]]),tensor(6,[],[0xff89]),'exclude-bf16-scalar'),
      (tensor(1,[],[1]),tensor(0,[1],[F[8]]),tensor(4,[],[0xfff0123456789abc]),'exclude-f64-narrowing'),
    ):
        neg.append(dict(label=f'{lab}-r{rev}',args=[m,*ordered(l,s,rev)],proposed_native_status=7))
    for t,w in ((5,0x7c35),(6,0x7f85)):
        neg.append(dict(label=f'exclude-dim{t}-r{rev}',args=[tensor(1,[],[1]),*ordered(tensor(t,[1],[w]),tensor(2,[],[1]),rev)],proposed_native_status=7))
p=P/'candidate'; report=json.loads((p/'report.json').read_bytes())
rr=list(frames(p/'requests.frames.bin',b'HTOWNR01'));rs=list(frames(p/'responses.frames.bin',b'HTOWNS01'))
old=[]
for e in report['rows']:
    if e.get('kind')!='refusal' or not e['label'].startswith('rank0-'):continue
    idx=e['frame']-1; cmd,=req(rr[idx])
    assert cmd[0]==129 and cmd[4:]==(0,0,0,0) and resp(rs[idx])==(7,[])
    assert e['pre']==e['post'] and sha(rr[idx])==e['request_sha256']
    a=e['pre']; snapshot=zlib.decompress((p/'owners'/(a['sha256']+'.zlib')).read_bytes())
    assert len(snapshot)==a['length'] and sha(snapshot)==a['sha256']
    own=Owner(snapshot);args=[]
    for h in cmd[1:4]:
        desc=own.node(h);tag=desc[72];dims=list(desc[12:12+desc[0]]);raw=own.storage(desc[27])[2];w=WIDTH[tag]
        assert desc[83]==desc[29]==desc[4]==desc[5]==0
        args.append(tensor(tag,dims,[int.from_bytes(raw[i:i+w],'little') for i in range(0,len(raw),w)]))
    m,x,y=args; ft=x if x[0]==0 else y; it=x if x[0]==2 else y
    assert m[0]==1 and ft[0]==0 and it[0]==2
    q0=dict(source_frame=idx+1,label=e['label'],args=args,request_sha256=sha(rr[idx]),response_sha256=sha(rs[idx]),owner_sha256=sha(snapshot),old_status=7)
    if ft[1] and not it[1]:
        row=case('historical-'+e['label'],*args)
        rows.append(row);q0['retire_only_if_physically_matched']=row['label']
    else:q0['proposed_native_status']=7
    old.append(q0)
assert len(rows)==438 and len(neg)==20 and len(old)==6 and sum('retire_only_if_physically_matched' in a for a in old)==2
body=dict(parent_head='aedb958dd9ed3b86bc1b45886442f2099fdac4c8',parent_tree='4eadcda7cad57e975f7d27193cd4be08f150e078',
 parent_runtime='400cd399360d1b846fc043932766ef9e121dfdb96795fc7bfde240fb0ec73e9d',
 shape_and_data_predictions_only=True,format_tensor='[public_dtype_tag,dimensions,list of unsigned raw words]',
 prior_rounding_digest=sha(ex),positive=rows,negatives=neg,archived=old,
 archive_identity=dict(elf=report['elf'],requests=sha((p/'requests.frames.bin').read_bytes()),responses=sha((p/'responses.frames.bin').read_bytes())),
 distinct_scalar_raw=values,traps=traps)
print(json.dumps(body,sort_keys=True,separators=(',',':')))
