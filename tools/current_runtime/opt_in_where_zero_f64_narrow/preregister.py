"""Pure integer references for rank-0 F64→F32. Native x86 NaN mapping is a hypothesis."""
import itertools,json,math,sys,zlib
from pathlib import Path
D=Path(__file__).parent
sys.path.insert(0,str(D))
from read_old import Owner,frames,req,resp,sha
F64MASK=(1<<52)-1
F=(0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x7f812345,0xff812345)
W={0:4,1:1,2:8,4:8,5:2,6:2}
def rn(a,b):
    q,r=divmod(a,b)
    return min((r,q&1,q),(b-r,(q+1)&1,q+1))[2]
def exp_frac(v):return (v>>52)&0x7ff,v&F64MASK
def isnan(v): e,f=exp_frac(v);return e==0x7ff and f!=0
def narrow(v,mode):
    sign=(v>>63)<<31;e,f=exp_frac(v)
    if e==0x7ff:
        if not f:return sign|0x7f800000
        if mode=='local-hia-warning':return sign|0x7fc00000
        assert mode=='x86-payload-hypothesis'
        return sign|0x7fc00000|(f>>29)
    if not e and not f:return sign
    a=((1<<52) if e else 0)|f
    power=(e or 1)-1023-52
    actual_exp=power+a.bit_length()-1
    wanted=max(-149,actual_exp-23)
    scale=power-wanted
    mant=(a<<scale) if scale>=0 else rn(a,1<<(-scale))
    if mant==1<<24:mant>>=1;wanted+=1
    if mant==0:return sign
    if mant<1<<23:return sign|mant
    e_after=wanted+23
    if e_after>=128:return sign|0x7f800000
    return sign|((e_after+127)<<23)|(mant-(1<<23))
for v,w in {0x3ff0000010000000:0x3f800000,0x3ff0000030000000:0x3f800002,0x3690000000000000:0,
            0x3690000000000001:1,0x380fffffe0000000:0x00800000,0x47effffff0000000:0x7f800000,
            0x47efffffefffffff:0x7f7fffff,0x7ff0000000000000:0x7f800000,
            0x7ff8123456789abc:0x7fc091a2,0xfff0123456789abc:0xffc091a2}.items():
    assert narrow(v,'x86-payload-hypothesis')==w
def t(tag,shape,raw):
    assert len(raw)==math.prod(shape)
    return [tag,shape,raw]
def canonical(shape):return [math.prod(max(1,v) for v in shape[i+1:]) for i in range(len(shape))]
def result(label,args,mode):
    m,x,y=args; rr=[a[1] for a in args]; rank=max(map(len,rr))
    candidates=[set(t)-{1} for t in zip(*[[1]*(rank-len(s))+s for s in rr])]
    assert all(len(v)<=1 for v in candidates)
    out=[next(iter(v),1) for v in candidates]
    assert math.prod(out)>0
    def get(k,c):
        _,ds,ws=k;pos=0
        for n,j in zip(ds,c[len(c)-len(ds):]):pos=pos*n+(0 if n==1 else j)
        return ws[pos]
    cooked=[]
    for c in itertools.product(*(range(d) for d in out)):
        selected=x if get(m,c) else y;raw=get(selected,c)
        cooked.append(raw if selected[0]==0 else narrow(raw,mode))
    return dict(label=label,args=args,expected=[out,canonical(out),cooked])
neighbors={
 'even-normal-mid':0x3ff0000010000000,'odd-normal-mid':0x3ff0000030000000,
 'zero-smallest-mid':0x3690000000000000,'subnormal-mid-odd':0x36a8000000000000,
 'subnormal-normal-mid':0x380fffffe0000000,'overflow-mid':0x47effffff0000000,
 'smallest-normal':0x3810000000000000,'max-f32-finite':0x47efffffe0000000,
 'smallest-f32-subnorm':0x36a0000000000000}
values=[(f'{k}-{j:+}',v+j) for k,v in neighbors.items() for j in (-2,-1,0,1,2)]
values += [('zero',0),('smallest-f64-subnorm',1),('largest-f64-subnorm',F64MASK),('smallest-f64-normal',1<<52),
           ('max-f64-finite',0x7fefffffffffffff),('infinity',0x7ff0000000000000)]
values=list(dict(values).items())
NAN=[('q-min',1<<51),('q-low-discarded',(1<<51)|1),('q-bit29',(1<<51)|(1<<29)),
     ('q-full',0x8123456789abc),('q-all',F64MASK),('s-low-discarded',1),('s-bottom-discarded',(1<<29)-1),
     ('s-bit29',1<<29),('s-high',1<<50),('s-full',0x0123456789abc),('s-all',(1<<51)-1)]
finite=[];nan=[];control=[];archived=[]
for rev in (0,1):
    def args(n,cond=None,dim=None):
        a,b=t(4,[],[n]),dim or t(0,[1,len(F)],F)
        return [cond or t(1,[2,1,1],[1-rev,rev]),*((b,a) if rev else (a,b))]
    for s in (0,1):
        for k,v in values:
            finite.append(result(f'finite-{k}-s{s}-r{rev}',args((s<<63)|v),'local-hia-warning'))
        for k,v in NAN:
            w=(s<<63)|(0x7ff<<52)|v;nargs=args(w)
            hypothetical=result(f'nan-{k}-s{s}-r{rev}',nargs,'x86-payload-hypothesis')
            old=result(hypothetical['label'],nargs,'local-hia-warning')
            nan.append(dict(**hypothetical,f64_word=w,current_hia_canonical=old['expected'],proposed_initial_native_status=7,
                            payload_hypothesis_differs=old['expected']!=hypothetical['expected']))
    for m,dim,l in ((t(1,[],[1]),t(0,[1],[F[-1]]),'scalar-mask-true'),(t(1,[],[0]),t(0,[1],[F[-1]]),'scalar-mask-false')):
        finite.append(result(f'{l}-r{rev}',args(0x3690000000000001,m,dim),'local-hia-warning'))
    for label,a in (
        ('empty-value',args(0,t(1,[],[1]),t(0,[2,0],[]))),
        ('empty-mask',args(0,t(1,[0,1,1],[]),t(0,[1,4],F[:4]))),
        ('incompatible',args(0,t(1,[2],[1,0]),t(0,[3],F[:3]))),
        ('both-zero',[t(1,[],[1]),*( (t(0,[],[F[-1]]),t(4,[],[0])) if rev else (t(4,[],[0]),t(0,[],[F[-1]])) )]),
    ):
        control.append(dict(label=f'{label}-r{rev}',args=a,proposed_native_status=7))
P=Path('/tmp/hepta-where-f32-f64-draft/candidate')
rq=list(frames(P/'requests.frames.bin',b'HTOWNR01')); rp=list(frames(P/'responses.frames.bin',b'HTOWNS01'))
report=json.loads((P/'report.json').read_bytes())
for e in report['rows']:
    if e.get('kind')!='refusal' or not e['label'].startswith('rank0'):continue
    i=e['frame']-1;a,b=rq[i],rp[i];cmd,=req(a)
    assert resp(b)==(7,[]) and cmd[0]==129 and cmd[4:]==(0,0,0,0) and sha(a)==e['request_sha256'] and e['pre']==e['post']
    st=e['pre'];raw=zlib.decompress((P/'owners'/(st['sha256']+'.zlib')).read_bytes())
    assert sha(raw)==st['sha256'] and len(raw)==st['length']
    o=Owner(raw);xx=[]
    for h in cmd[1:4]:
        d=o.node(h);tag=d[72];shape=list(d[12:12+d[0]]);bb=o.storage(d[27])[2];w=W[tag]
        assert d[83]==d[29]==d[4]==d[5]==0
        xx.append(t(tag,shape,[int.from_bytes(bb[k:k+w],'little') for k in range(0,len(bb),w)]))
    m,x,y=xx;f=x if x[0]==4 else y;fv=y if x[0]==4 else x
    rec=dict(frame=i+1,label=e['label'],args=xx,request_sha256=sha(a),response_sha256=sha(b),owner_sha256=sha(raw),old_status=7)
    if f[1]==[] and fv[1] and not isnan(f[2][0]):
        fix=result(f'historical-{i+1}',xx,'local-hia-warning');finite.append(fix);rec['retire_if_actual_match']=fix['label']
    else:rec['already_changed_or_still_excluded']=('already-approved-widening' if f[1] and not fv[1] else 'both-rank0-outside')
    archived.append(rec)
assert len(archived)==6 and sum('retire_if_actual_match' in x for x in archived)==2
print(json.dumps(dict(parent_head='768f7819b0b1248a9e5fb656e3431e4365b009b4',parent_tree='6fa2a0165395c522ea474ce3102b60ad1f018b8d',
 scope='pure predictions only, ordinary canonical nonempty rank-positive F32 and exactly one rank0 F64, no lock/runtime run',
 finite_and_infinity=finite,nan_unmeasured_hypothesis=nan,policy_guards=control,archived=archived,
 prediction='portable finite signed round-even integer distances; x86 payload mode is a hypothesis, do not rewrite corpus after oracle'),sort_keys=True,separators=(',',':')))
