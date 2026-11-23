"""Stdlib frozen true-rank0 reduced/F32 prediction; NaN policy hypotheses."""
import itertools,json,math,sys,zlib
from pathlib import Path
D=Path(__file__).parent
sys.path.insert(0,str(D))
from read_old import Owner,frames,req,resp,sha
WIDTH={0:4,1:1,2:8,4:8,5:2,6:2}
F=[0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x7f812345,0xff812345]
P={
5:[0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x7bff,0xfbff,0x7c00,0xfc00,
   0x7e01,0xfe35,0x7e53,0xfe59,0x7c01,0xfc35,0x7c53,0xfc59,0x3c00,0xbc00],
6:[0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x7f7f,0xff7f,0x7f80,0xff80,
   0x7fc1,0xffd5,0x7fc3,0xffc9,0x7f81,0xff95,0x7f83,0xff89,0x3f80,0xbf80],
}
def conv(tag,raw,reference_tail):
    assert 0<=raw<=65535
    if tag==6:return raw<<16 # hypothesis includes signaling-bit retention.
    assert tag==5
    s=(raw>>15)<<31;e=(raw>>10)&31;f=raw&1023
    if e==31:
        if f and reference_tail:return 0x7fffffff
        return s|0x7f800000|((f<<13)|0x400000 if f else 0)
    if e==0 and f==0:return s
    a=f|(1024 if e else 0)
    bl=a.bit_length()
    return s|(((e or 1)-15-10+bl-1+127)<<23)|((a<<(24-bl))&0x7fffff)
def ten(tag,sh,vals):
    assert math.prod(sh)==len(vals)
    w=WIDTH[tag];return [tag,sh,[v&((1<<(8*w))-1) for v in vals]]
def makecase(label,m,x,y):
    rs=[m[1],x[1],y[1]];rank=max(map(len,rs));join=[set(a)-{1} for a in zip(*[[1]*(rank-len(s))+s for s in rs])]
    assert all(len(a)<=1 for a in join)
    sh=[next(iter(a),1) for a in join];assert math.prod(sh)>0
    def load(v,c):
        _,dims,words=v;i=0
        for d,j in zip(dims,c[len(c)-len(dims):]):i=i*d+(0 if d==1 else j)
        return words[i]
    parts=[]
    for c in itertools.product(*(range(d) for d in sh)):
        a=x if load(m,c) else y;v=load(a,c)
        parts.append(v if a[0]==0 else conv(a[0],v,True))
    return dict(label=label,args=[m,x,y],expected=[sh,[math.prod(sh[i+1:]) for i in range(rank)],parts])
positives=[];policy=[];profiles=[];archive=[]
for tag,words in P.items():
 for rev in (0,1):
    def pair(dim,scalar):return (dim,scalar) if rev else (scalar,dim)
    for i,v in enumerate(words):
        positives.append(makecase(f'mixed-mask-tag{tag}-r{rev}-word{i}',ten(1,[2,1,1],[1-rev,rev]),
                        *pair(ten(0,[1,len(F)],F),ten(tag,[],[v]))))
    for flag in (0,1):
        positives.append(makecase(f'scalar-condition-tag{tag}-r{rev}-sel{flag}',ten(1,[],[flag]),
                        *pair(ten(0,[1],[F[-1]]),ten(tag,[],[words[19]]))))
    for lab,m,x,y in (
        ('reverse-role',ten(1,[],[1]),ten(tag,[1],[words[18]]),ten(0,[],[F[8]])),
        ('both-rank0',ten(1,[],[1]),ten(0,[],[F[8]]),ten(tag,[],[words[18]])),
        ('empty-value',ten(1,[],[0]),ten(0,[1,0],[]),ten(tag,[],[words[18]])),
        ('empty-mask',ten(1,[0,1,1],[]),ten(0,[1,4],F[:4]),ten(tag,[],[words[18]])),
        ('incompatible',ten(1,[2],[1,0]),ten(0,[3],F[:3]),ten(tag,[],[words[18]])),
        ('no-f64-narrowing',ten(1,[],[1]),ten(0,[1],F[:1]),ten(4,[],[0xfff0123456789abc])),
    ):
        policy.append(dict(label=f'{lab}-tag{tag}-r{rev}',args=[m,*pair(x,y)],proposed_native_status=7))
 for n in (1,7,8,9):
    for k,w in enumerate(words[12:20]):
        # Same raw first and tail word, different expected path only at F16 vector cut.
        src=[w]+[words[(i+2*k)%len(words)] for i in range(1,n)]
        src[-1]=w
        profiles.append(dict(label=f'cast-tag{tag}-rank1-len{n}-nan{k}',input=ten(tag,[n],src),
                             expected_f32_hypothesis=[conv(tag,v,i>=n//8*8) for i,v in enumerate(src)]))
 for k,w in enumerate(words):
    profiles.append(dict(label=f'cast-tag{tag}-true-rank0-word{k}',input=ten(tag,[],[w]),
                         expected_f32_hypothesis=[conv(tag,w,True)]))
for old in ('f16-tail-draft','bf16-f32-draft'):
    p=Path('/tmp')/f'hepta-where-{old}'/'candidate'
    report=json.loads((p/'report.json').read_bytes())
    requests=list(frames(p/'requests.frames.bin',b'HTOWNR01'));responses=list(frames(p/'responses.frames.bin',b'HTOWNS01'))
    for e in report['rows']:
        if e.get('kind')!='refusal' or not e['label'].startswith('rank0'):continue
        j=e['frame']-1; q,r=requests[j],responses[j];cmd,=req(q)
        assert resp(r)==(7,[]) and cmd[0]==129 and cmd[4:]==(0,0,0,0) and e['pre']==e['post']
        snap=zlib.decompress((p/'owners'/(e['pre']['sha256']+'.zlib')).read_bytes())
        assert len(snap)==e['pre']['length'] and sha(snap)==e['pre']['sha256'] and sha(q)==e['request_sha256']
        o=Owner(snap);args=[]
        for handle in cmd[1:4]:
            a=o.node(handle);tag=a[72];w=WIDTH[tag];sh=list(a[12:12+a[0]]);data=o.storage(a[27])[2]
            assert a[4]==a[5]==a[29]==a[83]==0
            args.append(ten(tag,sh,[int.from_bytes(data[i:i+w],'little') for i in range(0,len(data),w)]))
        m,x,y=args
        rec=dict(label=e['label'],origin=old,source_frame=j+1,elf=report['elf'],args=args,status=7,
                 request_sha256=sha(q),response_sha256=sha(r),owner_sha256=sha(snap))
        t=x if x[0]==0 else y;sc=y if x[0]==0 else x
        if t[0]==0 and t[1] and sc[0] in (5,6) and not sc[1]:
            row=makecase(f'historical-{old}-{j+1}',*args);positives.append(row);rec['matched_positive_prediction']=row['label']
        else:rec['retain_policy_status']=7
        archive.append(rec)
assert len(positives)==100 and len(policy)==24 and len(profiles)==108 and len(archive)==18
assert sum('matched_positive_prediction' in a for a in archive)==4
print(json.dumps(dict(parent_head='7e0cca2ecd1afd63190ae9f1545d6ea21ab35ad7',parent_tree='af1c96727ca2f69496743f48692b6f7bc7f84dd6',
 format_tensor='[dtype_tag,shape,raw_words]',
 scope='source-independent predictions, no official or native execution; genuine rank0 and nonempty canonical only',
 f16_nan_hypothesis='FBGEMM reference tail (true scalar count1) canonical positive 0x7fffffff; vector8 hardware preserves sign/shifted payload, quiets',
 bf16_nan_hypothesis='raw BF16 shifted 16, including sign and signaling bit; preserve selected F32 raw',
 positive=positives,reference_direct_profiles=profiles,policy=policy,archived=archive),sort_keys=True,separators=(',',':')))
