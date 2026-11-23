"""No-Torch replay of published #66 scopes; retire only signed, verified exact old witnesses."""
import json,os,resource,sys,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from pure import Owner,frames,req,resp,sha
H,WIRE,elf,old,original_where,suite,out=map(Path,sys.argv[1:])
sys.path.insert(0,str(H))
from test_runtime_candidate import load_wire
assert 'torch' not in sys.modules
assert sha(elf.read_bytes())=='ad9a5969fefb2f0484a86bbb9211c3aa93f19c206d82dd9880519a5a8d924c98'
W=load_wire(WIRE)
suit=json.loads(suite.read_bytes())
fixtures={q['label']:q for q in suit['rows']}
history=suit['retired']
assert sha(suite.read_bytes())=='ef1f8673289c68d7b86cadc3104c60e3098709485b068b702cb0fd29d5f89264'
assert sum(map(len,history.values()))==20
expected=json.loads((old/'retain-final/replay.json').read_bytes())
runs=[('i64_f32',old/'candidate'),*((x,old/'retain-final'/x) for x in expected)]
pressure,=[x for x in json.loads((original_where/'report.json').read_bytes())['rows'] if x['kind']=='pressure']
assert pressure['request_sha256']=='6951fb8100d6b9cad1735612a10955a51306e8e68a9ca7f524c26939fe3e77b8'
summary={};out.mkdir()
for run,p in runs:
    qs=list(frames(p/'requests.frames.bin',b'HTOWNR01'))
    rs=list(frames(p/'responses.frames.bin',b'HTOWNS01'))
    assert len(qs)==len(rs)
    if run in expected:
        x=expected[run]
        assert len(qs)==x['frames']
        assert sha((p/'requests.frames.bin').read_bytes())==x['request_sha256']
        assert sha((p/'responses.frames.bin').read_bytes())==x['response_sha256']
    else:
        x=json.loads((old/'pure.json').read_bytes())['candidate']
        assert len(qs)==x['frames']
        assert sha((p/'requests.frames.bin').read_bytes())==x['request_stream']
        assert sha((p/'responses.frames.bin').read_bytes())==x['response_stream']
    retire={int(i):v for i,v in history.get(run,{}).items()}
    cap_start=cap_failure=cap_recovery=None
    if run=='where':
        retired_before={int(i) for i in expected[run]['retired']}
        oq=list(frames(original_where/'requests.frames.bin',b'HTOWNR01'))
        rr=list(frames(original_where/'responses.frames.bin',b'HTOWNS01'))
        mapping={raw:j for j,raw in enumerate((i for i in range(1,len(oq)+1) if i not in retired_before),1)}
        assert qs==[q for i,q in enumerate(oq,1) if i in mapping] and rs==[r for i,r in enumerate(rr,1) if i in mapping]
        start=pressure['frame']+1-pressure['attempt']
        cap_start,cap_failure,cap_recovery=(mapping[i] for i in (start,pressure['frame'],pressure['recover_frame']))
        match=[i for i,(q,r) in enumerate(zip(qs,rs,strict=True),1) if sha(q)==pressure['request_sha256'] and resp(r)==(3,[])]
        assert match==[cap_failure]
        assert resp(rs[cap_recovery-1])[0]==0 and not {cap_start,cap_failure,cap_recovery}&set(retire)
        print('pressure archived-coordinate map',start,pressure['frame'],pressure['recover_frame'],
              '=>',cap_start,cap_failure,cap_recovery,flush=True)
    ss=W.Session(elf,directory=out/run)
    limit=None; roll=0; cap_witness=0
    try:
        for j,(q,r) in enumerate(zip(qs,rs,strict=True),1):
            if run=='where':
                if j==cap_start:
                    limit=resource.prlimit(ss.pid,resource.RLIMIT_AS)
                    size=os.sysconf('SC_PAGE_SIZE')
                    vmem=int(Path(f'/proc/{ss.pid}/statm').read_text().split()[0])*size
                    length=ss.owner()['length']
                    candidate=vmem+((length+16+size-1)//size)*size+65536
                    print('pressure archived/candidate-relative',pressure['limit_bytes'],candidate,flush=True)
                    assert abs(pressure['limit_bytes']-candidate)<=size*8
                    resource.prlimit(ss.pid,resource.RLIMIT_AS,(pressure['limit_bytes'],limit[1]))
                    assert resource.prlimit(ss.pid,resource.RLIMIT_AS)[0]==pressure['limit_bytes']
                    print('pressure exact absolute cap',pressure['limit_bytes'],flush=True)
                if j==cap_recovery:
                    resource.prlimit(ss.pid,resource.RLIMIT_AS,limit)
                    assert resource.prlimit(ss.pid,resource.RLIMIT_AS)==limit
                    limit=None
            status=resp(r)[0]
            if j in retire:
                h=retire[j]; words=ss.owner()['bytes']
                assert resp(r)==(7,[]) and sha(q)==h['request_sha256'] and sha(r)==h['response_sha256'] and sha(words)==h['owner_sha256']
                cmd,=req(q); assert cmd[0]==129 and cmd[4:]==(0,0,0,0)
                data=fixtures[h['label']]; o=Owner(words)
                for handle,k in zip(cmd[1:4],('mask','left','right'),strict=True):
                    node=o.node(handle)
                    assert node[72]==data[k+'_dtype'] and list(node[12:12+node[0]])==data[k+'_shape']
                    assert o.storage(node[27])[2].hex()==data[k+'_hex']
                old_dir=out/run/'historical';old_dir.mkdir(exist_ok=True)
                (old_dir/f'{j}.request.bin').write_bytes(q)
                (old_dir/f'{j}.response.bin').write_bytes(r)
                (old_dir/f'{j}.owner.zlib').write_bytes(zlib.compress(words,6))
                continue
            if status: before=ss.owner()
            assert ss.exchange(q)==r,(run,j,status,'exact reply mismatch')
            if status:
                assert before==ss.owner(),(run,j,status,'rollback mismatch')
                roll+=1
                if run=='where' and j==cap_failure: cap_witness+=1
    finally:
        if limit is not None:resource.prlimit(ss.pid,resource.RLIMIT_AS,limit)
        ss.finish()
    qf=out/run/'requests.frames.bin'; rf=out/run/'responses.frames.bin'
    assert list(frames(qf,b'HTOWNR01'))==[q for j,q in enumerate(qs,1) if j not in retire]
    assert list(frames(rf,b'HTOWNS01'))==[r for j,r in enumerate(rs,1) if j not in retire]
    assert roll==sum(bool(resp(r)[0]) for j,r in enumerate(rs,1) if j not in retire)
    assert ss.frames==len(qs)-len(retire)
    summary[run]=dict(previous_frames=len(qs),frames=ss.frames,exact_rollbacks=roll,retired=retire,
                      pressure_witness=cap_witness,request_sha256=sha(qf.read_bytes()),response_sha256=sha(rf.read_bytes()))
    print(run,ss.frames,roll,len(retire),flush=True)
assert summary['where']['pressure_witness']==1 and 'torch' not in sys.modules
(out/'replay.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
