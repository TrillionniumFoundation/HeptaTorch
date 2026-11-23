"""No-Torch replay of published #66 scopes; retire only signed, verified exact old witnesses."""
import json,os,resource,sys,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from pure import Owner,frames,req,resp,sha
H,WIRE,elf,old,original_where,suite,out=map(Path,sys.argv[1:])
sys.path.insert(0,str(H))
from test_runtime_candidate import load_wire
assert 'torch' not in sys.modules
assert sha(elf.read_bytes())=='4827d1eb731a55540cbd2fcb64252f7485e7505f1152225e293868679fb63017'
W=load_wire(WIRE)
suit=json.loads(suite.read_bytes())
fixtures={r['label']:r for r in suit['rows']}
archive=suit['archived']
supp=json.loads(suite.with_name('supplemental.json').read_bytes())
assert sha(suite.with_name('supplemental.json').read_bytes())=='92e411d05862787a60cc6b0f51747d40218529f80edc903da0e634a27b294fff'
fixtures.update({f['label']:f for f in supp['rows']});archive+=supp['archived']
expected_bytes=(old/'retain/replay.json').read_bytes()
assert sha(expected_bytes)=='709082b74716f04ab7dde8c4796a63dab373ea090976b86dde4175b093270d1e'
expected=json.loads(expected_bytes)
runs=[('zero_i64_f32',old/'candidate'),*((x,old/'retain'/x) for x in expected)]
pressure,=[x for x in json.loads((original_where/'report.json').read_bytes())['rows'] if x['kind']=='pressure']
assert pressure['request_sha256']=='6951fb8100d6b9cad1735612a10955a51306e8e68a9ca7f524c26939fe3e77b8'
assert pressure['limit_bytes']==17321984 and expected['where']['pressure_witness']==1
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
    pairs={(sha(q),sha(r)):[] for q,r in zip(qs,rs,strict=True)}
    for j,(q,r) in enumerate(zip(qs,rs,strict=True),1):pairs[(sha(q),sha(r))].append(j)
    retire={};still={}
    if run in ('f16_tail','bf16_f32','zero_i64_f32'):
        source={'f16_tail':'f16-tail-draft','bf16_f32':'bf16-f32-draft','zero_i64_f32':'immediate-zero-i64-f32'}[run]
        for e in archive:
            if e['origin']!=source:continue
            j,=pairs[e['request_sha256'],e['response_sha256']]
            assert resp(rs[j-1])==(7,[])
            rec=dict(e)
            if 'matched_positive_prediction' in e:
                f=fixtures[e['matched_positive_prediction']]
                for k,t in zip(('mask','left','right'),e['operands'],strict=True):
                    assert (f[k+'_dtype'],f[k+'_shape'],f[k+'_hex'])==(t['dtype'],t['shape'],t['byte_hex'])
                retire[j]=rec
            else:still[j]=rec
        expected_counts={'f16_tail':(2,10),'bf16_f32':(2,4),'zero_i64_f32':(4,0)}[run]
        assert (len(retire),len(still))==expected_counts
    cap_start=cap_failure=cap_recovery=None
    if run=='where':
        orig=list(frames(original_where/'requests.frames.bin',b'HTOWNR01'))
        old_reply=list(frames(original_where/'responses.frames.bin',b'HTOWNS01'))
        old_failure,=[j for j,(q,r) in enumerate(zip(orig,old_reply,strict=True),1) if sha(q)==pressure['request_sha256'] and resp(r)==(3,[])]
        assert old_failure==pressure['frame']
        start=old_failure+1-pressure['attempt']; recovery=pressure['recover_frame']
        assert 1<=start<=old_failure<recovery<=len(orig)
        cap_failure,=pairs[pressure['request_sha256'],sha(old_reply[old_failure-1])]
        cap_start=cap_failure-(old_failure-start)
        cap_recovery=cap_failure+(recovery-old_failure)
        assert qs[cap_start-1:cap_recovery]==orig[start-1:recovery]
        assert rs[cap_start-1:cap_recovery]==old_reply[start-1:recovery]
        assert resp(rs[cap_recovery-1])[0]==0 and not set(range(cap_start,cap_recovery+1))&set(retire)
        print('pressure verified by request digest and exact lead/recovery packets',cap_start,cap_failure,cap_recovery,flush=True)
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
                data=h; o=Owner(words)
                for handle,t in zip(cmd[1:4],data['operands'],strict=True):
                    node=o.node(handle)
                    assert node[72]==t['dtype'] and list(node[12:12+node[0]])==t['shape']
                    assert o.storage(node[27])[2].hex()==t['byte_hex']
                old_dir=out/run/'historical';old_dir.mkdir(exist_ok=True)
                (old_dir/f'{j}.request.bin').write_bytes(q)
                (old_dir/f'{j}.response.bin').write_bytes(r)
                (old_dir/f'{j}.owner.zlib').write_bytes(zlib.compress(words,6))
                continue
            if status: before=ss.owner()
            if j in still:
                h=still[j];assert status==7 and sha(before['bytes'])==h['owner_sha256']
                cmd,=req(q);oo=Owner(before['bytes'])
                for handle,t in zip(cmd[1:4],h['operands'],strict=True):
                    node=oo.node(handle)
                    assert node[72]==t['dtype'] and list(node[12:12+node[0]])==t['shape']
                    assert oo.storage(node[27])[2].hex()==t['byte_hex']
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
                      pressure_witness=cap_witness,preserved_explicit_rank0_controls=still,request_sha256=sha(qf.read_bytes()),response_sha256=sha(rf.read_bytes()))
    print(run,ss.frames,roll,len(retire),flush=True)
assert sum(len(a['retired']) for a in summary.values())==8
assert sum(len(a['preserved_explicit_rank0_controls']) for a in summary.values())==14
assert summary['where']['pressure_witness']==1 and 'torch' not in sys.modules
(out/'replay.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
