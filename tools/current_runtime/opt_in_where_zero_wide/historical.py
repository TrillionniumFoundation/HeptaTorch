"""Pure verification of exact existing rank0 witness operands and rollback."""
from pathlib import Path
import json,sys,zlib
sys.path.insert(0,str(Path(__file__).parent))
from preregister import W,array,fixture
from read_old import Owner,frames,req,resp,sha
out = Path(__file__).parent
parents={'half-f64':'hepta-where-half-f64','f32-f64':'hepta-where-f32-f64-draft','i64-f64':'hepta-where-i64-f64-draft'}
rows=[]; controls=[]; provenance={}
for label,local in parents.items():
    p=Path('/tmp')/local/'candidate'
    rq=Path(p/'requests.frames.bin');rp=Path(p/'responses.frames.bin')
    qs=list(frames(rq,b'HTOWNR01'));rs=list(frames(rp,b'HTOWNS01'))
    ledger=json.loads((p/'report.json').read_bytes()); image=ledger['elf']
    assert len(qs)==len(rs)==ledger['native_frames']
    provenance[label]=dict(elf_sha256=image,request_stream_sha256=sha(rq.read_bytes()),response_stream_sha256=sha(rp.read_bytes()),
                           ledger_sha256=sha((p/'report.json').read_bytes()))
    count=0
    for e in ledger['rows']:
        if e.get('kind')!='refusal' or not e['label'].startswith('rank0-'):continue
        count+=1
        n=e['frame'];q,r=qs[n-1],rs[n-1]
        cq,=req(q)
        assert cq[0]==129 and cq[4:]==(0,0,0,0) and resp(r)==(7,[]) and sha(q)==e['request_sha256']
        assert e['pre']==e['post']
        a=e['pre']
        snap=zlib.decompress((p/'owners'/(a['sha256']+'.zlib')).read_bytes())
        assert len(snap)==a['length'] and sha(snap)==a['sha256']
        owner=Owner(snap);args=[]
        for tok in cq[1:4]:
            d=owner.node(tok);tag=d[72];shape=list(d[12:12+d[0]])
            assert d[83]==d[4]==d[5]==d[29]==0
            raw=owner.storage(d[27])[2];w=W[tag]
            args.append(array(tag,shape,[int.from_bytes(raw[j:j+w],'little') for j in range(0,len(raw),w)]))
        m,x,y=args
        assert m['dtype']==1 and {x['dtype'],y['dtype']} & {4} and x['dtype']!=y['dtype']
        name=f'historical-{label}-{n}'
        proof=dict(label=name,prior_label=e['label'],proof_family=label,source_frame=n,
                   request_sha256=sha(q),response_sha256=sha(r),owner_sha256=sha(snap),prior_status=7)
        old_f64=x if x['dtype']==4 else y
        low=y if x['dtype']==4 else x
        assert low['dtype'] in (0,2,5,6)
        if old_f64['shape'] and not low['shape']:
            proof['expected_if_admitted']=fixture(name,m,x,y,'inputs-first' if x is low else 'result-first')
            rows.append(proof)
        else:
            assert not old_f64['shape']
            proof.update(mask=m,x=x,y=y,policy_expected_status=7)
            controls.append(proof)
    assert count==({'half-f64':12,'f32-f64':6,'i64-f64':6}[label])
result=dict(description='Exact archived controls verified from raw native receipts; none executed by this generator',
            retiring_candidates=rows,retained_policy_candidates=controls,provenance=provenance)
assert len(rows)==8 and len(controls)==16
print(json.dumps(result,sort_keys=True,separators=(',',':')))
