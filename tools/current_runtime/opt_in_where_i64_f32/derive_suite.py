from pathlib import Path
import json,sys,zlib
from preregister import array,case
from pure import frames,req,resp,Owner,sha
D=Path(__file__).parent
P=Path('/tmp/hepta-where-half-f64')
old_paths={x:Path('/tmp')/p/'candidate' for x,p in dict(
half_f64='hepta-where-half-f64',i64_f64='hepta-where-i64-f64-draft',f16_tail='hepta-where-f16-tail-draft',bf16_f32='hepta-where-bf16-f32-draft',wide='hepta-where-f32-f64-draft',bool_float='hepta-where-bool-float-draft',bool_i64='hepta-where-bool-i64-draft',full='hepta-where-full-draft',general='hepta-where-general-mask-draft',trailing='hepta-where-trailing-draft',scalar='hepta-where-broadcast-draft').items()}
old_paths['where']=Path('/tmp/hepta-where-draft/candidate-prime')
R=json.loads((D/'expectation.json').read_bytes()); cases=list(R['positives']); known={}
for run,p in old_paths.items():
    retained=p if run=='half_f64' else P/'retain'/run
    old_proof=json.loads((P/'retain/replay.json').read_text())
    retired={x['frame'] for x in old_proof.get(run,{}).get('retired',[])}
    qs=list(frames(p/'requests.frames.bin',b'HTOWNR01'))
    rs=list(frames(p/'responses.frames.bin',b'HTOWNS01'))
    qprev=list(frames(retained/'requests.frames.bin',b'HTOWNR01'))
    rprev=list(frames(retained/'responses.frames.bin',b'HTOWNS01'))
    for e in json.loads((p/'report.json').read_text()).get('rows',[]):
        if e.get('label') not in ('f32-i64-forward','f32-i64-reverse','mixed-value-dtype','mixed-dtype'):continue
        j=e['frame']
        found=[ii for ii,(a,b) in enumerate(zip(qprev,rprev,strict=True),1) if a==qs[j-1] and b==rs[j-1]]
        assert len(found)==1,(run,j,found)
        ix,=found
        q,r=qs[j-1],rs[j-1]; cmd,=req(q)
        assert cmd[0]==129 and cmd[4:]==(0,0,0,0) and resp(r)==(7,[])
        assert qprev[ix-1]==q and rprev[ix-1]==r
        snap=zlib.decompress((p/'owners'/(e['pre']['sha256']+'.zlib')).read_bytes())
        assert sha(snap)==e['pre']['sha256']==e['post']['sha256']
        assert sha(q)==e['request_sha256']
        o=Owner(snap); args=[]
        for h in cmd[1:4]:
            a=o.node(h); tag=a[72]
            assert a[83]==a[4]==a[5]==a[29]==0
            bs=o.storage(a[27])[2]; w={0:4,1:1,2:8}[tag]
            args.append(array(tag,list(a[12:12+a[0]]),[int.from_bytes(bs[i:i+w],'little') for i in range(0,len(bs),w)]))
        assert {args[1]['dtype'],args[2]['dtype']}=={0,2} and args[0]['dtype']==1 and args[1]['shape'] and args[2]['shape']
        label=f'historical-{run}-{j}'
        row=case(label,*args,'inputs-first' if len(cases)%2==0 else 'result-first')
        cases.append(row)
        known.setdefault(run,{})[str(ix)]=dict(label=label,old_label=e['label'],original_frame=j,
            owner_sha256=sha(snap),request_sha256=sha(q),response_sha256=sha(r))
def flatten(f):
    out=dict(label=f['label'],dtype=0,width=4,shape=f['result']['shape'],result_hex=f['result']['byte_hex'])
    shp=out['shape'];stride=1; st=[]
    for dim in reversed(shp):
        st.append(stride);stride*=max(1,dim)
    out['strides']=list(reversed(st))
    for name,key in (('mask','mask'),('left','x'),('right','y')):
        a=f[key];out[name]=a['words'];out[name+'_hex']=a['byte_hex'];out[name+'_shape']=a['shape'];out[name+'_dtype']=a['dtype']
    return out
(D/'suite.json').write_text(json.dumps(dict(rows=list(map(flatten,cases)),retired=known,rank0=R['rank0_native_policy_denials']),indent=2,sort_keys=True)+'\n')
print('retire', {k:len(v) for k,v in known.items()},'positive',len(cases))
