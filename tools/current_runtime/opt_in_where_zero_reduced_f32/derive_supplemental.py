"""Freeze four exactly matched immediate-old scalar reduced exclusions."""
import json,zlib,sys
from pathlib import Path
D=Path(__file__).parent;OLD=Path('/tmp/hepta-where-zero-i64-f32-prereg/candidate')
sys.path.insert(0,str(D));from pure import Owner,frames,req,resp,sha
exec(compile((D/'preregister.py').read_text().split('positives=[]')[0],str(D/'preregister.py'),'exec'))
p=json.loads((OLD/'report.json').read_bytes())
r=list(frames(OLD/'responses.frames.bin',b'HTOWNS01'));q=list(frames(OLD/'requests.frames.bin',b'HTOWNR01'))
wanted={8370:'exclude-f16-scalar-r0',8377:'exclude-bf16-scalar-r0',8440:'exclude-f16-scalar-r1',8447:'exclude-bf16-scalar-r1'}
out=[]
snapshot_dir=D/'supplemental-prior';snapshot_dir.mkdir()
for e in p['rows']:
    if e.get('label') not in wanted.values():continue
    j=e['frame'];a,b=q[j-1],r[j-1];assert wanted[j]==e['label'] and resp(b)==(7,[]) and e['pre']==e['post']
    assert sha(a)==e['request_sha256']
    cmd,=req(a);assert cmd[0]==129 and cmd[4:]==(0,0,0,0)
    meta=e['pre'];comp=(OLD/'owners'/(meta['sha256']+'.zlib')).read_bytes()
    raw=zlib.decompress(comp);assert sha(raw)==meta['sha256'] and len(raw)==meta['length']
    o=Owner(raw);args=[]
    for h in cmd[1:4]:
        d=o.node(h);tag=d[72];width=WIDTH[tag];sh=list(d[12:12+d[0]]);s=o.storage(d[27])[2]
        assert d[83]==d[29]==d[4]==d[5]==0
        args.append(ten(tag,sh,[int.from_bytes(s[i:i+width],'little') for i in range(0,len(s),width)]))
    m,x,y=args;assert m[0]==1 and any(d[0]==0 and d[1] for d in (x,y)) and any(d[0] in (5,6) and d[1]==[] for d in (x,y))
    name=f'immediate-old-{j}'
    c=makecase(name,*args);shape,strides,words=c['expected'];assert shape==[1]
    row=dict(label=name,shape=shape,strides=strides,torch_strides=strides,dtype=0,width=4,
             result_hex=b''.join(w.to_bytes(4,'little') for w in words).hex())
    operands=[]
    for k,(tag,sh,v) in zip(('mask','left','right'),args):
        ss=b''.join(w.to_bytes(WIDTH[tag],'little') for w in v).hex()
        row[k]=v;row[k+'_shape']=sh;row[k+'_dtype']=tag;row[k+'_hex']=ss
        operands.append(dict(dtype=tag,shape=sh,words=v,byte_hex=ss))
    rec=dict(origin='immediate-zero-i64-f32',source_frame=j,matched_positive_prediction=name,prior_label=e['label'],
        owner_sha256=sha(raw),request_sha256=sha(a),response_sha256=sha(b),operands=operands)
    for ext,body in (('request.bin',a),('response.bin',b),('owner.zlib',comp)):(snapshot_dir/f'{j}.{ext}').write_bytes(body)
    out.append((row,rec))
assert len(out)==4
(D/'supplemental.json').write_text(json.dumps(dict(rows=[r for r,k in out],controls=[],archived=[k for r,k in out]),sort_keys=True,separators=(',',':'))+'\n')
