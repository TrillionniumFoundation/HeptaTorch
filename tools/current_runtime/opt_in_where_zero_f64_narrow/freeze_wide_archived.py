"""Retain eight exact predecessor witnesses; this does not run Torch or HIA."""
import hashlib,itertools,json,math,sys,zlib
from pathlib import Path
D=Path(__file__).parent
sys.path.insert(0,str(D))
from read_old import Owner,frames,req,resp,sha
text=(D/'preregister.py').read_text()
exec(compile(text[text.index('F64MASK='):text.index('neighbors=')], 'preregister-definitions', 'exec'))
previous={'zero-wide':('zero-wide-prereg/nonempty',{3374:'reverse-role-t0-r0',3388:'reverse-role-t0-r1',
                                   3542:'historical-f32-f64-5151',3549:'historical-f32-f64-5152'})}
p=json.loads((D/'preregister.json').read_bytes())
prior={x['frame']:x for x in p['archived'] if 'retire_if_actual_match' in x}
positive={x['label']:x for x in p['finite_and_infinity']}
archive=D/'exact-wide-old'; archive.mkdir(mode=0o700)
rows=[]
for origin,(base,wanted) in previous.items():
    src=Path(f'/tmp/hepta-where-{base}/candidate')
    q=list(frames(src/'requests.frames.bin',b'HTOWNR01'))
    r=list(frames(src/'responses.frames.bin',b'HTOWNS01'))
    rawreport=(src/'report.json').read_bytes()
    report=json.loads(rawreport)
    found=[e for e in report['rows'] if e.get('frame') in wanted and e.get('kind')=='refusal']
    assert len(found)==len(wanted)
    for e in found:
        j=e['frame']; a,b=q[j-1],r[j-1]
        assert e['label']==wanted[j] and e['pre']==e['post'] and sha(a)==e['request_sha256'] and resp(b)==(7,[])
        meta=e['pre']
        comp=(src/'owners'/(meta['sha256']+'.zlib')).read_bytes()
        raw=zlib.decompress(comp); assert sha(raw)==meta['sha256'] and len(raw)==meta['length']
        cmd,=req(a);assert cmd[0]==129 and cmd[4:]==(0,0,0,0)
        o=Owner(raw); args=[]; strides=[]
        for h in cmd[1:4]:
            d=o.node(h);tag=d[72];w=W[tag];sh=list(d[12:12+d[0]])
            st=list(d[16:16+d[0]]); bb=o.storage(d[27])[2]
            assert d[83]==d[29]==d[4]==d[5]==0 and d[1]==math.prod(sh)>0 and len(bb)==w*d[1]
            assert st==canonical(sh)
            args.append(t(tag,sh,[int.from_bytes(bb[k:k+w],'little') for k in range(0,len(bb),w)]));strides.append(st)
        m,left,right=args
        f32,= [x for x in (left,right) if x[0]==0]
        f64,= [x for x in (left,right) if x[0]==4]
        assert m[0]==1 and f32[1] and f64[1]==[]
        v,=f64[2]; assert isnan(v)==(j in (3374,3388))
        label=f'exact-old-{origin}-{j}'
        want=result(label,args,'x86-payload-hypothesis')
        duplicate={3542:5151,3549:5152}.get(j)
        if duplicate:
            same=prior[duplicate];old=positive[same['retire_if_actual_match']]
            assert same['args']==old['args']==args and old['expected']==want['expected']
            assert same['request_sha256']!=sha(a) and same['owner_sha256']!=sha(raw)
        for ext,body in [('request.bin',a),('response.bin',b),('owner.zlib',comp)]:
            (archive/f'{j}.{ext}').write_bytes(body)
        rows.append(dict(origin=origin,frame=j,retained_frame=j-8,exact_operands_duplicate_frame=duplicate,old_label=e['label'],old_status=7,owner_rollback_exact=True,
            owner_sha256=sha(raw),owner_length=len(raw),compressed_owner_sha256=sha(comp),request_sha256=sha(a),response_sha256=sha(b),
            parent_report_sha256=sha(rawreport),input_strides=strides,f64_word=v,nan=isnan(v),new_prediction=want))
assert len(rows)==4
(D/'exact-wide-old.json').write_text(json.dumps(dict(status='frozen-original-and-pure-predictions-no-new-physical',witnesses=rows),indent=2,sort_keys=True)+'\n')
print('actual frozen refusals',len(rows),'finite',sum(not x['nan'] for x in rows),'NaN',sum(x['nan'] for x in rows))
