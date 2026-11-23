"""Independent wire capture reader for pinned Cat/Clone fan-in diagnostics."""
import argparse,collections,json,struct
from hashlib import sha1,sha256
from pathlib import Path
from types import ModuleType
ABI='9074399826154287c2a97bd7172d6d86562eb5dfe1f5ab641ac33d6a03fb59de'
PARSER='45783766aebeb4b183cd2aff993f27e5a40bf7db'

def check(ok):
    if not ok:raise ValueError('incorrect physical fan-in observation')
def digest(b):return sha256(b).hexdigest()
def inspect(root,script):
    raw=script.read_bytes()
    check(sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==PARSER)
    m=ModuleType('physical_transport');exec(compile(raw,str(script),'exec'),m.__dict__)
    o=json.loads(m.bound(root/'observations.json'))
    d=root/'native'
    tr=json.loads(m.bound(d/'transport.json'))
    Q=m.bound(d/'requests.frames.bin');R=m.bound(d/'responses.frames.bin')
    qs=m.frames(Q,b'HTOWNR01');rs=m.frames(R,b'HTOWNS01')
    check(len(qs)==len(rs)==tr['frames']==o['native_frames']==4296)
    check(tr['exit_code']==0 and tr['responses_before_stdin_eof'] and
          tr['executable_sha256']==o['native_sha256']==ABI)
    for i,(q,r,info) in enumerate(zip(qs,rs,tr['rows']),1):
        cmd=m.request(q);s,rows=m.response(r)
        check(s==0 and len(cmd)==len(rows)==1 and cmd[0][0]==rows[0][0] and info['frame']==i and
              info['pid']==tr['pid'] and info['response_bytes']==len(r) and info['stdin_open'])
    get=lambda i:m.request(qs[i-1])[0]
    ret=lambda i:m.response(rs[i-1])[1][0]
    from_wire=lambda i,n:list(struct.unpack(f'<{n}I',ret(i)[2]))
    counts=collections.defaultdict(collections.Counter)
    witness=[]
    check(len(o['cases'])==o['total_requested']==204)
    retained_count=0
    for row in o['cases']:
        n=row['native']; ca=row['cat_arms']; ck=row['creation']; sz=len(ca[0]); k=len(ca)
        tc=row['torch_arrival']; by_tag={}
        start=n['case_start_frame']
        leaf,lreq=m.factory_words(qs[start-1]);h=ret(start)[1]
        check(list(leaf)==[0x3f800000]*sz and lreq==1)
        check([i for i,_ in tc]==list(reversed(ck)) and all(v==ca[i] for i,v in tc))
        for j,tag in enumerate(ck):
            ci=start+1+2*j
            check(get(ci)[:2]==(18,h))
            by_tag[tag]=ret(ci)[1]
            check(get(ci+1)[:2]==(93,by_tag[tag]))
        cat=start+1+2*k
        cmd=get(cat)
        check(cmd[:2]==(34,0) and m.unpack(qs[cat-1][cmd[2]:])==(k,*(by_tag[i] for i in range(k))))
        seeds,sreq=m.factory_words(qs[cat])
        check(list(seeds)==[z for a in ca for z in a] and sreq==0)
        bw=cat+2
        check(bw==n['backward_frame'] and get(bw)[:4]==(58,ret(cat)[1],ret(cat+1)[1],0) and ret(bw)[1]==0)
        for i,(ix,ws) in n['retained_frames'].items():
            tag=int(i); check(get(ix-1)[:2]==(91,by_tag[tag]))
            check(get(ix)[:4]==(39,ret(ix-1)[1],0,sz) and from_wire(ix,sz)==ws==ca[tag])
            retained_count+=sz
        ix=n['leaf_frame'];check(get(ix)[:4]==(40,h,0,sz) and from_wire(ix,sz)==n['leaf'])
        if k==3:
            check(n['leaf']==row['torch_leaf'])
            witness.append({'creation':ck,'arrival':[i for i,_ in tc],'leaf_word':n['leaf'][0]})
            continue
        for a,b,t,v in zip(tc[0][1],tc[1][1],row['torch_leaf'],n['leaf']):
            ma=a&0x7fffffff;mb=b&0x7fffffff;na=ma>0x7f800000;nb=mb>0x7f800000
            if na and nb:
                category='same_raw_nan' if a==b else 'distinct_dual_nan'
                check(t==(b|0x400000))
            elif na or nb:
                category='sole_nan'
                check(t==((a if na else b)|0x400000))
            elif ma==mb==0x7f800000 and a!=b:
                category='opposite_inf'
                check(t==0xffc00000)
            else:
                category='other'
                check(t==v)
            counts[category]['compared']+=1
            counts[category]['native_mismatch']+=t!=v
            if category!='other':check(v==0x7fc00000)
    check(witness==[{'creation':[0,1,2],'arrival':[2,1,0],'leaf_word':0},
                     {'creation':[2,0,1],'arrival':[1,0,2],'leaf_word':1065353216}])
    check(retained_count==7342)
    return {'status':'INDEPENDENT_FAN_IN_READBACK_COMPLETE_EXISTING_NATIVE_PARITY_FAILED',
            'cases':len(o['cases']),'native_frames':len(qs),'branch_words_exact':retained_count,
            'torch_oracle':o['oracle'],'three_arm_arrival_witness':witness,
            'two_arm_leaf_cases':dict(counts),'hashes':{'requests':digest(Q),'responses':digest(R)},
            'no_new_native_executable':True}
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('physical',type=Path);p.add_argument('parser',type=Path);a=p.parse_args()
    print(json.dumps(inspect(a.physical,a.parser),indent=2,sort_keys=True))
