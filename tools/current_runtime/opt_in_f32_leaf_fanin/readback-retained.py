"""Parse retained physical F32 abs frames; does not launch Torch, a toolchain, or a runtime."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct

PRIOR='5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72'
CANDIDATE='1f0c6efc64e96fe25106acee7bf77a2ef86dd3d5b129b55bb773f5e5ff46e768'
CASES=('finite-dense','finite-chain','signed-zero-and-subnormal','inf-and-nan',
       'scalar','empty','forward-only','original-refusal','saved-version',
       'same-frame-and-rollback','real-address-space-refusal-retry','distinct-seed-accumulation',
       'special-nonleaf','fan-in-abs-first','fan-in-identity-first','special-mul-same-shape-both-slots')

def check(ok,msg):
    if not ok: raise ValueError(msg)
def sha(raw): return sha256(raw).hexdigest()
def bound(path):
    check(path.is_file() and not path.is_symlink() and path.stat().st_nlink==1,
          'not an ordinary unaliased physical file: '+path.name)
    return path.read_bytes()
def unpack(raw):
    check(len(raw)%8==0,'bad aligned body')
    return struct.unpack('<'+'Q'*(len(raw)//8),raw)
def frames(raw,magic):
    out=[]
    while raw:
        check(len(raw)>=16 and raw[:8]==magic,'bad outer magic')
        n,=unpack(raw[8:16])
        check(32<=n<=1048576 and 16+n<=len(raw),'bad physical length')
        out.append(raw[16:16+n]);raw=raw[16+n:]
    return out
def request(raw):
    check(raw[:8]==b'HTORCH03','bad request magic')
    n,size,unused=unpack(raw[8:32])
    check(32+64*n<=size==len(raw) and unused==0,'bad request envelope')
    return [unpack(raw[32+i*64:96+i*64]) for i in range(n)]
def response(raw):
    check(raw[:8]==b'HTORSP03','bad response magic')
    v,n,status=unpack(raw[8:32]);check(v==3,'wrong response version')
    if status:
        check(n==0 and len(raw)==32,'failure returned materials')
        return status,[]
    rows=[];raw=raw[32:]
    for i in range(n):
        check(len(raw)>=32,'short record')
        op,error,val,size=unpack(raw[:32])
        check(error==0 and len(raw)>=32+size,'bad result record')
        rows.append((op,val,raw[32:32+size]));raw=raw[32+size:]
    check(not raw,'unparsed response suffix')
    return status, rows

def factory_words(raw):
    c=request(raw)
    check(len(c)==1 and c[0][0:2]==(97,0),'not an F32 factory frame')
    offset=c[0][2]
    rank,grad,n,*tail=unpack(raw[offset:offset+64])
    data_at,data_n=tail[-2:]
    check(rank==1 and data_n==n*4 and data_at+data_n==len(raw),'invalid matrix factory extent')
    return struct.unpack(f'<{n}I',raw[data_at:]),grad

def classify_mul(left,right):
    a=left & 0x7fffffff
    b=right & 0x7fffffff
    inf=0x7f800000
    if a>inf and b>inf:return 'dual_nan'
    if a>inf or b>inf:return 'sole_nan'
    if (a,b) in ((inf,0),(0,inf)):return 'inf_times_zero'
    if a==0 or b==0:return 'ordinary_zero'
    return 'other'

def read(base):
    check({p.name for p in base.iterdir()}=={*CASES,'summary.json'},'case census')
    result={'status':'READBACK_COMPLETE_PARITY_FAILED','candidate':CANDIDATE,'prior':PRIOR,
            'cases':[],'total_frames':0,'raw_words_compared':0,'raw_value_frames':0}
    for name in CASES:
        b=base/name
        checks=json.loads(bound(b/'checks.json'))
        transport=json.loads(bound(b/'transport.json'))
        check(checks['name']==name and transport['executable_sha256']==(PRIOR if name=='original-refusal' else CANDIDATE),'wrong case identity')
        check(transport['exit_code']==checks['exit']==0 and transport['pid']==checks['pid'] and
              transport['responses_before_stdin_eof'] is True,'failed physical close')
        qr=bound(b/'requests.frames.bin')
        rr=bound(b/'responses.frames.bin')
        qs=frames(qr,b'HTOWNR01');rs=frames(rr,b'HTOWNS01')
        check(len(qs)==len(rs)==len(transport['rows'])==len(checks['steps'])==transport['frames']==checks['frames'],'physical count mismatch')
        findings={'case':name,'frames':len(qs),'raw_words':0,'raw_word_failures':[],'raw_value_frames':[],
                  'physical_owner_failures':[],'owner_rollbacks':[],'inner_58_results':[],
                  'request_sha256':sha(qr),'response_sha256':sha(rr)}
        owner_multiples={}
        matrix_classes={}
        for i,(raw_q,raw_r,step,tr) in enumerate(zip(qs,rs,checks['steps'],transport['rows']),1):
            check(step['frame']==tr['frame']==i and tr['pid']==transport['pid'] and
                  tr['response_bytes']==len(raw_r) and tr['stdin_open'] is True,'physical sequencing mismatch')
            q=request(raw_q);status,rows=response(raw_r)
            check((status,[r[0] for r in rows])==(step['status'],step['opcodes']),'wrong response semantics')
            if status==0:
                check([r[0] for r in rows]==[u[0] for u in q],'request/response opcode mismatch')
            if 'value' in step:
                check(len(rows)==1 and rows[0][1]==step['value'],'wrong used result value')
            if q[0][0]==58:
                check(len(rows)==1 and rows[0][0]==58 and len(rows[0][2])==32,'wrong inner backward result shape')
                findings['inner_58_results'].append(rows[0][1])
            if 'metadata' in step:
                expect=step['metadata']
                vals=unpack(rows[0][2])
                check(len(vals)==16 and (vals[0],vals[1],vals[2],list(vals[3:7]),list(vals[7:11]),vals[13],vals[14])==
                      (expect['rank'],expect['numel'],expect['requires_grad'],expect['shape'],
                       expect['strides'],expect['version'],expect['storage_offset']),'wrong physical metadata')
                if rows[0][1] not in owner_multiples:
                    owner_multiples[rows[0][1]]=(vals[11],vals[12])
                else:
                    check(owner_multiples[rows[0][1]]==(vals[11],vals[12]),'handle changed recorded owners')
            if 'payload_sha256' in step:
                expect=bound(b/f'oracle-{i:04d}.bin');actual=rows[0][2]
                check(sha(expect)==step['payload_sha256'] and len(expect)==len(actual)==step['payload_size'] and
                      len(actual)%4==0 and q[0][3]==len(actual)//4,'wrong physical value extent')
                n=len(actual)//4;findings['raw_words']+=n
                findings['raw_value_frames'].append({'frame':i,'label':step['label'],'words':n,'equal':expect==actual})
                if 'multiply' in step:
                    mat=step['multiply']
                    (lhs,lreq),(peer,preq),(grad,greq)=[factory_words(qs[i-k]) for k in (6,5,4)]
                    check(list(peer)==mat['saved_peer_words'] and list(grad)==mat['grad_words']
                          and set(lhs)=={0x40000000} and (lreq,preq,greq)==(1,0,0),'mismatched physical matrix inputs')
                    (leaf,other,seed,product)=[response(rs[i-k])[1][0][1] for k in (6,5,4,3)]
                    mul=request(qs[i-3])[0]
                    back=request(qs[i-2])[0]
                    check(mul[0:4]==(2,3,*((other,leaf) if mat['reverse'] else (leaf,other))) and
                          back[:4]==(58,product,seed,0) and q[0][:4]==(40,leaf,0,n),'matrix route not physically bound')
                    for (g,other_word,a,e) in zip(grad,peer,struct.unpack(f'<{n}I',actual),struct.unpack(f'<{n}I',expect)):
                        cat=classify_mul(g,other_word)
                        v=matrix_classes.setdefault(cat,{'compared':0,'mismatched':0})
                        v['compared']+=1
                        if a!=e:
                            v['mismatched']+=1
                            check(cat=='dual_nan' and a==0x7fc00000,'unexpected failure outside explicit fallback')
                        if cat=='sole_nan':
                            word=g if (g&0x7fffffff)>0x7f800000 else other_word
                            check(a==e==(word|0x400000),'bad sole-NaN treatment')
                        if cat=='inf_times_zero':
                            check(a==e==0xffc00000,'bad Inf times zero treatment')
                if expect!=actual:
                    delta=[(j,f'{a:08x}',f'{e:08x}') for j,(a,e) in enumerate(zip(struct.unpack(f'<{n}I',actual),struct.unpack(f'<{n}I',expect))) if a!=e]
                    findings['raw_word_failures'].append({'frame':i,'label':step['label'],'diff':delta,'native_sha256':sha(actual),'torch_sha256':sha(expect)})
            if 'owner_equal' in step:
                item=step['owner_equal']; label=item['name']
                before=bound(b/f'owner-{label}-before.bin');after=bound(b/f'owner-{label}-after.bin')
                for key,raw in ((item['before'],before),(item['after'],after)):
                    check(len(raw)==key['length'] and key['live']==1 and sha(raw)==key['sha256'],'owner custody mismatch')
                if before==after and item['before']==item['after']:
                    findings['owner_rollbacks'].append(label)
                else:
                    findings['physical_owner_failures'].append({
                        'frame':i,'label':label,'different_bytes':abs(len(before)-len(after))+sum(a!=b for a,b in zip(before,after)),
                        'pointer_changed':item['before']['pointer']!=item['after']['pointer'],
                        'before_sha256':sha(before),'after_sha256':sha(after)})
        if name=='real-address-space-refusal-retry':
            check(qs[1]==qs[2] and response(rs[1])==(3,[]) and response(rs[2])[0]==0,
                  'missing byte-identical OOM refusal/retry')
        if name in CASES[:6]:
            check(len(owner_multiples)==2 and len(set(owner_multiples.values()))==2 and
                  all(len({a[i] for a in owner_multiples.values()})==2 for i in (0,1)),
                  'not disjoint forward storage and version')
            check(findings['inner_58_results']==[0,0,10],'wrong retain/default/release sequence')
        if name=='saved-version':
            check(findings['inner_58_results']==[6],'wrong stale operand sequence')
        if name=='special-mul-same-shape-both-slots':
            findings['multiply_word_classes']=matrix_classes
        result['total_frames']+=findings['frames']
        result['raw_words_compared']+=findings['raw_words']
        result['raw_value_frames']+=len(findings['raw_value_frames'])
        result['cases'].append(findings)
    bad_numeric=[c['case'] for c in result['cases'] if c['raw_word_failures']]
    bad_owner=[c['case'] for c in result['cases'] if c['physical_owner_failures']]
    check(bad_numeric==list(CASES[-3:]) and bad_owner==[*CASES[:6],'saved-version'],
          'diff does not reproduce the observed focused successor/negative graph outcome')
    check(result['total_frames']==2443 and sum(c['frames'] for c in result['cases'][:15])==422,'physical total mismatch')
    distinct=result['cases'][11]
    labels=[f'{n}/{rev}:{stage}' for n in (1,7,8,9,15,16,17,31,32,33) for rev in (False,True) for stage in ('first','repeat')]
    check(distinct['inner_58_results']==[0]*40 and [r['label'] for r in distinct['raw_value_frames']]==labels and
        all(r['equal'] for r in distinct['raw_value_frames']),'incomplete distinct accumulation')
    nonleaf=result['cases'][12]
    check([(r['label'],r['equal']) for r in nonleaf['raw_value_frames']]==[('retained abs operand',True),
        ('public leaf gradient',True)],'nonleaf route differential changed')
    check(all(len(row['raw_word_failures'])==1 and len(row['raw_word_failures'][0]['diff'])==2
             for row in result['cases'][-3:-1]),'wrong fan-in difference count')
    matrix=result['cases'][-1]
    check(matrix['inner_58_results']==[0]*202 and len(matrix['raw_value_frames'])==202 and
          matrix['multiply_word_classes']['dual_nan']['mismatched']==314 and
          all(x['mismatched']==0 for k,x in matrix['multiply_word_classes'].items() if k!='dual_nan'),
          'wrong native fallback boundary')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw',type=Path); a=parser.parse_args()
    print(json.dumps(read(a.raw),indent=2,sort_keys=True))
