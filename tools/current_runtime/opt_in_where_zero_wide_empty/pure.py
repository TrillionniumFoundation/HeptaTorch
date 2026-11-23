"""Pure native trace/owner verifier without Torch, runtime, or fixture generator."""
import hashlib,json,math,sys,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from read_old import Owner,frames,req,resp,sha,q
W={0:4,1:1,2:8,4:8,5:2,6:2}
PIN={'baseline':'77e2ee440e0cb2f568d350a19e2bedd3138ffb407767f4d224f4567ea7fd393b',
     'candidate':'322b419c867210439f49313ee8d43ff1c220c4f15ae4d4ee673c4df78434683b'}
def same_inputs(o,opp,handles,axs):
    for h,e in zip(handles,axs,strict=True):
        a=o.node(h); tag=e['dtype']; dims=e['shape']; raw=bytes.fromhex(e['byte_hex']); n=len(e['words'])
        assert a[72]==tag and a[0:2]==(len(dims),n) and list(a[12:12+len(dims)])==dims
        assert a[4]==a[5]==a[29]==a[83]==0
        assert o.storage(a[27])[1:]==(n*W[tag],raw)
        if opp:
            assert a==opp.node(h) and o.row(1,h)[1]==opp.row(1,h)[1] and o.row(2,a[27])[1]==opp.row(2,a[27])[1]
            assert o.row(3,a[28])[1]==opp.row(3,a[28])[1] and o.storage(a[27])==opp.storage(a[27])
def snap(p,x):
    b=zlib.decompress((p/'owners'/(x['sha256']+'.zlib')).read_bytes())
    assert sha(b)==x['sha256'] and len(b)==x['length']
    return Owner(b)
def inspect(p,rows):
    led=json.loads((p/'report.json').read_bytes()); transport=json.loads((p/'transport.json').read_bytes())
    mode=led['mode']
    assert led['no_torch'] and PIN[mode]==led['elf']==transport['executable_sha256']
    assert transport['exit_code']==0 and transport['responses_before_stdin_eof'] and not (p/'error.json').exists()
    rawq=list(frames(p/'requests.frames.bin',b'HTOWNR01')); qs=list(map(req,rawq))
    rs=list(map(resp,frames(p/'responses.frames.bin',b'HTOWNS01')))
    assert len(qs)==len(rs)==led['frames']==transport['frames']
    counts={k:0 for k in ('positive','explicit','incompatible','all_baseline_denials','owner_rollbacks','aliases','life_inputs_first','life_result_first','weak_expired','collision','oob_read','oob_write','canonical_old','unsupported_112','unsupported_113','repeat')}
    seen=set()
    def check_deny(g,expected,commands):
        i=g['frame']-1
        assert qs[i]==commands and rs[i]==(expected,[]) and sha(rawq[i])==g['request_sha256']
        a,b=snap(p,g['pre']),snap(p,g['post'])
        assert a.b==b.b and g['pre']['ptr']==g['post']['ptr']
        counts['owner_rollbacks']+=1
        return a
    def read(i,h,n,raw):
        i-=1
        assert qs[i]==[(13,h,0,0,0,0,0,0),(98,h,0,n,0,0,0,0)]
        stat,data=rs[i]
        assert stat==0 and data[0][:2]==(13,h) and data[1]==(98,n,raw)
        return q(data[0][2],16)
    for e in led['rows']:
        label=e['label'];f=rows[label]; assert label not in seen;seen.add(label)
        h=e['handles']; cmd=(129,*h,0,0,0,0)
        classification=f['classification']
        if e['kind']=='refusal':
            assert mode=='baseline' or classification in ('explicit','incompatible')
            a=check_deny(e,7,[cmd]);same_inputs(a,None,h,f['operands'])
            counts['all_baseline_denials' if mode=='baseline' else classification]+=1
            continue
        assert mode=='candidate' and classification=='canonical'
        i=e['op']-1;out=e['out']
        assert qs[i]==[cmd] and rs[i]==(0,[(129,out,b'')]) and out not in h
        pre=snap(p,e['pre']);post=snap(p,e['post'])
        same_inputs(pre,post,h,f['operands'])
        d=post.node(out);sh=f['shape']; st=f['strides'];rank=len(sh)
        assert d[72]==4 and d[83]==d[29]==d[4]==d[5]==0 and d[:2]==(rank,0)
        assert list(d[12:12+rank])==sh and list(d[16:16+rank])==st
        assert st==[math.prod(max(1,x) for x in sh[i+1:]) for i in range(rank)]
        ss,vv=d[27:29]; ids=[post.node(a)[27:29] for a in h]
        assert ss not in {j[0] for j in ids} and vv not in {j[1] for j in ids}
        assert post.storage(ss)[1:]==(0,b'') and post.row(3,vv)[1][2]==0
        owner=post.row(1,out)[1];assert owner[2:6]==(ss,vv,0,0)
        md=read(e['read'],out,0,b'')
        assert md[:3]==(rank,0,0) and list(md[3:3+rank])==sh and list(md[7:7+rank])==st
        assert md[11:16]==(ss,vv,0,0,0)
        if 'alias' in e:
            a=e['alias']; j=e['alias_call']-1; record,=qs[j]
            assert record[:2]==(111,out) and rs[j]==(0,[(111,a,b'')])
            assert record[3]==80 and q(rawq[j],10,record[2])==(rank,*sh,*[0]*(4-rank),*st,*[0]*(4-rank),0)
            assert read(e['alias_read'],a,0,b'')==read(e['alias_survives_root'],a,0,b'')==md
            counts['aliases']+=1
            j=e['repeat_call']-1; h2=e['repeat']
            assert qs[j]==[cmd] and rs[j]==(0,[(129,h2,b'')])
            rmd=read(e['repeat_read'],h2,0,b'')
            assert rmd[:11]==md[:11] and rmd[11]!=md[11] and rmd[12]!=md[12] and rmd[13:]==(0,0,0)
            counts['repeat']+=1
            a,b,c,d,g=e['checks']
            check_deny(a,2,[(98,out,0,1,0,0,0,0)]);counts['oob_read']+=1
            check_deny(b,7,[(12,out,0,0x3f800000,0,0,0,0)]);counts['oob_write']+=1
            check_deny(c,4,[cmd,(65500,0,0,0,0,0,0,0)]);counts['collision']+=1
            for op,item,shape in ((112,d,sh),(113,g,[0,1] if rank==1 else [0])):
                readq,=qs[item['frame']-1]
                assert readq[:2]==(op,out) and readq[3]==40 and q(rawq[item['frame']-1],5,readq[2])==(len(shape),*shape,*[0]*(4-len(shape)))
                check_deny(item,7,[readq]);counts['unsupported_'+str(op)]+=1
        elif 'result_survives_inputs' in e:
            assert read(e['result_survives_inputs'],out,0,b'')==md
            counts['life_inputs_first']+=1
        else:
            x=f['operands'][1];read(e['inputs_survive_result'],h[1],len(x['words']),bytes.fromhex(x['byte_hex']))
            counts['life_result_first']+=1
        j=e['weak_frame']-1
        assert qs[j]==[(42,out,0,0,0,0,0,0)] and rs[j]==(0,[(42,e['weak'],b'')])
        check_deny(e['expired'],6,[(43,e['weak'],0,0,0,0,0,0)])
        counts['weak_expired']+=1
        counts['positive']+=1
        counts['canonical_old']+=bool(f.get('inherited'))
    assert seen==set(rows)
    if mode=='baseline':assert counts['all_baseline_denials']==192
    else:
        assert counts['positive']==112 and counts['canonical_old']==8 and counts['explicit']==32 and counts['incompatible']==48
        assert counts['aliases']==counts['repeat']==counts['unsupported_112']==counts['unsupported_113']==4 and counts['oob_read']==counts['oob_write']==counts['collision']==4
    return dict(**counts,frames=len(qs),request_stream=sha((p/'requests.frames.bin').read_bytes()),response_stream=sha((p/'responses.frames.bin').read_bytes()))
if __name__=='__main__':
    f=Path(sys.argv[1]).read_bytes()
    assert sha(f)=='10648ec9e86ee7b5cd7763b1ee088f6299dae2a796ebc30b991e39fee3955c51'
    rows={r['label']:r for r in json.loads(f)}
    print(json.dumps({Path(d).name:inspect(Path(d),rows) for d in sys.argv[2:]},sort_keys=True,indent=2))
