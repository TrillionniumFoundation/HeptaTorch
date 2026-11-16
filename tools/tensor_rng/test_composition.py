#!/usr/bin/env python3
"""One native process: RNG -> forward -> backward -> SGD -> graph reclamation."""
import argparse,hashlib,json,struct
from pathlib import Path
from protocol import Session,full,bits,meta,reference as ref
from reference import Reference
from reference_fp import uniform,parts,add,rne

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--executable',type=Path,required=True)
    p.add_argument('--cpp-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backward-code',type=int,choices=(7,57,58),default=7)
    p.add_argument('--presence',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    image=a.executable.resolve();pin=hashlib.sha256(image.read_bytes()).hexdigest()
    s=Session(image,directory=out/'frames');r=s.run([full((8,),0,1),(60,1729)])
    t,g=[x['value'] for x in r];model=Reference(a.cpp_reference.resolve(),1729);checks=[]
    for iteration in range(12):
        backward=(a.backward_code,ref(4)) if a.backward_code==7 else (a.backward_code,ref(4),0,0)
        commands=[(8,t),(68,t,g,0,bits(1),1),(2,3,t,t),(4,7,ref(3)),backward,
            (9,t,bits(.125)),(39,t,0,8),(40,t,0,8),(13,t),(63,g),(33,1),(53,)]
        if a.presence:commands.append((59,t))
        r=s.run(commands);expected=[];gradients=[]
        for _ in range(8):
            x=uniform(model.word(),0,bits(1));xp=parts(x);grad=rne((xp[0]*2,xp[1]));gp=parts(grad)
            quarter=parts(rne((gp[0],gp[1]-3)));updated=rne(add(xp,(-quarter[0],quarter[1])))
            expected.append(updated);gradients.append(grad)
        assert r[6]['payload']==struct.pack('<8I',*expected)
        assert r[7]['payload']==struct.pack('<8I',*gradients)
        assert meta(r[8])['version']==2*(iteration+1) and r[9]['payload']==model.state()
        if a.presence:assert r[-1]['value']==1
        if a.backward_code==58:
            assert r[4]['value']==0 and len(r[4]['payload'])==32 and struct.unpack_from('<Q',r[4]['payload'])[0]==0
        checks.append({'update':iteration+1,'version':2*(iteration+1),'parameter_bits':expected,'gradient_bits':gradients})
    s.run([(66,g),(53,)])
    s.finish();assert hashlib.sha256(image.read_bytes()).hexdigest()==pin
    report={'schema':'heptatorch.tensor-rng-training-composition.v1','status':'PASS','executable_sha256':pin,
        'backward_code':a.backward_code,'gradient_presence_checked':a.presence,'frames':s.frames,'updates':12,
        'exact_values':192,'full_generator_states':12,'single_native_process':True,'checks':checks}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()
