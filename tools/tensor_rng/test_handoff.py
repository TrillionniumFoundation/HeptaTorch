#!/usr/bin/env python3
"""Move genuine producer-owned states through both complete native runtimes."""
import argparse,hashlib,json,struct
from pathlib import Path
from protocol import Session,packet,parse,full,bits,meta,reference as ref

SHA=lambda b:hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    image=a.build.resolve()/'state-handoff-session';build=json.loads((a.build/'report.json').read_text())
    assert SHA(image.read_bytes())==build['executable_sha256']
    s=Session(image,directory=out/'frames');checks=[]
    def route(which,commands=(),error=0):
        before=s.owner() if error else None
        body=struct.pack('<Q',which)+(packet(commands) if which<2 else b'')
        status,records=parse(s.exchange(body));assert status==error,(status,error)
        if error:assert s.owner()==before
        return records
    old=route(0,[full((3,),7,1),(46,ref(1)),(53,)])
    t,saved=old[0]['value'],old[1]['value'];old_owner=s.owner();assert old_owner['bytes'][64:72]==b'HTSTATE3'
    route(1,[(60,123),(68,t,ref(1),0,bits(1),1)],7)
    returned=route(0,[(39,t,0,3),(49,saved,0,3),(53,)])
    assert returned[0]['payload']==returned[1]['payload']==struct.pack('<3f',7,7,7)
    checks.append('old-state-new-reject-owner-unchanged-old-recovery')
    route(2);assert s.owner()['length']==0 and not s.mapped(old_owner['pointer'])
    new=route(1,[full((3,),0),(60,99),(68,ref(1),ref(2),0,bits(1)),(63,ref(2)),(13,ref(1)),(53,)])
    t,g=new[0]['value'],new[1]['value'];state=new[3]['payload'];new_owner=s.owner()
    assert new_owner['bytes'][64:72]==b'HTSTRNG1'
    route(0,[(12,t,0,bits(99)),(33,0)],7)
    recovered=route(1,[(63,g),(13,t),(39,t,0,3),(53,)])
    assert recovered[0]['payload']==state and meta(recovered[1])['version']==1
    checks.append('new-state-old-reject-owner-unchanged-new-recovery')
    route(3);assert s.owner()['length']==0 and not s.mapped(new_owner['pointer'])
    route(0,[full((1,),13),(53,)]);route(2);route(1,[(60,5489),(62,ref(1)),(53,)]);route(3)
    checks.append('both-close-exports-consume-and-reinitialize-empty-owner')
    s.finish();report={'schema':'heptatorch.tensor-rng-owned-handoff.v1','status':'PASS','executable_sha256':build['executable_sha256'],
        'build_receipt_sha256':SHA((a.build/'report.json').read_bytes()),'checks':checks,'cases':len(checks),'frames':s.frames,
        'raw_owner_pointer_import':False,'producer_states_created_by_actual_hia':True,'closed_owner_mappings_absent':True}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
