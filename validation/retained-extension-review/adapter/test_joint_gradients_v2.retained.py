#!/usr/bin/env python3
"""Independent finite contracts for the real gradient/Adam/distribution owner.

Only a pinned packet/own-child observer is imported. All numerical expectations
are exact dyadic mathematics or previously observed fixed Torch bytes. This is
not a new Torch execution or a claim to complete any additional PyTorch schema.
"""
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import random
import signal
import struct
import traceback

sha = lambda data: hashlib.sha256(data).hexdigest()
packq = lambda values: struct.pack('<' + 'Q' * len(values), *values)
WIRE_SHA = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT_SHA = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
OBSERVATIONS_SHA = 'c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467'


class MT:
    """Independent CPython MT engine with the standard 624-word scalar seed."""
    def __init__(self, seed):
        self.seed = seed
        words = [seed & 0xffffffff]
        for i in range(1, 624):
            old = words[-1]
            words.append((1812433253 * (old ^ (old >> 30)) + i) & 0xffffffff)
        self.engine = random.Random(0)
        self.engine.setstate((3, tuple(words + [624]), None))
        self.draws = 0

    def next(self):
        self.draws += 1
        return self.engine.getrandbits(32)

    def advance(self, n):
        for _ in range(n):
            self.next()

    def state(self):
        words = self.engine.getstate()[1]
        index = words[-1] if self.draws else 0
        raw = bytearray(5056)
        struct.pack_into('<QIIQ', raw, 0, self.seed, 625-index if self.draws else 1, 1, index)
        struct.pack_into('<624Q', raw, 24, *words[:624])
        return bytes(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for field in ['binary', 'source', 'protocol', 'distribution-observations', 'output']:
        ap.add_argument('--' + field, type=Path, required=True)
    ap.add_argument('--expected-native-sha256', required=True)
    ap.add_argument('--expected-source-sha256', required=True)
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary, source, protocol = a.binary.resolve(), a.source.resolve(), a.protocol.resolve()
    client = protocol.with_name('owner_client.py')
    paths = {binary:a.expected_native_sha256, source:a.expected_source_sha256,
             protocol:WIRE_SHA, client:CLIENT_SHA, a.distribution_observations.resolve():OBSERVATIONS_SHA}
    input_hashes = {str(p):sha(p.read_bytes()) for p in paths}
    for path, expected in paths.items():
        assert input_hashes[str(path)] == expected, str(path)
    script = Path(__file__).resolve()
    script_bytes = script.read_bytes()
    (out/'executed-test.py').write_bytes(script_bytes)
    input_hashes[str(script)] = sha(script_bytes)
    spec = importlib.util.spec_from_file_location('independent_joint_wire', protocol)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    observed = json.loads(a.distribution_observations.read_bytes())
    assert observed['status']=='OBSERVED' and observed['cases']==183 and observed['probe_error_count']==0
    assert len(observed['oracle_provenance']['checks'])==11 and all(observed['oracle_provenance']['checks'].values())
    bernoulli = next(row for row in observed['rows'] if row['name']=='bernoulli-functional-autograd')
    before_rng = bytes.fromhex(observed['states'][bernoulli['state_before']]['hex'])
    after_rng = bytes.fromhex(observed['states'][bernoulli['state_after']]['hex'])
    reference_seed = struct.unpack_from('<Q',before_rng)[0]
    calibration = MT(reference_seed)
    assert calibration.state()==before_rng
    calibration.advance(4)
    assert calibration.state()==after_rng

    report = {'schema':'heptatorch.independent-gradient-adam-distributions-contracts.v1',
              'status':'RUNNING','native_source_sha256':a.expected_source_sha256,
              'native_executable_sha256':a.expected_native_sha256,'native_executed':True,
              'Torch_executed_here':False,'production_host_numerics':False,
              'test_numerics':'Exact dyadic Fractions and raw binary32 bits; no floating tolerance.',
              'scope':'Six finite shared-owner contracts; no repeated profile executions counted as API coverage.',
              'existing_actual_reference_sha256':OBSERVATIONS_SHA,
              'existing_actual_case_used':'bernoulli-functional-autograd',
              'independent_MT_calibration':{'seed':reference_seed,'raw32_draws':4,
                  'before_sha256':sha(before_rng),'after_sha256':sha(after_rng)},
              'source_custody_before':input_hashes,'cases':[], 'frames_before_EOF':0,
              'exact_assertions':0,'exact_binary32_words':0,'whole_MT_state_comparisons':0,
              'whole_MT_bytes_compared':0,'whole_owner_rollback_bytes':0,
              'private_committed_node_audits':0,'full_PyTorch_rewrite_complete':False,
              'pytorch_api_denominator':2590}
    active = None

    def eq(actual, expected, label):
        assert actual == expected, {'label':label,'actual':repr(actual)[:1600],'expected':repr(expected)[:1600]}
        active['exact_assertions'] += 1

    def one(s, *args):
        return s.run([args])[0]

    def make(s, shape, values, requires=0):
        return s.run([wire.tensor(shape, [float(v) for v in values], requires)])[0]['value']

    def meta(s, t):
        return wire.meta(one(s,13,t))

    def identity(s, t):
        m = meta(s,t)
        return (m['storage_id'],m['version_id'])

    def words(s, t, expected, code=39, raw=False):
        expected_bits = list(expected) if raw else [wire.bits(float(v)) for v in expected]
        rec=one(s,code,t,0,len(expected_bits))
        eq(rec['value'],len(expected_bits),'logical value count')
        got=list(struct.unpack('<'+'I'*len(expected_bits),rec['payload']))
        eq(got,expected_bits,'raw binary32 words including signed zero')
        active['exact_binary32_words'] += len(got)

    def versions(s, tensors, expected):
        eq([meta(s,t)['version'] for t in tensors],list(expected),'absolute Version counters')

    def grad(s, p, expected=None):
        g=one(s,91,p)['value']
        eq(one(s,59,p)['value'],int(g!=0),'presence agrees with real public gradient')
        if expected is not None:
            eq(g,expected,'same actual public gradient Tensor token')
        return g

    def opt(s, params, algorithm=1, eps=1, wd=0, momentum=.5):
        if algorithm==0:
            cfg=[wire.bits(v) for v in (.125,momentum,0,wd)]+[0,0]
        else:
            cfg=[wire.bits(v) for v in (.125,0,0,eps,wd)]+[0,0,0]
        groups=[1,len(params),0]+[0]*len(cfg)+list(params)
        rec=one(s,71,algorithm,wire.Payload(packq(cfg)),wire.Payload(packq(groups)))
        eq((rec['value']>>24)&255,8,'actual kind8 Optimizer')
        eq(struct.unpack('<4Q',rec['payload']),(1,len(params),len(set(params)),len(params)-len(set(params))),
           'actual optimizer membership inventory')
        return rec['value']

    def state(s, optimizer, occurrence=0):
        return tuple(rec['value'] for rec in s.run([(77,optimizer,0,occurrence,k) for k in (1,2,3)]))

    def generator(s, seed):
        g=one(s,60,seed)['value']; reference=MT(seed)
        rng(s,g,reference)
        return g,reference

    def rng(s, g, reference):
        raw=one(s,63,g)['payload']
        eq(raw,reference.state(),'whole 5056-byte independently advanced MT state')
        active['whole_MT_state_comparisons']+=1
        active['whole_MT_bytes_compared']+=len(raw)

    def refusal(s, commands, expected):
        before=s.owner()
        status,records=wire.parse(s.exchange(wire.packet(commands)))
        after=s.owner()
        eq((status,records),(expected,[]),'whole-frame ordinary refusal')
        eq(after,before,'entire owner pointer length and bytes rolled back')
        index=len(active['rollbacks'])+1
        path=out/active['name']
        (path/f'rollback-{index:02d}-before.bin').write_bytes(before['bytes'])
        (path/f'rollback-{index:02d}-after.bin').write_bytes(after['bytes'])
        active['rollbacks'].append({'status':expected,'bytes':before['length'],
             'before_sha256':sha(before['bytes']),'after_sha256':sha(after['bytes'])})
        active['whole_owner_rollback_bytes']+=before['length']

    def partial(s, commands, expected_payload, completed):
        status,records=wire.parse(s.exchange(wire.packet(commands)))
        eq(status,0,'structured runtime error commits outer frame')
        eq([r['opcode'] for r in records],[c[0] for c in commands[:completed]],'exact executed prefix; suffix absent')
        eq(records[-1]['value'],expected_payload[0],'runtime error scalar')
        eq(struct.unpack('<4Q',records[-1]['payload']),expected_payload,'exact runtime-error payload')
        return records

    def weak_live(s, weak, present):
        if not present:
            refusal(s,[(43,weak)],6)
            return
        hold=one(s,43,weak)['value']
        eq((hold>>24)&255,6,'weak locks to actual Hold')
        one(s,45,hold)

    def audit(s):
        eq(one(s,53)['value'],1,'native graph Storage Version Gradient Generator Optimizer audit')
        raw=s.owner()['bytes']
        u=lambda at:struct.unpack_from('<Q',raw,at)[0]
        eq(raw[64:72],b'HTRET001','exact combined numeric state tag')
        eq(raw[139264:139272],b'HTORET01','exact combined backing state tag')
        eq([u(i) for i in (112,3968,3976,3984,3992,4000,4008)],[0]*7,'all committed derivative/GC/staging controls cleared')
        slots=u(139264+16)
        poolbase=139264+144+552*slots
        checked=0
        for node in range(1,u(8)):
            page=u(4096+(node>>8)*8)
            if not page:
                continue
            eq(page>>63,1,'descriptor page is actual relative pool offset')
            at=poolbase+(page&((1<<63)-1))+(node&255)*576
            assert at+576<=len(raw)
            if u(at+304):
                eq([u(at+i) for i in range(448,576,8)],[0]*16,'no incoming cache/Storage/Version/origin survives commit')
                checked+=1
        active['private_committed_node_audits']+=checked
        active['private_audits'].append({'owner_bytes':len(raw),'owner_sha256':sha(raw),'live_nodes_checked':checked})

    def run(name, action):
        nonlocal active
        active={'name':name,'status':'RUNNING','frames':0,'exact_assertions':0,
                'exact_binary32_words':0,'whole_MT_state_comparisons':0,'whole_MT_bytes_compared':0,
                'whole_owner_rollback_bytes':0,'private_committed_node_audits':0,'private_audits':[], 'rollbacks':[]}
        report['cases'].append(active)
        directory=out/name;directory.mkdir()
        s=wire.Session(binary,directory=directory/'transport')
        try:
            active['contract']=action(s)
            audit(s)
            s.finish()
            active['status']='PASS'
        except BaseException as error:
            active.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
            if s.poll() is None:
                os.kill(s.pid,signal.SIGKILL)
            try:s.finish(expected=-signal.SIGKILL)
            except BaseException as cleanup:active['cleanup_error']=repr(cleanup)
        active['frames']=s.frames
        report['frames_before_EOF']+=s.frames
        for field in ['exact_assertions','exact_binary32_words','whole_MT_state_comparisons',
                      'whole_MT_bytes_compared','whole_owner_rollback_bytes','private_committed_node_audits']:
            report[field]+=active[field]
        (directory/'assertions.json').write_text(json.dumps(active,indent=2)+'\n')
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

    def strided_layout(s):
        original=[Fraction(i) for i in range(1,7)]
        p=make(s,(2,3),original,1)
        gb=make(s,(3,2),[1,-1,-1,1,0,0])
        g=one(s,6,gb,0,1)['value']; expected=[1,-1,0,-1,1,0]
        gid=identity(s,g);weak=one(s,42,g)['value'];saved=one(s,46,g)['value']
        eq(meta(s,g)['strides'][:2],[1,2],'assigned actual transposed grad layout')
        one(s,92,p,g);grad(s,p,g);words(s,p,expected,40)
        adam=opt(s,[p]);sgd=opt(s,[p],0)
        one(s,72,adam);st=state(s,adam)
        one(s,72,sgd);mom=one(s,77,sgd,0,0,0)['value']
        words(s,p,[x-Fraction(3*d,16) for x,d in zip(original,expected)])
        words(s,st[1],expected);words(s,st[2],[v*v for v in expected]);words(s,mom,expected)
        eq(meta(s,st[1])['strides'][:2],[3,1],'Adam first moment follows parameter layout')
        eq(meta(s,st[2])['strides'][:2],[3,1],'Adam second moment follows parameter layout')
        eq(meta(s,mom)['strides'][:2],[1,2],'first SGD momentum follows actual G layout')
        versions(s,[p,g,*st,mom],[2,0,1,1,2,0])
        kept=[identity(s,t) for t in [p,*st,mom]]
        s.run([(41,gb),(41,g),(41,p)]+[(41,t) for t in st]+[(41,mom),(33,0)])
        p2=one(s,75,adam,0,0)['value'];g2=grad(s,p2)
        st2=state(s,adam);mom2=one(s,77,sgd,0,0,0)['value']
        eq(identity(s,g2),gid,'actual gradient survives private-root reset')
        eq(g2!=g,True,'closed gradient public token never reactivated')
        eq([identity(s,t) for t in [p2,*st2,mom2]],kept,'parameter and all real states retain S/V through reset')
        words(s,p2,expected,40);audit(s)
        one(s,70,p2);eq(grad(s,p2),0,'clear removes binding but not externally held G')
        s.run([(72,adam),(72,sgd)])
        versions(s,[p2,*st2,mom2],[2,1,1,2,0])
        words(s,g2,expected);one(s,41,g2);weak_live(s,weak,True)
        one(s,48,saved);weak_live(s,weak,False)
        s.run([(78,adam),(78,sgd),(44,weak)])
        return {'real_strided_grad_reads':True,'Adam_parameter_layout_vs_SGD_gradient_layout':True,
                'private_reset_binding_and_fresh_public_generation':True,'None_skip_preserves_real_states':True}

    def moment_alias(s):
        values=[Fraction(1),Fraction(-2),Fraction(1,2),Fraction(-1,2)]
        first=[1,-1,1,-1]
        p=make(s,(4,),values,1);g0=make(s,(4,),first)
        one(s,92,p,g0);adam=opt(s,[p]);one(s,72,adam);st=state(s,adam)
        avg=st[1];avgid=identity(s,avg);weak=one(s,42,avg)['value']
        one(s,92,p,avg);grad(s,p,avg);one(s,41,g0)
        saved=one(s,46,avg)['value'];gen,mt=generator(s,719)
        prob=make(s,(4,),[1,0,float('nan'),1])
        mt.advance(2)
        partial(s,[(72,adam),(86,avg,gen,prob,1),(95,st[0],wire.bits(.5)),(78,adam)],(7,avg,2,2),2)
        grad(s,p,avg);eq(identity(s,avg),avgid,'state/gradient alias has one actual S/V')
        words(s,p,[x-Fraction(d,8) for x,d in zip(values,first)])
        words(s,p,[1,0,1,-1],40);versions(s,[p,*st],[2,2,2,4]);rng(s,gen,mt)
        refusal(s,[(47,saved)],6);snapshot=one(s,46,avg)['value'];audit(s)
        one(s,72,adam)
        words(s,p,[x-Fraction(d,8)-Fraction(e,16) for x,d,e in zip(values,first,[1,0,1,-1])])
        words(s,avg,[1,0,1,-1]);words(s,st[2],[1,0,1,1]);versions(s,[p,*st],[3,3,3,6])
        refusal(s,[(47,snapshot)],6)
        one(s,70,p);eq(grad(s,p),0,'None unbinds actual moment without destroying optimizer state')
        refusal(s,[(92,p,avg),(72,adam),(67,gen,1),(86,avg,gen,prob,2)],7)
        eq(grad(s,p),0,'ordinary rollback restores previous None binding');rng(s,gen,mt)
        s.run([(41,avg),(78,adam)])
        weak_live(s,weak,True);one(s,48,saved);weak_live(s,weak,True)
        one(s,48,snapshot);weak_live(s,weak,False)
        s.run([(44,weak),(66,gen)])
        return {'public_gradient_is_same_real_Adam_state_Tensor':True,'partial86_changes_visible_through_binding':True,
                'no_unperformed_Version_bump':True,'whole_owner_rollback_restores_binding_and_Adam_and_Generator':True}

    def graph41(s):
        p=make(s,(4,),[.5]*4,1);p_identity=identity(s,p)
        gen,mt=generator(s,reference_seed)
        y=one(s,82,p,gen,0)['value'];mt.advance(4)
        words(s,y,bernoulli['result']['words'],raw=True)
        seed=make(s,(4,),[-0.,float('nan'),float('inf'),-1.])
        one(s,57,y,seed,0);g=grad(s,p)
        actual=meta(s,g);wanted=bernoulli['gradient']
        eq(actual['shape'][:actual['rank']],wanted['shape'],'existing actual Bernoulli grad shape')
        eq(actual['strides'][:actual['rank']],wanted['strides'],'existing actual Bernoulli grad strides')
        eq(actual['version'],wanted['version'],'existing actual Bernoulli first gradient Version0')
        words(s,g,wanted['words'],raw=True)
        eq(identity(s,g) not in [identity(s,p),identity(s,y),identity(s,seed)],True,'fresh zero gradient owns distinct S/V')
        weak=one(s,42,g)['value'];saved=one(s,46,g)['value'];gid=identity(s,g)
        adam=opt(s,[p]);one(s,72,adam);st=state(s,adam)
        versions(s,[p,g,*st],[1,0,1,1,2]);words(s,p,[.5]*4)
        rng(s,gen,mt);audit(s)
        s.run([(41,p),(41,g),(41,y),(41,seed)]+[(41,t) for t in st]+[(33,0)])
        p2=one(s,75,adam,0,0)['value'];g2=grad(s,p2)
        eq(identity(s,p2),p_identity,'private parameter real identity retained')
        eq(identity(s,g2),gid,'defined actual zero gradient retained through reset')
        s.run([(70,p2),(41,g2),(78,adam)])
        weak_live(s,weak,True);one(s,48,saved);weak_live(s,weak,False)
        s.run([(44,weak),(66,gen)])
        return {'existing_actual183_Bernoulli_gradient_metadata_matched':True,
                'fresh_gradient_Storage_Version_separate_from_seed':True,'zero_gradient_participates_in_Adam':True,
                'no_new_strided_Torch_observation_claim':True}

    def seed_alias_rng(s):
        initial=[Fraction(i) for i in range(1,7)]
        p=make(s,(2,3),initial,1);y=one(s,17,p,1,6,0,0,0)['value']
        seed=make(s,(6,),[1,-1,0,1,-1,0]);seedid=identity(s,seed)
        one(s,57,y,seed,0);g=grad(s,p)
        eq(identity(s,g),seedid,'view backward gives actual G shared seed Storage and Version')
        eq(meta(s,g)['strides'][:2],[3,1],'shared seed gradient actual reshaped layout')
        eq(g!=seed,True,'separate Tensor identity with shared real S/V')
        saved=one(s,46,g)['value'];weak=one(s,42,g)['value']
        gen,mt=generator(s,83);prob=make(s,(6,),[1,0,1,0,1,0])
        one(s,81,seed,gen,prob,1);mt.advance(6)
        words(s,g,[1,0,1,0,1,0]);versions(s,[seed,g],[1,1]);refusal(s,[(47,saved)],6)
        adam=opt(s,[p]);one(s,72,adam);st=state(s,adam)
        words(s,p,[x-Fraction(d,16) for x,d in zip(initial,[1,0,1,0,1,0])])
        words(s,st[1],[1,0,1,0,1,0]);versions(s,[p,*st],[1,1,1,2])
        one(s,70,p);one(s,95,seed,wire.bits(.5));one(s,72,adam)
        eq(grad(s,p),0,'clear holds None while external gradient alias changes')
        words(s,g,[Fraction(3,2),Fraction(1,2)]*3);versions(s,[p,*st],[1,1,1,2]);versions(s,[seed,g],[2,2])
        rng(s,gen,mt);audit(s)
        s.run([(41,seed),(41,g),(41,y),(78,adam)])
        weak_live(s,weak,True);one(s,48,saved);weak_live(s,weak,False)
        s.run([(44,weak),(66,gen)])
        return {'view_seed_S_V_are_actually_adopted':True,'RNG_mutation_reaches_real_gradient_alias_and_Adam':True,
                'clear_None_does_not_destroy_externally_held_gradient':True}

    def partial58_and86(s):
        p=make(s,(2,),[2,3],1);q=make(s,(2,),[5,6],1)
        coefficient=make(s,(2,),[3,3]);goodscale=make(s,(2,),[1,1])
        bad=one(s,2,3,p,coefficient)['value'];good=one(s,2,3,q,goodscale)['value']
        joined=one(s,2,1,bad,good)['value'];loss=one(s,4,7,joined)['value']
        adam=opt(s,[p,q]);one(s,95,coefficient,wire.bits(1))
        gen,mt=generator(s,412)
        # The frozen command58 contract counts successful leaf accumulators in
        # completed_nodes: sum, add, good multiplication, then q's accumulator.
        partial(s,[(58,loss,0,0),(72,adam),(67,gen,5)],(6,4,1,3),1)
        eq(grad(s,p),0,'failed dependency leaves bad leaf gradient None')
        g=grad(s,q);words(s,g,[1,1]);eq(meta(s,g)['version'],0,'successful mul contribution adopts fresh Version0')
        eq(one(s,76,adam,0,0)['value'],0,'stopped suffix did not create bad parameter state')
        eq(one(s,76,adam,0,1)['value'],0,'stopped suffix did not create good parameter state')
        rng(s,gen,mt);audit(s)
        saved=one(s,46,g)['value'];prob=make(s,(2,),[0,float('nan')])
        mt.advance(1)
        partial(s,[(72,adam),(86,g,gen,prob,1),(73,adam,1),(999,)],(7,g,1,1),2)
        words(s,q,[Fraction(79,16),Fraction(95,16)]);words(s,q,[0,1],40)
        eq(grad(s,q),g,'partial G prefix retains actual binding')
        eq(meta(s,g)['version'],0,'failed Bernoulli prefix has no extra Version')
        eq(one(s,47,saved)['value'],1,'Saved observes changed prefix under unchanged Version')
        words(s,saved,[0,1],49)
        eq(one(s,76,adam,0,0)['value'],0,'bad None parameter still has no Adam state')
        st=state(s,adam,1);words(s,st[1],[1,1]);words(s,st[2],[1,1]);versions(s,[q,*st],[1,1,1,2])
        rng(s,gen,mt);audit(s)
        one(s,72,adam);words(s,q,[Fraction(79,16),Fraction(94,16)])
        words(s,st[1],[0,1]);words(s,st[2],[0,1]);versions(s,[q,*st],[2,2,2,4])
        refusal(s,[(92,p,g),(72,adam),(67,gen,1),(999,)],4)
        eq(grad(s,p),0,'ordinary rollback restores failed leaf None')
        rng(s,gen,mt);s.run([(48,saved),(78,adam),(66,gen)])
        return {'58_commits_only_ready_leaf_real_gradient':True,'58_error_stops_optimizer_and_rng_suffix':True,
                '86_prefix_commit_updates_same_G_read_by_next_Adam':True,'temporary_incoming_owners_released_on_partial_exit':True}

    def overlapping_grad87(s):
        initial=[Fraction(1),Fraction(-2),Fraction(1,2),Fraction(-1,2)]
        p=make(s,(4,),initial,1);base=make(s,(1,),[1])
        cfg=wire.Payload(packq([1,4,0,0,0,0,0,0,0,0]))
        g=one(s,69,base,cfg,80)['value'];one(s,92,p,g);grad(s,p,g)
        saved=one(s,46,g)['value'];adam=opt(s,[p],2,wd=.25)
        gen,mt=generator(s,333);mt.advance(8)
        partial(s,[(72,adam),(87,g,gen,0x3fe0000000000000,1,1),(999,)],(2,g,0,8),2)
        st=state(s,adam)
        expected=[v*Fraction(31,32)-Fraction(1,16) for v in initial]
        words(s,p,expected);words(s,st[1],[1]*4);words(s,st[2],[1]*4)
        versions(s,[p,g,*st],[2,0,1,1,2]);rng(s,gen,mt);audit(s)
        mt.advance(8)
        partial(s,[(73,adam,1),(87,g,gen,0x3fe0000000000000,1,1),(72,adam)],(2,g,0,8),2)
        eq(grad(s,p),0,'clear prefix commits None before later overlap error')
        one(s,72,adam);versions(s,[p,*st],[2,1,1,2]);words(s,base,[1]);rng(s,gen,mt);audit(s)
        one(s,92,p,g);one(s,73,adam,0)
        words(s,g,[0]*4);words(s,base,[0]);versions(s,[base,g],[1,1]);refusal(s,[(47,saved)],6)
        one(s,72,adam)
        words(s,p,[v*Fraction(31,32) for v in expected]);versions(s,[p,*st],[4,2,2,4])
        s.run([(48,saved),(78,adam),(66,gen)])
        return {'overlapping_actual_G_is_consumed_logically_by_AdamW':True,
                '87_draws_commit_while_noise_and_failed_write_are_reclaimed':True,
                'zero_grad_False_mutates_actual_G_alias_once':True,'defined_zero_still_applies_AdamW_decay':True}

    for name,fn in [
        ('assigned-strided-G-Adam-vs-SGD-layout-private-roots',strided_layout),
        ('actual-G-is-Adam-moment-partial86-and-rollback',moment_alias),
        ('Bernoulli41-fresh-zero-gradient-objects-and-Adam',graph41),
        ('view-seed-gradient-alias-RNG-and-Adam',seed_alias_rng),
        ('partial58-ready-gradient-then-partial86-Adam',partial58_and86),
        ('overlapping-real-gradient-partial87-AdamW-clear',overlapping_grad87)]:
        run(name,fn)
    after={str(p):sha(p.read_bytes()) for p in paths}
    after[str(script)]=sha(script.read_bytes())
    assert after==input_hashes
    report.update(status='PASS' if all(c['status']=='PASS' for c in report['cases']) else 'FAIL',
                  source_custody_after=after,case_count=len(report['cases']),
                  passed=sum(c['status']=='PASS' for c in report['cases']),
                  failed=sum(c['status']!='PASS' for c in report['cases']))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','case_count','passed','failed','frames_before_EOF',
          'exact_assertions','exact_binary32_words','whole_MT_state_comparisons','whole_owner_rollback_bytes']},indent=2))
    if report['status']!='PASS':
        raise SystemExit(1)


if __name__=='__main__':
    main()
