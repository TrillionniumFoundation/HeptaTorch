#!/usr/bin/env python3
"""Replay nine immutable actual-Torch empty NativeDropout recipes in native HIA.

Python only encodes requests, reads the child-owned bytes, and compares observations.
No Tensor arithmetic, AD execution, lifetime decision, or expected numeric result is
implemented here. Native and Python diagnostic surfaces are identified separately.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import traceback

REFERENCE = 'de7db2f985873be797bd7e167fb374f82b5690021e0a0541d2fb4f7a36b7ad83'
PROBE = '973421e3f6a239bad6f46a076b68666108ba976b3d1c7e0a4064a7d3ebaa33b6'
SOURCE = '6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3'
IMAGE = '830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51'
WIRE = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches', 'direct_url_matches',
          'direct_url_archive_sha256_matches', 'pip_report_url_matches',
          'pip_report_archive_sha256_matches', 'pip_report_version_matches', 'pip_report_requested'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def status_for(event):
    if event['accepted']:
        return 0
    message = event['error']['message']
    assert event['error']['type'] == 'RuntimeError'
    if 'moved into the graph interior' in message:
        return 11
    if 'second time' in message:
        return 10
    if 'modified by an inplace operation' in message:
        return 6
    if 'only change requires_grad flags of leaf' in message:
        return 5
    raise AssertionError(('unmapped actual exception category', message))


class Replay:
    def __init__(self, wire, args, directory, reference, case, variant=None):
        directory.mkdir(parents=True, exist_ok=False)
        self.w, self.args, self.directory = wire, args, directory
        self.reference, self.case, self.variant = reference, case, variant
        self.session = wire.Session(args.binary.resolve(), directory=directory/'transport')
        self.assertions = self.rng_bytes = self.rollback_bytes = self.snapshots = 0
        self.tensor_ids, self.storage_ids, self.sequences = {}, {}, {}
        self.public, self.weak_handles, self.optimizers = set(), set(), set()
        self.trace, self.owners, self.graph_observations = [], [], []
        self.phase = 'setup'
        self.generator = self.make(60, 1729)
        self.seed = self.full([2, 0, 3], 1)

    def eq(self, got, expected, label):
        self.assertions += 1
        if got != expected:
            def short(value):
                return {'bytes':len(value), 'sha256':sha(value)} if isinstance(value, bytes) else value
            raise AssertionError({'phase':self.phase, 'field':label, 'native':short(got), 'actual':short(expected)})

    def one(self, *command):
        status, records = self.w.parse(self.session.exchange(self.w.packet([command])))
        self.eq(status, 0, 'wire frame status')
        self.eq(len(records), 1, 'one result')
        self.eq(records[0]['opcode'], command[0], 'wire result opcode')
        def value(v):
            return {'payload_bytes':len(v.data), 'sha256':sha(v.data)} if isinstance(v, self.w.Payload) else v
        self.trace.append({'phase':self.phase, 'command':[value(v) for v in command],
                           'value':records[0]['value'], 'payload_hex':records[0]['payload'].hex()})
        return records[0]

    def make(self, *command):
        return self.one(*command)['value']

    def tensor_make(self, *command):
        t = self.make(*command)
        self.public.add(t)
        return t

    def full(self, shape, value, requires=0):
        return self.tensor_make(*self.w.full(shape, value, requires))

    def drop(self, tensor):
        self.eq(tensor in self.public, True, 'drop an owned public Tensor')
        self.one(41, tensor)
        self.public.remove(tensor)

    def empty(self, requires=True):
        base = self.full([0], 0)
        cfg = struct.pack('<10Q', 3, 2, 0, 3, 0, 17, 101, 2, 0, 7)
        view = self.tensor_make(69, base, self.w.Payload(cfg), 80)
        # The actual recipe explicitly detaches its as_strided view.
        result = self.tensor_make(19, view)
        self.drop(view)
        self.drop(base)
        if requires:
            self.one(90, result, 1)
        return result

    def multiply(self, tensor):
        scalar = self.full([], 2)
        result = self.tensor_make(2, 3, tensor, scalar)
        self.eq(self.node(tid=result)[1][40], 3, 'literal x * 2 uses native Mul operation 3')
        self.drop(scalar)
        return result

    def dropout(self, tensor):
        result = self.one(99, tensor, self.generator, 0x3fe0000000000000, 1)
        self.eq(result['value'], tensor, 'empty NativeDropout returns the same actual Tensor')
        self.eq(len(result['payload']), 8, 'one actual mask Tensor token')
        mask = struct.unpack('<Q', result['payload'])[0]
        self.eq(mask not in self.public, True, 'new mask public identity')
        self.public.add(mask)
        return result['value'], mask

    def save_owner(self, label, owner=None):
        owner = self.session.owner() if owner is None else owner
        name = f'owner-{len(self.owners):03d}-{label}.bin'
        (self.directory/name).write_bytes(owner['bytes'])
        record = {'path':name, 'bytes':owner['length'], 'sha256':sha(owner['bytes']),
                  'pointer':owner['pointer'], 'live':owner['live']}
        self.owners.append(record)
        return owner

    def node(self, tid=None, node=None, raw=None):
        raw = self.session.owner()['bytes'] if raw is None else raw
        def u(at):
            assert 0 <= at <= len(raw)-8, ('bounded native read', at, len(raw))
            return struct.unpack_from('<Q', raw, at)[0]
        self.eq(raw[64:72], b'HTDTYP03', 'fixed native state tag')
        base = 139264
        self.eq(raw[base:base+8], b'HTODTY03', 'fixed backing tag')
        slots = u(base+16)
        if tid is not None:
            self.eq((tid >> 24) & 255, 1, 'Tensor token kind')
            slot = tid & 0xffffff
            assert 0 < slot <= slots
            row = base+128+(slot-1)*64
            self.eq(u(row), tid >> 32, 'public token generation')
            self.eq(u(row+8), 1, 'public token live')
            node = u(row+48)
        assert node is not None and 0 < node < u(8)
        page = u(4096+(node >> 8)*8)
        self.eq(page >> 63, 1, 'pool-relative descriptor page')
        at = base+144+552*slots+(page & ((1 << 63)-1))+(node & 255)*640
        fields = {offset:u(at+offset) for offset in range(0,640,8)}
        self.eq(fields[304], 1, 'native descriptor remains live')
        if tid is not None:
            self.eq(fields[192], tid, 'public Tensor resolves to actual node')
            self.eq(fields[200], 1, 'public owner exists')
        return node, fields

    def identity(self, mapping, actual, native, label):
        assert isinstance(actual, str)
        if actual in mapping:
            self.eq(native, mapping[actual], label+' stable identity')
        else:
            self.eq(native not in mapping.values(), True, label+' distinct identity')
            mapping[actual] = native

    def tensor(self, tid, expected, label):
        self.eq(tid == 0, expected is None, label+' absence')
        if expected is None:
            return None
        meta = self.w.meta(self.one(13,tid))
        dtype, width, incarnation, is_leaf = struct.unpack('<4Q', self.one(96,tid)['payload'])
        self.eq(dtype, 0, label+' actual F32 dtype')
        self.eq(expected['dtype'], 'torch.float32', label+' reference F32 dtype')
        self.eq(width, expected['element_size'], label+' element bytes')
        rank = len(expected['shape'])
        for key, got, want in (
            ('rank',meta['rank'],rank), ('numel',meta['numel'],len(expected['binary_words'])),
            ('shape',meta['shape'][:rank],expected['shape']), ('strides',meta['strides'][:rank],expected['strides']),
            ('requires_grad',bool(meta['requires_grad']),expected['requires_grad']),
            ('storage_bytes',meta['storage_bytes'],expected['storage_bytes']),
            ('storage_offset',meta['storage_offset'],expected['storage_offset_elements']*width),
            ('version',meta['version'],expected['version']), ('is_leaf',bool(is_leaf),expected['is_leaf']),
            ('retains_grad',bool(self.make(94,tid)),expected['retains_grad'])):
            self.eq(got,want,label+'.'+key)
        node, fields = self.node(tid=tid)
        self.eq(fields[320],incarnation,label+' real Tensor incarnation')
        self.eq(fields[216],meta['storage_id'],label+' real Storage identity')
        self.eq(fields[224],meta['version_id'],label+' real Version identity')
        native_kind = {0:None, 3:'MulBackward0',42:'NativeDropoutBackward0'}[fields[40]]
        self.eq(native_kind,expected['grad_fn'],label+' native history operation kind')
        self.identity(self.tensor_ids,expected['tensor_identity'],incarnation,label+' Tensor')
        self.identity(self.storage_ids,expected['storage_identity'],meta['storage_id'],label+' Storage')
        raw = self.one(98,tid,0,meta['numel'])['payload']
        self.eq(list(struct.unpack('<'+'I'*meta['numel'],raw)),expected['binary_words'],label+' exact logical words')
        self.snapshots += 1
        return {'token':tid,'node':node,'incarnation':incarnation,'metadata':meta,'native_op':fields[40]}

    def gradient(self, tid, expected, label):
        self.eq(expected['observation']['accepted'],True,label+' actual observation accepted')
        g = self.make(91,tid)
        was_public = g in self.public
        if g:
            self.public.add(g)
        result = self.tensor(g,expected['tensor'],label)
        self.eq(bool(self.make(59,tid)),expected['tensor'] is not None,label+' actual binding presence')
        if g and not was_public:
            self.drop(g)
        return result

    def hold_gradient(self, tid):
        g = self.make(91,tid)
        if g:
            self.public.add(g)
        return g

    def rng(self, key, label):
        expected = bytes.fromhex(self.reference['states'][key]['hex'])
        raw = self.one(63,self.generator)['payload']
        self.eq(len(raw),5056,label+' complete MT/cache extent')
        self.eq(raw,expected,label+' exact full MT/cache bytes')
        self.rng_bytes += len(raw)

    def snapshot(self, tid, mask, expected, label):
        self.tensor(tid,expected['input'],label+'.input')
        self.gradient(tid,expected['gradient'],label+'.gradient')
        self.tensor(0 if mask is None else mask,expected['mask'],label+'.mask')
        self.rng(expected['state'],label+'.rng')

    def backward(self, x, root, retain, expected, label):
        self.phase=label
        result=self.one(58,root,self.seed,int(retain))
        wanted=status_for(expected['call'])
        self.eq(result['value'],wanted,label+' actual backward outcome category')
        self.eq(len(result['payload']),32,label+' partial backward diagnostics extent')
        self.eq(struct.unpack('<4Q',result['payload'])[0],wanted,label+' repeated runtime category')
        self.snapshot(x,None,expected['after'],label+'.after')
        self.eq(self.make(53),1,label+' native object/edge invariant audit')
        self.save_owner(label)

    def refuse(self, command, wanted, label):
        before=self.save_owner(label+'-before')
        status,rows=self.w.parse(self.session.exchange(self.w.packet([command])))
        after=self.save_owner(label+'-after')
        self.eq(status,wanted,label+' precise admission category')
        self.eq(rows,[],label+' no records on rejection')
        for field in ('pointer','length','live','bytes'):
            self.eq(after[field],before[field],label+' exact whole-owner rollback '+field)
        self.rollback_bytes += before['length']

    def event(self, command, expected, label):
        self.phase=label
        wanted=status_for(expected)
        if wanted:
            self.refuse(command,wanted,label)
        else:
            self.one(*command)

    def graph(self, expected, label, tid=None, node=None):
        raw=self.save_owner(label)['bytes']
        if tid is not None:
            node=self.node(tid=tid,raw=raw)[0]
        def walk(n,ref,path):
            if ref is None:
                self.eq(n,0,path+' no gradient edge')
                return None
            self.eq(n!=0,True,path+' actual edge exists')
            _,f=self.node(node=n,raw=raw)
            op=f[40]
            name={0:'AccumulateGrad',3:'MulBackward0',42:'NativeDropoutBackward0'}[op]
            self.eq(name,ref['type'],path+' exact native operation kind')
            order=f[632] or f[320]
            result={'node':n,'incarnation':f[320],'order':order,'native_op':op,'next':[]}
            if op==0:
                self.eq(ref['sequence_nr'],(1 << 64)-1,path+' actual accumulator sentinel')
                self.eq(ref['next'],[],path+' accumulator has no next edges')
                target=f[608] if f[240]==0 else n
                self.eq(0<target<65536,True,path+' accumulator owns an actual Tensor target')
                _,target_fields=self.node(node=target,raw=raw)
                self.tensor(target_fields[192],ref['accumulator_variable'],path+'.variable')
                result['target_incarnation']=target_fields[320]
            else:
                actual_seq=ref['sequence_nr']
                if actual_seq in self.sequences:
                    self.eq(order,self.sequences[actual_seq],path+' immutable captured history order')
                else:
                    self.eq(order not in self.sequences.values(),True,path+' distinct native history')
                    for seq,previous in self.sequences.items():
                        self.eq(order>previous,actual_seq>seq,path+' exact relative creation order')
                    self.sequences[actual_seq]=order
                count=1 if op==42 else 2
                self.eq(len(ref['next']),count,path+' actual edge arity')
                if count==1:
                    self.eq((f[56],f[80]),(0,0),path+' no hidden second gradient edge')
                for i,edge in enumerate(ref['next']):
                    self.eq(edge['output_nr'],0,path+' fixed single-output edge number')
                    enabled=f[72+8*i]
                    self.eq(enabled in (0,1),True,path+' native enable bit')
                    child=f[48+8*i] if enabled else 0
                    result['next'].append(walk(child,edge['node'],path+'.'+str(i)))
            return result
        result=walk(node,expected,label)
        self.graph_observations.append({'label':label,'native':result})
        return result

    def saved(self, tid, expected, label, private_node=None):
        self.phase=label
        if private_node is not None:
            # A captured AD node is not a public Tensor token. Observe its real
            # released flag and Saved handle without fabricating a wire command.
            self.save_owner(label)
            _,f=self.node(node=private_node)
            self.eq(status_for(expected['observation']),10,label+' actual Saved-freed category')
            self.eq(expected['tensor'],None,label+' actual Saved absent')
            self.eq(f[328],1,label+' actual native AD node is spent')
            self.eq(f[288],0,label+' actual native Saved record released')
            self.eq(f[584],0,label+' actual native strong mask Tensor released')
            return
        wanted=status_for(expected['observation'])
        if wanted:
            self.refuse((96,tid,1),wanted,label)
            self.eq(expected['tensor'],None,label+' no returned Saved Tensor')
        else:
            mask=self.make(96,tid,1)
            previous=mask in self.public
            self.public.add(mask)
            self.tensor(mask,expected['tensor'],label+'.mask')
            if not previous:
                self.drop(mask)

    def weak(self, tid):
        handle=self.make(42,tid)
        self.weak_handles.add(handle)
        return handle

    def weak_snapshot(self, handles, expected, label):
        self.phase=label
        observed={}
        for name,handle in handles.items():
            wanted=expected['Storage_expired'][name]
            if wanted:
                self.refuse((43,handle),6,label+'-'+name)
            else:
                hold=self.make(43,handle)
                self.eq(hold!=0,True,label+' actual live Storage lock '+name)
                self.one(45,hold)
            observed[name]=wanted
        self.eq(self.make(53),1,label+' invariant audit after final public releases')
        self.save_owner(label+'-final')
        return observed

    def run_lifetime(self, o):
        x=self.empty()
        g=self.full([2,0,3],0)
        self.one(92,x,g)
        self.drop(g)
        action=o['action_name']
        extra=None
        if action=='drop_with_older_child_then_drop_child':
            extra=self.multiply(x)
        if action=='drop_with_optimizer_then_drop_optimizer':
            cfg=[self.w.bits(.125),0,0,0,0,0]
            groups=[1,1,0,*([0]*6),x]
            pack=lambda v:self.w.Payload(struct.pack('<'+'Q'*len(v),*v))
            extra=self.make(71,0,pack(cfg),pack(groups))
            self.optimizers.add(extra)
        out,mask=self.dropout(x)
        self.snapshot(x,mask,o['before_action'],'before_action')
        g=self.hold_gradient(x)
        handles={'input':self.weak(x),'gradient':self.weak(g),'mask':self.weak(mask)}
        self.drop(g)
        if action=='drop_after_default_failure':
            self.backward(x,out,False,o['action'],'action')
        elif action=='detach_history_then_drop':
            self.event((101,x),o['action'],'action')
        else:
            self.eq(o['action']['outcome'],'NO_BACKWARD_REQUESTED','literal action has no backward')
        self.snapshot(x,mask,o['before_drop'],'before_drop')
        self.drop(x)
        self.drop(mask)
        remaining=['older_child'] if 'older_child' in action else ['optimizer'] if 'optimizer' in action else []
        self.eq(remaining,o['after_Tensor_locals_dropped']['explicit_remaining_roots'],'actual remaining strong roots')
        self.weak_snapshot(handles,o['after_Tensor_locals_dropped'],'after_Tensor_locals_dropped')
        if extra is not None:
            if action=='drop_with_optimizer_then_drop_optimizer':
                self.one(78,extra)
                self.optimizers.remove(extra)
            else:
                self.drop(extra)
        self.weak_snapshot(handles,o['after_remaining_roots_dropped'],'after_remaining_roots_dropped')

    def run(self):
        self.rng(self.case['state_before'],'initial')
        i=self.case['index']
        o=self.case['observation']
        if i==7:
            self.run_lifetime(o['variants'][self.variant])
        else:
            x=self.empty(i!=4)
            if i==0:
                self.snapshot(x,None,o['before'],'before')
                out,mask=self.dropout(x)
                self.snapshot(x,mask,o['after_forward'],'after_forward')
                self.graph(o['graph'],'graph',tid=x)
                self.eq(out==x,o['output_is_input_Tensor'],'same output Tensor')
                self.backward(x,out,False,o['first_default'],'first_default')
                self.saved(out,o['saved_after_first'],'saved_after_first')
                self.event((95,mask,self.w.bits(.5)),o['mask_mutation'],'mask_mutation')
                self.snapshot(x,mask,o['after_mask_mutation'],'after_mask_mutation')
                self.backward(x,out,False,o['retry_default'],'retry_default')
                self.saved(out,o['saved_after_retry'],'saved_after_retry')
            elif i==1:
                self.backward(x,x,False,o['materialize_before_forward'],'materialize_before_forward')
                held=self.hold_gradient(x)
                self.tensor(held,o['held_before'],'held_before')
                out,mask=self.dropout(x)
                self.snapshot(x,mask,o['after_forward'],'after_forward')
                self.eq(self.make(91,x)==held,o['same_existing_G_after_forward'],'same existing G after forward')
                self.backward(x,out,True,o['retained_backward'],'retained_backward')
                self.tensor(held,o['held_after'],'held_after')
                self.eq(self.make(91,x)==held,o['same_existing_G_after_failure'],'same existing G after failure')
                self.saved(out,o['saved_after_failure'],'saved_after_failure')
            elif i==2:
                out1,mask1=self.dropout(x)
                first=self.graph(o['first_graph'],'first_graph',tid=x)
                out2,mask2=self.dropout(x)
                second=self.graph(o['second_graph'],'second_graph',tid=x)
                self.eq(x==out1==out2,o['all_outputs_same_T'],'all outputs same Tensor')
                self.eq(first['order']!=second['order'],o['different_history_nodes'],'distinct AD histories')
                self.eq(second['next'][0]['order']==first['order'],o['second_next_is_first'],'second captures first')
                self.tensor(mask1,o['mask1'],'mask1')
                self.tensor(mask2,o['mask2'],'mask2')
                self.snapshot(x,None,o['after_forward'],'after_forward')
                self.backward(x,out2,False,o['default_backward'],'default_backward')
                self.saved(None,o['saved_first_after'],'saved_first_after',private_node=second['next'][0]['node'])
                self.saved(out2,o['saved_second_after'],'saved_second_after')
                self.backward(x,out2,False,o['retry'],'retry')
            elif i==3:
                old=self.multiply(x)
                first=self.graph(o['old_graph_before'],'old_graph_before',tid=old)
                out,mask=self.dropout(x)
                second=self.graph(o['old_graph_after'],'old_graph_after',tid=old)
                new=self.graph(o['new_graph'],'new_graph',tid=x)
                self.eq(first['next'][0]['target_incarnation'],second['next'][0]['target_incarnation'],'old edge keeps accumulator target')
                self.eq(second['next'][0]['node'],new['next'][0]['node'],'old and new roots capture same accumulator')
                self.snapshot(x,mask,o['after_forward'],'after_forward')
                self.backward(x,old,True,o['old_root_backward'],'old_root_backward')
                self.backward(x,out,False,o['new_root_backward'],'new_root_backward')
            elif i==4:
                self.snapshot(x,None,o['before'],'before')
                out,mask=self.dropout(x)
                self.snapshot(x,mask,o['after_forward'],'after_forward')
                self.eq(out==x,o['same_T'],'same output Tensor')
                self.event((90,x,1),o['enable_requires'],'enable_requires')
                self.snapshot(x,mask,o['after_enable'],'after_enable')
                self.backward(x,x,False,o['direct_backward'],'direct_backward')
            elif i==5:
                out,mask=self.dropout(x)
                self.snapshot(x,mask,o['after_forward'],'after_forward')
                self.event((90,x,0),o['disable_requires'],'disable_requires')
                self.snapshot(x,mask,o['after_disable_attempt'],'after_disable_attempt')
                self.event((90,x,1),o['enable_requires'],'enable_requires')
                self.snapshot(x,mask,o['after_enable_attempt'],'after_enable_attempt')
                self.backward(x,out,True,o['backward'],'backward')
            elif i==6:
                out,mask=self.dropout(x)
                self.event((93,x),o['retain_call'],'retain_call')
                self.snapshot(x,mask,o['before_backward'],'before_backward')
                self.backward(x,out,True,o['retained_failure'],'retained_failure')
                held=self.hold_gradient(x)
                self.tensor(held,o['held_after_first'],'held_after_first')
                self.backward(x,out,False,o['default_failure'],'default_failure')
                new=self.hold_gradient(x)
                self.eq(new==held,o['G_same_as_held_after_default'],'old held G preserved after default')
                self.drop(new)
                self.tensor(held,o['held_after_default'],'held_after_default')
                self.backward(x,out,False,o['released_retry'],'released_retry')
                new=self.hold_gradient(x)
                self.eq(new==held,o['G_same_as_held_after_retry'],'old held G preserved after retry')
                self.drop(new)
                self.tensor(held,o['held_after_retry'],'held_after_retry')
            elif i==8:
                q=self.multiply(x)
                self.snapshot(x,None,o['leaf_before'],'leaf_before')
                self.snapshot(q,None,o['nonleaf_before'],'nonleaf_before')
                previous=self.graph(o['previous_graph'],'previous_graph',tid=q)
                out,mask=self.dropout(q)
                self.snapshot(x,None,o['leaf_after_forward'],'leaf_after_forward')
                self.snapshot(q,mask,o['nonleaf_after_forward'],'nonleaf_after_forward')
                self.eq(out==q,o['same_input_T'],'same nonleaf Tensor')
                self.eq(x!=q,o['leaf_is_different_T'],'distinct original leaf')
                current=self.graph(o['current_graph'],'current_graph',tid=q)
                self.eq(current['next'][0]['order']==previous['order'],o['next_is_previous'],'previous Mul history captured')
                self.backward(x,out,False,o['backward'],'backward')
                self.snapshot(q,mask,o['nonleaf_after_backward'],'nonleaf_after_backward')
                self.backward(x,out,False,o['retry'],'retry')
            else:
                raise AssertionError(i)
        self.phase='final'
        self.rng(self.case['state_after'],'final')
        self.eq(self.make(53),1,'final native object audit')
        for t in list(self.public):
            self.drop(t)
        for handle in self.weak_handles:
            self.one(44,handle)
        self.one(66,self.generator)
        self.eq(self.make(53),1,'audit after releasing all explicit native public owners')
        self.save_owner('final')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','build-report','protocol','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),a.binary,a.source,a.build_report,a.protocol,a.protocol.with_name('owner_client.py'),a.reference]
    build=json.loads(a.build_report.read_bytes())
    for unit in build['units']:
        paths.extend((a.build_report.parent/(unit['name']+'.hepta'),a.build_report.parent/(unit['name']+'.o')))
    inputs={path.resolve():path.read_bytes() for path in paths}
    assert sha(inputs[a.source.resolve()])==SOURCE==build['source_sha256']
    assert sha(inputs[a.binary.resolve()])==IMAGE==build['native_sha256']
    assert sha(inputs[a.protocol.resolve()])==WIRE
    assert sha(inputs[a.protocol.with_name('owner_client.py').resolve()])==CLIENT
    assert sha(inputs[a.reference.resolve()])==REFERENCE
    assert build['status']=='PASS' and build['link_exit']==0 and len(build['units'])==11
    assert (build['functions'],build['blocks'],build['instructions'],build['descriptor_bytes'])==(523,6881,25112,640)
    for unit in build['units']:
        assert unit['compiler_exit']==0
        assert sha(inputs[(a.build_report.parent/(unit['name']+'.hepta')).resolve()])==unit['source_sha256']
        assert sha(inputs[(a.build_report.parent/(unit['name']+'.o')).resolve()])==unit['object_sha256']
    reference=json.loads(inputs[a.reference.resolve()])
    assert reference['probe_sha256']==PROBE
    assert reference['status']=='OBSERVED' and reference['torch_executed'] is True
    assert reference['cases']==reference['expected_cases']==len(reference['rows'])==9
    assert reference['probe_error_count']==0 and reference['probe_errors']==[]
    assert reference['oracle_provenance']['status']=='PASS'
    assert set(reference['oracle_provenance']['checks'])==CHECKS
    assert all(reference['oracle_provenance']['checks'][key] is True for key in CHECKS)
    for state,entry in reference['states'].items():
        raw=bytes.fromhex(entry['hex'])
        assert len(raw)==entry['bytes']==5056 and sha(raw)==state
    input_dir=out/'inputs'
    input_dir.mkdir()
    copied=[]
    for path,raw in inputs.items():
        target=input_dir/path.name
        assert not target.exists(),target
        target.write_bytes(raw)
        target.chmod(path.stat().st_mode & 0o777)
        copied.append({'path':str(path),'copy':str(target.relative_to(out)),'bytes':len(raw),'sha256':sha(raw),'mode':path.stat().st_mode & 0o777})
    spec=importlib.util.spec_from_file_location('identity9_wire',a.protocol)
    wire=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    rows=[]
    for case in reference['rows']:
        assert case['status']=='OBSERVED' and case['state_unchanged'] is True
        variants=range(5) if case['index']==7 else (None,)
        for variant in variants:
            suffix='' if variant is None else '-variant-'+str(variant)
            directory=out/(f'case-{case["index"]:02d}'+suffix)
            row={'index':case['index'],'name':case['name'],'variant':variant,'status':'RUNNING'}
            replay=None
            try:
                replay=Replay(wire,a,directory,reference,case,variant)
                replay.run()
                row['status']='PASS'
            except BaseException as error:
                row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
            finally:
                if replay is not None:
                    try:
                        replay.session.finish()
                        assert replay.session.exit_code==0
                    except BaseException as error:
                        row.update(status='FAIL',transport_error=repr(error))
                    row.update(assertions=replay.assertions,frames=replay.session.frames,
                               rollback_bytes=replay.rollback_bytes,rng_bytes=replay.rng_bytes,
                               tensor_snapshots=replay.snapshots,native_exit_code=replay.session.exit_code,
                               trace=replay.trace,owners=replay.owners,graphs=replay.graph_observations)
                write_json(directory/'report.json',row)
                rows.append(row)
    assert all(path.read_bytes()==raw for path,raw in inputs.items())
    summary={'schema':'heptatorch.actual-empty-dropout-identity9-native-replay.v1',
             'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
             'source_sha256':SOURCE,'native_sha256':IMAGE,'reference_sha256':REFERENCE,
             'probe_sha256':reference['probe_sha256'],'runner_sha256':sha(inputs[Path(__file__).resolve()]),
             'native_executed':True,'Torch_executed_here':False,'reference_unchanged':True,
             'inputs_unchanged':True,'inputs':copied,'recipes':9,'native_sessions':len(rows),
             'passing_recipes':sum(all(r['status']=='PASS' for r in rows if r['index']==i) for i in range(9)),
             'passed_sessions':sum(r['status']=='PASS' for r in rows),
             'frames':sum(r.get('frames',0) for r in rows),
             'assertions':sum(r.get('assertions',0) for r in rows),
             'rollback_bytes':sum(r.get('rollback_bytes',0) for r in rows),
             'rng_bytes':sum(r.get('rng_bytes',0) for r in rows),
             'scope':{'literal_multiply':'Native binary operation 3, verified from real descriptor; never sub operation 2',
                      'graph':'Actual captured edges, immutable relative order, native operation kinds, accumulator target identity',
                      'saved_old_private_node':'Original spent flag, Saved handle and strong-mask release read from native owner; no synthetic public handle',
                      'python_diagnostics_unmapped':['Python wrapper identity/aliveness','Python warning type/text','Python exception text','absolute PyTorch graph sequence counters'],
                      'native_result_substitution':False,'full_python_surface_parity':False},
             'rows':rows,'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590}
    write_json(out/'report.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','recipes','native_sessions','passing_recipes','passed_sessions','frames','assertions','rollback_bytes','rng_bytes')}))
    return int(summary['status']!='PASS')


if __name__=='__main__':
    raise SystemExit(main())
