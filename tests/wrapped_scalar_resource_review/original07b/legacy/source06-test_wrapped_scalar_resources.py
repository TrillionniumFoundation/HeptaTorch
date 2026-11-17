"""Finite wrapped-Scalar64 entry, ownership, and allocation controls.

This driver executes no Torch and derives no floating-point expected values.
Its three recipes are native resource contracts. Only the explicitly selected
positive_one row supplies immutable original numeric bytes. Weak lifetime here
is a native ownership control, not a new Torch compatibility observation.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import resource
import struct
import traceback
from pathlib import Path

from case_transport import Case as OriginalCase
from owner_state import State
import owner_view05 as observer
from support import Custody, create_packet, dump, raw_file, sha

HERE = Path(__file__).resolve().parent
IMAGE = 'f05020afd1f6d80551713a031e1976137d8e3ea25e9501f44f1449dde85a44ea'
BUILD = '29b9d7246859ae79c228090b6f8b14fdf92a95a49816e9d74ddc95eae8564c2b'
SOURCE = '743c6de8c4f2e9ea6dbb880aca5a27c719c2b14ce05e7262020ece63b1ccd148'
REFERENCE = 'f3ecc9dd7db2ce8fbd0e424557ad5b95ac2e899a1c09cf38dad47e4fecc2b0c6'
PROBE = 'c0ddafbae6ea81bd98f6066bc867f71e0b6e8d4f73e6c5d7f15f7fe3d29569a3'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches',
          'direct_url_matches', 'direct_url_archive_sha256_matches',
          'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested'}
CASES = ('entry_guards_and_relative_tokens', 'saved_scalar_and_temporary_lifetime',
         'result_allocation_oom_recovery')
ALIGN = lambda n: (n + 15) & ~15
PAGE = lambda n: (n + 4095) & ~4095

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_inputs(a, custody, out):
    manifest = json.loads(custody.bind(HERE / 'source-manifest.json'))
    actual_files={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}
    assert actual_files == set(manifest['files']) | {'source-manifest.json'}
    assert all(not p.is_symlink() for p in HERE.rglob('*'))
    for name, rec in manifest['files'].items():
        rel=Path(name)
        assert not rel.is_absolute() and '..' not in rel.parts
        p = HERE / name
        b = custody.bind(p, rec['sha256'])
        assert len(b) == rec['bytes'] and p.stat().st_mode & 511 == rec['mode']
    raw = custody.bind(a.native_build / 'runtime.hepta', SOURCE)
    functions = {m.group(1).decode(): m.group(0) for m in re.finditer(
        rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n', raw, re.M | re.S)}
    assert len(functions) == 656
    build = json.loads(custody.bind(a.native_build / 'build.json', BUILD))
    assert build['status'] == 'PASS'
    assert build['source_sha256'] == SOURCE
    assert build['native_sha256'] == a.expected_native_sha256
    assert build['functions'] == 656 and len(build['units']) == 13
    assert build['descriptor_bytes'] == 672
    assert build['state_format'] == 'HTVSHP02' and build['backing_format'] == 'HTOVSHP2'
    assert build['compiler_sha256'] == COMPILER and build['linker_sha256'] == LINKER
    assert build['link_exit'] == 0
    assert len({x['name'] for x in build['units']}) == 13
    for u in build['units']:
        assert u['compiler_exit'] == 0
        for suffix, field in (('.hepta','source_sha256'), ('.o','object_sha256')):
            custody.bind(a.native_build / (u['name'] + suffix), u[field])
    binary = a.binary.resolve()
    custody.bind(binary, a.expected_native_sha256)
    assert binary.parent == a.native_build.resolve()
    evidence = json.loads(custody.bind(HERE / 'source-obligations.json'))
    bound = out / 'bound-functions'; bound.mkdir()
    for name, pin in evidence['functions'].items():
        assert sha(functions[name]) == pin, name
        (bound / (name + '.hepta')).write_bytes(functions[name])
    custody.bind(a.protocol, PROTOCOL)
    custody.bind(a.protocol.with_name('owner_client.py'), CLIENT)
    wire = load_module(a.protocol, 'wrapped_scalar_resource_wire')
    data = json.loads(custody.bind(a.reference, REFERENCE))
    custody.bind(a.probe, PROBE)
    assert data['schema'] == 'heptatorch.wrapped-f64-multiply-reference32.v1'
    assert data['status'] == 'OBSERVED' and data['cases'] == len(data['rows']) == 32
    assert data['torch_executed'] is True and data['native_executed'] is False
    assert not data['probe_errors'] and data['probe_sha256'] == PROBE
    assert data['source_files_unchanged'] and data['inputs_sha256'] == data['inputs_after_sha256']
    assert data['oracle_provenance']['status'] == 'PASS'
    assert set(data['oracle_provenance']['checks']) == CHECKS
    assert all(x is True for x in data['oracle_provenance']['checks'].values())
    assert data['environment_before'] == data['environment_after'] == {
        'deterministic_algorithms': True, 'deterministic_fill': True,
        'grad_enabled': True, 'num_threads': 1, 'cpu_capability': 'AVX512',
        'default_dtype': 'torch.float32'}
    assert data['environment_unchanged'] and data['outer_rng_restored']
    for h, s in data['full_rng_states'].items():
        b = bytes.fromhex(s['hex'])
        assert len(b) == s['bytes'] == 5056 and sha(b) == h
    row = data['rows'][2]
    assert row['index'] == 2 and row['name'] == 'positive_one'
    assert row['scalar_binary64_hex'] == '3ff0000000000000'
    assert row['scalar_little_endian_hex'] == '000000000000f03f'
    assert row['shape'] == [4] and row['status'] == 'OBSERVED'
    assert all(e['accepted'] is True for e in row['events'])
    assert [e['label'] for e in row['events']] == [
        'construct_input','input_requires_grad','multiply_python_float',
        'retain_output_gradient','construct_seed','backward_default_release']
    # Runtime identity is supplied by the reviewed full execution intent, and
    # is checked against the actual ELF/build/13 products before any Session.
    assert observer.SOURCE == SOURCE
    assert a.expected_native_sha256 == IMAGE and observer.IMAGE == IMAGE
    observer.pin_profile(binary, a.native_build / 'runtime.hepta')
    return wire, binary, build, row, evidence

class Case(OriginalCase):
    def mark(self, label):
        snap = super().mark(label)
        state = self.state(); state.partition()
        self.row['committed_84Q_snapshots'] = self.row.get('committed_84Q_snapshots', 0) + 1
        if hasattr(self, 'rng0') and self.gen in state.table(7):
            self.check('full5056 RNG '+label, state.rng(self.gen), self.rng0)
            self.row['full_rng_bytes'] = self.row.get('full_rng_bytes', 0) + 5056
        return snap

    def batch(self, label, commands):
        status, recs = self.w.parse(self.s.exchange(self.w.packet(commands)))
        self.check(label+' header', status, 0)
        self.check(label+' opcode sequence', [r['opcode'] for r in recs], [x[0] for x in commands])
        self.mark(label)
        return recs

    def full(self, shape, value=1., requires=0):
        return self.one('real full Tensor', *self.w.full(shape, value, requires))['value']

    def logical(self, tid):
        return self.read(tid, self.meta(tid)['numel'])

    def dtype(self, tid):
        rec = self.one('actual dtype metadata', 96, tid)
        self.check('dtype metadata width', len(rec['payload']), 32)
        return struct.unpack('<4Q', rec['payload'])

    def scalar(self, output):
        rec = self.one('publish real original Saved scalar', 107, output, 1)
        self.check('107 record has no payload', rec['payload'], b'')
        self.check('107 publishes real token', rec['value'] != 0, True)
        return rec['value']

    def backward(self, output, seed):
        r = self.one('real backward default release', 58, output, seed, 0)
        self.check('backward result value', r['value'], 0)
        q = struct.unpack('<4Q', r['payload'])
        self.check('backward error field', q[0], 0)
        self.row.setdefault('native_backward_diagnostics', []).append(list(q))

    def expected_tensor(self, t, expected, label):
        m = self.meta(t)
        self.check(label+' shape', m['shape'][:m['rank']], expected['shape'])
        self.check(label+' strides', m['strides'][:m['rank']], expected['strides'])
        self.check(label+' Version', m['version'], expected['version'])
        self.check(label+' Storage bytes', m['storage_bytes'], expected['storage_bytes'])
        self.check(label+' Storage offset', m['storage_offset'], expected['storage_offset_bytes'])
        self.check(label+' requires', bool(m['requires_grad']), expected['requires_grad'])
        self.check(label+' literal logical bytes', self.read(t, m['numel']),
                   bytes.fromhex(expected['logical_bytes_hex']))
        self.check(label+' literal physical Storage', self.state().storage_bytes(m['storage_id']),
                   bytes.fromhex(expected['storage_raw_hex']))

    def reset_and_close(self):
        self.one('release all public graph roots', 33, 0)
        self.closed()

def shape_payload(shape):
    return struct.pack('<5Q', len(shape), *shape, *([0]*(4-len(shape))))

def layout_payload(shape, strides, offset=0):
    return struct.pack('<10Q', len(shape), *shape, *([0]*(4-len(shape))),
                       *strides, *([0]*(4-len(strides))), offset)

def selected_words(row, field):
    return b''.join(int(x,16).to_bytes(4,'little') for x in row[field])

def entry_guards_and_relative_tokens(c, row, evidence):
    f = c.typed(0,[2,2],raw=selected_words(row,'input_binary32_hex'))
    b = c.typed(1,[2,2],raw=b'\0\1\0\1')
    i = c.typed(2,[],raw=(7).to_bytes(8,'little'))
    by_dtype = {0:f,1:b,2:i}
    cfg = layout_payload([2,2],[2,1])
    reshape = shape_payload([4])
    for relative in (False,True):
        c.row.setdefault('valid_entry_routes', []).append('relative' if relative else 'literal')
        for op,target,dtype,args in [
            (99,f,0,(c.gen,0,0)), # train=False: no random draw; no arithmetic oracle
            (100,f,0,(0x3f800000,0)),
            (100,b,1,(1,0)),
            (100,i,2,(7,0)),
            (101,f,0,()),
            (102,b,1,()),
            (108,f,0,(0,0,0)),
            (109,f,0,(0,0)),
            (111,f,0,(c.w.Payload(cfg),80)),
            (112,f,0,(c.w.Payload(reshape),40)),
            (113,f,0,(c.w.Payload(reshape),40)),
            (6,f,0,(0,1))
        ]:
            old = c.meta(target)
            ref = c.w.reference(1) if relative else target
            command = (op,ref,*args)
            records = c.batch('legal public route %d/%s'%(op,relative),
                              [(108,target,dtype,0,0),command] if relative else [command])
            record = records[-1]
            if op in (100,101,102):
                c.check('mutation return value',record['value'],1 if op == 100 else target)
                c.check('mutation has no payload',record['payload'],b'')
                now = c.meta(target)
                for key in ('storage_id','version_id','storage_bytes','storage_offset'):
                    c.check('inplace identity '+key,now[key],old[key])
                c.check('actual Version increment',now['version'],
                        old['version']+(1 if op in (100,102) else 0))
            if op == 99:
                c.check('dropout mask payload width',len(record['payload']),8)
                mask,=struct.unpack('<Q',record['payload'])
                c.one('drop real mask',41,mask)
            result = record['value']
            if op != 100 and result not in by_dtype.values():
                c.one('drop published result',41,result)
    # Raw finite request errors and request-local out-of-range refs preserve
    # the real parent precedence; no broad error wildcard is accepted.
    for op,args in [(99,(c.gen,0,0)),(100,(0,0)),(101,()),(102,()),
                    (108,(0,0,0)),(109,(0,0)),
                    (111,(c.w.Payload(cfg),80)),
                    (112,(c.w.Payload(reshape),40)),(113,(c.w.Payload(reshape),40))]:
        c.refusal('invalid_public_%d'%op,[(op,1,*args)],1)
        c.refusal('relative_out_of_range_%d'%op,
                  [(108,f,0,0,0),(op,c.w.reference(2),*args)],1)
        bad=list((op,1,*args));bad += [0]*(8-len(bad));bad[7]=1
        c.refusal('reserved_before_invalid_%d'%op,[tuple(bad)],7)
    c.refusal('transpose_invalid_internal_admission',[(6,1,0,1)],1)
    for target in (b,i):
        c.refusal('114_nonfloat_'+str(target),[(114,target,0x3ff0000000000000)],7)
    c.refusal('114_invalid_Tensor',[(114,1,0x3ff0000000000000)],1)
    for slot in range(3,8):
        cmd=[114,f,0x3ff0000000000000,0,0,0,0,0];cmd[slot]=1
        c.refusal('114_reserved_'+str(slot),[tuple(cmd)],7)
    # True dtype3 is available only through the original Saved Tensor.
    x = c.typed(0,[4],raw=selected_words(row,'input_binary32_hex'),requires=1)
    y = c.one('genuine114 guarded graph',114,x,int(row['scalar_binary64_hex'],16))['value']
    scalar = c.scalar(y)
    c.check('original Saved dtype/width',c.dtype(scalar)[:2],(3,8))
    scalar_raw = c.read(scalar,1)
    blocked = [(99,(c.gen,0,0)),(100,(0,0)),(101,()),(102,()),
               (108,(0,0,0)),(109,(0,0)),
               (111,(c.w.Payload(layout_payload([],[])),80)),
               (112,(c.w.Payload(shape_payload([])),40)),
               (113,(c.w.Payload(shape_payload([])),40)),
               (18,()),(19,()),(6,(0,0)),
               (69,(c.w.Payload(layout_payload([],[])),80)),
               (114,(0x3ff0000000000000,))]
    for op,args in blocked:
        c.refusal('dtype3_literal_refusal_%d'%op,[(op,scalar,*args)],7)
        c.refusal('dtype3_relative_refusal_%d'%op,
                  [(107,y,1),(op,c.w.reference(1),*args)],7)
    c.refusal('dtype3_generic103',[(103,1,f,scalar)],7)
    c.refusal('dtype3_generic104',[(104,1,f,scalar)],7)
    for opcode in (97,110):
        packet=create_packet(3,[],scalar_raw,0,opcode)
        c.refusal('dtype3_constructor_%d'%opcode,[],7,packet=packet)
    c.refusal('107_bad_slot',[(107,y,2)],7)
    c.refusal('107_reserved',[(107,y,1,1)],7)
    c.check('all refusals preserve original8B',c.read(scalar,1),scalar_raw)
    c.reset_and_close()

def scalar_shape(c, scalar, row):
    meta=c.meta(scalar);node,words=c.node(scalar)
    c.check('Scalar rank/count/requires',(meta['rank'],meta['numel'],meta['requires_grad']),(0,1,0))
    c.check('Scalar original Storage extent/offset/Version',
            (meta['storage_bytes'],meta['storage_offset'],meta['version']),(8,0,0))
    c.check('Scalar descriptor dtype3',words[72],3)
    c.check('Scalar original literal8B',c.read(scalar,1),bytes.fromhex(row['scalar_little_endian_hex']))
    c.check('Scalar physical8B',c.state().storage_bytes(meta['storage_id']),
            bytes.fromhex(row['scalar_little_endian_hex']))
    return node,meta

def saved_scalar_and_temporary_lifetime(c, row, evidence):
    literal=selected_words(row,'input_binary32_hex')
    seedraw=selected_words(row,'seed_binary32_hex')
    f=c.typed(0,[4],raw=literal)
    baseline=c.state().counts()
    y=c.one('no-grad114 temporary cleanup',114,f,int(row['scalar_binary64_hex'],16))['value']
    expected=list(baseline)
    for index in (0,1,2):expected[index]+=1
    c.check('no-grad result owns one T/S/V only',c.state().counts(),expected)
    c.check('no original/promoted temporary nodes remain',
            [n for n in range(1,c.state().next) if c.state().node(n)[38] and c.state().node(n)[72]==3],[])
    c.one('drop no-grad result',41,y);c.one('drop no-grad input',41,f)
    for mode in ('default-release-no-public','default-release-held-public','graph-drop-held-public'):
        x=c.typed(0,[4],raw=literal,requires=1)
        seed=c.typed(0,[4],raw=seedraw)
        y=c.one('114 with genuine original Saved',114,x,int(row['scalar_binary64_hex'],16))['value']
        scalar=c.scalar(y);scalar_node,meta=scalar_shape(c,scalar,row)
        snap=c.mark('original8B '+mode)
        weak=c.one('Weak original Scalar Storage',42,scalar)['value']
        c.check('Weak remembers dtype3',c.state().row(weak,4)[3],3)
        # These copied corruptions do not mutate the live process.
        if mode=='default-release-no-public':
            snap=c.mark('copied-control baseline')
            c.negative('scalar_rank_corruption',snap,
                lambda s:s['nodes'][str(scalar_node)].__setitem__(0,1),'wrapped-Scalar64-role')
            vid=meta['version_id'];slot=(vid&0xffffff)-1
            c.negative('scalar_Version_corruption',snap,
                lambda s:s['tables']['3'][slot].__setitem__(2,1),'wrapped-Scalar64-Version')
            wslot=(weak&0xffffff)-1
            c.negative('Weak_dtype_corruption',snap,
                lambda s:s['tables']['4'][wslot].__setitem__(3,4),'Weak-dtype-fields')
            c.one('close temporary107 public',41,scalar)
            hold=c.one('Saved alone preserves Scalar',43,weak)['value']
            c.check('locked Hold remembers dtype3',c.state().row(hold,6)[4],3)
            c.one('close lock',45,hold)
        if mode.startswith('default-release'):
            c.one('retain output as in original actual row',93,y)
            c.backward(y,seed)
            expected=row['events'][-1]['after']['tensors']
            g=c.one('publish original expected input G',91,x)['value']
            c.expected_tensor(g,expected['input']['gradient']['tensor'],'original positive_one input G')
            c.one('close observed G public',41,g)
            c.refusal('spent_Saved_'+mode,[(107,y,1)],10)
        else:
            c.one('drop graph public root',41,y)
        if mode!='default-release-no-public':
            scalar_shape(c,scalar,row)
            hold=c.one('held public Scalar survives graph release',43,weak)['value']
            c.one('close held lock',45,hold)
            c.one('drop last Scalar public',41,scalar)
        c.refusal('Scalar Storage expires '+mode,[(43,weak)],6)
        c.one('drop Weak tombstone',44,weak)
        c.one('release whole recipe roots',33,0)
        c.check('recipe native audit',c.one('audit recipe',53)['value'],1)
    c.reset_and_close()

def result_allocation_oom_recovery(c, row, evidence):
    count=262144
    x=c.full([count],1.,0)
    canary=c.typed(0,[4],raw=selected_words(row,'input_binary32_hex'))
    source_sid=c.meta(x)['storage_id']
    source_before=c.state().storage_bytes(source_sid)
    canary_before=c.logical(canary)
    initial=c.state();pool=initial.pool;fillers=[]
    for _ in range(24):
        state=c.state();free=sum(n for _,n in state.free())
        if 60*1024<=free<=68*1024:break
        assert free>68*1024,('SETUP_FAIL free too small',free)
        amount=min(262144,((free-64*1024)//4)&~3)
        assert amount>0
        fillers.append(c.full([amount],1.,0))
        c.check('filler stays within existing pool',c.state().pool,pool)
    state=c.state();high=state.next-1
    xnode,xwords=c.node(x)
    assert xwords[5]==0 and xwords[4]==0 and not any(xwords[80:84])
    assert state.q(120) <= (1<<64)-4
    assert high<240 and all(state.q(4096+8*((high+1+j)>>8)) for j in range(3))
    # Match so_slot_available: a dead record at exhausted generation is not reusable.
    remaining={}
    for k in range(1,9):
        rows=[struct.unpack_from('<8Q',state.raw,139264+128+((k-1)*state.slots+s)*64)
              for s in range(state.slots)]
        remaining[k]=sum(r[1]==0 and r[0]<4294967295 for r in rows)
    assert min(remaining[k] for k in (1,2,3,6))>=4
    free=sum(n for _,n in state.free());largest=max(n for _,n in state.free())
    # Three fresh descriptors fit existing pages and slots. Only the two
    # private Scalars allocate before the result: align16(8)+align16(4)=32.
    # Source bindings establish these orders; this is not inferred from3.
    assert 60*1024<=free<=68*1024
    assert largest>=32 and free<ALIGN(4*count)
    c.row['oom_setup']={'count':count,'highest':high,'pool':pool,
        'source_node':xnode,'public_source_requires':0,'ViewInfo_words':list(xwords[80:84]),
        'pre_result_requested_bytes':[8,4],'pre_result_rounded_bytes':[16,16],
        'pre_result_payload_exact_under_checked_conditions':32,'target_result_requested':4*count,
        'target_result_rounded':ALIGN(4*count),'total_free':free,'largest_free':largest,
        'table_free_slots':remaining,'fresh_descriptor_pages_preexisting':True,
        'filler_count':len(fillers),'source_obligations':evidence['forward_oom']}
    owner=c.s.owner();old_limit=resource.prlimit(c.s.pid,resource.RLIMIT_AS)
    before_status=Path('/proc/%d/status'%c.s.pid).read_bytes()
    (c.out/'before-limit.status').write_bytes(before_status)
    vm=int(next(l.split()[1] for l in before_status.decode().splitlines() if l.startswith('VmSize:')))*1024
    soft=vm+PAGE(owner['length']+16)+PAGE(5056+16)+65536
    assert old_limit[1]==resource.RLIM_INFINITY or soft<old_limit[1]
    c.row['oom_setup'].update(current_vm=vm,original_limit=list(old_limit),soft_limit=soft)
    resource.prlimit(c.s.pid,resource.RLIMIT_AS,(soft,old_limit[1]))
    try:
        (c.out/'limited.limits').write_bytes(Path('/proc/%d/limits'%c.s.pid).read_bytes())
        c.check('same-capacity query under limit',c.one('94 while growth denied',94,x)['value'],0)
        c.refusal('114_result_growth_OOM',[(114,x,int(row['scalar_binary64_hex'],16))],3,maps=True)
    finally:
        resource.prlimit(c.s.pid,resource.RLIMIT_AS,old_limit)
        c.row['original_limit_restored']=resource.prlimit(c.s.pid,resource.RLIMIT_AS)==old_limit
    c.check('native audit after OOM',c.one('post-OOM audit',53)['value'],1)
    c.check('original whole source bytes after OOM',c.state().storage_bytes(source_sid),source_before)
    c.check('canary after OOM',c.logical(canary),canary_before)
    before_counts=c.state().counts()
    y=c.one('identical114 succeeds after restoring limit',114,x,int(row['scalar_binary64_hex'],16))['value']
    expected=list(before_counts)
    for idx in (0,1,2):expected[idx]+=1
    c.check('recovery result owns T/S/V without temporary leak',c.state().counts(),expected)
    c.check('recovery result actual Storage size',c.meta(y)['storage_bytes'],4*count)
    c.check('recovery grew pool',c.state().pool>pool,True)
    c.check('whole original source unchanged',c.state().storage_bytes(source_sid),source_before)
    c.check('canary unchanged',c.logical(canary),canary_before)
    c.reset_and_close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-build','binary','protocol','reference','probe','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-native-sha256',default=IMAGE)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--case',choices=('all',)+CASES,default='all')
    a=p.parse_args()
    assert a.expected_native_sha256 == IMAGE
    a.native_build=a.native_build.resolve()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    custody=Custody()
    report={'schema':'heptatorch.wrapped-scalar06.resources.v1','status':'RUNNING',
        'source_sha256':SOURCE,'native_sha256':a.expected_native_sha256,
        'native_executed':False,'Torch_executed_here':False,'compiler_executed':False,
        'emitter_executed':False,'whole_pytorch_gate':'HOLD','closed_schemas':0,
        'expected_case_names':list(CASES),'requested_case':a.case,'cases':[],
        'numeric_scope':'Only immutable actual32 positive_one literal output/gradient bytes when named; no numerical expectation model.',
        'weak_scope':'Native ownership control only. The immutable actual32 contains no Weak-expiration oracle.',
        'observer_scope':'All84 committed words and real table/root/extent equations; old isolated-transition gate is not invoked.',
        'old_resource_cases_executed':False}
    try:
        wire,binary,build,row,evidence=load_inputs(a,custody,out)
        report['provenance_checks']=json.loads(a.reference.read_bytes())['oracle_provenance']['checks']
        report['unit_count']=len(build['units'])
        report['native_build_sha256']=BUILD
        report['component_status']='LIMITED_CONTROLS_ONLY_06_ORIGINAL26_HAS_TWO_SEMANTIC_FAILURES'
        if a.prepare_only:
            report['status']='PREPARED_NOT_RUN'
        else:
            for name in CASES:
                if a.case!='all' and a.case!=name:continue
                c=None
                try:
                    c=Case.__new__(Case);report['native_executed']=True
                    c.__init__(name,wire,binary,out,report,128<<20)
                    globals()[name](c,row,evidence)
                    c.row['status']='PASS'
                except BaseException as e:
                    cr=c.row if c and hasattr(c,'row') else {'name':name}
                    if not c or not hasattr(c,'row'):report['cases'].append(cr)
                    cr.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
                    if c and hasattr(c,'s'):
                        try:cr['failed_owner']=raw_file(c.out,c.s.owner()['bytes'])
                        except BaseException as e2:cr['failed_owner_error']=repr(e2)
                finally:
                    if c and hasattr(c,'s'):
                        try:c.finish()
                        except BaseException as e2:c.row.update(status='FAIL',transport_error=repr(e2),transport_traceback=traceback.format_exc())
                        dump(c.out/'report.json',c.row)
                    dump(out/'report.json',report)
            expected=list(CASES) if a.case=='all' else [a.case]
            report['status']='PASS' if [r['name'] for r in report['cases']]==expected and all(r['status']=='PASS' and r.get('EOF_exit_code')==0 for r in report['cases']) else 'FAIL'
    except BaseException as e:
        report.update(status='FAIL',preparation_error=repr(e),traceback=traceback.format_exc())
    try:report['inputs']=custody.finish();report['all_inputs_unchanged']=True
    except BaseException as e:report.update(status='FAIL',custody_error=repr(e))
    report.update(case_count=len(report['cases']),passed=sum(r['status']=='PASS' for r in report['cases']),
        frames_before_EOF=sum(r.get('frames_before_EOF',0) for r in report['cases']),
        checks=sum(r.get('checks',0) for r in report['cases']),
        full_owner_rollback_bytes=sum(r.get('rollback_bytes',0) for r in report['cases']),
        full_rng_bytes=sum(r.get('full_rng_bytes',0) for r in report['cases']),
        committed_84Q_snapshots=sum(r.get('committed_84Q_snapshots',0) for r in report['cases']))
    dump(out/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','case_count','passed','frames_before_EOF','checks','preparation_error')}))
    return int(report['status'] not in ('PASS','PREPARED_NOT_RUN'))

if __name__=='__main__':
    raise SystemExit(main())
