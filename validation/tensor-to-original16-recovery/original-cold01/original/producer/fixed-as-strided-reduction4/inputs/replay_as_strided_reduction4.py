"""Strict replay of the additive, fixed Torch AsStrided reduction-bit observations.

This is a wire client and read-only observer. All expected values, metadata,
identities, edges, Versions and outcomes come from the immutable actual report.
The original conversion54, view4 and toggle4 readers are not imported or edited.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import lzma
from pathlib import Path
import struct
import traceback

PROBE = 'cea71a50c31d181c8e6615ed18e12cf4b138ea13974257336f7fe7b85bf3850d'
REFERENCE = '6cb9f88cbcc5d079f6b7c97e5dfa92bfa468478f927b1f57dc13a81dc565f6fc'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
OWNER_CLIENT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches',
          'direct_url_matches', 'direct_url_archive_sha256_matches',
          'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested'}
NAMES = {0:'AccumulateGrad', 1:'AddBackward0', 2:'SubBackward0',
         3:'MulBackward0', 4:'DivBackward0', 7:'SumBackward0',
         10:'TransposeBackward0', 16:'ViewBackward0', 17:'CloneBackward0',
         42:'NativeDropoutBackward0', 43:'ToCopyBackward0',
         44:'AsStridedBackward0'}
EVENTS = ['create_base', 'create_true_view', 'base_requires_true', 'base_add_one',
          'real_to_copy', 'retain_view_gradient', 'create_raw_seed', 'backward']
AFTER = {'create_true_view':'before_requires',
         'base_requires_true':'derived_requires_before_version',
         'base_add_one':'after_version_and_ordered_getters',
         'create_raw_seed':'before_backward', 'backward':'after_backward'}
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def safe_json(value):
    if isinstance(value, bytes):
        return {'bytes':len(value), 'sha256':sha(value),
                'hex':value.hex() if len(value) <= 128 else None}
    return str(value)


def normalized_status(event):
    if event['accepted']:
        return 0
    message = event['error']['message']
    if 'leaf variable has been moved into the graph interior' in message:
        return 11
    if 'does not require grad' in message:
        return 5
    if 'already been freed' in message or 'backward through the graph a second time' in message:
        return 10
    if 'modified by an inplace' in message:
        return 6
    raise AssertionError(('unmapped observed error', event))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('binary','source','build-report','protocol','observer','reference','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('expected-source-sha256','expected-native-sha256','expected-observer-sha256'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), args.binary, args.source, args.build_report,
             args.protocol, args.protocol.with_name('owner_client.py'),
             args.observer, args.reference]
    inputs = {path.resolve():path.read_bytes() for path in paths}
    for path,pin in ((args.binary,args.expected_native_sha256),
                     (args.source,args.expected_source_sha256),
                     (args.observer,args.expected_observer_sha256),
                     (args.reference,REFERENCE), (args.protocol,PROTOCOL),
                     (args.protocol.with_name('owner_client.py'),OWNER_CLIENT)):
        assert sha(inputs[path.resolve()]) == pin, (str(path), 'immutable input identity')
    reference = json.loads(inputs[args.reference.resolve()])
    build = json.loads(inputs[args.build_report.resolve()])
    assert reference['schema'] == 'heptatorch.as-strided-reduction-bits-reference4.v1'
    assert reference['status'] == 'OBSERVED' and reference['cases'] == 4
    assert reference['probe_error_count'] == 0 and reference['probe_errors'] == []
    assert reference['probe_sha256'] == PROBE and reference['source_files_unchanged'] is True
    assert reference['torch_executed'] is True and reference['native_executed'] is False
    assert reference['environment_restored'] is True
    assert reference['environment_before'] == reference['environment_after']
    assert reference['environment_before']['deterministic_algorithms'] is True
    assert reference['environment_before']['deterministic_fill'] is True
    provenance = reference['oracle_provenance']
    assert provenance['status'] == 'PASS' and set(provenance['checks']) == CHECKS
    assert all(value is True for value in provenance['checks'].values())
    assert build['status'] == 'PASS' and build['source_sha256'] == args.expected_source_sha256
    assert build['native_sha256'] == args.expected_native_sha256
    assert len(build['units']) == 12 and build['descriptor_bytes'] == 672
    for unit in build['units']:
        for extension,field in (('.hepta','source_sha256'),('.o','object_sha256')):
            path = args.build_report.parent/(unit['name']+extension)
            raw = path.read_bytes(); assert sha(raw) == unit[field]
            inputs[path.resolve()] = raw
    spec = importlib.util.spec_from_file_location('as_strided_copy_bits_observer',args.observer)
    observer = importlib.util.module_from_spec(spec); spec.loader.exec_module(observer)
    copied = output/'inputs'; copied.mkdir()
    for path in paths:
        (copied/path.name).write_bytes(inputs[path.resolve()])
    rows = []
    for case in reference['rows']:
        assert case['status'] == 'OBSERVED'
        assert [event['label'] for event in case['events']] == EVENTS
        assert [snap['label'] for snap in case['snapshots']] == list(AFTER.values())
        directory = output/('case-'+str(case['index']).zfill(2)); directory.mkdir()
        native = observer.Native(args.binary,args.source,args.protocol,directory)
        row = {key:case[key] for key in ('index','name','shape','stride','seed_kind','seed_words')}
        row.update(status='RUNNING',checks=0,exact_words=0,rng_bytes=0,rollback_bytes=0,
                   differences=[],host_errors=[],snapshots=[],events=[],owners=[])
        live = {}; identities = {}; generator = None

        def eq(label, actual, expected):
            row['checks'] += 1
            if actual != expected:
                row['differences'].append({'field':label,'actual':actual,'expected':expected})

        def one(*command):
            return native.one(*command)

        def archive_owner(raw):
            digest = sha(raw); path = directory/(digest+'.owner.xz')
            if not path.exists(): path.write_bytes(lzma.compress(raw))
            return {'file':path.name,'bytes':len(raw),'sha256':digest}

        def raw_at(memory,index):
            got = memory.tensor(index)
            logical = bytes.fromhex(got['logical_bytes_hex'])
            width = got['element_size']
            got.update(width=width, stride=got['strides'],offset=got['storage_offset_elements'],
                       storage_hex=got['storage_bytes_hex'],logical_hex=got['logical_bytes_hex'],
                       storage_sha256=sha(bytes.fromhex(got['storage_bytes_hex'])),
                       logical_sha256=sha(logical),
                       binary_words=[int.from_bytes(logical[i:i+width],'little')
                                     for i in range(0,len(logical),width)])
            return got

        def raw_token(token):
            memory = native.memory()
            return raw_at(memory,memory.public_node(token))

        def bind(label, kind, actual_identity, reference_identity):
            if (kind,reference_identity) in identities:
                eq(label+'.'+kind+'_stable',actual_identity,identities[(kind,reference_identity)])
            else:
                # Distinct live reference objects must never collapse onto a
                # single native incarnation or Storage. No address equality is
                # used as a substitute for either identity.
                eq(label+'.'+kind+'_fresh', actual_identity not in
                   [value for (saved_kind,_),value in identities.items() if saved_kind==kind], True)
                identities[(kind,reference_identity)] = actual_identity

        def compare_raw(label, got, want, current):
            for key in ('dtype','width','shape','stride','offset','storage_bytes',
                        'version','requires_grad','storage_hex','storage_sha256',
                        'logical_hex','logical_sha256','binary_words'):
                eq(label+'.'+key,got[key],want[key])
            row['exact_words'] += len(want['binary_words'])
            bind(label,'Tensor',got['native_incarnation'],want['tensor_identity'])
            bind(label,'Storage',got['native_storage_id'],want['storage_identity'])
            current[label] = (got['native_incarnation'],got['native_storage_id'],
                              want['tensor_identity'],want['storage_identity'])
            actual_root = got['native_base_incarnation']; wanted_root = want['base_tensor_identity']
            eq(label+'.base_None',actual_root is None,wanted_root is None)
            if actual_root is not None and wanted_root is not None:
                bind(label+'._base','Tensor',actual_root,wanted_root)
                current[label+'._base'] = (actual_root,got['native_storage_id'],
                                          wanted_root,want['storage_identity'])

        def node_info(memory,index):
            node = memory.node(index)
            if node[5] == 0: return None
            count = 2 if node[5] in (1,2,3,4) else 1
            edges = []
            for edge in range(count):
                target = node[6+edge] if node[9+edge] else 0
                item = {'kind':None,'output_nr':0}
                if target:
                    child = memory.node(target)
                    item['kind'] = NAMES.get(child[5],'UNMAPPED-'+str(child[5]))
                    if child[5] == 0:
                        variable = target if child[30] else (child[76] & 65535)
                        item['native_variable_incarnation'] = memory.node(variable)[40]
                edges.append(item)
            return {'kind':NAMES.get(node[5],'UNMAPPED-'+str(node[5])),'next_edges':edges}

        def compare_node(label, actual, expected):
            eq(label+'.None',actual is None,expected is None)
            if actual is None or expected is None: return
            eq(label+'.kind',actual['kind'],expected['kind'])
            eq(label+'.edge_count',len(actual['next_edges']),len(expected['next_edges']))
            for index,(edge,wanted) in enumerate(zip(actual['next_edges'],expected['next_edges'])):
                prefix=label+'.edge'+str(index)
                eq(prefix+'.kind',edge['kind'],wanted['kind'])
                eq(prefix+'.output_nr',edge['output_nr'],wanted['output_nr'])
                eq(prefix+'.has_Accum_target','native_variable_incarnation' in edge,
                   'variable_tensor_identity' in wanted)
                if 'variable_tensor_identity' in wanted and 'native_variable_incarnation' in edge:
                    bind(prefix,'Tensor',edge['native_variable_incarnation'],wanted['variable_tensor_identity'])

        def rng(key,label):
            raw = one(63,generator)['payload']
            eq(label+'.full_rng5056',raw.hex(),reference['states'][key]['hex'])
            row['rng_bytes'] += len(raw)

        def inspect_tensor(label, token, want, current):
            # Match the probe's ordering exactly. Its requires_grad precedes
            # is_leaf and grad_fn, both of which may trigger lazy ViewInfo.
            before = raw_token(token)
            compare_raw(label,before,want,current)
            meta = native.w.meta(one(13,token))
            typed = struct.unpack('<4Q',one(96,token,0)['payload'])
            memory = native.memory(); index = memory.public_node(token); node = memory.node(index)
            eq(label+'.public_dtype',typed[0],{'torch.float32':0,'torch.bool':1,'torch.int64':2}[want['dtype']])
            eq(label+'.public_width',typed[1],want['width'])
            eq(label+'.public_incarnation',typed[2],before['native_incarnation'])
            eq(label+'.is_leaf',bool(typed[3]),want['is_leaf'])
            eq(label+'.metadata_shape',meta['shape'][:meta['rank']],want['shape'])
            eq(label+'.metadata_strides',meta['strides'][:meta['rank']],want['stride'])
            eq(label+'.metadata_byte_offset',meta['storage_offset'],want['offset']*want['width'])
            eq(label+'.metadata_version',meta['version'],want['version'])
            gotfn = node_info(memory,index); compare_node(label+'.grad_fn',gotfn,want['grad_fn'])
            eq(label+'.retains_grad',bool(one(94,token)['value']),want['retains_grad'])
            result = {'before_lazy_getter':before,'grad_fn':gotfn}
            if 'grad' in want:
                memory = native.memory(); node = memory.node(memory.public_node(token))
                gradient = node[48]; was_public = bool(gradient and memory.node(gradient)[25])
                handle = one(91,token)['value']; present = one(59,token)['value']
                eq(label+'.grad_None',handle==0,want['grad'] is None)
                eq(label+'.grad_presence',present,int(want['grad'] is not None))
                result['grad'] = None
                if handle:
                    if want['grad'] is not None:
                        result['grad'] = inspect_tensor(label+'.grad',handle,want['grad'],current)
                    if not was_public: one(41,handle)
            return result

        def snapshot(want):
            label = want['label']; current = {}
            eq(label+'.ordered_names',list(live),list(want['tensors']))
            before = native.s.owner()
            details = {'label':label,'before':archive_owner(before['bytes']),'tensors':{}}
            for name,token in live.items():
                details['tensors'][name] = inspect_tensor(label+'.'+name,token,want['tensors'][name],current)
            for (left,a),(right,b) in itertools.combinations(current.items(),2):
                eq(label+'.'+left+'|'+right+'.Tensor_identity',a[0]==b[0],a[2]==b[2])
                eq(label+'.'+left+'|'+right+'.Storage_identity',a[1]==b[1],a[3]==b[3])
            rng(want['rng'],label)
            header,records = native.w.parse(native.s.exchange(native.w.packet([(53,)])))
            eq(label+'.audit_status',header,0)
            if header == 0: eq(label+'.audit_value',records[0]['value'],1)
            details['after'] = archive_owner(native.s.owner()['bytes'])
            row['snapshots'].append(details)

        def create(shape,raw):
            cfg = struct.pack('<8Q',len(shape),0,*shape,*([0]*(4-len(shape))),160,len(raw))
            packet = (b'HTORCH03'+struct.pack('<3Q',1,160+len(raw),0)
                      +struct.pack('<8Q',110,0,96,0,0,0,0,0)+cfg+raw)
            header,records = native.w.parse(native.s.exchange(packet))
            assert header == 0 and len(records) == 1,('raw fixture',header,records)
            return records[0]['value']

        def view(token,shape,strides,offset):
            cfg = struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),
                              *strides,*([0]*(4-len(shape))),offset)
            return one(111,token,native.w.Payload(cfg),80)['value']

        try:
            generator = one(60,1729)['value']
            initial = bytes.fromhex(reference['states'][case['state_before']]['hex'])
            one(64,generator,native.w.Payload(initial),len(initial))
            rng(case['state_before'],'initial')
            snapshots = {snapshot['label']:snapshot for snapshot in case['snapshots']}
            for event in case['events']:
                label = event['label']; before = native.s.owner()
                rng(event['state_before'],label+'.before')
                status=0; returned=None; records=[]; publish=None
                if label == 'create_base':
                    wanted=snapshots['before_requires']['tensors']['base']
                    returned=create(wanted['shape'],bytes.fromhex(wanted['storage_hex'])); publish='base'
                elif label == 'create_true_view':
                    returned=view(live['base'],case['shape'],case['stride'],case['offset']); publish='view'
                elif label == 'base_requires_true':
                    returned=one(90,live['base'],1)['value']
                elif label == 'base_add_one':
                    returned=one(95,live['base'],0x3f800000)['value']
                elif label == 'real_to_copy':
                    returned=one(109,live['view'],0,0)['value']; publish='copy'
                elif label == 'retain_view_gradient':
                    rec=one(93,live['view']); eq(label+'.native_success',rec['value'],1)
                elif label == 'create_raw_seed':
                    # Torch's clone().reshape() exposes a real hidden flat
                    # _base. The native view holds that Tensor strongly after
                    # its public token is withdrawn, rather than pretending
                    # a directly shaped allocation is the same object graph.
                    wanted=snapshots['before_backward']['tensors']['seed']
                    flat=create([len(case['seed_words'])],bytes.fromhex(case['seed_hex']))
                    returned=view(flat,wanted['shape'],wanted['stride'],wanted['offset'])
                    one(41,flat); publish='seed'
                elif label == 'backward':
                    header,records=native.w.parse(native.s.exchange(native.w.packet([(58,live['copy'],live['seed'],0)])))
                    status=header if header else records[0]['value']
                    if header==0:
                        eq(label+'.record_status',struct.unpack('<4Q',records[0]['payload'])[0],status)
                    elif normalized_status(event)==5:
                        after=native.s.owner()
                        eq(label+'.rollback_pointer',after['pointer'],before['pointer'])
                        eq(label+'.rollback_length',after['length'],before['length'])
                        eq(label+'.rollback_owner',sha(after['bytes']),sha(before['bytes']))
                        row['rollback_bytes']+=len(before['bytes'])
                else: raise AssertionError(('unmapped real event',label))
                eq(label+'.normalized_status',status,normalized_status(event))
                eq(label+'.returns_Tensor',returned is not None,event['result_tensor_identity'] is not None)
                if returned is not None and event['result_tensor_identity'] is not None:
                    bind(label,'Tensor',raw_token(returned)['native_incarnation'],event['result_tensor_identity'])
                if publish is not None: live[publish]=returned
                rng(event['state_after'],label+'.after')
                row['events'].append({'label':label,'normalized_status':status,'records':records,
                                      'before':archive_owner(before['bytes']),
                                      'after':archive_owner(native.s.owner()['bytes'])})
                if label in AFTER: snapshot(snapshots[AFTER[label]])
            rng(case['state_after'],'final')
            for name in reversed(list(live)):
                token=live.pop(name)
                if token not in live.values(): one(41,token)
            one(66,generator); generator=None
            eq('final.audit',one(53)['value'],1)
            row['status']='PASS' if not row['differences'] else 'FAIL'
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
            row['host_errors'].append({'error':repr(error),'traceback':traceback.format_exc()})
            try: row['failed_owner']=archive_owner(native.s.owner()['bytes'])
            except BaseException as nested: row['owner_error']=repr(nested)
        finally:
            row['frames']=native.s.frames
            try: row['transport']=native.finish()
            except BaseException as error: row.update(status='FAIL',transport_error=repr(error))
            (directory/'report.json').write_text(json.dumps(row,indent=2,allow_nan=False,default=safe_json)+'\n')
            rows.append(row)
    unchanged=all(path.read_bytes()==raw for path,raw in inputs.items()); assert unchanged
    report={'schema':'heptatorch.as-strided-reduction-bits-native4.v1',
            'status':'PASS' if all(row['status']=='PASS' for row in rows) else 'FAIL',
            'cases':len(rows),'passed':sum(row['status']=='PASS' for row in rows),
            'failed':sum(row['status']!='PASS' for row in rows),
            **{key:sum(row[key] for row in rows) for key in ('frames','checks','exact_words','rollback_bytes','rng_bytes')},
            'reference_sha256':REFERENCE,'probe_sha256':PROBE,
            'source_sha256':args.expected_source_sha256,'native_sha256':args.expected_native_sha256,
            'observer_sha256':args.expected_observer_sha256,
            'reader_sha256':sha(inputs[Path(__file__).resolve()]),
            'all_inputs_unchanged':unchanged,
            'inputs_sha256':{str(path):sha(raw) for path,raw in inputs.items()},
            'provenance_checks':provenance['checks'],'rows':rows,
            'scope':'The fixed four actual recipes: raw logical and complete Storage bytes, Tensor/Storage identities and true _base, Version, lazy current grad_fn and captured immediate edges, retained grad, complete RNG and real native owner audit.',
            'unexposed_frontend_fields':['Python warning text','Python wrapper reference counts','Python exception text; the existing normalized runtime status is compared'],
            'weak_lifetime_claimed':False,'full_pytorch_rewrite_complete':False}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=safe_json)+'\n')
    print(json.dumps({key:report[key] for key in ('status','cases','passed','failed','frames','checks','exact_words','rollback_bytes','rng_bytes')}))
    return int(report['status']!='PASS')


if __name__=='__main__': raise SystemExit(main())
