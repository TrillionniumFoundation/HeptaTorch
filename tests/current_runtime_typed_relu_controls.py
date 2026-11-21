"""Four exact unary-op5 successors in the original371/405 negation groups.

The complete actual776-case Torch observation is admitted before selecting
the four literal inputs. Expected tensor values come only from its retained
blobs. The original negation cases and all other protocol controls remain in
their original runners; this module never executes Torch or the native image.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


CASE_SHA256 = '117a548388a0ca08715e5922abb2ea0bf4b1d3b346ee3212238316400378367f'
GROUPS = {
    'half_neg': {
        'cases': 371,
        'source_sha256': '643da7cd50f0b9364444bac11a091b99b1fd6670b91979dca40e3227ee07cd69',
        'slots': ((369, 774, 'successor-unsupported-half5-neg', 5),
                  (370, 775, 'successor-unsupported-half6-neg', 6)),
    },
    'wide_neg': {
        'cases': 405,
        'source_sha256': '5796a0561637004f3cd5a5ac16cc0612e0ed51bd0c381c0440c2ced666ec0b84',
        'slots': ((403, 403, 'i64-protocol-controls', 2),
                  (404, 404, 'successor-unsupported-F64-neg', 4)),
    },
}
LABELS = {group: tuple(name + ':unary-5' for _, _, name, _ in value['slots'])
          for group, value in GROUPS.items()}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def module(path, name, inputs, expected=None):
    inputs.bind(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    require(Path(result.__file__).resolve(strict=True) == path.resolve(strict=True),
            'the bound observer module is the actual imported file')
    return result


def map_recipe(group, original_index, original, successor_index, successor,
               original_raw, successor_raw):
    """Admit literal inputs; do not calculate any ReLU output."""
    require(group in GROUPS, 'only the two original negation groups have ReLU controls')
    slots = [slot for slot in GROUPS[group]['slots'] if slot[0] == original_index]
    require(len(slots) == 1, 'one of the four original unary-op5 control positions')
    old_index, new_index, name, dtype = slots[0]
    require(original_index == old_index and successor_index == new_index
            and original['name'] == name and original['operation'] == 'neg'
            and original.get('controls') is True and original['input']['dtype'] == dtype,
            'the exact original negation control name, position, dtype and operation')
    expected = copy.deepcopy(original)
    expected.pop('legacy_refusal_label', None)
    expected['name'] = 'relu-' + name
    expected['operation'] = 'relu'
    require(successor == expected,
            'the whole reference recipe differs only in its ReLU name/operation and origin label')
    source = original['input']
    require(set(source) == {'dtype', 'shape', 'strides', 'offset', 'storage_count', 'words'}
            and source['shape'] == [2] and source['strides'] is None
            and source['offset'] == 0 and source['storage_count'] == 2,
            'the exact original dense two-element control geometry')
    width = 2 if dtype in (5, 6) else 8
    literal = b''.join(word.to_bytes(width, 'little') for word in source['words'])
    require(type(original_raw) is bytes and type(successor_raw) is bytes
            and original_raw == successor_raw == literal and len(literal) == 2 * width,
            'all original raw input bits and their complete physical extent are identical')
    return dict(group=group, label=name + ':unary-5', original_case_index=old_index,
        original_case_name=name, reference_case_index=new_index,
        reference_case_name=successor['name'], opcode=3, operation=5, native_dtype=dtype,
        shape=source['shape'], strides=source['strides'], storage_offset=source['offset'],
        storage_count=source['storage_count'], element_size=width,
        raw_input_bytes=len(literal), raw_input_sha256=sha(literal),
        original_recipe_sha256=sha((json.dumps(original, sort_keys=True, separators=(',', ':'))+'\n').encode()),
        reference_recipe_sha256=sha((json.dumps(successor, sort_keys=True, separators=(',', ':'))+'\n').encode()))


def read_reference(repository, directory, execution, inputs):
    """Read the same complete actual776 reference used by the new ReLU group."""
    root = Path(repository).resolve(strict=True)
    require(root == Path(__file__).resolve().parents[1], 'actual ReLU-successor source repository')
    directory, execution = Path(directory).resolve(strict=True), Path(execution).resolve(strict=True)
    inputs.bind(Path(__file__).resolve())
    reader = module(root/'tests/typed_relu/run.py', 'typed_relu_control_reference', inputs)
    for group in ('typed_relu', 'half_neg', 'wide_neg', 'numeric_predicates', 'half_logical'):
        for path in sorted((root/'tests'/group).glob('*.py')):
            inputs.bind(path)
    reference = reader.read_reference(directory, execution, inputs)
    require(reference['report']['cases'] == 776
            and reference['report']['case_sha256'] == reader.CASE_SHA256 == CASE_SHA256,
            'the complete unchanged776-case actual ReLU reference')
    inverse = module(root/'tools/current_runtime/typed_relu_observer_transition.py',
                     'typed_relu_control_observer_inverse', inputs)
    _, observer_proof = inverse.restore_observers(inputs.bind)
    controls, mappings = {}, []
    for group, domain in GROUPS.items():
        cases = module(root/'tests'/group/'cases.py', 'typed_relu_original_'+group+'_cases',
                       inputs, domain['source_sha256'])
        require(len(cases.CASES) == domain['cases']
                and [index for index, spec in enumerate(cases.CASES) if spec.get('controls')]
                    == [slot[0] for slot in domain['slots']],
                'every original negation control position is retained')
        for old_index, new_index, name, _ in domain['slots']:
            original = cases.CASES[old_index]
            row = reference['report']['rows'][new_index]
            mapping = map_recipe(group, old_index, original, row['index'], row['spec'],
                cases.raw_input(original['input']), reader.raw_input(row['spec']['input']))
            require(row['accepted'] is True and row['fresh_tensor'] is True
                    and row['fresh_storage'] is True and row['input_unchanged'] is True
                    and row['rng_unchanged'] is True,
                    'an actual accepted ReLU observation with fresh ownership and unchanged input/RNG')
            controls[group, old_index] = dict(original_spec=original, row=row, mapping=mapping)
            mappings.append(mapping)
    require(len(controls) == len(mappings) == 4, 'exactly four numerical control successors')
    admission = dict(schema='heptatorch.typed-relu-four-negation-controls.reference-admission.v1',
        reference_directory=str(directory), reference_process_directory=str(execution),
        reference_sha256=reference['report_sha256'],
        reference_process_sha256=sha(inputs.bind(execution/'process.json')),
        reference_case_sha256=CASE_SHA256, reference_cases=776,
        actual_reference_pid=reference['report']['actual_process']['pid'],
        observer_transition=observer_proof, original_control_mappings=mappings,
        historical_half239_partition_unchanged=True, historical_f6487_partition_unchanged=True,
        original_control_slots_per_group={'half_neg': 46, 'wide_neg': 46},
        numerical_successors_per_group={'half_neg': 2, 'wide_neg': 2},
        host_expected_values_computed=False, native_executed_by_reader=False,
        Torch_executed_by_reader=False)
    return dict(reference, controls=controls, admission=admission)


def expected(reference, value):
    return {key: reference['blob'](item) if key in ('logical', 'storage') else item
            for key, item in value.items() if key not in ('tensor_identity', 'storage_identity')}


def compare(n, row, source, negation, generator, reference, group):
    """Execute the original request and observe its explicit numerical successor."""
    from numeric_predicates.native import Memory
    require((group, row['index']) in reference['controls'], 'only an admitted original ReLU control')
    selected = reference['controls'][group, row['index']]
    require(row['spec'] == selected['original_spec'], 'complete original negation recipe identity')
    successor, mapping = selected['row'], selected['mapping']
    original, _ = n.tensor(source)
    negated, _ = n.tensor(negation)
    n.check('original op5 operand equals actual independent Torch ReLU input',
            original, expected(reference, successor['input']))
    n.check('the same input also equals the original Torch negation input', original, n.expected(row['input']))
    rng = n.one(63, generator)['payload']
    n.check('complete Generator bytes before the original ReLU control', len(rng), 5056)
    request = n.wire.packet([(3, 5, source)])
    exact = b'HTORCH03'+struct.pack('<3Q', 1, 96, 0)+struct.pack('<8Q', 3, 5, source, 0, 0, 0, 0, 0)
    n.check('complete original op5 request envelope, capability and padding', request, exact)
    status, records, before, after = n.issue(request)
    n.check('original unary-op5 refusal now matches actual Torch ReLU success', status, 0)
    n.check('the original ReLU request returns one matching opcode', [item['opcode'] for item in records], [3])
    before_memory, after_memory = Memory(before['bytes']), Memory(after['bytes'])
    source_node, negation_node = before_memory.node(source), before_memory.node(negation)
    n.check('ReLU preserves all84 original source descriptor words', after_memory.node(source), source_node)
    n.check('ReLU preserves all84 live negation descriptor words', after_memory.node(negation), negation_node)
    n.check('ReLU allocates only one fresh Tensor, Storage and Version',
            [new-old for old, new in zip(before_memory.live_counts(), after_memory.live_counts())],
            [1, 1, 1, 0, 0, 0, 0, 0])
    result = records[0]['value']
    actual, node = n.tensor(result)
    n.check('all original-control ReLU bytes and metadata equal actual Torch',
            actual, expected(reference, successor['result']))
    n.check('ReLU owns fresh Tensor Storage Version identities independent of both live tensors',
            [result not in (source, negation), node[27] not in (source_node[27], negation_node[27]),
             node[28] not in (source_node[28], negation_node[28])], [True]*3)
    no_edges = (32, 40, 48, 56, 72, 80, 88, 176, 184, 248, 280, 288, 296, 312, 360,
                384, 392, 400, 408, 584, 592, 600, 608, 616, 624, 640, 648, 656, 664)
    n.check('ReLU retains no AD Saved input or view edges',
            [node[offset//8] for offset in no_edges], [0]*len(no_edges))
    n.one(41, result)
    released = Memory(n.s.owner()['bytes'])
    n.check('ReLU-result release restores every original live object count',
            released.live_counts(), before_memory.live_counts())
    n.check('ReLU-result release preserves all84 original source words', released.node(source), source_node)
    n.check('ReLU-result release preserves all84 original negation words', released.node(negation), negation_node)
    n.check('both complete original tensors survive ReLU-result release',
            [n.tensor(source)[0], n.tensor(negation)[0]], [original, negated])
    n.check('all complete Generator bytes survive ReLU and result release', n.one(63, generator)['payload'], rng)
    n.check('original owner audit after releasing the ReLU successor', n.one(53)['value'], 1)
    return dict(mapping, original_expected_status=7, actual_status=status,
        original_request_bytes=len(request), original_request_sha256=sha(request),
        original_request_hex=request.hex(), actual_source_capability=source,
        reference_sha256=reference['report_sha256'], elements=actual['numel'],
        result_logical_bytes=len(actual['logical']), result_logical_sha256=sha(actual['logical']),
        result_storage_bytes=len(actual['storage']), result_storage_sha256=sha(actual['storage']),
        fresh_tensor_storage_version=True, complete_source_and_negation_preserved=True,
        complete_generator_preserved=True, result_released=True,
        all_other_original_control_slots_retained=True, host_expected_values_computed=False)


def qualify_successors(reference, records, group):
    require(group in GROUPS and [item['label'] for item in records] == list(LABELS[group])
            and len(records) == 2, 'both exact original ReLU control slots physically replayed in order')
    for saved, slot in zip(records, GROUPS[group]['slots']):
        mapping = reference['controls'][group, slot[0]]['mapping']
        require(all(saved[key] == value for key, value in mapping.items())
                and saved['reference_sha256'] == reference['report_sha256']
                and saved['original_expected_status'] == 7 and saved['actual_status'] == 0
                and saved['elements'] == 2,
                'complete original recipe mapping and actual shared reference binding')
    return dict(original_control_slots=46, explicit_numerical_successors=2,
        other_original_control_slots=44, bitwise_elements=sum(row['elements'] for row in records),
        labels=list(LABELS[group]), records=records,
        reference_sha256=reference['report_sha256'],
        reference_process_sha256=reference['admission']['reference_process_sha256'],
        original_inputs_and_request_bytes_preserved=True,
        every_other_original_predicate_retained=True, all_original_EOF_checks_retained=True,
        physical_replay=True, host_expected_values_computed=False)
