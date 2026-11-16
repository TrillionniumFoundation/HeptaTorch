"""Pinned real-PyTorch oracle for the separately versioned owned Tensor protocol.

The frozen 21-case session suite is read, verified, and compiled in memory. Its
numerical statements, inputs, tolerances, and registrations are retained. Only
the transport constants change, and explicitly audited observation calls are
inserted. No target numerical operation is implemented by this test adapter.
"""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import json
import os
import runpy
import select
import struct
import subprocess
import sys
import time
import traceback

SESSION_SHA = '35fa5fa3e1b740f901c11d46dca484079b77805004586c03ae42b35be2265577'
SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
INITIAL_NATIVE_SHA = '05c5544da2b7b34a8db9bd019a3ad0fa930f9ed5c18fd4b7582e7150f21eaeea'
MIB = 1 << 20
ROWS = []
DEFINITION_NAMES = ('Session', 'case', 'aliases', 'joined', 'indexed', 'empty',
                    'cumulative', 'transaction', 'optimizer_session', 'network',
                    'matrix', 'channels_last')
OBSERVERS = ('_v2_alias_strict', '_v2_observe_optimizer', '_v2_observe_network',
             '_v2_assert_versions')
EXTRA_CASES = [
    ('owned-v2-paged-descriptors-nonuniform-VJP-and-generation', 'descriptor_pages', 0., 0.),
    ('owned-v2-two-MiB-tensor-chunked-values-gradients-and-SGD', 'large_chunked', 0., 0.),
    ('owned-v2-300-prefix-SGD-across-capacity-growth', 'prefix_SGD', 5e-5, 5e-6),
    ('owned-v2-300-prefix-Adam-across-capacity-growth', 'prefix_Adam', 1e-4, 4e-6),
    ('owned-v2-300-prefix-AdamW-across-capacity-growth', 'prefix_AdamW', 1e-4, 4e-6),
    ('owned-v2-late-zero-broadcast-matmul-and-empty-VJP', 'latezero_matmul', 0., 0.),
    ('owned-v2-late-zero-gather-and-empty-VJP', 'latezero_gather', 0., 0.),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def call_is(node, name):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == name)


def injection(source):
    return ast.parse(source).body[0]


def adapt_session(source, filename):
    """Return definitions/registrations with a reversible, checked AST overlay."""
    tree = ast.parse(source, filename=filename)
    definitions = [copy.deepcopy(n) for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert tuple(n.name for n in definitions) == DEFINITION_NAMES
    original = ast.Module(body=copy.deepcopy(definitions), type_ignores=[])
    by_name = {n.name: n for n in definitions}
    literals = {b'HTOSES01': b'HTOWNR01', b'HTOSRP01': b'HTOWNS01',
                b'HTORSP01': b'HTORSP02'}
    changed = {'transport_magic_literals': 0, 'response_version_literal': 0,
               'alias_strict_assertion_call': 0, 'optimizer_observation_calls': 0,
               'network_observation_calls': 0, 'end_of_case_version_assertion_call': 0}

    class Transport(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, bytes) and node.value in literals:
                changed['transport_magic_literals'] += 1
                node.value = literals[node.value]
            return node

        def visit_Compare(self, node):
            self.generic_visit(node)
            if (isinstance(node.left, ast.Name) and node.left.id == 'version'
                    and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)
                    and len(node.comparators) == 1
                    and isinstance(node.comparators[0], ast.Constant)
                    and node.comparators[0].value == 1):
                node.comparators[0].value = 2
                changed['response_version_literal'] += 1
            return node

    Transport().visit(by_name['Session'])
    by_name['aliases'].body.append(injection('_v2_alias_strict(s)'))
    changed['alias_strict_assertion_call'] = 1
    optimizer = by_name['optimizer_session']
    initial = [i for i, n in enumerate(optimizer.body) if call_is(n, 's.exchange')]
    assert len(initial) == 1
    optimizer.body.insert(initial[0]+1, injection('_v2_observe_optimizer(s, locals(), True)'))
    loop, = [n for n in optimizer.body if isinstance(n, ast.For)]
    updates = [i for i, n in enumerate(loop.body)
               if call_is(n, 's.exchange') and len(n.value.args) == 2
               and isinstance(n.value.args[1], ast.Name) and n.value.args[1].id == 'expected']
    assert len(updates) == 1
    loop.body.insert(updates[0]+1, injection('_v2_observe_optimizer(s, locals(), False)'))
    changed['optimizer_observation_calls'] = 2
    network = by_name['network']
    initial = [i for i, n in enumerate(network.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'optimizer' for t in n.targets)]
    assert len(initial) == 1
    network.body.insert(initial[0]+1, injection('_v2_observe_network(s, locals(), True)'))
    loop, = [n for n in network.body if isinstance(n, ast.For)]
    step_index, = [i for i, n in enumerate(loop.body) if call_is(n, 'optimizer.step')]
    update = next(i for i in range(step_index+1, len(loop.body)) if call_is(loop.body[i], 's.exchange'))
    loop.body.insert(update+1, injection('_v2_observe_network(s, locals(), False)'))
    changed['network_observation_calls'] = 2
    case_try, = [n for n in by_name['case'].body if isinstance(n, ast.Try)]
    close, = [i for i, n in enumerate(case_try.body) if call_is(n, 'session.close')]
    case_try.body.insert(close+1, injection('_v2_assert_versions(session)'))
    changed['end_of_case_version_assertion_call'] = 1
    assert changed == {
        'transport_magic_literals': 3, 'response_version_literal': 1,
        'alias_strict_assertion_call': 1, 'optimizer_observation_calls': 2,
        'network_observation_calls': 2, 'end_of_case_version_assertion_call': 1}
    adapted = ast.Module(body=definitions, type_ignores=[])

    class Reverse(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id in OBSERVERS):
                return None
            return self.generic_visit(node)

    reversed_tree = Reverse().visit(copy.deepcopy(adapted))
    reverse_literals = {v: k for k, v in literals.items()}
    for node in ast.walk(reversed_tree.body[0]):
        if isinstance(node, ast.Constant) and isinstance(node.value, bytes) and node.value in reverse_literals:
            node.value = reverse_literals[node.value]
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id == 'version' and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 2):
            node.comparators[0].value = 1
    assert ast.dump(reversed_tree, include_attributes=False) == ast.dump(original, include_attributes=False)
    ast.fix_missing_locations(adapted)
    start, = [i for i, n in enumerate(tree.body) if call_is(n, 'case')
               and n.value.args[0].value == 'cross-frame-storage-alias-and-version']
    end, = [i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'after' for t in n.targets)]
    registrations = ast.Module(body=tree.body[start:end], type_ignores=[])
    plan = []
    def collect(name, fn, rtol=4e-5, atol=5e-6):
        plan.append({'case': name, 'rtol': rtol, 'atol': atol, 'source': 'frozen original registration'})
    scope = {name: None for name in DEFINITION_NAMES}
    scope['case'] = collect
    exec(compile(registrations, filename, 'exec'), scope)
    assert len(plan) == 21 and len({v['case'] for v in plan}) == 21
    audit = {'original_session_sha256': sha(source.encode()), 'original_ast_sha256': sha(ast.dump(original).encode()),
             'adapted_ast_sha256': sha(ast.dump(adapted).encode()), 'changes': changed,
             'reverse_overlay_restores_every_original_definition_ast': True,
             'registration_ast_sha256': sha(ast.dump(registrations).encode()),
             'original_definitions': {n.name: sha(ast.get_source_segment(source, n).encode())
                                      for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))},
             'original_case_plan': plan}
    return adapted, registrations, audit


def _v2_alias_strict(s):
    observation = s.row['version_observation']
    observation['profile'] = 'owned Tensor v2; former protocol remains separately qualified'
    observation['former_profile_absolute_counter_gap_remains_open'] = True
    observation['scope'] = 'Exact absolute initial counters and one explicit shared write, with independent clone.'
    equal = (observation['native_initial'] == observation['pytorch_initial']
             and observation['native_after_write'] == observation['pytorch_after_write'])
    observation['absolute_counter_parity_qualified'] = equal
    assert equal, observation


def _version_records(s, parameter_pairs, state_pairs, label, step):
    p = s.program()
    parameter_indices = [p.command(13, native) for native, _ in parameter_pairs]
    state_indices = [p.command(13, native) for _, native, _ in state_pairs]
    records = s.exchange(p)
    observation = {'optimizer': label, 'step': step, 'parameters': [], 'states': [],
                   'direct_parameter_absolute_versions_asserted': True}
    for (native, reference), index in zip(parameter_pairs, parameter_indices):
        s.metadata(records[index], reference)
        fields = struct.unpack('<13Q', records[index][2])
        row = {'native_handle': native, 'native_owner': fields[11], 'native_version': fields[12],
               'pytorch_version': int(reference._version)}
        row['equal'] = row['native_version'] == row['pytorch_version']
        observation['parameters'].append(row)
        if not row['equal']:
            s.row.setdefault('parameter_version_mismatches', []).append(dict(row, step=step, optimizer=label))
    for (name, native, reference), index in zip(state_pairs, state_indices):
        fields = struct.unpack('<13Q', records[index][2])
        row = {'state': name, 'native_handle': native, 'native_version': fields[12],
               'pytorch_version': None if reference is None else int(reference._version),
               'pytorch_state_exists': reference is not None,
               'native_lifecycle': 'explicit zero tensor supplied before the first optimizer step',
               'pytorch_lifecycle': 'state allocated by torch.optim when its update path first needs it'}
        if reference is not None:
            s.metadata(records[index], reference)
            row['absolute_versions_equal'] = fields[12] == int(reference._version)
            if not row['absolute_versions_equal']:
                s.row.setdefault('explicit_state_version_differences', []).append(dict(row, step=step, optimizer=label))
        else:
            row['absolute_versions_equal'] = None
        observation['states'].append(row)
    s.row.setdefault('optimizer_version_observations', []).append(observation)


def _v2_observe_optimizer(s, context, initial):
    parameter, native, optimizer, mode = (context[key] for key in ('parameter', 'native', 'optimizer', 'mode'))
    state = optimizer.state.get(parameter, {})
    if mode == 'SGD':
        pairs = [('momentum_buffer', context['state'], state.get('momentum_buffer'))]
    else:
        pairs = [('exp_avg', context['first'], state.get('exp_avg')),
                 ('exp_avg_sq', context['second'], state.get('exp_avg_sq'))]
    _version_records(s, [(native, parameter)], pairs, mode, 0 if initial else int(context['step']))


def _v2_observe_network(s, context, initial):
    _version_records(s, list(zip(context['ids'], context['parameters'])), [],
                     'MLP-SGD', 0 if initial else int(context['_'])+1)


def _v2_assert_versions(s):
    failures = s.row.get('parameter_version_mismatches', [])
    s.row['direct_parameter_absolute_version_status'] = 'FAIL' if failures else (
        'PASS' if s.row.get('optimizer_version_observations') else 'NOT_APPLICABLE')
    s.row['explicit_optimizer_state_absolute_version_contract'] = (
        'OPEN: explicit native state and torch.optim-created state have distinct construction lifecycles; '
        'all observed absolute counters and differences are retained, without claiming full state parity.')
    assert not failures, {'absolute_parameter_version_mismatches': len(failures), 'first_mismatches': failures[:4]}


def descriptor_pages(s):
    left = torch.tensor([[1., -2., 3.], [4., -5., 6.]], requires_grad=True)
    right = torch.tensor([[.5, -1.], [2., .25], [-1., 1.5]], requires_grad=True)
    seed = torch.tensor([[.5, -1.25, 2.], [3., -.75, 1.5]])
    p = s.program(); nl, nr, ns = p.tensor(left, True), p.tensor(right, True), p.tensor(seed)
    leaf_ids = [nl, nr, ns]+[p.full((0,), 0.) for _ in range(297)]
    records = s.exchange(p)
    assert leaf_ids == list(range(1, 301))
    assert [record[1] for record in records] == leaf_ids
    p = s.program(); output = p.matmul(nl, nr); transposed = p.transpose(output, 0, 1)
    selected = p.index_select(transposed, -1, torch.tensor([1, 1, 0], dtype=torch.int64))
    index = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.int64)
    gathered = p.gather(selected, 0, index); squared = p.binary(3, gathered, gathered)
    reference = torch.gather(torch.index_select((left@right).transpose(0, 1), -1,
                                                torch.tensor([1, 1, 0], dtype=torch.int64)), 0, index).square()
    s.exchange(p, {p.read(squared): reference})
    for _ in range(2):
        reference.backward(seed, retain_graph=True)
        p = s.program(); p.seeded(squared, ns)
        s.exchange(p, {p.read(nl, True): left.grad, p.read(nr, True): right.grad})
    p = s.program(); p.reset_graph(300); s.exchange(p)
    p = s.program(); p.read(squared); s.exchange(p, status=1)
    p = s.program(); fresh = p.tensor(.25); assert fresh == (1 << 16) | 301
    left_metadata = p.command(13, nl); empty_metadata = p.command(13, leaf_ids[-1])
    records = s.exchange(p, {p.read(fresh): torch.tensor(.25), p.read(nl): left})
    s.metadata(records[left_metadata], left)
    s.metadata(records[empty_metadata], torch.empty(0))
    s.row['retained_leaf_prefix'] = 300
    s.row['fresh_generation_handle'] = fresh


def read_chunks(s, native, reference, gradient=False):
    flattened = reference.detach().reshape(-1)
    for start in range(0, flattened.numel(), 131072):
        count = min(131072, flattened.numel()-start)
        p = s.program(); at = p.window(native, start, count, gradient)
        s.exchange(p, {at: flattened[start:start+count]})


def large_chunked(s):
    parameter = torch.nn.Parameter(torch.full((512, 1024), .125))
    p = s.program(); native = p.full(parameter.shape, .125, True); metadata = p.command(13, native)
    records = s.exchange(p); s.metadata(records[metadata], parameter)
    _version_records(s, [(native, parameter)], [], 'large-SGD', 0)
    p = s.program(); product = p.binary(3, native, native); loss = p.reduce(7, product)
    reference = parameter.square(); reference_loss = reference.sum()
    s.exchange(p, {p.read(loss): reference_loss})
    reference_loss.backward()
    p = s.program(); p.backward(loss); s.exchange(p)
    p = s.program(); p.read(native); s.exchange(p, status=3)
    read_chunks(s, native, parameter)
    read_chunks(s, product, reference)
    read_chunks(s, native, parameter.grad, True)
    optimizer = torch.optim.SGD([parameter], lr=.25, foreach=False, fused=False)
    optimizer.step()
    p = s.program(); p.sgd(native, .25); s.exchange(p)
    _version_records(s, [(native, parameter)], [], 'large-SGD', 1)
    read_chunks(s, native, parameter)
    s.row['tensor_payload_bytes'] = parameter.numel()*parameter.element_size()
    s.row['all_value_and_gradient_elements_read_in_bounded_chunks'] = True


def prefix_optimizer(s, mode):
    parameter = torch.nn.Parameter(torch.tensor([-.7, .3, 1.2]))
    lr, b1, b2, eps, wd, momentum = map(f32, [.02, .9, .999, 1e-8, .07, .9])
    p = s.program(); native = p.tensor(parameter, True)
    if mode == 'SGD':
        state = p.tensor(torch.zeros_like(parameter)); used = 2
        optimizer = torch.optim.SGD([parameter], lr=lr, momentum=momentum, weight_decay=wd,
                                    foreach=False, fused=False)
    else:
        first, second = p.tensor(torch.zeros_like(parameter)), p.tensor(torch.zeros_like(parameter)); used = 3
        cls = torch.optim.AdamW if mode == 'AdamW' else torch.optim.Adam
        optimizer = cls([parameter], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd,
                        foreach=False, fused=False)
    padding = [p.full((0,), 0.) for _ in range(300-used)]
    assert padding[-1] == 300
    s.exchange(p)
    _v2_observe_optimizer(s, locals(), True)
    generator = torch.Generator().manual_seed(8237)
    growth_rows = []
    for step in range(1, 33):
        gradient = torch.randn(parameter.shape, generator=generator)
        optimizer.zero_grad(set_to_none=False)
        p = s.program(); p.zero(native); s.exchange(p)
        p = s.program(); ng = p.tensor(gradient); loss = p.reduce(7, p.binary(3, native, ng))
        reference = (parameter*gradient).sum()
        expected = {p.read(loss): reference.detach()}
        if step in (1, 17):
            shape = (256 if step == 1 else 512, 1024)
            filler = p.full(shape, .03125)
            expected[p.window(filler, shape[0]*shape[1]-3, 3)] = torch.full((3,), .03125)
            growth_rows.append({'step': step, 'new_tensor_shape': list(shape), 'new_tensor_payload_bytes': shape[0]*shape[1]*4})
        s.exchange(p, expected)
        reference.backward()
        p = s.program(); p.backward(loss); s.exchange(p, {p.read(native, True): parameter.grad})
        optimizer.step()
        p = s.program()
        if mode == 'SGD':
            p.momentum(native, state, step, lr, momentum, 0., wd, False)
            expected = {p.read(native): parameter.detach().clone(),
                        p.read(state): optimizer.state[parameter]['momentum_buffer'].clone()}
        else:
            p.adam(native, first, second, step, lr, b1, b2, eps, wd, mode == 'AdamW')
            expected = {p.read(native): parameter.detach().clone(),
                        p.read(first): optimizer.state[parameter]['exp_avg'].clone(),
                        p.read(second): optimizer.state[parameter]['exp_avg_sq'].clone()}
        s.exchange(p, expected)
        _v2_observe_optimizer(s, locals(), False)
        p = s.program(); p.reset_graph(300); s.exchange(p)
    p = s.program(); metadata = p.command(13, padding[-1]); records = s.exchange(p)
    s.metadata(records[metadata], torch.empty(0))
    s.row.update(optimizer_steps=32, retained_leaf_prefix=300, capacity_growth_inputs=growth_rows)


def latezero_matmul(s):
    left = torch.empty((262144, 262144, 0, 1), requires_grad=True)
    right = torch.empty((1, 1, 1, 0), requires_grad=True)
    reference = torch.matmul(left, right)
    assert tuple(reference.shape) == (262144, 262144, 0, 0)
    seed = torch.empty(reference.shape)
    p = s.program(); nl, nr = p.tensor(left, True), p.tensor(right, True); s.exchange(p)
    p = s.program(); output = p.matmul(nl, nr); metadata = p.command(13, output)
    records = s.exchange(p, {p.read(output): reference}); s.metadata(records[metadata], reference)
    reference.backward(seed)
    p = s.program(); ns = p.tensor(seed); p.seeded(output, ns)
    metas = [p.command(13, value) for value in (nl, nr)]
    records = s.exchange(p, {p.read(nl, True): left.grad, p.read(nr, True): right.grad})
    for index, tensor in zip(metas, (left, right)): s.metadata(records[index], tensor)
    s.row['reference_shapes'] = {'left': list(left.shape), 'right': list(right.shape), 'output': list(reference.shape),
                               'left_gradient': list(left.grad.shape), 'right_gradient': list(right.grad.shape)}


def latezero_gather(s):
    source = torch.empty((262144, 262144, 0, 1), requires_grad=True)
    index = torch.empty((262144, 262144, 0, 0), dtype=torch.int64)
    reference = torch.gather(source, 3, index)
    seed = torch.empty(reference.shape)
    p = s.program(); native = p.tensor(source, True); s.exchange(p)
    p = s.program(); output = p.gather(native, 3, index); metadata = p.command(13, output)
    records = s.exchange(p, {p.read(output): reference}); s.metadata(records[metadata], reference)
    reference.backward(seed)
    p = s.program(); ns = p.tensor(seed); p.seeded(output, ns); metadata = p.command(13, native)
    records = s.exchange(p, {p.read(native, True): source.grad}); s.metadata(records[metadata], source)
    s.row['reference_shapes'] = {'source': list(source.shape), 'index': list(index.shape),
                               'output': list(reference.shape), 'gradient': list(source.grad.shape)}


def emit(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--session-reference', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True,
                        help='Repository containing compatibility/upstream-lock.json and the locked pip report path.')
    parser.add_argument('--expected-native-sha256', default=INITIAL_NATIVE_SHA)
    parser.add_argument('--prepare-only', action='store_true', help='Check sources and freeze case plan without importing Torch.')
    args = parser.parse_args()
    for name in ('executable', 'report', 'session_reference', 'reference', 'lock', 'repo'):
        setattr(args, name, getattr(args, name).resolve())
    global EXE, REPORT, torch, F, bits, f32, Program, PROVENANCE
    EXE, REPORT = args.executable, args.report
    started = time.monotonic()
    custody = {'session_reference_sha256': sha(args.session_reference.read_bytes()),
               'reference_support_sha256': sha(args.reference.read_bytes()),
               'oracle_script_sha256': sha(Path(__file__).read_bytes()),
               'oracle_lock_sha256': sha(args.lock.read_bytes())}
    assert custody['session_reference_sha256'] == SESSION_SHA, custody
    assert custody['reference_support_sha256'] == SUPPORT_SHA, custody
    before = sha(EXE.read_bytes())
    assert before == args.expected_native_sha256, (before, args.expected_native_sha256)
    adapted, registrations, audit = adapt_session(args.session_reference.read_text(), str(args.session_reference))
    plan = audit['original_case_plan']+[
        {'case': name, 'rtol': rtol, 'atol': atol, 'source': 'owned Tensor v2 extension'}
        for name, _, rtol, atol in EXTRA_CASES]
    assert len(plan) == 28 and len({v['case'] for v in plan}) == 28
    base = {'schema': 'heptatorch.live-pytorch-owned-tensor-v2.v1', 'status': 'NOT_RUN',
            'native_executable_sha256': before, 'source_custody': custody, 'overlay_audit': audit,
            'expected_cases': 28, 'inherited_cases': 21, 'extension_cases': 7, 'case_plan': plan,
            'full_rewrite_complete': False, 'pytorch_api_denominator': 2590,
            'former_profile_absolute_counter_gap_remains_open': True,
            'scope': 'CPU binary32 rank 0..4, paged offset arena and real owned interactive state. '
                     'No claim that independent Storage/Version objects or all PyTorch are integrated.'}
    if args.prepare_only:
        emit(REPORT, dict(base, reason='Case plan and reversible source overlay prepared; real pinned PyTorch has not run.',
                          source_overlay_checks='PASS', cases=0, compared_values=0, frames=0))
        return 0
    os.chdir(args.repo)
    provenance_path = REPORT.with_name(REPORT.stem+'-provenance.json')
    previous_argv = sys.argv
    try:
        sys.argv = [str(args.reference), str(EXE), str(provenance_path), str(args.lock)]
        support = runpy.run_path(str(args.reference), run_name='heptatorch_owned_tensor_reference_support')
    except SystemExit as error:
        observed = json.loads(provenance_path.read_text()) if provenance_path.exists() else {'error': str(error)}
        emit(REPORT, dict(base, reason='Pinned reference support declined execution; no native case was run.',
                          reference_initialization=observed, cases=0, compared_values=0, frames=0))
        return 2
    finally:
        sys.argv = previous_argv
    torch, F, bits, f32, PROVENANCE = (support[key] for key in ('torch', 'F', 'bits', 'f32', 'PROVENANCE'))
    assert PROVENANCE['status'] == 'PASS' and len(PROVENANCE['checks']) == 11
    assert all(value is True for value in PROVENANCE['checks'].values())
    original_program = support['Program']

    class OwnedProgram(original_program):
        def handle(self, slot):
            return slot if slot <= self.keep else (self.generation << 16) | slot

        def encode(self):
            raw = super().encode()
            assert raw[:8] == b'HTORCH01' and len(raw) <= MIB
            return b'HTORCH02'+raw[8:]

        def full(self, shape, value, requires=False):
            shape = tuple(shape)
            assert len(shape) <= 4
            return self.make(38, len(shape), int(requires), *shape, *([0]*(4-len(shape))), bits(value))

        def window(self, native, start, count, gradient=False):
            return self.command(40 if gradient else 39, native, start, count)

    Program = OwnedProgram
    exec(compile(adapted, str(args.session_reference), 'exec'), globals())
    exec(compile(registrations, str(args.session_reference), 'exec'), globals())
    assert len(ROWS) == 21
    extra_functions = {'descriptor_pages': descriptor_pages, 'large_chunked': large_chunked,
                       'prefix_SGD': lambda s: prefix_optimizer(s, 'SGD'),
                       'prefix_Adam': lambda s: prefix_optimizer(s, 'Adam'),
                       'prefix_AdamW': lambda s: prefix_optimizer(s, 'AdamW'),
                       'latezero_matmul': latezero_matmul, 'latezero_gather': latezero_gather}
    for name, function, rtol, atol in EXTRA_CASES:
        case(name, extra_functions[function], rtol, atol)
    after = sha(EXE.read_bytes())
    assert [r['case'] for r in ROWS] == [r['case'] for r in plan]
    assert [(r['rtol'], r['atol']) for r in ROWS] == [(r['rtol'], r['atol']) for r in plan]
    versions = sum(len(r.get('parameter_version_mismatches', [])) for r in ROWS)
    states = sum(len(r.get('explicit_state_version_differences', [])) for r in ROWS)
    source_unchanged = {
        'original_session': sha(args.session_reference.read_bytes()) == SESSION_SHA,
        'reference_support': sha(args.reference.read_bytes()) == SUPPORT_SHA,
        'oracle_script': sha(Path(__file__).read_bytes()) == custody['oracle_script_sha256'],
        'oracle_lock': sha(args.lock.read_bytes()) == custody['oracle_lock_sha256'],
    }
    passed = (len(ROWS) == 28 and all(r['status'] == 'PASS' for r in ROWS)
              and before == after and all(source_unchanged.values()))
    report = dict(base, status='PASS' if passed else 'FAIL', oracle_provenance=PROVENANCE,
                  native_unchanged_during_run=before == after, cases=len(ROWS),
                  failed_cases=sum(r['status'] == 'FAIL' for r in ROWS),
                  compared_values=sum(r['compared_values'] for r in ROWS),
                  frames=sum(r['frames'] for r in ROWS), rows=ROWS,
                  direct_parameter_absolute_version_mismatches=versions,
                  explicit_state_absolute_version_differences=states,
                  source_files_unchanged_during_run=source_unchanged,
                  explicit_optimizer_state_parity='OPEN; observations retained with lifecycle differences',
                  wall_seconds=time.monotonic()-started)
    emit(REPORT, report)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
