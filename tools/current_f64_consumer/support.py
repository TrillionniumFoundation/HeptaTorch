"""Independent input, source, native object and original protocol readers."""
from pathlib import Path
import ast
import hashlib
import json
import re
import runpy
import struct
import textwrap
import types

from build import (Inputs, COMPILER, LINKER, SOURCE, PROFILE, INSPECTOR, adapter,
                   current_admission, dump, identity, require, sha)

ORACLE = 'bb639f7ef669d34787986ed62e870cc61071adcba1ae4e44e325673e2f75bb03'
ADAPTER_OBJECT = 'dd1c27326e7fc8e501d8be63d48edfee04770ae6a0bf0a66a3c86a1dd2f4fea4'
REVIEWED_IMAGES = {
    'aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c':
        ('1f61565bddf8c3b49e4406f581c4085d7971b2ce8b401ebc784992dfb21ae832', 4067328, 4042576),
}


def read_process(directory, name, executable, payload, wanted, timeout=120):
    """Read all physical process bytes and reconstruct intent without execution."""
    directory = Path(directory)
    report = json.loads((directory/(name+'.process.json')).read_bytes())
    intent = json.loads((directory/(name+'.intent.json')).read_bytes())
    require(set(intent) == {'name', 'argv', 'cwd', 'environment', 'timeout_seconds', 'executable', 'supervisor_pid', 'started'}, 'complete original process intent')
    require(intent['name'] == name and intent['argv'] == [str(Path(executable).absolute())]
            and intent['cwd'] == str(directory.absolute()) and intent['environment'] == {'PATH': ''}
            and intent['timeout_seconds'] == timeout and intent['executable'] == identity(executable)
            and type(intent['supervisor_pid']) is int and intent['supervisor_pid'] > 0 and intent['started'] is False,
            'exact process command, image, environment and cwd')
    require(all(report.get(k) == v for k, v in intent.items() if k != 'started'), 'process receipt matches original intent')
    require(report['started'] is True and type(report['pid']) is int and report['pid'] > 0
            and report['exit_code'] == 0 and report['timed_out'] is False
            and report['stdout_EOF'] is True and report['stderr_EOF'] is True, 'complete process/EOF')
    request = (directory/(name+'.request.bin')).read_bytes()
    response = (directory/(name+'.stdout.bin')).read_bytes()
    stderr = (directory/(name+'.stderr.bin')).read_bytes()
    require(request == payload and response == wanted and stderr == b'', 'complete actual process bytes')
    for prefix, raw in [('request', request), ('stdout', response), ('stderr', stderr)]:
        require(report[prefix+'_bytes'] == len(raw) and report[prefix+'_sha256'] == sha(raw), 'process byte identity')
    return report


def admit(repo, current, directory):
    repo, current, directory = map(lambda p: Path(p).absolute(), (repo, current, directory))
    inputs = Inputs()
    built = current_admission(repo, current, inputs)
    raw, proof = adapter(inputs.bind(repo/'tools/dtype-f64/f64.hepta', SOURCE),
                         inputs.bind(current/'runtime.hepta', built['source_sha256']),
                         inputs.bind(repo/'tools/dtype-f64/profile.json', PROFILE))
    report = json.loads((directory/'build.json').read_bytes())
    require(report['schema'] == 'heptatorch.current-object-f64-build.v1' and report['status'] == 'PASS_NATIVE_BUILD_ONLY', 'successful build report')
    require(report['current_build_sha256'] == sha(inputs.bind(current/'build.json'))
            and report['current_source_sha256'] == built['source_sha256']
            and report['current_native_sha256'] == built['native_sha256']
            and report['current_function_count'] == built['functions'], 'exact canonical cold build')
    require(report['source_transition'] == proof and (directory/'packet_adapter.hepta').read_bytes() == raw, 'full original source inverse')
    require(report['actual_current_objects_reused'] is True and report['legacy143_restored'] is False
            and report['full_current_image_executed'] is False and report['runtime_qualified'] is False
            and report['whole_pytorch_rewrite_complete'] is False, 'build scope')
    inspector = types.ModuleType('f64_reader_stack_inspector')
    exec(compile(inputs.bind(repo/'tools/stack_budget/inspect_stack.py', INSPECTOR), '<fixed-stack-inspector>', 'exec'), inspector.__dict__)
    objects, views, units = [], [], []
    for unit in built['units']:
        name = unit['name']; source = inputs.bind(current/(name+'.hepta'), unit['source_sha256'])
        obj = inputs.bind(current/(name+'.o'), unit['object_sha256'])
        objects.append(obj); views.append(inspector.inspect(source, obj))
        units.append({'name': name, 'source_sha256': sha(source), 'object_sha256': sha(obj), 'bytes': len(obj)})
    require(report['current_units'] == units, 'every actual current object and source')
    obj = (directory/'packet_adapter.o').read_bytes()
    require(sha(obj) == report['adapter_object_sha256'] == ADAPTER_OBJECT, 'independently fixed adapter object')
    views.append(inspector.inspect(raw, obj)); objects.append(obj)
    spec = json.loads(inputs.bind(repo/'tools/stack_budget/profile.json'))
    stack = inspector.longest(views)
    require(report['stack'] == stack and stack['maximum_path_bytes'] <= spec['linkers']['runtime']['stack_budget'], 'whole linked stack graph')
    require(spec['compiler']['executable_sha256'] == COMPILER and spec['linkers']['runtime']['executable_sha256'] == LINKER, 'native tool profile')
    for name, pin in [('compiler', COMPILER), ('linker', LINKER)]:
        require(identity(directory/name)['sha256'] == pin and identity(directory/name)['mode'] == 0o500, 'exact executed tool')
    entry = b'f64_packet_entry'
    request = b'HIASTKR1'+struct.pack('<3Q', len(objects), len(entry), 0)+entry
    request += b''.join(struct.pack('<Q', len(o))+o for o in objects)
    expected = REVIEWED_IMAGES.get(built['source_sha256'])
    require(expected is not None, 'current source requires its own reviewed adapter image')
    image = (directory/'f64-native').read_bytes()
    digest, size, start = expected
    require((sha(image), len(image), report['entry_offset']) == expected
            and report['native_sha256'] == digest and report['native_bytes'] == size,
            'independently fixed linked image')
    require(image[477] == 0xe8 and 482+struct.unpack_from('<i', image, 478)[0] == start, 'actual adapter entry target')
    decoded = next(r for r in views[-1]['records'] if r['name'] == 'f64_packet_entry')
    offset = views[-1]['text_at']+decoded['offset']
    require(image[start:start+15] == obj[offset:offset+15], 'actual adapter entry prologue')
    physical = []
    for i in range(2):
        physical.append(read_process(directory, 'compile-'+str(i), directory/'compiler', raw, obj))
    for i in range(2):
        physical.append(read_process(directory, 'link-'+str(i), directory/'linker', request, image))
    require(report['processes'] == physical and len({p['pid'] for p in physical}) == 4, 'complete real tool process inventory')
    origins = report['producer_paths']
    require(origins['repository'] == str(repo) and origins['current_build'] == str(current)
            and origins['output'] == str(directory), 'actual producer paths')
    expected_inputs = dict(inputs.records)
    for path in (Path(__file__).with_name('build.py'), Path(origins['flat_linker'])):
        expected_inputs[str(path.absolute())] = identity(path)
    require(report['inputs'] == expected_inputs, 'complete actual build input closure')
    inputs.recheck()
    return report


def original(repo):
    path = Path(repo)/'tests/f64_oracle.py'
    raw = path.read_bytes(); require(sha(raw) == ORACLE, 'complete original oracle')
    context = runpy.run_path(str(path), run_name='original_f64_corpus_and_predicates')
    text = raw.decode()
    start = text.index('    rows = []\n', text.index('def main():'))
    end = text.index('    unchanged = ', start)
    body = text[start:end]
    code = textwrap.dedent(body)
    require(textwrap.indent(code, '    ') == body, 'entire original comparison inverse')
    parsed = ast.parse(text)
    proof = {'original_oracle_sha256': ORACLE, 'original_comparison_sha256': sha(body.encode()),
             'original_comparison_unmodified': True,
             'preserved_functions': {n.name: sha(ast.get_source_segment(text, n).encode())
                                    for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name != 'main'},
             'original_expected_cases': 55, 'original_bad_packet_cases': 16,
             'original_limit': context['LIMIT'], 'old_compile_and_source_admission_replaced_explicitly': True}
    return context, code, proof


def comparisons(repo, invoke, reference):
    context, body, proof = original(repo)
    context.update(invoke=invoke, reference=reference)
    exec(compile(body, str(Path(repo)/'tests/f64_oracle.py')+'#unchanged-comparisons', 'exec'), context)
    return context['rows'], proof


def comparable_rows(rows):
    """Traceback locations vary in pure replay; every comparison field stays exact."""
    return [{k: v for k, v in row.items() if k != 'error'} for row in rows]


def unchanged_block(path, pin, start, end):
    raw = Path(path).read_bytes(); require(sha(raw) == pin, 'complete frozen auxiliary reader')
    text = raw.decode(); begin = text.index(start); finish = text.index(end, begin)
    body = text[begin:finish]; code = textwrap.dedent(body)
    require(textwrap.indent(code, '    ') == body, 'complete auxiliary comparison inverse')
    return code, {'original_source_sha256': pin, 'original_comparison_sha256': sha(body.encode()),
                  'entire_comparison_body_unchanged': True}


def diagnostic(repo, native, invoke, reference):
    """All original rational checks and shape observations, on actual new calls."""
    path = Path(repo)/'tests/f64_sqrt_probe.py'
    pin = 'fa59ecdbeae9e40025517a5226e67816a40f8fb38ff29494e9e74e91e2d76b47'
    code, proof = unchanged_block(path, pin, '    rows, mismatches, native_errors, reference_errors = [], [], [], []\n', '    mismatch_raw = ')
    scope, _, _ = original(repo)
    context = runpy.run_path(str(path), run_name='original_rational_sqrt_functions')

    def native_call(argv, *, input, capture_output, timeout, env):
        require(argv == [str(Path(native).resolve())] and capture_output is True
                and timeout == 30 and env == {'PATH': ''}, 'original native diagnostic invocation')
        return types.SimpleNamespace(returncode=0, stderr=b'', stdout=invoke(input))

    context.update(scope=scope, reference=lambda words: reference(4, [(v, 0) for v in words]),
                   a=types.SimpleNamespace(native=Path(native)), subprocess=types.SimpleNamespace(run=native_call))
    exec(compile(code, str(path)+'#unchanged-comparisons', 'exec'), context)
    result = {name: context[name] for name in ('rows', 'mismatches', 'native_errors', 'reference_errors', 'shape_observations')}
    result['status'] = 'FAIL' if result['native_errors'] else ('OBSERVED_DIFFERENCE' if result['mismatches'] else 'PASS')
    result['original_comparison'] = proof
    return result


def accuracy(repo, strict_rows, cpython_rows, invoke, reference):
    """Repeat the original55 RNE/accuracy predicates without relaxing strict bits."""
    path = Path(repo)/'tests/f64_rne_qualification.py'
    pin = '235206e6c490a30a5f26586c7452a423e6a29b0638839168425dc52252e707d6'
    code, proof = unchanged_block(path, pin, '    rows, mismatches = [], []\n', '    assert all(path.read_bytes()')
    integer_path = Path(repo)/'tests/f64_sqrt_integer.py'
    require(sha(integer_path.read_bytes()) == '24fca49ddf6a4a93a3c81e6e69ed969b69512ae072c58697da09140a77e5b375', 'unchanged original integer checker')
    integer = runpy.run_path(str(integer_path), run_name='original_integer_sqrt_proofs')
    scope, _, _ = original(repo)
    context = dict(integer, hashlib=hashlib, struct=struct, SHA=sha,
                   scope=scope, invoke=invoke, reference=reference,
                   old={r['case']: r for r in strict_rows}, cp_rows={r['case']: r for r in cpython_rows})
    controls = integer['self_check']()
    exec(compile(code, str(path)+'#unchanged-comparisons', 'exec'), context)
    return {'status': 'PARTIAL', 'qualification_status': 'PASS', 'cases': len(context['rows']),
            'integer_checker_boundary_controls': controls, 'original_comparison': proof,
            'integer_checker_sha256': sha(integer_path.read_bytes()),
            'rows': context['rows'], 'sqrt_bitwise_mismatches': context['mismatches'],
            'strict_pytorch_bitwise_gate_closed': False, 'full_dtype_gate_closed': False,
            'whole_pytorch_rewrite_complete': False}
