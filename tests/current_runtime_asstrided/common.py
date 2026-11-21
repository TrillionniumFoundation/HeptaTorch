"""Exact identity admission for the four unchanged original AsStrided readers."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import struct
import sys
import types

HERE = Path(__file__).resolve().parent
SOURCE = 'aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c'
IMAGE = '083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5'
SOURCES_SHA = '28ebdd8182315916c6caa58220f3225fae30050038b2419f674d28ddcad9639b'
SUITES = ('copy4', 'reduction4', 'lanes2', 'branches6')
COUNTS = dict(copy4=4, reduction4=4, lanes2=2, branches6=6)
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
    'git_matches_oracle_build_pin', 'distribution_version_matches', 'direct_url_matches',
    'direct_url_archive_sha256_matches', 'pip_report_url_matches',
    'pip_report_archive_sha256_matches', 'pip_report_version_matches', 'pip_report_requested'}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


class Inputs:
    def __init__(self):
        self.rows = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                'unique regular input required: '+str(path))
        raw = path.read_bytes()
        after = path.stat()
        require(all(getattr(before, key)==getattr(after, key) for key in
                ('st_dev','st_ino','st_size','st_mode','st_mtime_ns','st_ctime_ns')),
                'input changed during read: '+str(path))
        row = dict(bytes=len(raw), sha256=sha(raw), mode=stat.S_IMODE(after.st_mode))
        require(expected is None or row['sha256']==expected, 'input SHA mismatch: '+str(path))
        require(str(path) not in self.rows or self.rows[str(path)]==row,
                'bound input changed: '+str(path))
        self.rows[str(path)] = row
        return raw

    def recheck(self):
        for path, row in list(self.rows.items()):
            self.bind(Path(path), row['sha256'])


def module(path, name, inputs):
    raw = inputs.bind(path)
    result = types.ModuleType(name)
    result.__file__ = str(path)
    sys.modules[name] = result
    exec(compile(raw, str(path), 'exec'), result.__dict__)
    return result


def definitions(raw):
    lines = raw.splitlines(keepends=True)
    return {node.name:b''.join(lines[node.lineno-1:node.end_lineno])
            for node in ast.parse(raw).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def derive(original, output, edits, changed_definitions, inputs):
    before = inputs.bind(original)
    after, history = before, []
    for label, old, new, count in edits:
        old = old.encode() if isinstance(old, str) else old
        new = new.encode() if isinstance(new, str) else new
        require(old != new and after.count(old)==count, 'non-exact edit: '+label)
        after = after.replace(old, new, count)
        history.append(dict(label=label, before_hex=old.hex(), after_hex=new.hex(), count=count))
    inverse = after
    for row in reversed(history):
        old, new = bytes.fromhex(row['before_hex']), bytes.fromhex(row['after_hex'])
        require(inverse.count(new)==row['count'], 'ambiguous source inverse: '+row['label'])
        inverse = inverse.replace(new, old, row['count'])
    require(inverse==before, 'complete original source inverse differs')
    old_defs, new_defs = definitions(before), definitions(after)
    require(old_defs.keys()==new_defs.keys(), 'definitions added or removed')
    changed = [name for name in old_defs if old_defs[name]!=new_defs[name]]
    require(set(changed)==set(changed_definitions), 'unexpected changed definitions: '+repr(changed))
    Path(output).write_bytes(after)
    inputs.bind(output, sha(after))
    proof = dict(original=str(original), original_sha256=sha(before), derived=str(output),
        derived_sha256=sha(after), whole_file_inverse_exact=True, edits=history,
        changed_definitions=changed,
        unchanged_definitions={name:sha(value) for name,value in old_defs.items() if name not in changed})
    return after, proof


def admit(repository, build, expected_build_sha, reference_root, suite, inputs):
    require(re.fullmatch('[0-9a-f]{64}', expected_build_sha) is not None,
            'explicit actual cold-build receipt SHA required')
    pins = json.loads(inputs.bind(HERE/'sources.json', SOURCES_SHA))
    require(pins['suites']==list(SUITES) and pins['cases']==COUNTS, 'original16 denominator changed')
    require(pins['target']==dict(source_sha256=SOURCE,native_sha256=IMAGE,functions=800,units=15),
            'current source profile changed')
    for relative, row in pins['repository_files'].items():
        path = repository/relative
        raw = inputs.bind(path, row['sha256'])
        require(len(raw)==row['bytes'] and stat.S_IMODE(path.stat().st_mode)==row['mode'],
                'original source bytes or mode changed: '+relative)
    for relative, row in pins['reference_files'].items():
        path = reference_root/relative
        raw = inputs.bind(path,row['sha256'])
        require(len(raw)==row['bytes'] and stat.S_IMODE(path.stat().st_mode)==row['mode'],
                'original reference-package bytes or mode changed: '+relative)
    package = json.loads(inputs.bind(reference_root/'package-manifest.json'))
    installation = json.loads(inputs.bind(reference_root/'installation-custody.json'))
    installed = {row['installed_path']:row for row in installation['entries']}
    require(installation['bytes_changed'] is False and installation['archive_member_modes_changed'] is False,
            'original recovery changed bytes or modes')
    recovered = {name:row for name,row in pins['repository_files'].items()
                 if name.startswith('tests/tensor_to_original16_review/')}
    recovered.update({'validation/tensor-to-original16-recovery/'+name:row
                      for name,row in pins['reference_files'].items()
                      if name.startswith(('reference/','original-probes/'))})
    for name,row in recovered.items():
        require(package['files'][name]==row and all(installed[name][key]==row[key]
                for key in ('bytes','sha256'))
                and installed[name]['original_mode']==installed[name]['installed_mode']==row['mode'],
                'fixed original recovery custody differs: '+name)
    current = module(repository/'tools/current_runtime/qualification.py',
                     'asstrided16_current_admission', inputs)
    require((current.SOURCE,current.IMAGE,current.FUNCTIONS,current.UNITS)==(SOURCE,IMAGE,800,15),
            'canonical current manifest changed')
    built = current.admit_build(build, inputs.bind, expected_build_sha)
    raw = inputs.bind(build/'runtime.hepta', SOURCE)
    view = module(repository/'tools/current_runtime/public_f64_view_source_transition.py',
                  'asstrided16_actual_f64_view_original_domain', inputs)
    view_route = view.prove_original_domain(raw, inputs.bind)
    found = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n',raw,re.M|re.S))
    functions = {m[1].decode():m[0] for m in found}
    require(len(found)==len(functions)==800, 'complete current function denominator changed')
    predicate = module(repository/'tools/current_runtime/predicate_source_transition.py',
                       'asstrided16_predicate_source_inverse', inputs)
    bf16 = module(repository/'tools/current_runtime/bf16_source_transition.py',
                  'asstrided16_bf16_source_inverse', inputs)
    mul = module(repository/'tools/current_runtime/half_mul_source_transition.py',
                 'asstrided16_half_mul_source_inverse', inputs)
    neg = module(repository/'tools/current_runtime/half_neg_source_transition.py',
                 'asstrided16_half_neg_source_inverse', inputs)
    addsub = module(repository/'tools/current_runtime/half_addsub_source_transition.py',
                    'asstrided16_half_addsub_source_inverse', inputs)
    wide = module(repository/'tools/current_runtime/wide_neg_source_transition.py',
                  'asstrided16_wide_neg_source_inverse', inputs)
    division = module(repository/'tools/current_runtime/half_div_source_transition.py',
                      'asstrided16_half_div_source_inverse', inputs)
    arithmetic = module(repository/'tools/current_runtime/public_f64_arithmetic_source_transition.py',
                        'asstrided16_public_f64_arithmetic_source_inverse', inputs)
    index_path = repository / 'tools/current_runtime/index_signbit_source_transition.py'
    inputs.bind(index_path)
    index_spec = importlib.util.spec_from_file_location('index_signbit_current_source', index_path)
    indexing = importlib.util.module_from_spec(index_spec); index_spec.loader.exec_module(indexing)
    raw, index_inverse = indexing.restore_source(raw, inputs.bind)
    require(index_inverse['functions'] == 800 and index_inverse['parent_functions'] == 775
            and index_inverse['whole_source_inverse_exact'] is True
            and index_inverse['whole_round_trip_exact'] is True,
            'complete800 source restores every original775 byte before earlier source proofs')
    relu = module(repository/'tools/current_runtime/typed_relu_source_transition.py',
                  'asstrided16_typed_relu_source_inverse', inputs)
    same773, relu_inverse = relu.restore_source(raw, inputs.bind)
    same = module(repository/'tools/current_runtime/same_f64_arithmetic_source_transition.py',
                  'asstrided16_same_f64_arithmetic_source_inverse', inputs)
    old773, same_inverse = same.restore_source(same773, inputs.bind)
    old771, arithmetic_inverse = arithmetic.restore_source(old773, inputs.bind)
    old767, div_inverse = division.restore_source(old771, inputs.bind)
    old765, wide_inverse = wide.restore_source(old767, inputs.bind)
    old761, addsub_inverse = addsub.restore_source(old765, inputs.bind)
    old758, neg_inverse = neg.restore_source(old761, inputs.bind)
    old742, mul_inverse = mul.restore_source(old758, inputs.bind)
    old740, bf16_inverse = bf16.restore_source(old742, inputs.bind)
    predecessor, predicate_inverse = predicate.restore_source(old740, inputs.bind)
    guard = predicate_inverse['dispatcher_guard']
    delegation = predicate_inverse['old_path_delegation']
    require(relu.SOURCE == indexing.PARENT_SOURCE and same.SOURCE == relu.PARENT_SOURCE
        and relu_inverse['functions'] == 775 and relu_inverse['parent_functions'] == 773
        and relu_inverse['changed_functions'] == ['unary']
        and relu_inverse['whole_source_inverse_exact'] is True and relu_inverse['whole_round_trip_exact'] is True
        and arithmetic.SOURCE == same.PARENT_SOURCE
        and same_inverse['functions'] == same_inverse['parent_functions'] == 773
        and same_inverse['changed_functions'] == ['half_mul_priority','public_f64_binary']
        and same_inverse['unchanged_parent_functions'] == 771
        and same_inverse['whole_source_inverse_exact'] is True and same_inverse['whole_round_trip_exact'] is True
        and division.SOURCE == arithmetic.PARENT_SOURCE
        and arithmetic_inverse['functions'] == 773 and arithmetic_inverse['parent_functions'] == 771
        and arithmetic_inverse['changed_functions'] == ['binary','half_mul_cast']
        and arithmetic_inverse['unchanged_parent_functions'] == 769
        and arithmetic_inverse['whole_source_inverse_exact'] is True
        and arithmetic_inverse['whole_round_trip_exact'] is True
        and wide.SOURCE == division.PARENT_SOURCE
        and div_inverse['functions'] == 771 and div_inverse['parent_functions'] == 767
        and div_inverse['changed_functions'] == ['binary']
        and div_inverse['unchanged_parent_functions'] == 766
        and div_inverse['whole_source_inverse_exact'] is True
        and div_inverse['whole_round_trip_exact'] is True
        and addsub.SOURCE == wide.PARENT_SOURCE
        and neg.SOURCE == addsub.PARENT_SOURCE
        and mul.SOURCE == neg.PARENT_SOURCE
        and bf16.SOURCE == mul.PARENT_SOURCE
        and predicate.SOURCE == bf16.PARENT_SOURCE
        and predicate.PARENT_SOURCE == 'bf3641b1a9061c6f92b8cbeef75adfac89e82ee16d32a9f650cb938d9e918386'
        and sha(predecessor) == predicate.PARENT_SOURCE
        and predicate_inverse['functions'] == 740
        and predicate_inverse['parent_functions'] == 736
        and predicate_inverse['unchanged_parent_functions'] == 735
        and predicate_inverse['changed_functions'] == ['entry']
        and predicate_inverse['entry_only_upper_bound_and_dispatch_target_changed'] is True
        and guard['opcode_min'] == 120 and guard['opcode_max'] == 124
        and guard['reads_before_guard'] == ['getr(request,at)']
        and guard['exact_closed_range_verified'] is True
        and delegation['target'] == 'to_command'
        and delegation['parameter_count'] == 6
        and delegation['argument_order_preserved'] is True
        and delegation['complete_parent_body_unchanged'] is True
        and delegation['parent_body_sha256'] == delegation['actual_body_sha256']
            == view_route['old_to_command_sha256']
        and view_route['source_sha256'] == SOURCE
        and view_route['actual_to_command_sha256'] == sha(functions['to_command'])
        and view_route['original_domain_dispatcher_guard_unchanged'] is True
        and view_route['original_non_f64_constructor_and_all_arguments_retained'] is True,
        'exact predicate source inverse and unchanged six-argument dispatch for the original AsStrided commands')
    layout = []
    for name,pin in pins['original_observer_layout_functions'].items():
        body = functions[name]
        require(sha(body)==pin['sha256'] and len(body)==pin['bytes'],
                'original observer layout function changed: '+name)
        layout.append(dict(name=name,sha256=sha(body),bytes=len(body),relation='byte-identical-to-original614'))
    tags = []
    for tag, bodies in pins['current_owner_tag_functions'].items():
        literal = ('{"value":'+str(struct.unpack('<Q',tag.encode())[0])+'}').encode()
        for name,pin in bodies.items():
            require(sha(functions[name])==pin and literal in functions[name],
                    'current owner tag source changed: '+name)
        tags.append(dict(tag=tag,functions=bodies))
    reference_pin = pins['references'][suite]
    reference_path = reference_root/reference_pin['reference_path']
    probe_path = reference_root/reference_pin['probe_path']
    reference = json.loads(inputs.bind(reference_path,reference_pin['reference_sha256']))
    inputs.bind(probe_path,reference_pin['probe_sha256'])
    require(reference['status']=='OBSERVED' and reference['cases']==len(reference['rows'])==COUNTS[suite]
            and reference['probe_sha256']==reference_pin['probe_sha256']
            and reference['native_executed'] is False and reference['torch_executed'] is True,
            'fixed original reference denominator or execution identity changed')
    provenance = reference['oracle_provenance']
    require(provenance['status']=='PASS' and set(provenance['checks'])==CHECKS
            and all(value is True for value in provenance['checks'].values()),
            'all eleven original oracle provenance checks required')
    require(provenance['lock_sha256']==pins['repository_files']['compatibility/pytorch-oracle-lock.json']['sha256'],
            'the original reference must use the canonical fixed Torch lock')
    proof = dict(source_sha256=SOURCE,native_sha256=IMAGE,actual_build_sha256=expected_build_sha,
        functions=800,units=15,original_layout_functions=layout,current_tag_functions=tags,
        predicate_source_inverse=predicate_inverse,
        bf16_source_inverse=bf16_inverse,
        half_mul_source_inverse=mul_inverse,
        half_neg_source_inverse=neg_inverse,
        half_addsub_source_inverse=addsub_inverse,
        wide_neg_source_inverse=wide_inverse,
        half_div_source_inverse=div_inverse,
        public_f64_arithmetic_source_inverse=arithmetic_inverse,
        typed_relu_source_inverse=relu_inverse,
        index_signbit_source_inverse=index_inverse,
        public_f64_view_original_domain_route=view_route,
        same_f64_arithmetic_source_inverse=same_inverse,
        original_source=pins['original_source'],reference_sha256=reference_pin['reference_sha256'],
        probe_sha256=reference_pin['probe_sha256'],reference_provenance=provenance['checks'],
        original_finish=pins['original_finish'],original_denominators=pins['original_denominators'][suite])
    return current, built, reference, reference_path, proof


def frames(raw, magic):
    result, at = [], 0
    while at<len(raw):
        require(at+16<=len(raw), 'truncated frame header')
        actual,size = struct.unpack_from('<8sQ', raw, at)
        require(actual==magic and at+16+size<=len(raw), 'frame magic or length mismatch')
        result.append((size,raw[at+16:at+16+size]))
        at += 16+size
    require(at==len(raw), 'unconsumed frame bytes')
    return result


def native_execution_fact(actual_directory, driver_entered):
    """Unknown is retained after an interrupted driver; no success is inferred."""
    if not driver_entered:
        return False
    for path in Path(actual_directory).glob('case-*/raw/responses.frames.bin'):
        raw = path.read_bytes()
        if len(raw)>=16:
            magic,size = struct.unpack_from('<8sQ',raw)
            if magic==b'HTOWNS01' and size>=32 and len(raw)>=16+size:
                return True
    return None
