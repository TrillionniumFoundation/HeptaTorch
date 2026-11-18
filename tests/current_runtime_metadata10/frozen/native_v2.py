#!/usr/bin/env python3
"""Observe the ten cat recipes on the admitted HIA image, then compare metadata.

The actual Torch report supplies comparison values only. Literal recipe inputs,
the same object repetitions, cat dimensions and seed construction are fixed by
the independently executed producer. This is a bounded comparison, not schema
closure or a replacement for any original failed legacy report.
"""
import argparse
import hashlib
import importlib.util
import json
import lzma
import os
from pathlib import Path
import signal
import struct
import sys
import traceback

sys.dont_write_bytecode = True
SOURCE = '8fcd7ac16f0b2bf09207120cd7806ff0384708ce0b59301881caec8cadb23794'
IMAGE = '3b11781b08fe1d5f28ec595292eac1123f652acc63be8149fd08b884608b4319'
REFERENCE = '2b80fe6a5a9c83d305b73a6eb86e3d788196eb9e11297402e7ba4e8a33ca6243'
PRODUCER = 'd9f55ee44f51d1800f6aa231f472fc0051f815d284b2f4590f383e443e52b57c'
FIELDS = ('shape', 'strides', 'storage_offset', 'numel', 'dtype', 'element_size',
          'requires_grad', 'is_leaf', 'version')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'build-directory', 'reference', 'producer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-build-sha256', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert not sys.flags.optimize
    repo, build, out = args.repository.resolve(), args.build_directory.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binding = load(repo / 'tests/current_runtime_original35/binding.py', 'cat10_binding')
    inputs = binding.Inputs()
    inputs.bind(Path(__file__).resolve())
    inputs.bind(repo / 'tests/current_runtime_original35/binding.py')
    reference = json.loads(inputs.bind(args.reference.resolve(), REFERENCE))
    inputs.bind(args.producer.resolve(), PRODUCER)
    recipes = load(args.producer.resolve(), 'cat10_literal_recipes').CASES
    assert reference['status'] == 'OBSERVED' and reference['cases'] == len(recipes) == 10
    assert reference['inputs_unchanged'] and reference['rng_unchanged']
    assert len(reference['oracle_provenance']['checks']) == 11
    assert all(reference['oracle_provenance']['checks'].values())
    assert reference['recipes'] == list(recipes)
    adapter_path = repo / 'tests/current_runtime270.py'
    fixture_path = repo / 'tests/current_runtime_fixtures.py'
    inputs.bind(adapter_path); inputs.bind(fixture_path)
    adapter = load(adapter_path, 'cat10_owner_binding')
    fixtures = load(fixture_path, 'cat10_fixture_package')
    package = fixtures.restore_current270(repo, out / 'original-owner-package', inputs)
    built, source_proof = adapter.make_bindings(package, build, args.expected_build_sha256, inputs)
    assert built['source_sha256'] == SOURCE and built['native_sha256'] == IMAGE
    owner, owner_proof, owner_raw = sys.modules['prepare'].owner_module(SOURCE, IMAGE)
    (out / 'owner-current.py').write_bytes(owner_raw)
    inputs.bind(out / 'owner-current.py')
    dump(out / 'source-binding.json', dict(current=source_proof, owner=owner_proof))
    logical = repo / 'tests/logical_reductions'
    for path in (logical / 'cases.py', logical / 'native.py',
                 repo / 'tests/differentiable_view_native_review/protocol.py',
                 repo / 'tests/differentiable_view_native_review/owner_client.py'):
        inputs.bind(path)
    sys.path.insert(0, str(logical))
    module = load(logical / 'native.py', 'cat10_exact_native_observer')
    wire = module.load_wire(repo)
    args.binary = build / 'heptatorch'
    rows = []
    report = dict(schema='heptatorch.current-cat.actual-reference10.v1', status='RUNNING',
                  source_sha256=SOURCE, native_sha256=IMAGE, actual_build_sha256=args.expected_build_sha256,
                  reference_sha256=REFERENCE, producer_sha256=PRODUCER, rows=rows,
                  native_executed=False, Torch_executed=False, compiler_executed=False,
                  old_reports_modified=False, full_pytorch_rewrite_complete=False,
                  scope='Ten fixed CPU F32 cat recipes and observable Tensor/gradient metadata; Python warning text and grad_fn class names are not compared.')

    def compare(actual, expected, path, differences):
        if actual is None or expected is None:
            if actual is not expected:
                differences.append(dict(path=path, native=actual, reference=expected))
            return
        for field in FIELDS:
            if actual[field] != expected[field]:
                differences.append(dict(path=path + '/' + field, native=actual[field], reference=expected[field]))
        for actual_key, expected_key in (('storage_bytes', 'storage_nbytes'),
                                         ('logical_bytes_hex', 'logical_data_hex')):
            if actual[actual_key] != expected[expected_key]:
                differences.append(dict(path=path + '/' + actual_key,
                                        native=actual[actual_key], reference=expected[expected_key]))
        if (actual['grad_fn'] is None) != (expected['grad_fn'] is None):
            differences.append(dict(path=path + '/grad_fn_presence',
                                    native=actual['grad_fn'], reference=expected['grad_fn']))
        if 'gradient' in actual:
            compare(actual['gradient'], expected['gradient'], path + '/gradient', differences)

    def execute(recipe, expected, ordinal):
        directory = out / ('%02d-' % ordinal + recipe['name'])
        native = module.Native(args, wire, directory)
        result = dict(name=recipe['name'], recipe=recipe, status='OBSERVING',
                      native_executed=True, differences=[], owner_observations=[])
        try:
            # Wait for the forked child to reach the native frame read after exec.
            native.s.control_base()
            assert sha(Path('/proc/%d/exe' % native.s.pid).read_bytes()) == IMAGE

            def snapshot_owner(label):
                observed = native.s.owner()
                decoded = owner.decode_owner(observed['bytes'], observed['pointer'])
                checked = owner.verify_snapshot(decoded)
                raw = observed['bytes']; digest = sha(raw)
                target = directory / (digest + '.owner.xz')
                if not target.exists():
                    target.write_bytes(lzma.compress(raw, preset=1))
                result['owner_observations'].append(dict(label=label, pointer=observed['pointer'],
                    raw_bytes=len(raw), raw_sha256=digest, file=target.name,
                    compressed_sha256=sha(target.read_bytes()), verified=checked))
                return decoded

            def snapshot(token, label):
                observed, unused = native.tensor(token)
                before = snapshot_owner(label + '/before-grad-read')
                index = owner.public_node(before, token)
                gradient_index = before['nodes'][str(index)][48]
                public_before = bool(gradient_index and before['nodes'][str(gradient_index)][25])
                gradient = native.one(91, token)['value']
                assert native.one(59, token)['value'] == bool(gradient)
                try:
                    observed['gradient'] = None if gradient == 0 else native.tensor(gradient)[0]
                finally:
                    if gradient and not public_before:
                        native.one(41, gradient)
                snapshot_owner(label + '/after-grad-read')
                return observed

            def capture(label, tokens):
                result[label] = [snapshot(token, label + '/' + str(i)) for i, token in enumerate(tokens)]

            tensors = []
            for i, shape in enumerate(recipe['shapes']):
                values = recipe.get('values', {}).get(str(i))
                if values is None:
                    assert 0 in shape
                    raw = b''
                else:
                    assert shape == [2, 1] and values == [[1.0], [2.0]]
                    raw = struct.pack('<2f', 1.0, 2.0)
                tensors.append(native.create(0, shape, raw, 1))
            capture('inputs_before_forward', tensors)
            repeated = [tensors[i] for i, count in recipe['runs'] for unused in range(count)]
            payload = wire.Payload(struct.pack('<Q', len(repeated)) + struct.pack('<' + 'Q' * len(repeated), *repeated))
            status, records, before, after = native.issue([(34, recipe['dim'], payload)], 'actual-cat')
            result['forward_status'] = status
            assert status == 0 and len(records) == 1
            output = records[0]['value']
            result['forward_output'] = snapshot(output, 'output-after-forward')
            capture('inputs_after_forward', tensors)
            seed = None
            if recipe['backward']:
                output_meta = native.tensor(output)[0]
                seed = native.create(0, output_meta['shape'], struct.pack('<f', 1.0) * output_meta['numel'], 0)
                result['seed_before'] = snapshot(seed, 'seed-before')
                status, records, before, after = native.issue([(58, output, seed, 0)], 'actual-backward')
                result['backward_status'] = status
                assert status == 0 and len(records) == 1 and records[0]['value'] == 0
                result['seed_after'] = snapshot(seed, 'seed-after')
                result['output_after_backward'] = snapshot(output, 'output-after-backward')
            capture('inputs_after_backward_or_skip', tensors)
            native.one(41, output)
            if seed is not None:
                native.one(41, seed)
            capture('inputs_after_releasing_output_and_seed', tensors)
            for token in tensors:
                native.one(41, token)
            snapshot_owner('after-releasing-inputs')
            native.finish()
            result.update(exit_code=0, stdout_EOF=True, stderr_EOF=True,
                          frames=native.s.frames, process_id=native.s.pid, forced_cleanup=False)
            for label in ('inputs_before_forward', 'inputs_after_forward', 'inputs_after_backward_or_skip',
                          'inputs_after_releasing_output_and_seed'):
                assert len(result[label]) == len(expected[label])
                for i, (actual, wanted) in enumerate(zip(result[label], expected[label])):
                    compare(actual, wanted, label + '/' + str(i), result['differences'])
            compare(result['forward_output'], expected['forward']['output'], 'forward/output', result['differences'])
            if recipe['backward']:
                for actual_key, ref_key in (('seed_before', 'seed_before'), ('seed_after', 'seed_after'),
                                            ('output_after_backward', 'output_after')):
                    compare(result[actual_key], expected['backward'][ref_key], 'backward/' + ref_key, result['differences'])
            result['status'] = 'MATCH_FIXED_REFERENCE_FIELDS' if not result['differences'] else 'DIFFERENT_FIXED_REFERENCE_FIELDS'
        except BaseException as error:
            result.update(status='FAIL', error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        finally:
            if native.s.poll() is None:
                native.s.close_input()
                try:
                    native.s.finish()
                except BaseException as error:
                    result['cleanup_error'] = repr(error)
                    if native.s.poll() is None:
                        os.kill(native.s.pid, signal.SIGKILL)
                        os.waitpid(native.s.pid, 0)
                        result['forced_cleanup'] = True
            dump(directory / 'case.json', result)
        return result

    if args.prepare_only:
        report['status'] = 'PREPARED_NOT_EXECUTED'
    else:
        for ordinal, (recipe, expected) in enumerate(zip(recipes, reference['rows'])):
            assert recipe['name'] == expected['name'] and expected['recipe'] == recipe
            rows.append(execute(recipe, expected, ordinal))
            report['native_executed'] = True
            dump(out / 'report.json', report)
        report['status'] = 'MATCH_ALL_TEN_FIXED_REFERENCE_FIELDS' if all(row['status'] == 'MATCH_FIXED_REFERENCE_FIELDS' for row in rows) else 'FAIL_OR_DIFFERENCE'
    errors = inputs.recheck()
    report.update(inputs=inputs.records, all_inputs_unchanged=not errors, input_recheck_errors=errors)
    if errors:
        report['status'] = 'FAIL_INPUT_RECHECK'
    dump(out / 'report.json', report)
    print(json.dumps(dict(status=report['status'], cases=len(rows),
                         differences=sum(len(row['differences']) for row in rows), all_inputs_unchanged=not errors)))
    return 0 if report['status'] in ('PREPARED_NOT_EXECUTED', 'MATCH_ALL_TEN_FIXED_REFERENCE_FIELDS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
