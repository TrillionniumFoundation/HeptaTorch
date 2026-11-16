#!/usr/bin/env python3
"""Observe native_dropout optional training, real mask dtype/layout and Saved use.

Reference-only: actual pinned Torch supplies every outcome. No HIA executes,
no outcome table is predicted, and the frozen 183-case probe is unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import struct
import sys
import traceback
import warnings

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
PARENT_PROBE_SHA = 'db00efdf0cca14b3b3f25545e8a59a27308ca128d861e04086c18bd1edb6d3a4'
PARENT_REPORT_SHA = 'c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467'
D64 = lambda value: struct.unpack('<Q', struct.pack('<d', value))[0]
CHECKS = ('python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches',
          'direct_url_matches', 'direct_url_archive_sha256_matches',
          'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested')
PROBABILITIES = (('half', .5), ('nan', math.nan), ('below_zero', -.25), ('above_one', 1.25))
TRAINING = (('none', None), ('false', False), ('true', True))
EXTRA_LAYOUTS = ('overlap', 'expanded', 'channels_last', 'extreme_size_one')
SOURCE_REF = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
SOURCE_PINS = {
    'aten/src/ATen/native/Dropout.cpp': '1d4ea50e36409cf4b7e13e719bab05ccad86a5f01412d7fc9825011200f9285a',
    'tools/autograd/derivatives.yaml': '4a4d1ae1ddea605f57594de991da2c92e89ae11d6aa6edf348107589b548082c',
    'aten/src/ATen/native/TensorFactories.cpp': 'b7a3bfc016a7497bd002e4c0d5da2c74c492e7b7d9af5a0bdbc846c1f6d2160d',
    'aten/src/ATen/ExpandUtils.cpp': '69c16cc97348ebb3135eb8c1d0f175736aa7190799266dee7421b0d642690f17',
    'c10/core/ScalarType.h': '0328f8ad7e04afa056df5959fb4337ff16a9e082b3d68b4902d93e2ab12f8379',
    'torch/csrc/autograd/FunctionsManual.cpp': 'a8603d6309822a3f93fa4fcb5ada05e2f0f7e29a1b96db64fe6f59ce0c730558',
    'torchgen/api/autograd.py': 'f1d569fab4ab61c96a9dd6f99e32a255e64b5322184df257d0be741f4528060d',
}


def plan():
    rows = []
    for layout in ('transpose', 'empty_nonstandard'):
        for train_name, train in TRAINING:
            for probability_name, probability in PROBABILITIES:
                rows.append({'name': f'matrix-{layout}-{train_name}-{probability_name}',
                             'family': 'optional-training-and-probability', 'layout': layout,
                             'training': train, 'probability_name': probability_name,
                             'probability_double_bits': D64(probability),
                             'backward_path': 'retain_then_input_mutation_then_mask_mutation'})
    for layout in EXTRA_LAYOUTS:
        rows.append({'name': 'layout-' + layout, 'family': 'mask-layout-and-default-release',
                     'layout': layout, 'training': True, 'probability_name': 'half',
                     'probability_double_bits': D64(.5), 'backward_path': 'default_then_retry'})
    assert len(rows) == 28 and len({row['name'] for row in rows}) == 28
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('support', 'lock', 'executable', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    a = parser.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    inputs = {Path(__file__).resolve(): Path(__file__).read_bytes(),
              a.support.resolve(): a.support.read_bytes(), a.lock.resolve(): a.lock.read_bytes()}
    assert SHA(inputs[a.support.resolve()]) == SUPPORT_SHA
    assert SHA(inputs[a.lock.resolve()]) == LOCK_SHA
    planned = plan()
    common = {'schema': 'heptatorch.pinned-torch-native-dropout-contract-probe.v1',
              'probe_sha256': SHA(inputs[Path(__file__).resolve()]),
              'parent_probe_sha256': PARENT_PROBE_SHA, 'parent_observation_sha256': PARENT_REPORT_SHA,
              'expected_cases': len(planned), 'case_plan': planned,
              'primary_source_ref': SOURCE_REF,
              'primary_source_pins': [{'path': path, 'sha256': value,
                  'url': f'https://github.com/pytorch/pytorch/blob/{SOURCE_REF}/{path}'}
                  for path, value in SOURCE_PINS.items()],
              'native_executed': False, 'native_values_compared': 0,
              'native_qualified': False, 'all_rng_complete': False,
              'full_pytorch_rewrite_complete': False, 'pytorch_api_denominator': 2590,
              'comparison_policy': 'OBSERVATION ONLY. No expected acceptance, layout, dtype, alias, gradient or exception is substituted.'}
    if a.prepare_only:
        common.update(status='PREPARED_NOT_RUN', torch_executed=False, cases=0, rows=[],
                      inputs_sha256={str(path): SHA(data) for path, data in inputs.items()})
        a.output.write_text(json.dumps(common, indent=2, allow_nan=False) + '\n')
        print(json.dumps(common, allow_nan=False))
        return 0
    old_argv = sys.argv
    try:
        sys.argv = [str(a.support), str(a.executable),
                    str(a.output.with_suffix('.provenance.json')), str(a.lock)]
        context = runpy.run_path(str(a.support), run_name='native_dropout_contract_provenance_only')
    finally:
        sys.argv = old_argv
    t, provenance = context['torch'], context['PROVENANCE']
    assert provenance['status'] == 'PASS' and set(provenance['checks']) == set(CHECKS)
    assert all(provenance['checks'][key] is True for key in CHECKS)
    states, control_cache, rows, setup_errors = {}, {}, [], []

    def raw_state(generator=None):
        state = generator.get_state() if generator is not None else t.get_rng_state()
        raw = bytes(state.tolist())
        assert len(raw) == 5056
        return raw

    def intern(raw):
        key = SHA(raw)
        if key not in states:
            states[key] = {'bytes': len(raw), 'hex': raw.hex(),
                           'left': struct.unpack_from('<i', raw, 8)[0],
                           'next': struct.unpack_from('<Q', raw, 16)[0],
                           'normal_cache_bytes': raw[5016:].hex()}
        return key

    def initial_state():
        generator = t.Generator(device='cpu').manual_seed(1729)
        raw = bytearray(raw_state(generator))
        struct.pack_into('<d', raw, 5024, 6.25)
        struct.pack_into('<I', raw, 5040, 1)
        struct.pack_into('<f', raw, 5048, 1.25)
        raw[5052] = 1
        generator.set_state(t.tensor(list(raw), dtype=t.uint8))
        return raw_state(generator)

    def consumption(before, after, elements):
        # Explicit Generator controls only observe equal complete state bytes.
        # They do not determine the acceptance or values of native_dropout.
        candidates = sorted({0, 1, elements, 2 * elements, 2 * elements + 1})
        matches = []
        for count in candidates:
            key = (SHA(before), count)
            if key not in control_cache:
                generator = t.Generator(device='cpu')
                generator.set_state(t.tensor(list(before), dtype=t.uint8))
                t.rand(count, dtype=t.float32, generator=generator)
                control_cache[key] = raw_state(generator)
            if control_cache[key] == after:
                matches.append(count)
        return matches

    def words(x):
        value = x.detach().contiguous().reshape(-1)
        if x.dtype == t.float32:
            return [int(n) & 0xffffffff for n in value.view(t.int32).tolist()]
        if x.dtype == t.float64:
            return [int(n) & ((1 << 64) - 1) for n in value.view(t.int64).tolist()]
        if x.dtype in (t.float16, t.bfloat16):
            return [int(n) & 65535 for n in value.view(t.int16).tolist()]
        return [int(n) for n in value.tolist()]

    def tensor(x):
        if x is None:
            return None
        return {'dtype': str(x.dtype), 'shape': list(x.shape), 'strides': list(x.stride()),
                'element_size': x.element_size(), 'storage_bytes': x.untyped_storage().nbytes(),
                'storage_offset_elements': x.storage_offset(),
                'storage_offset_bytes': x.storage_offset() * x.element_size(),
                'tensor_identity': str(x._cdata), 'storage_identity': str(x.untyped_storage()._cdata),
                'version': x._version, 'requires_grad': x.requires_grad, 'is_leaf': x.is_leaf,
                'grad_fn': type(x.grad_fn).__name__ if x.grad_fn is not None else None,
                'binary_words': words(x)}

    def gradient(x):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = tensor(x.grad)
        return {'tensor': result, 'warnings': [
            {'type': type(row.message).__name__, 'message': str(row.message)} for row in caught]}

    def capture(operation):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                value = operation()
                result = {'accepted': True, 'outcome': 'SUCCEEDED', 'error': None}
            except Exception as error:
                value = None
                result = {'accepted': False, 'outcome': 'RAISED',
                          'error': {'type': type(error).__name__, 'message': str(error)}}
        result['warnings'] = [{'type': type(row.message).__name__, 'message': str(row.message)}
                              for row in caught]
        return value, result

    def make_input(layout):
        if layout == 'transpose':
            base = t.tensor([-2., -0., 0., 1., 3., 9.], dtype=t.float32).reshape(2, 3)
            x = base.t()
        elif layout == 'empty_nonstandard':
            base = t.empty((0,), dtype=t.float32)
            x = base.as_strided((2, 0, 3), (17, 101, 2), 7)
        elif layout == 'overlap':
            base = t.tensor([-2., -0., 1., 3.], dtype=t.float32)
            x = base.as_strided((2, 3), (1, 1))
        elif layout == 'expanded':
            base = t.tensor([-2., 3.], dtype=t.float32)
            x = base.as_strided((3, 2), (0, 1))
        elif layout == 'channels_last':
            x = t.arange(12, dtype=t.float32).reshape(1, 2, 2, 3)
            x = x.contiguous(memory_format=t.channels_last)
        elif layout == 'extreme_size_one':
            base = t.tensor([-2., -0., 0., 1., 3., 9.], dtype=t.float32)
            x = base.as_strided((1, 2, 3), (1 << 61, 1, 2))
        else:
            raise AssertionError(layout)
        return x.detach().requires_grad_(True)

    def saved(output, mask):
        fn = output.grad_fn
        if fn is None:
            return {'grad_fn': None, 'attributes': {}}
        attributes = {}
        for name in sorted(n for n in dir(fn) if n.startswith('_saved_')):
            value, observation = capture(lambda name=name: getattr(fn, name))
            if observation['accepted']:
                if isinstance(value, t.Tensor):
                    observation['tensor'] = tensor(value)
                    observation['same_Tensor_as_mask'] = value._cdata == mask._cdata
                    observation['same_Storage_as_mask'] = value.untyped_storage()._cdata == mask.untyped_storage()._cdata
                elif isinstance(value, float):
                    observation['double_bits'] = D64(value)
                elif value is None or isinstance(value, (bool, int, str)):
                    observation['value'] = value
                else:
                    observation['diagnostic_repr'] = repr(value)
            attributes[name] = observation
        return {'grad_fn': type(fn).__name__, 'attributes': attributes}

    def backward_observation(x, output, mask, seed, *, retain):
        _, event = capture(lambda: output.backward(seed, retain_graph=retain))
        event.update(input_after=tensor(x), output_after=tensor(output), mask_after=tensor(mask),
                     input_gradient=gradient(x), state_after=intern(raw_state()))
        return event

    def observe(spec):
        probability = struct.unpack('<d', struct.pack('<Q', spec['probability_double_bits']))[0]
        x = make_input(spec['layout'])
        before = initial_state()
        t.set_rng_state(t.tensor(list(before), dtype=t.uint8))
        row = {**spec, 'input_before': tensor(x), 'state_before': intern(before)}
        result, forward = capture(lambda: t.ops.aten.native_dropout(x, probability, spec['training']))
        after = raw_state()
        row.update(forward=forward, input_after_forward=tensor(x), state_after_forward=intern(after),
                   state_unchanged=before == after,
                   matching_raw32_uniform_counts=consumption(before, after, x.numel()),
                   normal_caches_unchanged=before[5016:] == after[5016:])
        if not forward['accepted']:
            row.update(status='OBSERVED', backward_not_attempted='No forward outputs were returned.')
            return row
        assert isinstance(result, tuple) and len(result) == 2
        output, mask = result
        row.update(output=tensor(output), mask=tensor(mask),
                   output_is_input_object=output is x, output_is_input_Tensor=output._cdata == x._cdata,
                   output_aliases_input_Storage=output.untyped_storage()._cdata == x.untyped_storage()._cdata,
                   mask_aliases_input_Storage=mask.untyped_storage()._cdata == x.untyped_storage()._cdata,
                   mask_aliases_output_Storage=mask.untyped_storage()._cdata == output.untyped_storage()._cdata,
                   saved_before_backward=saved(output, mask))
        # Direct backward avoids fabricating an int64 mask.sum target surface.
        _, mask_event = capture(lambda: mask.backward(t.ones_like(mask)))
        mask_event.update(input_gradient=gradient(x), mask_after=tensor(mask), state_after=intern(raw_state()))
        row['mask_as_output_backward'] = mask_event
        x.grad = None
        seed_values = [-0., 1., -1., 2., -3., .5]
        if output.numel() == 0:
            seed = t.empty_like(output)
        else:
            values = [seed_values[i % len(seed_values)] for i in range(output.numel())]
            seed = t.tensor(values, dtype=t.float32).reshape(tuple(output.shape))
        row['seed'] = tensor(seed)
        if spec['backward_path'] == 'default_then_retry':
            row['first_backward'] = backward_observation(x, output, mask, seed, retain=False)
            row['saved_after_default_backward'] = saved(output, mask)
            row['mask_still_public_after_release'] = tensor(mask)
            x.grad = None
            row['retry_after_default_release'] = backward_observation(x, output, mask, seed, retain=False)
        else:
            row['first_backward'] = backward_observation(x, output, mask, seed, retain=True)
            x.grad = None
            def mutate_input():
                with t.no_grad():
                    x.add_(.25)
            _, mutation = capture(mutate_input)
            mutation.update(input_after=tensor(x), output_after=tensor(output), mask_after=tensor(mask))
            row['input_mutation'] = mutation
            row['backward_after_input_mutation'] = backward_observation(x, output, mask, seed, retain=True)
            x.grad = None
            def mutate_mask():
                with t.no_grad():
                    mask.zero_()
            _, mutation = capture(mutate_mask)
            mutation.update(mask_after=tensor(mask), output_after=tensor(output),
                            state_after=intern(raw_state()))
            row['mask_mutation'] = mutation
            row['saved_after_mask_mutation'] = saved(output, mask)
            row['backward_after_mask_mutation'] = backward_observation(x, output, mask, seed, retain=False)
        row.update(status='OBSERVED', state_after_all=intern(raw_state()),
                   all_backward_and_mutations_consume_no_rng=raw_state() == after)
        return row

    with t.random.fork_rng(devices=[]):
        for index, spec in enumerate(planned):
            try:
                rows.append({'index': index, **observe(spec)})
            except BaseException as error:
                setup_errors.append({'index': index, 'name': spec['name'],
                                     'error': repr(error), 'traceback': traceback.format_exc()})
                rows.append({'index': index, **spec, 'status': 'PROBE_ERROR',
                             'error': repr(error), 'traceback': traceback.format_exc()})
    assert all(path.read_bytes() == raw for path, raw in inputs.items())
    report = {**common, 'status': 'OBSERVED' if not setup_errors else 'PROBE_ERROR',
              'torch_executed': True, 'oracle_provenance': provenance,
              'build_configuration': t.__config__.show(), 'mkl_available': t.backends.mkl.is_available(),
              'cpu_capability': t.backends.cpu.get_cpu_capability(), 'threads': t.get_num_threads(),
              'source_files_unchanged': True, 'cases': len(rows), 'rows': rows,
              'states': states, 'unique_complete_states': len(states),
              'probe_errors': setup_errors, 'probe_error_count': len(setup_errors),
              'inputs_sha256': {str(path): SHA(data) for path, data in inputs.items()},
              'consumption_reference': 'Separate explicit Generator restored from each complete5056-byte before state; torch.rand float32 controls compared byte-for-byte.'}
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False))
    return int(bool(setup_errors))


if __name__ == '__main__':
    raise SystemExit(main())
