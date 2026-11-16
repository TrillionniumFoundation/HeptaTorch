#!/usr/bin/env python3
"""Observe pinned CPU Bernoulli/dropout dispatch, state, layout and autograd.

Only actual Torch operations produce observations. No HIA target executes and
no expected outcome is substituted. Complete 5056-byte states are deduplicated
by SHA256; raw32-consumption controls use independent explicit CPU Generators
and the already source-audited float32 uniform primitive.
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

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
D64 = lambda value: struct.unpack('<Q', struct.pack('<d', value))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('support', 'lock', 'executable', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    assert SHA(a.support.read_bytes()) == SUPPORT_SHA and SHA(a.lock.read_bytes()) == LOCK_SHA
    old_argv = sys.argv
    try:
        sys.argv = [str(a.support), str(a.executable), str(a.output.with_suffix('.provenance.json')), str(a.lock)]
        context = runpy.run_path(str(a.support), run_name='bernoulli_dropout_provenance_only')
    finally:
        sys.argv = old_argv
    t, provenance = context['torch'], context['PROVENANCE']
    assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
    assert all(provenance['checks'].values())
    states, rows, controls, setup_errors = {}, [], {}, []

    def raw_state(generator=None):
        return bytes((generator.get_state() if generator is not None else t.get_rng_state()).tolist())

    def intern(raw):
        key = SHA(raw)
        if key not in states:
            assert len(raw) == 5056
            states[key] = {'bytes': len(raw), 'hex': raw.hex(),
                           'left': struct.unpack_from('<i', raw, 8)[0],
                           'next': struct.unpack_from('<Q', raw, 16)[0],
                           'normal_cache_bytes': raw[5016:].hex()}
        return key

    def generator(seed=1729, cached=False, zero_engine=False):
        g = t.Generator(device='cpu').manual_seed(seed)
        if cached or zero_engine:
            raw = bytearray(raw_state(g))
            if cached:
                struct.pack_into('<d', raw, 5024, 6.25)
                struct.pack_into('<I', raw, 5040, 1)
                struct.pack_into('<f', raw, 5048, 1.25)
                raw[5052] = 1
            if zero_engine:
                # Legal set_state input, not a seed claim: forces exact uniform0
                # so rare positive-probability/large-scale boundaries execute.
                raw[24:5016] = bytes(624 * 8)
            g.set_state(t.tensor(list(raw), dtype=t.uint8))
        return g

    def consumption(before, after, elements):
        # This records matching controls, not a required dispatch verdict.
        candidates = {0, 1, elements, 2 * elements}
        candidates |= {n for n in (elements-1, elements+1, 2*elements-1, 2*elements+1, 2, 3, 4, 5, 6, 7) if 0 <= n <= 8192}
        match = []
        initial = SHA(before)
        for count in sorted(candidates):
            key = (initial, count)
            if key not in controls:
                g = t.Generator(device='cpu')
                g.set_state(t.tensor(list(before), dtype=t.uint8))
                t.rand(count, dtype=t.float32, generator=g)
                controls[key] = raw_state(g)
            if controls[key] == after:
                match.append(count)
        return match

    def words(x):
        value = x.detach().contiguous().reshape(-1)
        if x.dtype == t.float64:
            return [int(n) & ((1 << 64) - 1) for n in value.view(t.int64).tolist()]
        if x.dtype == t.float32:
            return [int(n) & 0xffffffff for n in value.view(t.int32).tolist()]
        if x.dtype in (t.float16, t.bfloat16):
            return [int(n) & 65535 for n in value.view(t.int16).tolist()]
        return [int(n) for n in value.tolist()]

    def tensor(x):
        return {'dtype': str(x.dtype), 'shape': list(x.shape), 'strides': list(x.stride()),
                'storage_offset': x.storage_offset(), 'version': x._version,
                'requires_grad': x.requires_grad, 'is_leaf': x.is_leaf,
                'grad_fn': type(x.grad_fn).__name__ if x.grad_fn is not None else None,
                'words': words(x)}

    def physical(x):
        n = x.untyped_storage().nbytes() // x.element_size()
        return words(x.as_strided((n,), (1,), 0))

    def record(name, operation, x, g, *, extra=None, global_rng=False, no_grad=True):
        row = {'name': name, 'input_before': tensor(x), 'physical_before': physical(x)}
        if extra:
            row.update(extra)
        if global_rng:
            t.set_rng_state(g.get_state())
        before = raw_state(None if global_rng else g)
        result, error = None, None
        try:
            with t.set_grad_enabled(not no_grad):
                result = operation()
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        after = raw_state(None if global_rng else g)
        row.update(error=error, accepted=error is None, input_after=tensor(x), physical_after=physical(x),
                   state_before=intern(before), state_after=intern(after),
                   state_unchanged=before == after,
                   matching_raw32_uniform_counts=consumption(before, after, x.numel()),
                   normal_caches_unchanged=before[5016:] == after[5016:])
        if isinstance(result, tuple):
            row['results'] = [tensor(y) for y in result]
            row['result0_is_input'] = result[0] is x
            row['result0_aliases_input_storage'] = result[0].untyped_storage()._cdata == x.untyped_storage()._cdata
        elif result is not None:
            row['result'] = tensor(result)
            row['result_is_input'] = result is x
            row['result_aliases_input_storage'] = result.untyped_storage()._cdata == x.untyped_storage()._cdata
        rows.append(row)
        return result

    def setup(name, body):
        try:
            body()
        except BaseException as exc:
            setup_errors.append({'name': name, 'error': repr(exc), 'traceback': traceback.format_exc()})

    lengths = (0, 1, 2, 7, 623, 624, 625, 2048)
    for n in lengths:
        for probability in (0., .5, 1.):
            def run(n=n, probability=probability):
                x = t.full((n,), -7., dtype=t.float32)
                g = generator()
                record(f'scalar-consumption-n{n}-p{probability}', lambda: x.bernoulli_(probability, generator=g), x, g,
                       extra={'family': 'scalar-p', 'probability_double_bits': D64(probability)})
            setup(f'scalar-{n}-{probability}', run)
        for dtype in (t.float32, t.float64):
            def run(n=n, dtype=dtype):
                x, probability, g = t.full((n,), -7., dtype=t.float32), t.full((n,), .5, dtype=dtype), generator()
                record(f'tensor-consumption-n{n}-{dtype}', lambda: x.bernoulli_(probability, generator=g), x, g,
                       extra={'family': 'tensor-p', 'probabilities': tensor(probability)})
            setup(f'tensor-{n}-{dtype}', run)

    # Scalar probabilities really are binary64, including close neighbors.
    scalar_probabilities = (0., -0., 1., .1, math.nextafter(.5, 0.), math.nextafter(.5, 1.),
                            2. ** -53, math.ldexp(1., -1074), math.nextafter(1., 0.), -0.1, 1.1, math.nan, math.inf)
    for n in (0, 7):
        for probability in scalar_probabilities:
            def run(n=n, probability=probability):
                x, g = t.full((n,), -7., dtype=t.float32), generator(cached=True)
                record(f'scalar-bound-n{n}-bits{D64(probability):016x}', lambda: x.bernoulli_(probability, generator=g), x, g,
                       extra={'family': 'scalar-bound', 'probability_double_bits': D64(probability)})
            setup(f'scalar-bound-{n}-{D64(probability)}', run)

    # Probability equality/adjacent-double observations distinguish strict <.
    for dtype in (t.float32, t.float64):
        control = generator(99)
        first = float(t.rand((), dtype=dtype, generator=control).item())
        probabilities = (math.nextafter(first, 0.), first, math.nextafter(first, 1.))
        for ordinal, probability in enumerate(probabilities):
            def run(dtype=dtype, ordinal=ordinal, probability=probability, first=first):
                x, g = t.full((1,), -7., dtype=t.float32), generator(99)
                if dtype == t.float64:
                    operation = lambda: x.bernoulli_(probability, generator=g)
                    extra = {'family': 'scalar-strict-compare', 'probability_double_bits': D64(probability)}
                else:
                    # Use float32 nextafter so all three Tensor values differ.
                    center = t.tensor(first, dtype=t.float32)
                    pvalue = t.nextafter(center, t.tensor(0. if ordinal == 0 else 1., dtype=t.float32)) if ordinal != 1 else center
                    operation = lambda: x.bernoulli_(pvalue, generator=g)
                    extra = {'family': 'tensor-strict-compare', 'probabilities': tensor(pvalue)}
                record(f'strict-{dtype}-{ordinal}', operation, x, g, extra=extra)
            setup(f'strict-{dtype}-{ordinal}', run)

    layouts = [
        ('contiguous', (2, 3), (3, 1), 0, 12),
        ('transpose', (3, 2), (1, 3), 0, 12),
        ('permuted3', (2, 3, 2), (1, 4, 2), 0, 24),
        ('channels-last', (1, 2, 2, 3), (12, 1, 6, 2), 0, 24),
        ('holes', (2, 3), (7, 2), 1, 20),
        ('offset-transpose', (3, 2), (2, 9), 3, 24),
        ('equal-stride-overlap', (2, 2), (1, 1), 0, 8),
        ('expanded', (2, 3), (0, 1), 0, 8),
        ('empty-outside-offset', (0, 3), (3, 1), 999, 8),
    ]
    for name, shape, strides, offset, capacity in layouts:
        def run(name=name, shape=shape, strides=strides, offset=offset, capacity=capacity):
            base = t.full((capacity,), -7., dtype=t.float32)
            x = base.as_strided(shape, strides, offset)
            g = generator(99)
            record('scalar-layout-' + name, lambda: x.bernoulli_(.3, generator=g), x, g,
                   extra={'family': 'scalar-layout', 'probability_double_bits': D64(.3)})
            base = t.full((capacity,), -7., dtype=t.float32)
            x = base.as_strided(shape, strides, offset)
            probability = t.tensor([.125, .5, .875], dtype=t.float32)
            if shape[-1] != 3:
                probability = t.tensor([.125, .875], dtype=t.float32)
            g = generator(99)
            record('tensor-layout-broadcast-' + name, lambda: x.bernoulli_(probability, generator=g), x, g,
                   extra={'family': 'tensor-layout-broadcast', 'probabilities': tensor(probability)})
            base = t.tensor([(i-7)*.25 for i in range(capacity)], dtype=t.float32)
            x = base.as_strided(shape, strides, offset)
            g = generator(99)
            record('dropout-layout-' + name, lambda: t.nn.functional.dropout(x, p=.3, training=True), x, g,
                   global_rng=True, extra={'family': 'dropout-layout', 'probability_double_bits': D64(.3)})
        setup('layouts-' + name, run)

    probability_vectors = [
        ('invalid-first', [math.nan, .5, .5, .5]), ('invalid-late', [.5, .5, math.nan, .5]),
        ('negative-late', [.5, .5, -.1, .5]), ('above-one-late', [.5, .5, 1.1, .5]),
        ('special-valid', [0., 1., -0., .5]),
    ]
    for name, probabilities in probability_vectors:
        for empty in (False, True):
            def run(name=name, probabilities=probabilities, empty=empty):
                x, probability, g = t.full((0 if empty else 4,), -7., dtype=t.float32), t.tensor(probabilities, dtype=t.float32), generator()
                record(f'tensor-probability-{name}-empty{empty}', lambda: x.bernoulli_(probability, generator=g), x, g,
                       extra={'family': 'tensor-invalid-and-empty', 'probabilities': tensor(probability)})
            setup(f'tensor-invalid-{name}-{empty}', run)
    for empty in (False, True):
        def run(empty=empty):
            x, probability, g = t.full((0 if empty else 4,), -7., dtype=t.float32), t.tensor([0, 1], dtype=t.int64), generator()
            record(f'tensor-wrong-dtype-shape-empty{empty}', lambda: x.bernoulli_(probability, generator=g), x, g,
                   extra={'family': 'tensor-invalid-dtype-shape', 'probabilities': tensor(probability)})
        setup(f'bad-tensor-{empty}', run)

    for alias in ('self', 'repeat-self', 'partial-input', 'transposed-input'):
        def run(alias=alias):
            base = t.tensor([.125, .375, .625, .875], dtype=t.float32)
            x = base.reshape(2, 2)
            if alias == 'repeat-self':
                x = base.as_strided((2, 2), (1, 1))
            probability = x
            if alias == 'partial-input':
                probability = base[0]
            if alias == 'transposed-input':
                probability = x.t()
            g = generator(19)
            record('tensor-alias-' + alias, lambda: x.bernoulli_(probability, generator=g), x, g,
                   extra={'family': 'tensor-alias', 'probabilities_before': tensor(probability)})
        setup('alias-' + alias, run)

    with t.random.fork_rng(devices=[]):
        for probability in (0., math.ldexp(1., -1074), 2. ** -54, math.nextafter(2. ** -54, 1.),
                            .1, .2, .3, math.nextafter(.5, 1.), .7, .9, math.nextafter(1., 0.), 1.):
            def run(probability=probability):
                x = t.tensor([-2., -0., 0., 1., 3., 9.], dtype=t.float32, requires_grad=True)
                g = generator(zero_engine=True)
                y = record(f'dropout-probability-bound-{D64(probability):016x}',
                           lambda: t.nn.functional.dropout(x, probability, True), x, g,
                           global_rng=True, no_grad=False,
                           extra={'family': 'dropout-probability-bound', 'probability_double_bits': D64(probability),
                                  'accepted_set_state_with_zero_MT_words': True})
                saved = getattr(y.grad_fn, '_saved_other', None) if y.grad_fn is not None else None
                rows[-1]['saved_scale_words'] = tensor(saved) if saved is not None else None
                y.sum().backward()
                rows[-1]['gradient'] = tensor(x.grad)
            setup(f'dropout-probability-bound-{D64(probability)}', run)
        for n in (0, 6):
            for probability in (0., .3, 1., -0.1, math.nan):
                for train in (False, True):
                    for inplace in (False, True):
                        def run(n=n, probability=probability, train=train, inplace=inplace):
                            x = t.tensor([-2., -0., 0., 1., 3., 9.][:n], dtype=t.float32)
                            g = generator(cached=True)
                            record(f'dropout-special-{n}-{D64(probability):016x}-{train}-{inplace}',
                                   lambda: t.nn.functional.dropout(x, p=probability, training=train, inplace=inplace), x, g,
                                   global_rng=True, extra={'family': 'dropout-special', 'probability_double_bits': D64(probability),
                                                           'training': train, 'inplace': inplace})
                        setup(f'dropout-special-{n}-{D64(probability)}-{train}-{inplace}', run)
        for n in (0, 6):
            for probability in (0., .3, 1., math.nan):
                def run(n=n, probability=probability):
                    x, g = t.ones((n,), dtype=t.float32), generator()
                    record(f'native-dropout-contrast-{n}-{D64(probability):016x}',
                           lambda: t.ops.aten.native_dropout(x, probability, True), x, g, global_rng=True,
                           extra={'family': 'native-dropout-contrast', 'probability_double_bits': D64(probability)})
                setup(f'native-contrast-{n}-{D64(probability)}', run)

        for probability in (0., .3, 1.):
            def run(probability=probability):
                x = t.tensor([-2., -0., 0., 1., 3., 9.], dtype=t.float32, requires_grad=True)
                g = generator(99)
                y = record(f'dropout-autograd-{probability}', lambda: t.nn.functional.dropout(x, probability, True),
                           x, g, global_rng=True, no_grad=False,
                           extra={'family': 'dropout-autograd', 'probability_double_bits': D64(probability)})
                row = rows[-1]
                saved = getattr(y.grad_fn, '_saved_other', None) if y.grad_fn is not None else None
                row['saved_other_before'] = tensor(saved) if saved is not None else None
                seed = t.tensor([-0., 1., -1., 2., -3., 0.], dtype=t.float32)
                y.backward(seed, retain_graph=True)
                row['first_gradient'] = tensor(x.grad)
                x.grad = None
                with t.no_grad():
                    x.add_(1.)
                y.backward(seed)
                row['gradient_after_input_mutation'] = tensor(x.grad)
                try:
                    y.backward(seed)
                    row['backward_after_release'] = {'accepted': True}
                except Exception as exc:
                    row['backward_after_release'] = {'accepted': False, 'type': type(exc).__name__, 'message': str(exc)}
                row['state_after_backward'] = intern(raw_state())
                row['backward_consumes_no_rng'] = row['state_after_backward'] == row['state_after']
            setup(f'dropout-autograd-{probability}', run)
        for probability in (0., .3, 1.):
            def run(probability=probability):
                x = t.ones((4,), dtype=t.float32, requires_grad=True)
                g = generator()
                record(f'dropout-inplace-grad-leaf-{probability}',
                       lambda: t.nn.functional.dropout(x, probability, True, True), x, g,
                       global_rng=True, no_grad=False,
                       extra={'family': 'dropout-inplace-grad-leaf', 'probability_double_bits': D64(probability)})
            setup(f'dropout-inplace-grad-leaf-{probability}', run)
        def mutated_mask():
            x, g = t.ones((4,), dtype=t.float32, requires_grad=True), generator()
            y = record('dropout-saved-mask-mutation', lambda: t.nn.functional.dropout(x, .3, True), x, g,
                       global_rng=True, no_grad=False, extra={'family': 'dropout-saved-mask-mutation'})
            mask = y.grad_fn._saved_other
            rows[-1]['saved_before_mutation'] = tensor(mask)
            mask.add_(1.)
            try:
                y.sum().backward()
                rows[-1]['backward_after_mask_mutation'] = {'accepted': True}
            except Exception as exc:
                rows[-1]['backward_after_mask_mutation'] = {'accepted': False, 'type': type(exc).__name__, 'message': str(exc)}
            rows[-1]['gradient_is_None'] = x.grad is None
            rows[-1]['state_after_backward'] = intern(raw_state())
        setup('dropout-saved-mask-mutation', mutated_mask)

    def bernoulli_ad():
        x, g = t.full((4,), .5, dtype=t.float32, requires_grad=True), generator()
        y = record('bernoulli-functional-autograd', lambda: t.bernoulli(x, generator=g), x, g,
                   no_grad=False, extra={'family': 'bernoulli-functional-autograd'})
        y.backward(t.tensor([-0., math.nan, math.inf, -1.], dtype=t.float32))
        rows[-1]['gradient'] = tensor(x.grad)
        rows[-1]['state_after_backward'] = intern(raw_state(g))
    setup('bernoulli-functional-autograd', bernoulli_ad)
    report = {'schema': 'heptatorch.pytorch-cpu-bernoulli-dropout-probe.v1', 'status': 'OBSERVED',
              'probe_sha256': SHA(Path(__file__).read_bytes()), 'oracle_provenance': provenance,
              'build_configuration': t.__config__.show(), 'mkl_available': t.backends.mkl.is_available(),
              'cpu_capability': t.backends.cpu.get_cpu_capability(), 'threads': t.get_num_threads(),
              'consumption_reference': 'separate Generator set_state + torch.rand(float32), compared against every byte of the observed state',
              'rows': rows, 'cases': len(rows), 'states': states, 'unique_complete_states': len(states),
              'probe_errors': setup_errors, 'probe_error_count': len(setup_errors),
              'native_executed': False, 'native_qualified': False,
              'all_rng_complete': False, 'full_pytorch_rewrite_complete': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False))
    if setup_errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
