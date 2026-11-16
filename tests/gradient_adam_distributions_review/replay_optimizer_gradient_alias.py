"""Execute the previously unmapped case20 from the frozen Optimizer reference.

The original 24-case reference and its original replay stay unchanged. This
additive recipe uses the now-existing real gradient getter, owned Optimizer
zero_grad and native no-grad scalar-add operations. No literal mutation
fixture, synthetic gradient identity or counter adjustment is used.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import traceback

REFERENCE = '7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160'
PROBE = 'f69340170dc2e55886bca73a8d37265ba2c95a8c793c7633c7e8a22234cfa386'
BASE = '8a9b56c837e9c7b3af47ed401a91e932fb3359a4c109157df738e5aa63e64f85'
LAYOUT = '67b8ec4e141e2837db1e7694dfd7684978ab79a0ca61989be63ad61b491462da'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path, pin):
    assert sha(path.read_bytes()) == pin
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'controls', 'observer', 'base-replay', 'layout-adapter', 'observations', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--expected-source-sha256', required=True)
    p.add_argument('--state-format', default='HTGAD002')
    p.add_argument('--backing-format', default='HTOGAD02')
    p.add_argument('--descriptor-bytes', type=int, default=576)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = {path.resolve(): path.read_bytes() for path in
               (Path(__file__), a.binary, a.source, a.controls, a.observer,
                a.base_replay, a.layout_adapter, a.observations)}
    assert sha(a.binary.read_bytes()) == a.expected_sha256
    assert sha(a.source.read_bytes()) == a.expected_source_sha256
    assert sha(a.observations.read_bytes()) == REFERENCE
    ref = json.loads(a.observations.read_bytes())
    assert ref['status'] == 'OBSERVED' and ref['cases'] == 24 and ref['probe_errors'] == 0
    assert ref['probe_sha256'] == PROBE
    assert len(ref['oracle_provenance']['checks']) == 11
    assert all(value is True for value in ref['oracle_provenance']['checks'].values())
    actual = ref['rows'][19]
    assert actual['case'] == 'external-gradient-alias-survives-zero-grad-True'
    assert actual['status'] == 'OBSERVED'
    base = load('optimizer_gradient_alias_original_observer', a.base_replay, BASE)
    layout = load('optimizer_gradient_alias_layout_observer', a.layout_adapter, LAYOUT)
    b = base.load('optimizer_gradient_alias_wire', a.controls, base.CONTROLS_SHA)
    transport = base.load('optimizer_gradient_alias_transport', a.observer, base.OBSERVER_SHA)
    Parent = layout.replay_class(base)

    class Replay(Parent):
        def observe_tensor(self, tensor, evidence):
            # The frozen Optimizer info helper explicitly records CPU float32
            # and is_leaf; it omits device/grad_fn strings. Every Tensor in
            # this fixed recipe is a leaf, which is also checked natively by
            # the unchanged observer rather than inferred from the native ID.
            assert evidence['dtype'] == 'torch.float32' and evidence['is_leaf'] is True
            mapped = dict(evidence, device='cpu', grad_fn_type=None)
            return self.tensor(tensor, mapped)

        def direct_backward(self, parameter, values):
            seed = self.c.new([2], values)
            self.back(parameter, seed)
            self.c.one(41, seed)

        def observe(self, evidence):
            p = self.c.new([2], [1., -2.], True)
            opt = self.opt([p])
            self.direct_backward(p, [-0., 2.])
            held = self.c.make(91, p)
            weak = self.c.make(42, held)
            self.observe_tensor(p, evidence['before']['parameter'])
            self.observe_tensor(held, evidence['before']['gradient'])
            self.c.one(73, opt, 1)
            self.observe_tensor(p, evidence['after_true']['parameter'])
            self.eq(self.c.make(91, p) == 0, evidence['after_true']['grad_is_None'], 'Optimizer cleared gradient binding')
            self.eq(self.c.make(59, p), 0, 'cleared binding has no gradient')
            self.observe_tensor(held, evidence['after_true']['held'])
            self.c.expiry(weak, evidence['after_true']['old_storage_expired'])
            self.c.one(95, held, b.bits(.5))
            self.observe_tensor(held, evidence['old_external_gradient_after_mutation'])
            self.direct_backward(p, [3., 4.])
            current = self.c.make(91, p)
            current_info = self.observe_tensor(current, evidence['after_rebuild']['new_gradient'])
            held_info = self.observe_tensor(held, evidence['after_rebuild']['held'])
            self.eq(current_info['incarnation'] == held_info['incarnation'],
                    evidence['after_rebuild']['same_grad_Tensor'], 'new gradient has a distinct Tensor identity')
            self.eq(current_info['metadata'][11] == held_info['metadata'][11],
                    evidence['after_rebuild']['same_grad_Storage'], 'new gradient has distinct Storage')
            self.c.one(73, opt, 0)
            self.observe_tensor(current, evidence['after_false']['current_gradient'])
            self.observe_tensor(held, evidence['after_false']['held'])
            self.c.one(41, held)
            self.c.expiry(weak, evidence['old_storage_expired_after_external_drop'])
            self.c.run([(44, weak), (41, current), (78, opt), (41, p), (33, 0)])
            self.c.audit()

    session = transport.OwnerSession(a.binary.resolve(), directory=out / 'transport', address_limit=64 << 20)
    check = b.Check(session)
    replay = Replay(check, b, a)
    report = {'schema': 'heptatorch.optimizer-gradient-alias-case20-native-replay.v1',
              'status': 'RUNNING', 'source_sha256': a.expected_source_sha256,
              'native_sha256': a.expected_sha256, 'reference_sha256': REFERENCE,
              'probe_sha256': PROBE, 'reference_case_index': 20, 'reference_case': actual['case'],
              'runner_sha256': sha(Path(__file__).read_bytes()),
              'input_source_hashes': {str(path): sha(data) for path, data in sources.items()},
              'oracle_provenance': ref['oracle_provenance'],
              'reference_cases': 1, 'native_executed': True, 'pinned_Torch_executed_during_replay': False,
              'counting_boundary': 'This is the previously unmapped case20 from the existing 24-case reference, not a newly generated upstream test.',
              'comparison': 'Exact words including signed zero, absolute Version, layout, opaque Tensor/Storage identity and Storage Weak expiry; real Optimizer73 and scalar-add95.',
              'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    try:
        replay.observe(actual['observation'])
        report['status'] = 'FAIL' if replay.mismatches else 'QUALIFIED_EXPOSED_SCOPE'
    except Exception as error:
        report.update(status='FAIL', exception_type=type(error).__name__, message=str(error)[:4096],
                      traceback=traceback.format_exc()[-12000:])
    finally:
        try:
            report['transport'] = session.finish()
        except Exception as error:
            report.update(status='FAIL', transport_error=repr(error))
        report.update(frames=session.frames, exact_comparisons=check.comparisons,
                      rollback_bytes=check.rollback_bytes, frame_records=check.frames,
                      mismatch_fields=len(replay.mismatches), mismatches=replay.mismatches)
        assert all(path.read_bytes() == data for path, data in sources.items())
        (out / 'executed-replay.py').write_bytes(Path(__file__).read_bytes())
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('input_source_hashes', 'oracle_provenance', 'frame_records', 'mismatches')}, indent=2))
    return int(report['status'] == 'FAIL')


if __name__ == '__main__':
    raise SystemExit(main())
