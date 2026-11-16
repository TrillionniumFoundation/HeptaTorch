#!/usr/bin/env python3
"""Add the formerly unexecuted retain admission from the original 24-case oracle.

This does not rewrite or reclassify the frozen original replay. It executes
only row 14's nonleaf-before-retain / same-nonleaf-after-retain observations
with the original SGD defaults. Additional identity checks are explicitly
native interface controls, not fabricated Torch observations.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import struct
import traceback

ACTUAL_SHA = '7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160'
PROBE_SHA = 'f69340170dc2e55886bca73a8d37265ba2c95a8c793c7633c7e8a22234cfa386'
OLD_REPLAY_SHA = 'ea748d4100b539e0144f032ca301b4886d0f9520c36676c980279e0bbc9bc034'
WIRE_SHA = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
OBSERVER_SHA = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('observations', 'probe', 'old-replay', 'protocol', 'binary', 'source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    a = parser.parse_args()
    paths = {name: getattr(a, name).resolve() for name in
             ('observations', 'probe', 'old_replay', 'protocol', 'binary', 'source')}
    before = {key: sha(path) for key, path in paths.items()}
    before['runner'] = sha(__file__)
    before['transport'] = sha(paths['protocol'].with_name('owner_client.py'))
    assert before['observations'] == ACTUAL_SHA and before['probe'] == PROBE_SHA
    assert before['old_replay'] == OLD_REPLAY_SHA and before['protocol'] == WIRE_SHA
    assert before['transport'] == OBSERVER_SHA
    assert before['binary'] == a.expected_sha256 and before['source'] == a.expected_source_sha256
    reference = json.loads(paths['observations'].read_bytes())
    assert reference['status'] == 'OBSERVED' and reference['cases'] == 24 and reference['probe_errors'] == 0
    assert len(reference['oracle_provenance']['checks']) == 11
    assert all(v is True for v in reference['oracle_provenance']['checks'].values())
    assert all(v is True for v in reference['source_files_unchanged'].values())
    row = reference['rows'][13]
    assert row['case'] == 'SGD-constructor-and-group-override-boundaries' and row['status'] == 'OBSERVED'
    observation = row['observation']
    assert observation['nonleaf']['outcome'] == 'RAISED' and observation['nonleaf']['exception_type'] == 'ValueError'
    assert observation['retained_nonleaf']['outcome'] == 'SUCCEEDED'
    assert observation['retained_nonleaf']['return_type'] == 'SGD'
    spec = importlib.util.spec_from_file_location('retained_admission_wire', paths['protocol'])
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'runner.py').write_bytes(Path(__file__).read_bytes())
    report = {'schema': 'heptatorch.original-optimizer-retained-admission-additive.v1',
              'status': 'RUNNING', 'inputs': before,
              'reference_json_pointers': ['/rows/13/observation/nonleaf', '/rows/13/observation/retained_nonleaf'],
              'original_frozen_replay_modified': False, 'original_frozen_report_reclassified': False,
              'other_original_24_observations_reexecuted': False,
              'torch_executed_by_this_runner': False, 'native_executed': False,
              'native_controls_are_separate_from_reference_outcomes': True,
              'full_rewrite_complete': False, 'calls': []}
    session = None
    try:
        session = wire.Session(paths['binary'], directory=a.output / 'transport')
        report['native_executed'] = True
        def one(*command, status=0):
            before_owner = session.owner() if status else None
            error, records = wire.parse(session.exchange(wire.packet([command])))
            assert error == status, (error, status)
            result = {'opcode': command[0], 'status': error}
            report['calls'].append(result)
            if status:
                assert not records and session.owner() == before_owner
                result['whole_owner_rollback_bytes'] = before_owner['length']
                return None
            assert len(records) == 1 and records[0]['opcode'] == command[0]
            result.update(value=records[0]['value'], payload_hex=records[0]['payload'].hex())
            return records[0]
        p = one(*wire.tensor([2], [1., -2.], 1))['value']
        two = one(*wire.tensor([], [2.]))['value']
        q = one(2, 3, p, two)['value']
        one(41, two)
        initial = one(13, q)
        assert one(94, q)['value'] == 0
        cfg = wire.Payload(struct.pack('<6Q', wire.bits(.125), 0, 0, 0, 0, 0))
        groups = wire.Payload(struct.pack('<10Q', 1, 1, 0, 0, 0, 0, 0, 0, 0, q))
        one(71, 0, cfg, groups, status=5)
        assert one(93, q)['value'] == 1
        assert one(94, q)['value'] == 1
        opt_record = one(71, 0, cfg, groups)
        opt = opt_record['value']
        assert struct.unpack('<4Q', opt_record['payload']) == (1, 1, 1, 0)
        assert one(75, opt, 0, 0)['value'] == q
        assert one(13, q)['payload'] == initial['payload']
        inventory = one(76, opt, 0, 0)
        assert inventory['value'] == 0
        assert struct.unpack('<4Q', inventory['payload'])[1:] == (0, 0, 0)
        one(78, opt)
        assert one(53)['value'] == 1
        report.update(status='ORIGINAL_RETAINED_NONLEAF_SUBOBSERVATION_QUALIFIED',
                      reference_outcomes_matched=2,
                      native_controls=['same public Tensor through retain and optimizer membership',
                          'unchanged metadata including Storage/Version identity and absolute counter',
                          'real retain flag', 'no lazy state creation at constructor',
                          'ordinary rejected constructor fully rolls back owner', 'ownership audit'])
    except BaseException as error:
        report.update(status='FAIL', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if session is not None:
            try:
                session.finish()
            except BaseException as error:
                report.update(status='FAIL', close_error=str(error), close_traceback=traceback.format_exc())
            report.update(frames=session.frames, exit_code=session.exit_code,
                          all_responses_before_stdin_eof=all(v['stdin_open'] for v in session.rows))
    after = {key: sha(path) for key, path in paths.items()}
    after['runner'] = sha(__file__)
    after['transport'] = sha(paths['protocol'].with_name('owner_client.py'))
    assert after == before
    report['all_inputs_unchanged'] = True
    emit(a.output / 'report.json', report)
    print(json.dumps({'status': report['status'], 'frames': report.get('frames', 0)}))
    return int(report['status'] == 'FAIL')


if __name__ == '__main__':
    raise SystemExit(main())
