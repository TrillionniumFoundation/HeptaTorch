#!/usr/bin/env python3
"""Regress both frozen owner guards without executing a numerical runtime."""

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import types


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def module(name):
    path = ROOT / 'tests' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Inputs:
    def bind(self, path, expected=None):
        raw = Path(path).read_bytes()
        require(expected is None or sha(raw) == expected, 'unchanged fixed input')
        return raw


def frozen_owner(fixtures):
    manifest, archive = fixtures.package(ROOT, 'source', Inputs())
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:xz') as reader:
        member = reader.getmember('owner736.py')
        raw = reader.extractfile(member).read()
    row = manifest['files']['owner736.py']
    require(member.isfile() and len(raw) == row['bytes']
            and sha(raw) == row['sha256'] and member.mode == row['mode'],
            'complete original owner bytes and mode')
    return raw


def bind_owner(helper, raw, kind):
    # All actual method code comes from the production adapter. Only the
    # immutable owner transport is in memory; no fixture file is rewritten.
    path = ROOT / ('owner-profile-control-' + kind) / 'owner736.py'
    bridge = types.SimpleNamespace(HERE=path.parent, OWNER=sha(raw),
        SOURCE=helper.SOURCE, IMAGE=helper.IMAGE, PRODUCER='',
        PRODUCER_MANIFEST='', FROZEN_CANDIDATE=ROOT, PRODUCER_CANDIDATE=ROOT)
    adapter_type = helper.current_adapter_type(object, bridge, ROOT, (), kind)
    adapter = adapter_type()
    adapter.repository = ROOT

    class OwnerInput:
        def bind(self, selected, expected):
            require(selected == path and sha(raw) == expected,
                    'exact owner input transport')
            return raw

    return adapter.make_owner(ROOT, helper.CURRENT_IMAGE, OwnerInput())


def rejected(call, message, error_type=AssertionError):
    try:
        call()
    except error_type:
        return
    raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    helper = module('current_runtime_f64_fixtures')
    half = module('current_runtime_half_fixtures')
    profile = json.loads((ROOT / 'tools/current_runtime/source/manifest.json').read_bytes())
    count = profile['runtime']['functions']
    require(profile['runtime']['sha256'] == helper.CURRENT_SOURCE
            and profile['expected_native_sha256'] == helper.CURRENT_IMAGE,
            'the actual current profile supplies the binding')
    results = []
    for kind, fixtures in (('f64', helper), ('half', half)):
        raw = frozen_owner(fixtures)
        require(raw.count(b'functions == 736') == 2,
                'both original profile-count guards remain in the frozen package')
        owner = bind_owner(helper, raw, kind)
        require(owner.SOURCE == helper.CURRENT_SOURCE and owner.IMAGE == helper.CURRENT_IMAGE,
                'both owner layers admit the exact current profile')
        for wrong_count in (736, count - 1, count + 1, True, float(count)):
            rejected(lambda: owner.bind_profile(helper.CURRENT_SOURCE,
                helper.CURRENT_IMAGE, wrong_count), 'incorrect function count accepted')
        rejected(lambda: owner.bind_profile('0' * 64, helper.CURRENT_IMAGE, count),
                 'incorrect source accepted')
        rejected(lambda: owner.bind_profile(helper.CURRENT_SOURCE, '1' * 64, count),
                 'changed image accepted')
        mutations = {
            'missing_base_guard': raw.replace(b'functions == 736, "sum02',
                                              b'functions == 735, "sum02', 1),
            'missing_wrapper_guard': raw.replace(b'F64_SOURCE_SHA256 and functions == 736,',
                                                 b'F64_SOURCE_SHA256 and functions == 735,', 1),
            'extra_guard': raw + b'\n# functions == 736\n',
            'already_modified_guard': raw.replace(b'functions == 736', b'functions == 740', 1),
            'weakened_type_guard': raw.replace(b'type(functions) is int and functions == 736',
                b'isinstance(functions, int) and functions == 736', 1),
            'weakened_source_guard': raw.replace(b'source_sha256 == F64_SOURCE_SHA256 and functions == 736',
                b'source_sha256 != F64_SOURCE_SHA256 and functions == 736', 1),
        }
        for name, altered in mutations.items():
            require(altered != raw, 'the negative control changes the input')
            rejected(lambda: bind_owner(helper, altered, kind),
                     'malformed owner guard accepted: ' + name, ValueError)
        results.append(dict(kind=kind, original_owner_sha256=sha(raw),
            guards=2, binding_controls=7, guard_controls=list(mutations), status='PASS'))
    report = dict(schema='heptatorch.current-owner-profile-guards.v1', status='PASS',
        source_sha256=helper.CURRENT_SOURCE, native_sha256=helper.CURRENT_IMAGE,
        runtime_executed=False, reference_expectations_changed=False, results=results)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
