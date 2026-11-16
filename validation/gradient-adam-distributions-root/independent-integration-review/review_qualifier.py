#!/usr/bin/env python3
"""Independently inspect source and actual receipts; do not rerun native suites."""
from pathlib import Path
import argparse
import ast
import gzip
import hashlib
import json
import os
import stat
import subprocess
import sys


SOURCE = '42295619cdd761b9bc16837aef612285c8defc6a788df06e8d4f4e3cc125baa9'
RECEIPT = 'ce41c9ac7492c802cbd551e595c87a6fe72f43e1b3ac3945def7569ed29ef6be'
CORE = '1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae'
IMAGE = '6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for data in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'receipt', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.source, a.receipt, a.output = (x.resolve() for x in (a.source, a.receipt, a.output))
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    code, raw = a.source.read_bytes(), a.receipt.read_bytes()
    assert sha(code) == SOURCE and sha(raw) == RECEIPT
    (a.output / 'reviewed-qualifier.py').write_bytes(code)
    (a.output / 'original-integration-report.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    assert gzip.decompress((a.output / 'original-integration-report.json.gz').read_bytes()) == raw
    report = json.loads(raw)
    checks = []

    def check(name, good):
        assert good, name
        checks.append(name)

    check('actual report is qualified with original open scope',
          report['status'] == 'QUALIFIED_COMPONENT_WITH_OPEN_SCOPE' and report['readiness'] == 'HOLD' and
          report['full_optimizer_complete'] is False and report['all_rng_complete'] is False and
          report['full_rewrite_complete'] is False and report['pytorch_api_denominator'] == 2590)
    check('actual producer reuse is explicit', report['producer_executed_this_invocation'] is False and report['producer']['mode'] == 'verified_existing_execution')
    check('actual qualifier source binding', report['integration_script_sha256'] == SOURCE and
          (a.receipt.parent / 'executed-integration.py').read_bytes() == code)
    check('runtime exact identity', report['source_sha256'] == CORE and report['native_sha256'] == IMAGE)
    tree = ast.parse(code)
    main_body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    text = code.decode()
    check('four required Make arguments exactly match parser loop',
          "('gradient-build', 'adam-distribution-build', 'stack-build', 'output')" in text)
    check('installed tests path resolves default repository',
          "parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])" in text)
    check('default cold output path is producer/native',
          "producer = args.producer_review.resolve() if args.producer_review else out / 'producer'" in text and
          "native = producer / 'native'" in text and "if args.producer_review is None:" in text)
    check('report is atomically saved at requested output root',
          "temporary = out / 'report.json.tmp'" in text and "temporary.replace(out / 'report.json')" in text)
    help_result = subprocess.run([sys.executable, '-B', str(a.source), '--help'], capture_output=True,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    (a.output / 'actual-help.stdout').write_bytes(help_result.stdout)
    (a.output / 'actual-help.stderr').write_bytes(help_result.stderr)
    check('actual argparse help succeeds without executing suites', help_result.returncode == 0 and
          all(x.encode() in help_result.stdout for x in ('--gradient-build', '--adam-distribution-build', '--stack-build', '--output')))
    inputs = report['checked_inputs']
    check('425 complete checked inputs', report['input_custody_after'] is True and report['input_file_count'] == len(inputs) == 425)
    total_bytes = 0
    for name, pin in inputs.items():
        path = Path(name)
        observed = {'bytes': path.stat().st_size, 'sha256': file_sha(path), 'mode': stat.S_IMODE(path.stat().st_mode)}
        check('current input bytes/mode ' + name, observed == pin)
        total_bytes += observed['bytes']
    producer = report['producer']
    producer_path = Path(producer['directory']) / 'report.json'
    producer_raw = producer_path.read_bytes()
    check('reused original17 report raw custody and decoded content',
          len(producer_raw) == producer['original_report_bytes'] and sha(producer_raw) == producer['original_report_sha256'] and
          json.loads(producer_raw) == producer['original_report'])
    check('original17 actual successful steps', len(producer['original_report']['steps']) == len(producer['children']) == 17 and
          all(step['exit_code'] == 0 for step in producer['original_report']['steps']))

    def child(item, label):
        data = Path(item['path']).read_bytes()
        check('raw child length/hash/content ' + label, len(data) == item['bytes'] and sha(data) == item['sha256'] and json.loads(data) == item['report'])
        value = item['report']
        hashes = [value[k] for k in ('source_sha256', 'native_source_sha256', 'runtime_source_sha256', 'current_source_sha256') if k in value]
        images = [value[k] for k in ('native_sha256', 'native_executable_sha256', 'executable_sha256', 'image_sha256', 'current_executable_sha256') if k in value]
        check('exact runtime child identity ' + label, hashes and images and all(x == CORE for x in hashes) and all(x == IMAGE for x in images))

    for name, item in producer['children'].items():
        child(item, 'producer/' + name)
    for step in producer['original_report']['steps']:
        for stream in ('stdout', 'stderr'):
            check('original producer stream ' + step['name'] + '/' + stream,
                  file_sha(Path(producer['directory']) / (step['name'] + '.' + stream)) == step[stream + '_sha256'])
        script = Path(step['command'][4])
        check('executed producer script is bound ' + step['name'], str(script.resolve()) in inputs and file_sha(script) == inputs[str(script.resolve())]['sha256'])
    expected_executions = ('native-review-3', 'zero-merge-mathematical', 'actual-zero-merge',
                           'actual-scalar-add-nan', 'native-cross', 'extreme-stride-observe', 'extreme-stride-compare')
    check('seven outer invocations stay separately labeled', tuple(x['name'] for x in report['execution_steps']) == expected_executions)
    for step in report['execution_steps']:
        check('outer exit and original report ' + step['name'], step['exit_code'] == 0 and file_sha(Path(step['report_path'])) == step['report_sha256'])
        for stream in ('stdout', 'stderr'):
            check('outer stream hash ' + step['name'] + '/' + stream,
                  file_sha(a.receipt.parent / step[stream + '_path']) == step[stream + '_sha256'])
        script = Path(step['command'][4])
        check('outer actual script pinned ' + step['name'], str(script.resolve()) in inputs and file_sha(script) == inputs[str(script.resolve())]['sha256'])
    for name, item in report['native_review']['children'].items():
        child(item, 'native/' + name)
    for name, item in report['peer_review'].items():
        child(item, 'peer/' + name)
    child(report['extreme_stride']['observation'], 'extreme/observation')
    child(report['extreme_stride']['comparison'], 'extreme/comparison')
    check('strict extreme8 actually consumes fresh native receipt',
          report['extreme_stride']['comparison']['report']['native_observations_sha256'] == report['extreme_stride']['observation']['sha256'])
    union = report['optimizer_reference_union']
    check('original Optimizer24 union is 23 exposed qualifications plus one partial',
          union['original_reference_cases'] == 24 and union['qualified_exposed_scope'] == 23 and union['partial_cases'] == 1 and
          union['partial_one_based_indices'] == [14] and union['original_18_plus_1_partial_plus_5_open_report_unchanged'] is True)
    ledger = report['distribution_reference_ledger']
    check('both183 contracts retain their exact open categories', ledger['original_cases'] == 183 and
          ledger['atomic'] == {'qualified': 161, 'transaction_differences': 4, 'unrepresented': 18} and
          ledger['explicit_partial_commands'] == {'qualified': 165, 'unrepresented': 18})
    check('no new liveTorch execution claimed', report['new_Torch_executed_here'] is False and report['full_32_live_Torch_executed_here'] is False and report['full_35_live_Torch_executed_here'] is False)
    check('four frozen peer runners retain exact scopes',
          report['peer_review']['actual-zero-merge']['report']['case_count'] == 2 and
          report['peer_review']['actual-scalar-add-nan']['report']['variant_count'] == 48 and
          report['peer_review']['native-cross']['report']['case_count'] == 6 and
          len(report['peer_review']['zero-merge-mathematical']['report']['cases']) == 2)
    check('reference11 check is mandatory in aggregator source',
          "set(value['checks']) == PROVENANCE and all(x is True for x in value['checks'].values())" in text)
    check('qualifier remains unchanged after review', a.source.read_bytes() == code)
    check('original raw aggregate remains unchanged', a.receipt.read_bytes() == raw)
    result = {
        'schema': 'heptatorch.gradient-owner-aggregate-independent-source-and-receipt-review.v1',
        'status': 'PASS_READ_ONLY_REVIEW', 'reviewed_source_sha256': SOURCE,
        'reviewed_actual_report_sha256': RECEIPT, 'reviewed_actual_report_bytes': len(raw),
        'actual_outer_mode_reviewed': 'verified_existing_execution', 'producer_steps_reused_and_verified': 17,
        'cold_default_branch_reviewed_from_source_not_rerun': True,
        'make_four_arguments_and_producer_native_output_match': True,
        'checked_input_files': 425, 'checked_input_bytes': total_bytes,
        'new_native_suites_executed_by_this_review': 0, 'new_Torch_execution_by_this_review': False,
        'actual_argparse_help_executed': True, 'source_or_raw_reports_modified': False,
        'no_blocker_found_in_reviewed_scope': True, 'full_rewrite_complete': False,
        'check_count': len(checks), 'checks': checks,
    }
    (a.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'check_count', 'checked_input_files', 'checked_input_bytes')}))


if __name__ == '__main__':
    main()
