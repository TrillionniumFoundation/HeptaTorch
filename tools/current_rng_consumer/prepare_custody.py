#!/usr/bin/env python3
"""Redirect only six historical143 helper comparisons to exact current rng03."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_evidence import record_actual_process

sha=lambda raw:hashlib.sha256(raw).hexdigest()
ORIGINAL='e33a6da5b8a32792d42dabe66bbbe712f6b63d5c6461db9c07af4d6d53182385'


def main():
    record_actual_process(__file__)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();repo=a.repository.resolve();out=a.output.resolve()
    original=(repo/'tests/check_rng_custody.py').read_text();assert sha(original.encode())==ORIGINAL
    raw=original;changes=[]
    def change(before,after,reason):
        nonlocal raw
        assert raw.count(before)==1
        raw=raw.replace(before,after)
        changes.append(dict(before=before,after=after,reason=reason))
    change('import re\n',
           'import re\nimport sys\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nfrom process_evidence import record_actual_process\nrecord_actual_process(__file__)\n',
           'explicit repository argument and actual process receipt')
    change('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(sys.argv[1]).resolve()', 'read original repository without restoration')
    change("for name, row in profile['reused_functions'].items():\n    assert bodies[name] == extract((ROOT/row['source']).read_bytes())[name]\n",
           "current_rng = (ROOT/'tools/current_runtime/source/units/rng03.hepta').read_bytes()\n"
           "assert sha(current_rng) == '7c32981c916bc277e16221cce1c3dd2381ce0018642e21b20836d861cf585865'\n"
           "current_bodies = extract(current_rng)\n"
           "assert len(current_bodies) == 18\n"
           "assert set(bodies) - set(current_bodies) == {'entry'}\n"
           "assert set(current_bodies) - set(bodies) == {'cg_putw'}\n"
           "assert all(bodies[name] == current_bodies[name] for name in set(bodies) & set(current_bodies))\n"
           "mapped_helpers = []\n"
           "for name, row in profile['reused_functions'].items():\n"
           "    if row['source'] == 'src/heptatorch.hepta':\n"
           "        assert name in {'getr','getw','putw','get4','read4','put4'}\n"
           "        assert bodies[name] == current_bodies[name]\n"
           "        mapped_helpers.append(name)\n"
           "    else:\n"
           "        assert name in {'round_f16','round_bf16'}\n"
           "        assert bodies[name] == extract((ROOT/row['source']).read_bytes())[name]\n",
           'only six source143 helper lookups now use identical current bodies; both dtype comparisons retain their original sources')
    change("sources = json.loads((ROOT/'tools/rng-cpu/upstream-sources.json').read_bytes())", "assert set(mapped_helpers) == {'getr','getw','putw','get4','read4','put4'}\nsources = json.loads((ROOT/'tools/rng-cpu/upstream-sources.json').read_bytes())",'enforce exactly all six migrated helpers')
    change("'whole_pytorch_rewrite_complete': False}", "'same_current_rng_unit_sha256': sha(current_rng),\n                  'all_17_non_entry_bodies_equal': True,\n                  'historical143_source_restored': False,\n                  'whole_pytorch_rewrite_complete': False}", 'accurate migration scope')
    inverse=raw
    for c in reversed(changes):
        assert inverse.count(c['after'])==1
        inverse=inverse.replace(c['after'],c['before'])
    assert inverse==original
    protected=original[original.index("assert sources['upstream_commit']"):original.index('print(json.dumps')]
    assert raw.count(protected)==1
    assert "    assert sha(bodies[name]) == row['body_sha256']" in raw
    ast.parse(raw)
    out.write_text(raw)
    proof=dict(schema='heptatorch.rng-consumer-custody-inverse.v1',status='PASS_SOURCE_ADMISSION_ONLY',
               original_sha256=ORIGINAL,executed_sha256=sha(raw.encode()),changes=changes,
               whole_file_inverse_exact=True,all_six_licensed_sources_and_notice_checks_unchanged=True,
               both_original_dtype_body_comparisons_retained=True,all_eight_original_body_hash_assertions_retained=True,
               protected_section_sha256=sha(protected.encode()),native_tests_executed=False)
    out.with_suffix('.inverse.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:proof[k] for k in ('status','executed_sha256','whole_file_inverse_exact')}))


if __name__=='__main__':main()
