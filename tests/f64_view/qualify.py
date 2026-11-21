#!/usr/bin/env python3
"""Independently replay complete public F64 view bytes; start no native or Torch process."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))


def audit(event, arguments):
    if event.startswith(('subprocess.', 'os.exec', 'os.spawn')) or event in ('os.system','os.fork','os.posix_spawn'):
        raise RuntimeError('the physical F64 view reader cannot execute a process')
    if event == 'import' and str(arguments[0]).split('.')[0] == 'torch':
        raise RuntimeError('the physical F64 view reader cannot import Torch')


sys.addaudithook(audit)
from f64_view.run import Inputs, qualify, require
from numeric_predicates.native import dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-directory', type=Path, required=True)
    p.add_argument('--expected-build-sha256', required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); inputs = Inputs(); directory = args.directory.resolve(strict=True)
    try:
        require(not sys.flags.optimize and not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),
                'all predicates enabled and no preloaded Torch')
        report = json.loads(inputs.bind(directory / 'report.json'))
        result = qualify(report, ROOT, args.build_directory.resolve(strict=True),
                         args.expected_build_sha256, directory, inputs)
        require(result['status'] == 'PASS_PHYSICAL_F64_VIEW', 'complete physical qualification')
        dump(args.output, result)
        print(json.dumps(result), flush=True)
        return 0
    except BaseException as error:
        result = dict(status='FAIL_PHYSICAL_F64_VIEW', error=str(error), traceback=traceback.format_exc(),
                      inputs=inputs.records, full_project_status='HOLD', full_pytorch_rewrite_complete=False)
        dump(args.output, result); print(json.dumps(result), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
