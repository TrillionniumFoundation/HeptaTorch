#!/usr/bin/env python3
"""Observe the fixed Torch source in the selected interpreter, then run one unchanged reader."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import runpy
import stat
import sys

FUNCTIONAL_SOURCE_SHA = '1229e7eeb316276164d5d7df28211e723e29751e827829341408a638766f334b'
FUNCTIONAL_SOURCE_BYTES = 45081
NUMPY_WARNING_SUFFIX = (
    ":377: UserWarning: Failed to initialize NumPy: No module named 'numpy' "
    "(Triggered internally at /__w/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:82.)\n"
    '  cpu = _conversion_method_template(device=torch.device("cpu"))\n'
)


def require(condition, rule):
    if not condition:
        raise ValueError(rule)


def snapshot(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'actual Torch source must be a regular file')
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_size,
                             item.st_mtime_ns, item.st_ctime_ns)
    require(identity(before) == identity(after), 'Torch source changed during read')
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                mode=stat.S_IMODE(after.st_mode), device=after.st_dev, inode=after.st_ino,
                mtime_ns=after.st_mtime_ns, ctime_ns=after.st_ctime_ns), raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-observation', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    require(args.command and args.command[0] == '--' and len(args.command) >= 2,
            'one explicit unchanged reader after --')
    target = Path(args.command[1]).resolve(strict=True)
    output = args.source_observation.absolute()
    require(not output.exists(), 'fresh Torch source observation')
    require(not sys.flags.optimize and sys.version_info[:2] == (3, 12)
            and sys.platform == 'linux' and platform.machine() == 'x86_64',
            'fixed CPython 3.12 Linux x86-64 interpreter')
    old_argv, old_path = sys.argv, sys.path[:]
    sys.argv = [str(target), *args.command[2:]]
    sys.path[0] = str(target.parent)
    receipt = dict(schema='heptatorch.current-runtime.live-torch-source.v1', status='FAIL',
                   python_executable=str(Path(sys.executable).absolute()),
                   python_executable_resolved=str(Path(sys.executable).resolve()),
                   target_argv=sys.argv[:], before=None, after=None, actual_loaded=None)
    try:
        require('torch' not in sys.modules, 'observe Torch source before its one original import')
        spec = importlib.util.find_spec('torch')
        require(spec is not None and spec.origin is not None, 'selected interpreter has an actual Torch package')
        torch_init = Path(spec.origin).resolve(strict=True)
        distribution = importlib.metadata.distribution('torch')
        require(Path(distribution.locate_file('torch/__init__.py')).resolve() == torch_init,
                'import resolution and installed Torch distribution are the same package')
        source = torch_init.parent / '_subclasses/functional_tensor.py'
        before, raw = snapshot(source)
        require(before['sha256'] == FUNCTIONAL_SOURCE_SHA and before['bytes'] == FUNCTIONAL_SOURCE_BYTES,
                'fixed wheel functional_tensor.py complete source bytes')
        receipt.update(torch_init=str(torch_init), before=before)
        copy = output.with_name('torch-functional_tensor.py')
        with copy.open('xb') as stream:
            stream.write(raw)
        copy.chmod(before['mode'])
        receipt['source_copy'] = dict(path=copy.name, bytes=len(raw), sha256=before['sha256'], mode=before['mode'])
        try:
            runpy.run_path(str(target), run_name='__main__')
        finally:
            after, _ = snapshot(source)
            receipt['after'] = after
            require(after == before, 'actual fixed Torch source unchanged after the original reader')
            loaded = {name: str(Path(getattr(sys.modules[name], '__file__')).resolve(strict=True))
                      for name in ('torch', 'torch._subclasses.functional_tensor')}
            receipt['actual_loaded'] = loaded
            receipt['cpu_capability'] = sys.modules['torch'].backends.cpu.get_cpu_capability()
            receipt['aten_cpu_capability_override'] = os.environ.get('ATEN_CPU_CAPABILITY')
            require(loaded['torch'] == str(torch_init)
                    and loaded['torch._subclasses.functional_tensor'] == before['path'],
                    'original reader actually imported the observed fixed Torch source')
            receipt['status'] = 'PASS_SOURCE_OBSERVATION_ONLY'
    finally:
        sys.argv, sys.path[:] = old_argv, old_path
        with output.open('x') as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
