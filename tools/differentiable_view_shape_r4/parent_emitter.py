"""R4 concrete view/reshape parent loader: no prior object is reused in the new link."""
import importlib.util
import json
import sys
from pathlib import Path
from authoring_support import Emitter, functions, sha, RNG, OWNERS
from shape_overlay import PARENT_SOURCE, PARENT_IMAGE


class ParentEmitter(Emitter):
    def __init__(self, previous, repository, upstream, compactor):
        self.inputs = {}
        def read(path, pin=None):
            path = Path(path).resolve()
            raw = path.read_bytes()
            if pin is not None:
                assert sha(raw) == pin, (path, sha(raw), pin)
            self.inputs[path] = raw
            return raw
        self.parent = functions(read(previous / 'runtime.hepta', PARENT_SOURCE))
        read(previous / 'tensor-differentiable-view-session', PARENT_IMAGE)
        build = json.loads(read(previous / 'build.json'))
        assert build['status'] == 'PASS' and build['source_sha256'] == PARENT_SOURCE
        assert build['native_sha256'] == PARENT_IMAGE and build['descriptor_bytes'] == 672
        assert build['functions'] == 627 and len(build['units']) == 13
        for unit in build['units']:
            read(previous / (unit['name'] + '.hepta'), unit['source_sha256'])
            read(previous / (unit['name'] + '.o'), unit['object_sha256'])
        self.rng_raw = read(previous / 'rng03.hepta', RNG)
        self.owner_raw = read(previous / 'owners.hepta', OWNERS)
        self.rng = functions(self.rng_raw)
        assert len(self.parent) == 627 and len(self.rng) == 17
        emitter_path = repository / 'tools/owned_tensors/authoring.py'
        read(emitter_path, '2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6')
        profile = json.loads(read(repository / 'tools/owned_tensors/profile.json',
            '69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242'))
        read(compactor, '36730b55b58c29b210d921a426c62eab50d9f579ad5d7a1e4199b38799218f37')
        for name, pin in profile['upstream_authoring_sources'].items():
            read(upstream / name, pin)
        sys.path.insert(0, str(upstream))
        spec = importlib.util.spec_from_file_location('direct_view_native_emitter', emitter_path)
        self.emitter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.emitter)
        spec = importlib.util.spec_from_file_location('direct_view_bounded_compactor', compactor)
        self.compactor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.compactor)
        self.context, self.signatures, self.checks = [], {}, []
        for name, body in {**self.parent, **self.rng}.items():
            self.add_context(name, body)
