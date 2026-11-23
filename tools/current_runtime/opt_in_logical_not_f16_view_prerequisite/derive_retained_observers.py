"""Regenerate the prior native observers, changing only the exact candidate ELF pin."""
from pathlib import Path
import hashlib,sys
previous,dest=map(Path,sys.argv[1:])
old=b'6c8e5d338f63f7cf2cd4437b1f49294e635d6f07e9927b4b1f0ccdd11d8a1095'
new=b'2196cd3811604609925ba6da965b105f35cd0336e3ac6f7f7b7fab44df745ebb'
dest.mkdir(exist_ok=False)
for name,expected in (('native_observe.py','5e0168a191fce8c498be923c6462bd806c58e88f4c5f6ab0a881f93042d0e93c'),
                       ('retained79_observe.py','a8afe0530c97c4e8447877c2faf6f8166ad2d744135bac9c54e3b03f30ed0eb4')):
 b=(previous/name).read_bytes()
 assert hashlib.sha256(b).hexdigest()==expected and b.count(old)==1
 (dest/name).write_bytes(b.replace(old,new))
fixture=(previous/'official.fixture.json').read_bytes()
assert hashlib.sha256(fixture).hexdigest()=='b7510694f0e9a2e236c1ed1e84a4c5dfcffc3028cf5337c7c9a474cfc9596076'
(dest/'official.fixture.json').write_bytes(fixture)
