"""Qualification-only real SIGUSR1 observer for the owned native wrapper.

This installs SA_SIGINFO | SA_RESTORER without SA_RESTART. Its signal handler
only copies kernel-supplied saved registers into otherwise unused control
slots; it never changes ucontext, any owner slot, or an I/O loop instruction.
"""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRODUCTION_SOURCE = (ROOT / 'wrapper.py').read_bytes()
assert hashlib.sha256(PRODUCTION_SOURCE).hexdigest() == 'ab66f4ca7308799df362a281f255709c2d3bb9ace61d97dd0edb7da0ee73438e'
spec = importlib.util.spec_from_file_location('owned_signal_production', ROOT / 'wrapper.py')
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)
assert (ROOT / 'wrapper.py').read_bytes() == PRODUCTION_SOURCE


def owned_session_template():
    original, original_step, original_close, base = production.owned_session_template()
    assert hashlib.sha256(original).hexdigest() == 'f68e28de33e2d4120b1a8a4b4463dc9b1d63f095eb8920955d43647ab3fa7f8b'
    source = inspect.getsource(production.owned_session_template)
    first = '    code=bytearray();labels={};branches=[]\n'
    assert source.count(first) == 1
    modified = source.replace(first, first + '    absolute=[]\n', 1)
    startup = '''    # QUALIFICATION ONLY: kernel action [320,352), away from all ABI slots.
    emit('48 b8');absolute.append((len(code),'signal_handler'));quad(0)
    emit('49 89 84 24');word(320)
    emit('49 c7 84 24');word(328);word(0x04000004)
    emit('48 b8');absolute.append((len(code),'signal_restorer'));quad(0)
    emit('49 89 84 24');word(336)
    emit('bf 0a 00 00 00 49 8d b4 24');word(320)
    emit('31 d2 41 ba 08 00 00 00 b8 0d 00 00 00 0f 05 48 85 c0')
    branch('0f 88','io')

'''
    needle = "    label('frame')\n"
    assert modified.count(needle) == 1
    modified = modified.replace(needle, startup + needle, 1)
    observer = '''    # QUALIFICATION ONLY: saved gregs start at ucontext+40. Observe only.
    label('signal_handler')
    emit('49 ff 84 24');word(256)
    emit('48 8b 82 90 00 00 00 49 89 84 24');word(264)
    emit('48 8b 82 a8 00 00 00 49 89 84 24');word(272)
    emit('48 8b 82 88 00 00 00 49 89 84 24');word(280)
    emit('48 8b 82 70 00 00 00 49 89 84 24');word(288)
    emit('c3')
    label('signal_restorer');emit('b8 0f 00 00 00 0f 05 0f 0b')
    for at,target in absolute:
        struct.pack_into('<Q',code,at,0x400000+176+labels[target])
'''
    needle = "    for at,target in branches:struct.pack_into('<i',code,at,labels[target]-at-4)\n"
    assert modified.count(needle) == 1
    modified = modified.replace(needle, observer + needle, 1)
    restored = modified.replace('    absolute=[]\n', '', 1).replace(startup, '', 1).replace(observer, '', 1)
    assert restored == source
    namespace = dict(production.__dict__)
    exec(compile(modified, 'qualification-owned-signal-authoring', 'exec'), namespace)
    prefix, step, close, metadata = namespace['owned_session_template']()
    old_loops = original[base['labels']['read_exact']:base['labels']['success']]
    new_loops = prefix[metadata['labels']['read_exact']:metadata['labels']['success']]
    assert old_loops == new_loops
    assert (ROOT / 'wrapper.py').read_bytes() == PRODUCTION_SOURCE
    metadata.update({'qualification_only': True,
        'production_prefix_sha256': base['prefix_sha256'],
        'production_template_sha256': hashlib.sha256(PRODUCTION_SOURCE).hexdigest(),
        'production_authoring_unchanged_except_signal_setup_and_observer': True,
        'production_io_loop_bytes_identical': True,
        'production_io_loop_bytes': len(old_loops),
        'production_io_loop_sha256': hashlib.sha256(old_loops).hexdigest(),
        'signal': 10, 'sigaction_flags': 0x04000004, 'sa_restart': False,
        'handler_never_changes_ucontext': True,
        'observer_controls': {'count': 256, 'saved_rax': 264, 'saved_rip': 272,
            'saved_rdx': 280, 'saved_rsi': 288, 'action': [320, 352]},
        'kernel_ucontext_greg_offsets': {'rax': 144, 'rip': 168, 'rdx': 136, 'rsi': 112}})
    return prefix, step, close, metadata


if __name__ == '__main__':
    prefix, step, close, metadata = owned_session_template()
    (ROOT / 'signal-prefix.bin').write_bytes(prefix)
    (ROOT / 'signal-prefix.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
