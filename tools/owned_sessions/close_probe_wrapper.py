"""Qualification-only post-close pause; never the production startup profile.

After checked HIA close has returned, fd3 receives HTOCLO01 + exit_status.
The process then waits for one byte on fd4 before clearing the dead argument
slot and exiting. The observer can prove munmap happened before process exit.
"""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PRODUCTION_SOURCE=(ROOT/'wrapper.py').read_bytes()
assert hashlib.sha256(PRODUCTION_SOURCE).hexdigest()=='ab66f4ca7308799df362a281f255709c2d3bb9ace61d97dd0edb7da0ee73438e'
spec=importlib.util.spec_from_file_location('owned_production_wrapper',ROOT/'wrapper.py')
production=importlib.util.module_from_spec(spec);spec.loader.exec_module(production)
assert (ROOT/'wrapper.py').read_bytes()==PRODUCTION_SOURCE


def owned_session_template():
    original,original_step,original_close,base=production.owned_session_template()
    assert hashlib.sha256(original).hexdigest()=='f68e28de33e2d4120b1a8a4b4463dc9b1d63f095eb8920955d43647ab3fa7f8b'
    source=inspect.getsource(production.owned_session_template)
    needle="    emit('48 83 c4 08 41 5c 31 c0')\n"
    assert source.count(needle)==1
    hook='''    # QUALIFICATION ONLY: close returned; no owner remains live.
    emit('48 b8');quad(int.from_bytes(b'HTOCLO01','little'))
    emit('49 89 84 24');word(208)
    emit('49 8b 44 24 30 49 89 84 24');word(216)
    label('probe_write')
    emit('b8 01 00 00 00 bf 03 00 00 00 49 8d b4 24');word(208)
    emit('ba 10 00 00 00 0f 05 48 83 f8 fc');branch('0f 84','probe_write')
    emit('48 83 f8 10');branch('0f 85','probe_bad')
    label('probe_wait')
    emit('31 c0 bf 04 00 00 00 49 8d b4 24');word(224)
    emit('ba 01 00 00 00 0f 05 48 83 f8 fc');branch('0f 84','probe_wait')
    emit('48 83 f8 01');branch('0f 85','probe_bad')
    emit('31 c0');branch('e9','probe_done')
    label('probe_bad');emit('bf 63 00 00 00 b8 3c 00 00 00 0f 05 0f 0b')
    label('probe_done')
'''
    modified=source.replace(needle,needle+hook,1)
    assert modified.replace(hook,'',1)==source
    namespace=dict(production.__dict__)
    exec(compile(modified,'qualification-post-close-authoring','exec'),namespace)
    prefix,step,close,metadata=namespace['owned_session_template']()
    assert step==original_step and close==original_close
    assert (ROOT/'wrapper.py').read_bytes()==PRODUCTION_SOURCE
    metadata.update({'qualification_only':True,
        'production_prefix_sha256':base['prefix_sha256'],
        'production_template_sha256':hashlib.sha256(PRODUCTION_SOURCE).hexdigest(),
        'production_authoring_unchanged_except_post_close_hook':True,
        'post_close_marker_fd':3,'post_close_ack_fd':4,
        'post_close_marker':'HTOCLO01 + u64 exit_status',
        'post_close_dead_argument_slot':[96,112],
        'post_close_live_owner_flag_must_be':0,
        'dead_argument_bits_are_observed_not_reused_as_an_owner':True})
    return prefix,step,close,metadata


if __name__=='__main__':
    prefix,step,close,metadata=owned_session_template()
    (ROOT/'close-probe-prefix.bin').write_bytes(prefix)
    (ROOT/'close-probe-prefix.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
