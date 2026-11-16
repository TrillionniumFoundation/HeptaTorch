#!/usr/bin/env python3
"""Independent CPython MT and native owner checks for explicit partial draws.

This is qualification code. The ELF executes all distributions and Tensor writes.
The original hosted observation is tested separately without changing its values.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import traceback

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
MODEL = 'a942a26c3f0a3286ec32e361aeff201574cadc0c52a855726ba52fe65f17b8b8'


def load(name, path, digest):
    assert SHA(path.read_bytes()) == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'atomic-tools', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    build, atomic, out = a.build.resolve(), a.atomic_tools.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    image = build / 'tensor-distributions-session'
    receipt = json.loads((build / 'build.json').read_text())
    assert receipt['status'] == 'PASS' and SHA(image.read_bytes()) == receipt['executable_sha256']
    model_path = Path(__file__).with_name('reference.py')
    inputs = {p: p.read_bytes() for p in (image, build/'build.json', atomic/'protocol.py',
              atomic/'owner_client.py', model_path, Path(__file__).resolve())}
    wire = load('partial_control_wire', atomic / 'protocol.py', PROTOCOL)
    model = load('partial_control_model', model_path, MODEL)
    report = {'schema': 'heptatorch.cpu-distributions-partial-native-controls.v1',
              'status': 'RUNNING', 'executable_sha256': receipt['executable_sha256'],
              'runtime_source_sha256': receipt['runtime_source_sha256'], 'checks': [],
              'frames_before_EOF': 0, 'exact_float32_words': 0, 'complete_state_comparisons': 0,
              'rollback_bytes_compared': 0, 'host_reference_is_production': False,
              'all_rng_complete': False, 'all_tensor_dtypes_complete': False,
              'full_pytorch_rewrite_complete': False}

    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')

    def begin(name):
        row = {'name': name, 'status': 'RUNNING'}
        report['checks'].append(row)
        return row, wire.Session(image, directory=out/name)

    def close(row, session):
        session.run([(53,)])
        session.finish()
        row.update(status='PASS', frames=session.frames)
        report['frames_before_EOF'] += session.frames
        save()

    def state(session, generator, expected=None):
        value = session.run([(63, generator)])[0]['payload']
        if expected is not None:
            assert value == expected
            report['complete_state_comparisons'] += 1
        return value

    def words(session, tensor, expected):
        raw = session.run([(39, tensor, 0, len(expected))])[0]['payload']
        assert raw == struct.pack('<'+'I'*len(expected), *expected)
        report['exact_float32_words'] += len(expected)

    def operation(session, commands, code, expected):
        status, records = wire.parse(session.exchange(wire.packet(commands)))
        assert status == 0 and records and records[-1]['opcode'] == code
        assert struct.unpack('<4Q', records[-1]['payload']) == expected
        assert records[-1]['value'] == expected[0]
        return records

    # Invalid probabilities at/beside MT twist boundaries: valid prefix advances
    # exactly once, failure does not advance, and both normal caches survive.
    for position in (0, 1, 623, 624, 625, 1249):
        row, session = begin('late-probability-'+str(position))
        try:
            size = 1251
            x, p, g = [r['value'] for r in session.run(
                [wire.full((size,), 7), wire.full((size,), .5), (60, 1001+position)])]
            session.run([(12, p, position, (0x7fc01234, 0xbf800000, 0x3f800001)[position%3])])
            raw = bytearray(state(session, g))
            struct.pack_into('<Q', raw, 5024, 0x3ff123456789abcd)
            struct.pack_into('<I', raw, 5040, 1)
            struct.pack_into('<I', raw, 5048, 0x3e123456)
            raw[5052] = 1
            session.run([(64, g, wire.Payload(raw), 5056)])
            reference = model.Reference(None, state=bytes(raw))
            expected = [0x3f800000 if (reference.word()&0xffffff) < 0x800000 else 0
                        for _ in range(position)] + [wire.bits(7)]*(size-position)
            report['rollback_bytes_compared'] += session.error([(81,x,g,p,1)],7)
            records = operation(session, [(60,9988), (86,x,g,p,1), (61,g,0), (999,)],
                                86, (7,x,position,position))
            assert len(records) == 2
            assert session.run([(62,records[0]['value'])])[0]['value'] == 9988
            words(session,x,expected)
            assert wire.meta(session.run([(13,x)])[0])['version'] == 0
            state(session,g,reference.state())
            # Ordinary admission remains atomic even after a successful 86.
            good = session.run([wire.full((size,),1)])[0]['value']
            report['rollback_bytes_compared'] += session.error([(86,x,g,good,1),(999,)],4)
            state(session,g,reference.state())
            row.update(invalid_logical_index=position, consumed_raw32=position,
                       prefix_committed=True, suffix_stopped=True, cache_bytes_preserved=True)
            close(row,session)
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc()); save(); raise

    # Nonzero overlapping strides are accepted; each p read sees earlier writes.
    row, session = begin('repeated-stride-self-probability-feedback')
    base, g = [r['value'] for r in session.run([wire.tensor((4,),[.5,.25,1.1,.75]),(60,781)])]
    config = struct.pack('<10Q',2,2,3,0,0,1,1,0,0,0)
    x = session.run([(69,base,wire.Payload(config),80)])[0]['value']
    reference = model.Reference(None,state=state(session,g))
    expected = [wire.bits(v) for v in (.5,.25,1.1,.75)]
    prefix = 0
    for at in (0,1,1,2,2,3):
        p = struct.unpack('<f',struct.pack('<I',expected[at]))[0]
        if not 0 <= p <= 1: break
        expected[at] = 0x3f800000 if (reference.word()&0xffffff)/(1<<24) < p else 0
        prefix += 1
    assert prefix == 3
    operation(session,[(86,x,g,x,1),(999,)],86,(7,x,3,3))
    words(session,base,expected);state(session,g,reference.state())
    assert wire.meta(session.run([(13,x)])[0])['version'] == 0
    row.update(source_model_only=True,physical_indices=[0,1,1,2,2,3],per_element_alias_feedback=True)
    close(row,session)

    # Dropout builds and retires a real noise Tensor before either late error.
    for name, size, tracked in [('tracked-leaf-twist',625,True),('tracked-leaf-pool-growth',100000,True),
                                ('expanded-overlap',6,False)]:
        row, session = begin(name)
        x,g=[r['value'] for r in session.run([wire.full((size,),1,int(tracked)),(60,83)])]
        backing=x
        if not tracked:
            config=struct.pack('<10Q',2,2,3,0,0,0,1,0,0,0)
            x=session.run([(69,x,wire.Payload(config),80)])[0]['value']
        old_size=session.owner()['length']
        reference=model.Reference(None,state=state(session,g))
        for _ in range(size*2):reference.word()
        stats=struct.unpack('<12Q',session.run([(54,)])[0]['payload'])
        error=5 if tracked else 2
        operation(session,[(87,x,g,0x3fe0000000000000,1,0 if tracked else 1),(999,)],
                  87,(error,x,0,size*2))
        new_size=session.owner()['length']
        words(session,backing,[0x3f800000]*size)
        assert wire.meta(session.run([(13,x)])[0])['version']==0
        state(session,g,reference.state())
        after=struct.unpack('<12Q',session.run([(54,)])[0]['payload'])
        assert stats[5:11]==after[5:11]
        if size==100000:assert new_size>old_size,(old_size,new_size)
        row.update(operation_error=error,consumed_raw32=size*2,
                   temporary_noise_live_rows_reclaimed=True,owner_bytes_before=old_size,owner_bytes_after=new_size,
                   source_model_only=True,capacity_retry_does_not_double_consume=True)
        close(row,session)

    # Successful overlapping inplace multiply follows output-first ordering and
    # naturally reuses values already written to a repeated physical address.
    row,session=begin('successful-repeated-stride-dropout')
    base,g=[r['value'] for r in session.run([wire.full((4,),1),(60,157)])]
    config=struct.pack('<10Q',2,2,3,0,0,1,1,0,0,0)
    x=session.run([(69,base,wire.Payload(config),80)])[0]['value']
    reference=model.Reference(None,state=state(session,g));expected=[1.0]*4
    for at in (0,1,1,2,2,3):
        unit=(reference.double_word()&((1<<53)-1))/(1<<53)
        expected[at]*=2 if unit<.5 else 0
    operation(session,[(87,x,g,0x3fe0000000000000,1,1)],87,(0,x,6,12))
    words(session,base,[wire.bits(v) for v in expected]);state(session,g,reference.state())
    assert wire.meta(session.run([(13,x)])[0])['version']==1
    row.update(source_model_only=True,physical_indices=[0,1,1,2,2,3],per_element_alias_feedback=True)
    close(row,session)

    row,session=begin('ordinary-admission-and-prefix-boundaries')
    x,p,g=[r['value'] for r in session.run([wire.full((4,),1),wire.full((4,),.5),(60,99)])]
    bad=[((86,x,g,p,2),7),((86,x,g,p,1,1),7),((86,x,g,p,1,0,1),7),
         ((87,x,g,0x3fe0000000000000,2,1),7),((87,x,g,0x3fe0000000000000,1,2),7),
         ((87,x,g,0x3fe0000000000000,1,1,1),7),((86,x,g,0,1),1),((87,x,0,0,1,1),1)]
    for command,error in bad:
        report['rollback_bytes_compared']+=session.error([(67,g,1),command],error)
    session.run([(12,p,2,0x7fc00000)])
    # Oversized response in a skipped suffix is never executed or admitted.
    records=operation(session,[(86,x,g,p,1)]+[(63,g)]*207,86,(7,x,2,2))
    assert len(records)==1
    row.update(admission_failures=len(bad),unreachable_response_overflow_skipped=True)
    close(row,session)

    row,session=begin('tensor-and-generator-prefix-commit')
    x,p,g=[r['value'] for r in session.run([wire.full((4,),2),wire.full((4,),1),(60,401)])]
    session.run([(12,p,2,0x7fc00000)])
    reference=model.Reference(None,state=state(session,g))
    for _ in range(3):reference.word()
    commands=[(12,x,3,wire.bits(9)),(67,g,1),(86,x,g,p,1),(12,x,0,wire.bits(8)),(61,g,0)]
    records=operation(session,commands,86,(7,x,2,2));assert len(records)==3
    words(session,x,[wire.bits(v) for v in (1,1,2,9)])
    assert wire.meta(session.run([(13,x)])[0])['version']==1
    state(session,g,reference.state())
    row.update(previous_tensor_write_committed=True,previous_generator_advance_committed=True,
               later_tensor_write_and_reseed_stopped=True,partial_operation_itself_did_not_bump_version=True)
    close(row,session)

    # The failed inplace operation itself does not bump Version. An already
    # saved alias therefore continues to read its now-modified prefix. This is
    # an explicit native lifecycle test, not an additional hosted observation.
    row,session=begin('saved-alias-keeps-version-after-late-error')
    x,p,g=[r['value'] for r in session.run([wire.full((4,),2,1),wire.full((4,),1),(60,612)])]
    y=session.run([(2,3,x,x)])[0]['value']
    session.run([(12,p,2,0x7fc00000)])
    reference=model.Reference(None,state=state(session,g));reference.word();reference.word()
    operation(session,[(86,x,g,p,1),(999,)],86,(7,x,2,2))
    assert wire.meta(session.run([(13,x)])[0])['version']==0
    seed=session.run([wire.full((4,),1)])[0]['value']
    session.run([(57,y,seed,0)])
    grad=session.run([(40,x,0,4)])[0]['payload']
    assert grad==struct.pack('<4f',2,2,4,4)
    report['exact_float32_words']+=4
    state(session,g,reference.state())
    row.update(source_contract_lifecycle_only=True,saved_alias_not_spuriously_invalidated=True,
               backward_reads_committed_prefix=True)
    close(row,session)

    for path,raw in inputs.items():assert path.read_bytes()==raw,path
    report.update(status='PASS',case_count=len(report['checks']),input_custody_after=True,
                  inputs={str(path):SHA(raw) for path,raw in inputs.items()})
    save()
    print(json.dumps({key:report[key] for key in ('status','case_count','frames_before_EOF',
         'exact_float32_words','complete_state_comparisons','rollback_bytes_compared')},indent=2))


if __name__=='__main__':main()
