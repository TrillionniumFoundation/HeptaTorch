"""Linux x86-64 owned-state session prefix, emitted by a native HIA linker.

The only owner origins are the trusted startup live-empty buffer and the
checked HIA step result. Frame bytes never supply an owner descriptor.
"""
import hashlib
import json
from pathlib import Path
import struct

MIB = 1 << 20
CONTROL = 4096
INPUT = CONTROL
OUTPUT = CONTROL + MIB
MAPPING = CONTROL + 2*MIB
REQUEST_MAGIC = b'HTOWNR01'
RESPONSE_MAGIC = b'HTOWNS01'


def owned_session_template():
    """Return prefix, step rel32 hole, close rel32 hole, and complete layout.

    Native object text must start at 4096 or later. The caller patches each
    E8 rel32 with target_offset-(hole+4), and ELF p_filesz/p_memsz at96/104.
    """
    code=bytearray();labels={};branches=[]
    def emit(value):code.extend(bytes.fromhex(value))
    def word(value):code.extend(struct.pack('<I',value))
    def quad(value):code.extend(struct.pack('<Q',value))
    def label(name):
        assert name not in labels
        labels[name]=len(code)
    def branch(op,target):
        emit(op);branches.append((len(code),target));word(0)
    def zero_slot(offset):
        # Caller supplies zero in RAX; this is clearing a dead ABI slot.
        if offset<128:
            emit('49 89 44 24');code.append(offset)
        else:
            emit('49 89 84 24');word(offset)
    def live(value):
        emit('49 c7 84 24');word(192);word(value)

    # Before this mapping exists, no HIA owner has entered the live state.
    emit('48 83 3c 24 01');branch('0f 85','unmapped_usage')
    emit('31 ff be');word(MAPPING)
    emit('ba 03 00 00 00 41 ba 22 00 00 00 49 c7 c0 ff ff ff ff')
    emit('45 31 c9 b8 09 00 00 00 0f 05 48 3d 01 f0 ff ff')
    branch('0f 83','unmapped_io')
    emit('49 89 c4')                    # r12 = private control/input/output
    emit('49 89 64 24 38')             # control+56 = initial aligned rsp
    live(1)                            # mmap-zeroed slot32 is live-empty(0,0)

    # A broken output pipe must return -EPIPE and reach the single close path.
    emit('49 c7 44 24 40 01 00 00 00')  # SIG_IGN; remaining sigaction is zero
    emit('bf 0d 00 00 00 49 8d 74 24 40 31 d2 41 ba 08 00 00 00')
    emit('b8 0d 00 00 00 0f 05 48 85 c0');branch('0f 88','io')

    label('frame')
    emit('4d 89 e5 41 be 10 00 00 00')
    branch('e8','read_exact')
    emit('85 c0');branch('0f 84','header')
    emit('49 83 fe 10');branch('0f 84','success')
    branch('e9','invalid')
    label('header')
    emit('48 b8');quad(int.from_bytes(REQUEST_MAGIC,'little'))
    emit('49 39 04 24');branch('0f 85','invalid')
    emit('4d 8b 74 24 08 49 81 fe');word(MIB)
    branch('0f 87','space')
    emit('4d 8d ac 24');word(INPUT)
    branch('e8','read_exact')
    emit('85 c0');branch('0f 85','invalid')

    # Private ABI arguments occupy [96,144): readbuf, mutbuf, owned buffer.
    emit('49 8d 84 24');word(INPUT)
    emit('49 89 44 24 60 49 8b 44 24 08 49 89 44 24 68')
    emit('49 8d 84 24');word(OUTPUT)
    emit('49 89 44 24 70 49 c7 44 24 78');word(MIB)
    # Move the sole live owner into slot2; invalidate its previous home.
    emit('49 8b 44 24 20 49 8b 54 24 28')
    emit('49 89 84 24');word(128)
    emit('49 89 94 24');word(136)
    emit('31 c0');zero_slot(32);zero_slot(40);live(0)
    for offset in (160,168,176,184):zero_slot(offset)
    # Result area [160,192): slot0 buffer, slot1 normalized u32 length.
    emit('49 8d 7c 24 60 49 8d b4 24');word(160)
    emit('41 54 48 83 ec 08 e8');step_hole=len(code);word(0)
    emit('48 83 c4 08 41 5c')
    # Adopt exactly the returned owner before checking response delivery.
    emit('49 8b 84 24');word(160)
    emit('49 8b 94 24');word(168)
    emit('49 89 44 24 20 49 89 54 24 28')
    emit('41 8b 84 24');word(176)
    emit('49 89 44 24 18')             # u32 zero-extended response length
    emit('31 c0')
    for offset in (128,136,160,168,176,184):zero_slot(offset)
    live(1)
    emit('41 8b 44 24 18 83 f8 20');branch('0f 82','invalid')
    emit('3d');word(MIB);branch('0f 87','invalid')
    emit('48 b8');quad(int.from_bytes(RESPONSE_MAGIC,'little'))
    emit('49 89 44 24 10')
    emit('4d 8d 6c 24 10 41 be 10 00 00 00')
    branch('e8','write_all')
    emit('4d 8d ac 24');word(OUTPUT)
    emit('4d 8b 74 24 18')
    branch('e8','write_all')
    branch('e9','frame')

    label('read_exact')
    emit('4d 85 f6');branch('0f 84','read_done')
    emit('31 c0 31 ff 4c 89 ee 4c 89 f2 0f 05 48 83 f8 fc')
    branch('0f 84','read_exact')
    emit('48 85 c0');branch('0f 88','io');branch('0f 84','read_eof')
    emit('49 01 c5 49 29 c6');branch('e9','read_exact')
    label('read_done');emit('31 c0 c3')
    label('read_eof');emit('b8 01 00 00 00 c3')
    label('write_all')
    emit('4d 85 f6');branch('0f 84','write_done')
    emit('b8 01 00 00 00 bf 01 00 00 00 4c 89 ee 4c 89 f2 0f 05')
    emit('48 83 f8 fc');branch('0f 84','write_all')
    emit('48 85 c0');branch('0f 8e','io')
    emit('49 01 c5 49 29 c6');branch('e9','write_all')
    label('write_done');emit('c3')

    # Every mapped termination converges on one close call. Restoring rsp is
    # essential: a failing I/O helper may still have its return address on it.
    for name,status in [('success',0),('invalid',2),('space',3),('io',4)]:
        label(name);emit('b8');word(status)
        emit('49 89 44 24 30');branch('e9','cleanup')
    label('cleanup')
    emit('49 8b 64 24 38')             # restore entry alignment before close
    emit('49 83 bc 24');word(192);emit('00')
    branch('0f 84','closed')
    emit('49 8b 44 24 20 49 8b 54 24 28')
    emit('49 89 44 24 60 49 89 54 24 68')
    emit('31 c0');zero_slot(32);zero_slot(40);live(0)
    emit('49 8d 7c 24 60 49 8d b4 24');word(160)
    emit('41 54 48 83 ec 08 e8');close_hole=len(code);word(0)
    emit('48 83 c4 08 41 5c 31 c0')
    zero_slot(96);zero_slot(104)
    label('closed')
    emit('41 8b 7c 24 30');branch('e9','exit')
    label('unmapped_io');emit('bf 04 00 00 00');branch('e9','exit')
    label('unmapped_usage');emit('bf 40 00 00 00');branch('e9','exit')
    label('exit');emit('b8 3c 00 00 00 0f 05 0f 0b')
    for at,target in branches:struct.pack_into('<i',code,at,labels[target]-at-4)
    header=(b'\x7fELF\x02\x01\x01'+bytes(9)
        +struct.pack('<HHIQQQIHHHHHH',2,62,1,0x4000b0,64,0,0,64,56,2,0,0,0)
        +struct.pack('<IIQQQQQQ',1,5,0,0x400000,0x400000,0,0,4096)
        +struct.pack('<IIQQQQQQ',0x6474e551,6,0,0,0,0,0,16))
    assert len(header)==176
    prefix=header+code
    assert len(prefix)<4096 and prefix[176+step_hole-1]==prefix[176+close_hole-1]==0xe8
    metadata={'schema':'heptatorch.owned-session-prefix.v1','architecture':'linux-x86_64',
        'prefix_bytes':len(prefix),'prefix_sha256':hashlib.sha256(prefix).hexdigest(),
        'elf_entry_offset':176,'elf_filesz_offset':96,'elf_memsz_offset':104,
        'object_text_origin':4096,'step_call_hole':176+step_hole,
        'close_call_hole':176+close_hole,'labels':{n:176+v for n,v in labels.items()},
        'request_magic':REQUEST_MAGIC.decode(),'response_magic':RESPONSE_MAGIC.decode(),
        'frame_header_bytes':16,'frame_body_limit':MIB,'minimum_response_body':32,
        'mapping_bytes':MAPPING,'input_offset':INPUT,'output_offset':OUTPUT,
        'control':{'owner_slot':[32,48],'pending_exit_status':48,'saved_initial_rsp':56,
            'sigpipe_action':[64,96],'argument_area':[96,144],
            'argument_slots':{'input':[96,112],'output':[112,128],'state':[128,144]},
            'result_area':[160,192],'result_slots':{'state':[160,176],'response_length':[176,192]},
            'owner_live_flag':192},
        'step':{'name':'step','parameters':['readbuf','mutbuf','buffer'],
                'results':['buffer','u32'],'effects_mask':15},
        'close':{'name':'close','parameters':['buffer'],'results':[],'effects_mask':9},
        'startup_owner':'trusted live-empty buffer(0,0)',
        'owner_from_frame_bytes':False,'owner_cloning':False,
        'normal_step_always_returns_one_owner':True,
        'response_status_opaque_to_wrapper':True,
        'sigpipe_disposition':'SIG_IGN','cleanup_restores_initial_stack':True,
        'single_close_call_site':True,
        'unrecoverable_process_death':'external fatal signal or HIA trap is not claimed to run close'}
    return prefix,176+step_hole,176+close_hole,metadata


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    prefix,step,close,metadata=owned_session_template()
    (root/'prefix.bin').write_bytes(prefix)
    (root/'prefix.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
