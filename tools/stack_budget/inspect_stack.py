"""Independent host qualification: source call graph and decoded machine frames."""
import hashlib
import re
import struct
from collections import deque


def inspect(source, raw):
    table=struct.unpack_from('<Q',raw,40)[0]
    count=struct.unpack_from('<H',raw,60)[0]
    assert count==9
    sections=[]
    for i in range(count):
        sections.append(struct.unpack_from('<IIQQQQIIQQ',raw,table+i*64))
    text_at,text_len=sections[1][4:6]
    sym_at,sym_len=sections[2][4:6]
    string_at,string_len=sections[3][4:6]
    meta_at,meta_len=sections[7][4:6]
    stack_at,stack_len=sections[8][4:6]
    assert raw[stack_at:stack_at+8]==b'HIASTK03'
    header=struct.unpack_from('<8Q',raw,stack_at)
    n,width=header[2],header[5]
    functions=[];body={};current=None
    for line in source.decode().splitlines():
        if line.startswith('fn '):
            name=line[3:line.index('(')];functions.append(name)
            current=None if line.endswith('external()') else name
            if current is not None:body[current]=[]
        elif current is not None:
            body[current].append(line)
            if line=='end':current=None
    assert n==len(functions)
    local_ids={name:i for i,name in enumerate(functions)}
    records=[];globals={};imports={}
    for i,name in enumerate(functions):
        at=stack_at+64+i*64
        ordinal,offset,size,declared_frame,symbol,imported,slots,reserved=struct.unpack_from('<8Q',raw,at)
        assert ordinal==i and not reserved
        called=set()
        for target in re.findall(r'\{"function"\s*:\s*"([A-Za-z_][A-Za-z_0-9]*)"\}', '\n'.join(body.get(name,[]))):
            if target!=name:called.add(local_ids[target])
        matrix=stack_at+64+n*64+i*width
        actual={j for j in range(n) if raw[matrix+j//8] & 1<<(j%8)}
        assert actual==called,(name,actual,called)
        frame=0
        if not imported:
            code=raw[text_at+offset:text_at+offset+size]
            assert code[:4]==bytes.fromhex('55 48 89 e5')
            cursor=4
            while code[cursor:cursor+3]==bytes.fromhex('48 81 ec'):
                amount=struct.unpack_from('<I',code,cursor+3)[0]
                assert 1<=amount<=4096 and code[cursor+7:cursor+11]==bytes.fromhex('48 8b 04 24')
                frame+=amount;cursor+=11
            assert frame==declared_frame==(slots+2)*16+576
        else:
            assert name not in body and declared_frame==offset==size==slots==0
        symbol_name=None
        if symbol:
            name_at=string_at+struct.unpack_from('<I',raw,sym_at+symbol*24)[0]
            symbol_name=raw[name_at:raw.index(0,name_at)].decode()
            assert symbol_name==name
            (imports if imported else globals)[symbol_name]=i
        records.append({'name':name,'ordinal':i,'offset':offset,'size':size,'decoded_frame':frame,
                        'imported':bool(imported),'calls_from_source':sorted(called),'symbol':symbol,
                        'stack_record_offset':at})
    return {'source_sha256':hashlib.sha256(source).hexdigest(),'object_sha256':hashlib.sha256(raw).hexdigest(),
            'text_sha256':hashlib.sha256(raw[text_at:text_at+text_len]).hexdigest(),'text_at':text_at,
            'metadata_at':meta_at,'metadata_bytes':meta_len,'stack_at':stack_at,'stack_bytes':stack_len,
            'function_count':n,'row_bytes':width,'records':records,'exports':globals,'imports':imports}


def longest(units):
    exports={}
    for obj,unit in enumerate(units):
        for name,ordinal in unit['exports'].items():
            assert name not in exports
            exports[name]=(obj,ordinal)
    edges={};weights={};parents={}
    for obj,unit in enumerate(units):
        for ordinal,record in enumerate(unit['records']):
            key=(obj,ordinal)
            edges[key]=({exports[record['name']]} if record['imported'] else
                        {(obj,target) for target in record['calls_from_source']})
            weights[key]=0 if record['imported'] else record['decoded_frame']+16
            for target in edges[key]:parents.setdefault(target,set()).add(key)
    pending={key:len(targets) for key,targets in edges.items()}
    ready=deque(key for key,count in pending.items() if not count)
    results={}
    while ready:
        key=ready.popleft()
        results[key]=weights[key]+max((results[target] for target in edges[key]),default=0)
        for parent in parents.get(key,()):
            pending[parent]-=1
            if pending[parent]==0:ready.append(parent)
    assert len(results)==len(edges),'source call graph cycle'
    maximum=max(results.values())
    return {'maximum_path_bytes':maximum,'reserved_helper_and_startup_bytes':4096,
            'required_stack_bytes':maximum+4096,
            'sum_all_frames_bytes':sum(r['decoded_frame']+16 for u in units for r in u['records'] if not r['imported']),
            'per_function_path_bytes':{f'{obj}:{i}':value for (obj,i),value in sorted(results.items())}}
