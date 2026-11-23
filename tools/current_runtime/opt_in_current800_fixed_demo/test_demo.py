"""Pure valid-protocol, fixture, and fixed call-plan tests; no child executable."""
import hashlib
import json
import stat
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_demo as d

class Wire(unittest.TestCase):
    def test_valid_public_request(self):
        body = d.request((32, 21, 22))
        self.assertEqual(struct.unpack_from('<8s3Q8Q',body),
                         (b'HTORCH03',1,96,0,32,21,22,0,0,0,0,0))
        self.assertEqual(len(body),96)

    def test_valid_typing_and_unpack(self):
        p=d.manifest('sgd')['input_tensors']['weight']
        b=d.factory(p)
        self.assertEqual(struct.unpack_from('<8Q',b,32),(97,0,96,0,0,0,0,0))
        self.assertEqual(struct.unpack_from('<8Q',b,96),(2,1,2,1,0,0,160,8))
        self.assertEqual(b[160:],bytes.fromhex('0000803f0000803f'))
        dat=struct.pack('<16Q',2,2,1,2,1,0,0,1,1,0,0,21,22,0,0,8)
        frame=struct.pack('<8s3Q4Q',b'HTORSP03',3,1,0,13,0,500,128)+dat
        self.assertEqual(d.parse(frame,13),(500,dat))
        self.assertEqual(d.unpack_meta(dat),(2,2,1,[2,1],[1,1],21,22,0,0,8))

    def test_private_copy_of_verified_bytes(self):
        with tempfile.TemporaryDirectory() as root1, tempfile.TemporaryDirectory() as root2:
            p=Path(root1)/'origin';a=b'verified test bytes'
            p.write_bytes(a);checked=d.pinned_bytes(p,hashlib.sha256(a).hexdigest(),32)
            p.write_bytes(b'different bytes after verification')
            exe=d.stage_executable(root2,checked)
            self.assertEqual(exe.read_bytes(),a)
            self.assertEqual(stat.S_IMODE(exe.stat().st_mode),0o500)

    def test_manifests_fixed_only(self):
        f=d.manifest('inference')
        self.assertEqual(len(f['admitted']),3)
        self.assertEqual(f['excluded']['original_frozen']['label'],'hidden-width-one-broadcast-two-outputs')
        self.assertTrue(all(tuple(tuple(s['command']) for s in c['original_frozen']['stages'])==d.INFER for c in f['admitted']))
        g=d.manifest('sgd')
        self.assertEqual(tuple(tuple(s['command']) for s in g['before_update']),d.TRAIN)
        self.assertEqual(tuple(tuple(s['command']) for s in g['after_update']),d.TRAIN_FRESH)
        self.assertEqual(g['update']['command'],[9,'weight',0x3e000000])

    def test_fixed_cases_on_independent_positive_mock(self):
        f=d.manifest('inference')
        plans=[(x['original_frozen']['inputs'],[s['expected'] for s in x['original_frozen']['stages']],None)
               for x in f['admitted']]
        g=d.manifest('sgd')
        plans.append((g['input_tensors'],[s['expected'] for s in g['before_update']+g['after_update']],g['update']['weight_after']))
        run=[]
        class PositiveMock:
            def __init__(self, exe):
                self.roots,self.stages,self.post=plans[len(run)]
                self.items={};self.frames=0;self.next=100;self.ops=[]
                self.roots_i=iter(self.roots.items());self.stages_i=iter(self.stages)
                self.weight=None;run.append(self)
            def __enter__(self):return self
            def __exit__(self,*_):assert not self.items
            def make(self,e):
                self.next+=1;self.items[self.next]=e;return self.next,b''
            def exchange(self, body, op):
                assert op==97
                name,e=next(self.roots_i)
                assert bytes.fromhex(e['raw_f32le_hex'])==body[160:]
                self.frames+=1
                pair=self.make(e)
                if name=='weight':self.weight=pair[0]
                self.ops.append(op);return pair
            def call(self, op,*a):
                self.frames+=1;self.ops.append(op)
                if op in (2,3,4,32):
                    # All routed tensor operands must be currently live returned handles.
                    start=1 if op in (2,3,4) else 0
                    assert all(x in self.items for x in a[start:])
                    return self.make(next(self.stages_i))
                h=a[0];e=self.items[h];b=bytes.fromhex(e['raw_f32le_hex']);s=e['shape'];st=e['strides']
                if op==13:
                    return h,struct.pack('<16Q',len(s),len(b)//4,int(e['requires_grad']),*s,*([0]*(4-len(s))),
                                         *st,*([0]*(4-len(st))),h+100,h+200,e['version'],0,len(b))
                if op==98:return len(b)//4,b
                if op==41:
                    self.items.pop(h);return 1,b''
                if op==7:return 1,b''
                if op==11:return 2,bytes.fromhex('000080c000000041')
                if op==9:
                    assert h==self.weight and a[1]==0x3e000000
                    self.items[h]=self.post;return 1,b''
                raise AssertionError(op)
        with patch.object(d,'Session',PositiveMock):
            inf=d.inference(None); training=d.sgd(None)
        self.assertEqual(len(run),4)
        self.assertEqual(inf[0]['prediction'],[9,-14,-10,17])
        self.assertEqual(training['loss_before'],8)
        self.assertEqual(training['loss_after'],2.25)
        self.assertEqual(training['gradient'],[-4,8])
        self.assertEqual(training['weight'],[1.5,0])
        self.assertEqual([r.ops.count(97) for r in run],[5,5,5,3])
        self.assertEqual([r.ops.count(9) for r in run],[0,0,0,1])
        self.assertEqual([r.ops.count(7) for r in run],[0,0,0,1])
        self.assertEqual([r.ops.count(91) for r in run],[0,0,0,0])

if __name__ == '__main__':
    unittest.main()
