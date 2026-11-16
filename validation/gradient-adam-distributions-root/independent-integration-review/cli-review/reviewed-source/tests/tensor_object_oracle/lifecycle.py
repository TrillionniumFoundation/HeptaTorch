"""Additional real-Torch object lifetime cases, separate from the frozen 28.

Reference values, gradients, counters and storage expiry are observed from the
locked Torch process. No native result is used to construct its own reference.
"""
import gc
import struct


CASES = [
    ('objects-v3-data-alias-separate-Version-and-saved-mutation', 'data_alias', 0., 0.),
    ('objects-v3-clone-dense-and-empty-transpose-layout-VJP', 'clone_layout', 0., 0.),
    ('objects-v3-public-Saved-survives-reset-and-checks-shared-Version', 'public_saved', 0., 0.),
    ('objects-v3-unsaved-nonleaf-payload-expires-before-backward', 'nonleaf', 0., 0.),
    ('objects-v3-leaf-AccumulateGrad-retains-Storage-until-root-drop', 'leaf', 0., 0.),
    ('objects-v3-saved-output-retention-policy-difference-explicit', 'retention', 4e-5, 5e-6),
    ('objects-v3-nongrad-coefficient-Saved-retains-Storage', 'coefficient', 0., 0.),
]


def build(scope):
    torch, bits = scope['torch'], scope['bits']

    class Weak:
        def __init__(self, tensor):
            self.token = tensor.untyped_storage()._weak_ref()

        def expired(self):
            gc.collect()
            return bool(torch.UntypedStorage._expired(self.token))

        def close(self):
            torch.UntypedStorage._free_weak_ref(self.token)

    def command(s, opcode, *words, status=0):
        p = s.program(); i = p.command(opcode, *words)
        records = s.exchange(p, status=status)
        return None if status else records[i]

    def weak_native(s, native):
        return command(s, 42, native)[1]

    def observe_expiry(s, weak, reference, expected, label):
        actual_reference = reference.expired()
        assert actual_reference is expected, (label, actual_reference, expected)
        if expected:
            command(s, 43, weak, status=6)
        else:
            hold = command(s, 43, weak)[1]
            command(s, 45, hold)
        s.row.setdefault('storage_expiry_observations', []).append({
            'label':label, 'pytorch_expired':actual_reference,
            'native_expired':expected, 'strictly_equal':True})

    def metadata(s, native, reference):
        p = s.program(); i = p.command(13, native)
        row = s.exchange(p)[i]
        s.metadata(row, reference)
        result = s.raw_metadata[native]
        assert result[13] == int(reference._version)
        return result

    def data_alias(s):
        x = torch.tensor([1., -2., 3.], requires_grad=True)
        alias = x.data
        product = x*x
        loss = product.sum()
        p = s.program(); nx = p.tensor(x, True); na = p.make(55, nx)
        np = p.binary(3, nx, nx); nl = p.reduce(7, np); s.exchange(p)
        before = [metadata(s, n, t) for n, t in ((nx,x),(na,alias))]
        assert before[0][11] == before[1][11] and before[0][12] != before[1][12]
        alias[1] = 4.
        command(s, 12, na, 1, bits(4.))
        after = [metadata(s, n, t) for n, t in ((nx,x),(na,alias))]
        assert [r[13] for r in after] == [0,1]
        loss.backward()
        p = s.program(); p.backward(nl)
        s.exchange(p, {p.read(nx):x, p.read(na):alias, p.read(nx, True):x.grad,
                       p.read(np):product, p.read(nl):loss})
        s.row['data_alias_is_separate_version_identity'] = True

    def clone_layout(s):
        layouts = []
        for shape in ((2,3,2),(0,3,2)):
            x = torch.arange(12, dtype=torch.float32).reshape(shape) if shape[0] else torch.empty(shape)
            x.requires_grad_(True)
            transposed = x.transpose(1,2)
            copied = transposed.clone()
            seed = torch.arange(copied.numel(), dtype=torch.float32).reshape(copied.shape)/4
            p = s.program(); nx = p.tensor(x, True); nt = p.transpose(nx, 1, 2)
            nc = p.clone(nt); ns = p.tensor(seed); s.exchange(p)
            mt, mc = metadata(s, nt, transposed), metadata(s, nc, copied)
            assert mt[11] != mc[11] and mt[12] != mc[12] and mc[13] == 0
            copied.backward(seed)
            p = s.program(); p.seeded(nc, ns)
            s.exchange(p, {p.read(nc):copied, p.read(nx, True):x.grad})
            layouts.append({'input_shape':list(shape), 'copied_shape':list(copied.shape),
                            'copied_strides':list(copied.stride()), 'Storage_and_Version_independent':True})
        s.row['clone_layout_observations'] = layouts

    def public_saved(s):
        x = torch.arange(6, dtype=torch.float32).reshape(2,3).requires_grad_(True)
        transposed = x.transpose(0,1)
        product = transposed*transposed
        actual_saved_variable = product.grad_fn
        p = s.program(); nx = p.tensor(x, True); nt = p.transpose(nx,0,1); s.exchange(p)
        original = metadata(s, nt, transposed)
        saved = command(s,46,nt)[1]
        p = s.program(); p.reset_graph(0); s.exchange(p)
        del x, transposed, product
        gc.collect()
        reference = actual_saved_variable._saved_self.detach()
        p = s.program(); nu = p.make(56,saved); ri=p.command(49,saved,0,reference.numel())
        records = s.exchange(p, {ri:reference, p.read(nu):reference})
        unpacked = metadata(s,nu,reference)
        assert unpacked[11:13] == original[11:13]
        assert unpacked[0:2] == original[0:2] and unpacked[3:11] == original[3:11]
        command(s,47,saved)
        reference[0,0] = 19.
        command(s,12,nu,0,bits(19.))
        try:
            actual_saved_variable._saved_self
        except RuntimeError as error:
            observed_error = {'type':type(error).__name__, 'message':str(error)}
        else:
            raise AssertionError('PyTorch SavedVariable accepted a changed Version')
        command(s,47,saved,status=6)
        command(s,49,saved,0,reference.numel(),status=6)
        p=s.program(); p.make(56,saved); s.exchange(p,status=6)
        p=s.program(); s.exchange(p,{p.read(nu):reference})
        command(s,48,saved)
        s.row['saved_variable_version_exception'] = observed_error
        s.row['public_saved_contract_scope'] = (
            'Logical layout and shared Storage/Version retention, checked against a real SavedVariable; '
            'public unpack is explicitly non-grad and does not claim to restore SavedVariable autograd metadata.')

    def nonleaf(s):
        x=torch.tensor([1.,-2.,3.],requires_grad=True)
        middle=x*2.; output=middle+1.; loss=output.sum()
        weak=Weak(middle)
        try:
            p=s.program(); nx=p.tensor(x,True); two=p.full((),2.); one=p.full((),1.)
            nm=p.binary(3,nx,two); no=p.binary(1,nm,one); nl=p.reduce(7,no); s.exchange(p)
            nw=weak_native(s,nm)
            command(s,41,nm); command(s,41,no)
            del middle,output
            observe_expiry(s,nw,weak,True,'after public nonleaf drop, before backward')
            loss.backward()
            p=s.program(); p.backward(nl); s.exchange(p,{p.read(nx,True):x.grad})
            observe_expiry(s,nw,weak,True,'gradient topology survives the expired payload')
            command(s,44,nw)
        finally:
            weak.close()

    def leaf(s):
        x=torch.tensor([1.,-2.,3.],requires_grad=True)
        loss=(x+1.).sum(); weak=Weak(x)
        try:
            p=s.program(); nx=p.tensor(x,True); one=p.full((),1.)
            middle=p.binary(1,nx,one); nl=p.reduce(7,middle); s.exchange(p)
            nw=weak_native(s,nx)
            command(s,41,middle); command(s,41,nx)
            del x
            observe_expiry(s,nw,weak,False,'leaf held by AccumulateGrad before backward')
            loss.backward(); command(s,7,nl)
            observe_expiry(s,nw,weak,False,'leaf held while loss root remains alive')
            command(s,10,nx,status=1)
            command(s,41,nl); del loss
            observe_expiry(s,nw,weak,True,'last root dropped')
            command(s,44,nw)
        finally:
            weak.close()

    def retention(s):
        x=torch.tensor([-.7,.2,1.1],requires_grad=True)
        output=torch.tanh(x); loss=(output*output).sum(); weak=Weak(output)
        try:
            p=s.program(); nx=p.tensor(x,True); no=p.unary(15,nx)
            square=p.binary(3,no,no); nl=p.reduce(7,square); s.exchange(p)
            nw=weak_native(s,no)
            command(s,41,no); command(s,41,square); del output
            observe_expiry(s,nw,weak,False,'Saved output retained before backward')
            loss.backward(retain_graph=True)
            p=s.program(); p.backward(nl); s.exchange(p,{p.read(nx,True):x.grad})
            observe_expiry(s,nw,weak,False,'retain_graph backward')
            loss.backward()
            p=s.program(); p.backward(nl); s.exchange(p,{p.read(nx,True):x.grad})
            assert weak.expired() is True
            hold=command(s,43,nw)[1]; command(s,45,hold)
            s.row['known_policy_difference'] = {
                'status':'OPEN', 'operation':'backward default graph retention',
                'pytorch_saved_output_expired_after_default_backward':True,
                'native_saved_output_expired_after_default_backward':False,
                'native_contract':'Backward retains graph and Saved data until public roots are dropped or graph_reset.',
                'this_is_not_default_backward_API_parity':True}
            command(s,41,nl); del loss
            observe_expiry(s,nw,weak,True,'explicit last-root disposal closes native retention')
            command(s,44,nw)
        finally:
            weak.close()

    def coefficient(s):
        x=torch.tensor([1.,-2.,3.],requires_grad=True)
        factor=torch.tensor([2.,3.,-4.]); loss=(x*factor).sum(); weak=Weak(factor)
        try:
            p=s.program(); nx=p.tensor(x,True); nf=p.tensor(factor)
            middle=p.binary(3,nx,nf); nl=p.reduce(7,middle); s.exchange(p)
            nw=weak_native(s,nf)
            command(s,41,nf); command(s,41,middle); del factor
            observe_expiry(s,nw,weak,False,'nongrad coefficient held only by Saved data')
            loss.backward(retain_graph=True)
            p=s.program(); p.backward(nl); s.exchange(p,{p.read(nx,True):x.grad})
            observe_expiry(s,nw,weak,False,'explicit retained backward preserves coefficient')
            command(s,41,nl); del loss
            observe_expiry(s,nw,weak,True,'last Saved edge removed with root')
            command(s,44,nw)
        finally:
            weak.close()

    return {
        'data_alias':data_alias, 'clone_layout':clone_layout, 'public_saved':public_saved,
        'nonleaf':nonleaf, 'leaf':leaf, 'retention':retention, 'coefficient':coefficient}
