"""Derive the exact 712 cold-build admission from its retained 701 source."""
import ast
import hashlib

FROZEN_SHA = '5b7b9a538877c06bce9daff0631657ae3b9ce79bcd7336d42839a01ef9cbcfde'
SOURCE = 'fac3d413c6c0f6874d224cbea312108ee2c8baa027102d41890d85094a5faa38'
IMAGE = '5b7aac203d2c913d43781f0da45d40e9bac2037c2bc074ec30f8dd663710a805'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def changes(source, image):
    require(source == SOURCE and image == IMAGE, 'exact actually cold-built Generator712 image required')
    return [
        ('de4fdf964a74be11acfeddcf5fe16656c6cfa671fc7917585a72364918387a92', source, 1),
        ('75e29e13c0a7f14395fd21acb82686279dc962615ae6bcde3176b72083cac971', image, 1),
        ('aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16',
         '761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e', 1),
        ('b9d039c365a8a3988e3add120e8cffa46a4e1297878f4cced30103953b030bfb',
         '9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d', 1),
        ('a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3',
         '7c32981c916bc277e16221cce1c3dd2381ce0018642e21b20836d861cf585865', 1),
        ('9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884',
         '867a27b3a86634ea6a873909eb8bc16d5d8027c98c7cb9340376fba51312ec86', 1),
        ('current701', 'current712', 4),
        ('CURRENT701', 'CURRENT712', 1),
        ('== 701', '== 712', 2),
        ("'functions': 701", "'functions': 712", 1),
        ("built['units_planned'] == 14", "built['units_planned'] == 15", 1),
        ("len(units) == len(manifest['units']) == 14 and len(processes) == 15",
         "len(units) == len(manifest['units']) == 15 and len(processes) == 16", 1),
        ("len({u['name'] for u in units}) == 14", "len({u['name'] for u in units}) == 15", 1),
        ("b'HIASTKS1' + struct.pack('<3Q', 14, 4, 0)",
         "b'HIACGS01' + struct.pack('<3Q', 15, 4, 0)", 1),
        ("'units': 14, 'compiler_processes': 14", "'units': 15, 'compiler_processes': 15", 1),
    ]


class LiteralShape(ast.NodeTransformer):
    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(value='literal:' + type(node.value).__name__), node)


def check_shape(raw):
    tree = LiteralShape().visit(ast.parse(raw))
    return ast.dump(tree, include_attributes=False)


def derive(frozen, source=SOURCE, image=IMAGE):
    require(sha(frozen) == FROZEN_SHA, 'complete original 701 cold admission bytes required')
    edits = changes(source, image)
    raw = frozen
    for before, after, count in edits:
        a, b = before.encode(), after.encode()
        require(raw.count(a) == count and b not in raw, 'exact cold-contract edit count: ' + before)
        raw = raw.replace(a, b)
    inverse = raw
    for before, after, count in reversed(edits):
        require(inverse.count(after.encode()) == count, 'exact cold-contract inverse count: ' + after)
        inverse = inverse.replace(after.encode(), before.encode())
    require(inverse == frozen, 'complete cold admission inverse differs')
    require(check_shape(raw) == check_shape(frozen), 'cold admission changed more than its literal contract')
    proof = {
        'schema': 'heptatorch.current712-cold-admission.identity-inverse.v1',
        'frozen_sha256': FROZEN_SHA, 'executed_sha256': sha(raw),
        'whole_file_inverse_exact': True, 'edits': edits,
        'identity_assignments_changed': 2, 'build_predicates_changed': True,
        'changed_predicates_limited_to_exact_current_generation_contract': True,
        'original_check_shape_preserved': True,
        'normalized_check_ast_sha256': sha(check_shape(raw).encode()),
        'functions': 712, 'units': 15, 'compiler_processes': 15, 'linker_processes': 1,
        'link_request_magic': 'HIACGS01', 'four_argument_step': True,
        'cg_model_counted_as_core_partition': True,
        'original_compiler_stack_limit_EOF_and_byte_checks_preserved': True,
        'native_executed_by_derivation': False, 'Torch_executed_by_derivation': False,
    }
    return raw, proof
