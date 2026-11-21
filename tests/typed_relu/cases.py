"""Reuse literal negation input geometries for independently observed ReLU."""
import copy
import hashlib
import json

from half_neg import cases as half_inputs
from wide_neg import cases as wide_inputs

DTYPES = {0: 'float32', 2: 'int64', 4: 'float64', 5: 'float16', 6: 'bfloat16'}
WIDTHS = {0: 4, 2: 8, 4: 8, 5: 2, 6: 2}


def raw_input(spec):
    module = half_inputs if spec['dtype'] in (5, 6) else wide_inputs
    return module.raw_input(spec)


def overwrite_words(spec):
    return wide_inputs.overwrite_words(spec)


def recipes():
    for module in (wide_inputs, half_inputs):
        for original in module.CASES:
            recipe = copy.deepcopy(original)
            recipe.pop('legacy_refusal_label', None)
            recipe['name'] = 'relu-' + recipe['name']
            recipe['operation'] = 'relu'
            yield recipe


CASES = list(recipes())
assert len(CASES) == 776 and len({row['name'] for row in CASES}) == 776
CASE_SHA256 = hashlib.sha256((json.dumps(CASES, sort_keys=True, separators=(',', ':'))+'\n').encode()).hexdigest()
