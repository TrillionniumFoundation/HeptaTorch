"""Literal inputs only; signbit expectations are always observed with Torch."""

import hashlib
from numeric_predicates.cases import INPUTS, DTYPES, WIDTHS, raw_input, canonical

OPERATIONS = {'signbit': 126}
CASES = [dict(spec, operation='signbit', opcode=126) for spec in INPUTS]
CASE_SHA256 = hashlib.sha256(canonical(CASES)).hexdigest()
