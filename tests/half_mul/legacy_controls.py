"""Exactly22 historical Half47 refusals replaced by native multiplication.

This table records the immutable original request recipes, never their expected
numeric outputs. The new reference executes every recipe with the fixed Torch
wheel. All other217 original refusals remain outside this successor table.
"""

ORIGINAL_PACKAGE = 'validation/current-half47-source'
ORIGINAL_MEMBER = 'frozen737/controls.py'
ORIGINAL_SOURCE_SHA256 = '5ea842f9ffcde2bab021773d61e78ea33023b64ad37bbcd7a7a52b6f112bb25b'
ORIGINAL_ENTRY_SHA256 = '7b1db2c47be5e94c23455ea49fd4a74fde9184516f1e4c400a9d46538222fc0a'
ORIGINAL_ENTRY_BYTES = 7497
ORIGINAL_REFUSALS = 239
RETAINED_REFUSALS = 217
PREDECESSOR_SOURCE_SHA256 = 'abbc84b8eca16670e72b8872d1d7ab479f8a1d44f7b2b65deea1649f7682b2c7'
COMMAND, OPERATION, PREDECESSOR_STATUS, SUCCESSOR_STATUS = 2, 3, 7, 0

# Public dtype tags and raw payloads are copied from the original setup.
WORDS = {
    0: (0x3f800000,0x80000000),
    1: (1,0),
    2: (1,0),
    4: (0x3ff0000000000000,0x8000000000000000),
    5: (0x3c00,0x8000),
    6: (0x3f80,0x8000),
}

# Original labels deliberately retain both repeated cross-Half operand routes.
SUCCESSORS = (
    ('half-binary-left-5-0-3',5,0),
    ('half-binary-right-5-0-3',0,5),
    ('half-binary-left-5-1-3',5,1),
    ('half-binary-right-5-1-3',1,5),
    ('half-binary-left-5-2-3',5,2),
    ('half-binary-right-5-2-3',2,5),
    ('half-binary-left-5-4-3',5,4),
    ('half-binary-right-5-4-3',4,5),
    ('half-binary-left-5-5-3',5,5),
    ('half-binary-left-5-6-3',5,6),
    ('half-binary-right-5-6-3',6,5),
    ('half-binary-left-6-0-3',6,0),
    ('half-binary-right-6-0-3',0,6),
    ('half-binary-left-6-1-3',6,1),
    ('half-binary-right-6-1-3',1,6),
    ('half-binary-left-6-2-3',6,2),
    ('half-binary-right-6-2-3',2,6),
    ('half-binary-left-6-4-3',6,4),
    ('half-binary-right-6-4-3',4,6),
    ('half-binary-left-6-5-3',6,5),
    ('half-binary-right-6-5-3',5,6),
    ('half-binary-left-6-6-3',6,6),
 )

assert len(SUCCESSORS)==22 and len({row[0] for row in SUCCESSORS})==22
assert ORIGINAL_REFUSALS==RETAINED_REFUSALS+len(SUCCESSORS)
