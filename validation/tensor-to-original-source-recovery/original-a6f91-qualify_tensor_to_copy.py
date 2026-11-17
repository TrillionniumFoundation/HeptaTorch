#!/usr/bin/env python3
"""Fresh native Tensor.to/_to_copy qualification with seven explicit outer gates.

Source-frozen entry. Qualification requires this invocation to execute all seven fresh gates.
Live32 and full35 remain separate required gates. No old result is rebound to a new image.
"""
import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile
import traceback
import importlib.util
import zlib
import ast
import re

CHECKS = ('python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches',
          'direct_url_matches', 'direct_url_archive_sha256_matches',
          'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested')

INTEGER_GATES = (
    ('storage12', {'cases': 12, 'passed': 12, 'failed': 0, 'frames': 235}),
    ('arithmetic30', {'cases': 30, 'passed': 30, 'failed': 0, 'frames': 1122}),
    ('dropout12', {'cases': 12, 'passed': 12, 'failed': 0, 'frames': 333}),
    ('alias8', {'cases': 8, 'passed': 8, 'native_frames': 516, 'exact_checks': 2792}),
    ('RNE1237', {'cases': 5, 'integers': 1237, 'native_frames': 60, 'exact_checks': 1282}),
    ('promotion20', {'cases': 20, 'passed': 20, 'original_mixed_operations': 40,
                     'native_frames': 2280, 'exact_checks': 10580}),
    ('view-base6', {'cases': 6, 'passed': 6, 'native_frames': 450,
                   'exact_checks': 2046, 'native_role_checks': 168}),
    ('inplace40', {'cases': 40, 'passed': 40}),
)

LEGACY_GATES = (
    ('typed-slice', 'PASS', {'frames': 48}),
    ('typed-lifetimes', 'PASS', {'frames': 343}),
    ('bool-logical-not', 'PASS', {'frames': 66}),
    ('dtype-admission101', 'PASS', {'frames': 126}),
    ('owned-allocation', 'PASS', {'frames': 120}),
    ('history-mul-controls8', 'PASS', {'cases': 8, 'passed': 8, 'frames': 216}),
    ('actual-native-dropout28', 'PASS', {'cases': 28, 'frames': 1816}),
    ('original-native-dropout8', 'PASS', {'cases': 8, 'passed': 8, 'frames': 140}),
    ('actual-existing-optimizer6', 'PASS', {'cases': 6, 'passed': 6, 'frames': 2114}),
    ('actual-sgd-effective-layout6', 'PASS', {'cases': 6, 'passed': 6, 'frames': 891}),
    ('actual-retained12', 'QUALIFIED_EXPOSED_SCOPE',
     {'variants_executed': 12, 'variants_qualified': 12, 'unexpected_failures': 0}),
    ('inherited16', 'PASS', {'step_count': 16, 'all_sources_unchanged': True}),
)

INHERITED_NAMES = ('original-public-ten', 'incoming-25', 'alias-edge-12',
    'bernoulli-gradient-layout-8', 'native-preserve-storage-4', 'native-private-lifetimes-5',
    'typed-gradient-references', 'signed-zero-13', 'partial-16', 'presence-8',
    'gradient-clear-10', 'distribution-atomic-183', 'distribution-explicit-183',
    'original-optimizer-24-mapping', 'owned-adam-cases21-24', 'optimizer-gradient-case20')

INHERITED_EXPECTED = (
    ('QUALIFIED_WITH_EXPLICIT_OPEN_BOUNDARIES', {'qualified_groups': 9, 'partial_groups': 1, 'failed_groups': 0}),
    ('QUALIFIED_WITH_UNREPRESENTED_APIS', {'qualified_variants': 22, 'unrepresented_variants': 3, 'failed_variants': 0}),
    ('QUALIFIED_WITH_LITERAL_MUTATION_BOUNDARY', {'qualified_variants': 12, 'failed_variants': 0}),
    ('QUALIFIED_EXPOSED_SCOPE', {'qualified': 8, 'failed': 0, 'mismatch_fields': 0}),
    ('PASS', {'cases': 4, 'passed': 4}),
    ('PASS', {'cases': 5, 'passed': 5}),
    ('PASS', {'refusal_cases': 12}),
    ('PASS', {'cases': 13, 'passed': 13, 'failed': 0}),
    ('PASS', {'cases': 16, 'passed': 16}),
    ('QUALIFIED_WITH_OPEN_STATE_LIFECYCLE', {'parameter_values_versions_gradient_presence_and_skip_passed': 8,
        'unexpected_failures': 0, 'state_version_difference_observations': 8,
        'state_creation_difference_observations': 65, 'adam_compared_values_with_different_bits': 0}),
    ('QUALIFIED_GRAD_CLEAR_WITH_OPEN_OBJECT_SCOPE', {'qualified': 10, 'unexpected_failures': 0,
        'state_version_difference_observations': 11, 'state_creation_difference_observations': 9,
        'native_step_Tensor_absence_observations': 30, 'Adam_different_bits': 6}),
    ('PARTIAL', {'counts': {'OPEN_TRANSACTION_DIFFERENCE': 4, 'OPEN_UNREPRESENTED': 18, 'PASS': 161}}),
    ('PARTIAL', {'counts': {'OPEN_UNREPRESENTED': 18, 'PASS': 165}}),
    ('QUALIFIED_WITH_EXPLICIT_OPEN_SCOPE', {'qualified_exposed_scope': 18, 'partial_cases': 1,
        'open_not_executed_cases': 5, 'unexpected_failures': 0}),
    ('QUALIFIED_EXPOSED_SCOPE', {'qualified_exposed_scope': 4, 'failures': 0}),
    ('QUALIFIED_EXPOSED_SCOPE', {'mismatch_fields': 0}),
)

PARENT_MANIFESTS = (
    ('validation/mixed-integer-autograd-evidence/handoff-manifest.json', '9fbb13cc0637057b5de90d8554cde7df1397710879db2a9302d8bd30fb1451c8', 118),
    ('tools/mixed_integer_autograd/package-manifest.json', '8e8f775ea42fbec4898af318bc0f51be6dcb1510a1eff13b96e2e8f74cf6daca', 85),
    ('validation/saved-public-lifetime/package-manifest.json', '144b0cfc7e11787581614403cbc2b3e643d4650846582440a10c676f9e072661', 17),
    ('validation/mixed-integer-owner-review/package-manifest.json', '4c2ac36f0e96390f47ef29a2d7361de4747736037c006d6a9fc98d5b8d289040', 55),
    ('tools/integer_tensors/package-manifest.json', 'ece14e766825fe0843dfe624fdab8a3b4ec407d27f7a0b5e155e2ee15ee77b71', 54),
    ('tools/native_dropout/package-manifest.json', '3e087c9ddeae0e34f29120ee9e05632e3048aaf5d17b449330faaaed15633708', 99),
    ('validation/int64-inplace-review/package-manifest.json', '60db4e5b20cb3f2ad83383fb4ed6c47a10741a5de9032a0b7db815197cd01e33', 14),
    ('tools/gradient_adam_distributions/package-manifest.json', '3e8c80053365f7e5de1d24725c511f49f256e8437f5a6d3556f09ff07682dc69', 45),
    ('tools/retained_gradients/package-manifest.json', 'd91e5fc032d6a38c262a09b73813668b5b087905d8d4ec27643c04546072dcad', 13),
)

FINAL = {'status': 'FROZEN',
 'source_sha256': '4f8f73f463f681b5eb29423d71a35a0b42d807fd92bf976bbf76c1970fbcee33',
 'native_sha256': '53fb43b869858f020c0b936a65f00f3b14b97d9e37038a74c357e1da471fe083',
 'profile_sha256': '37957659ab5b4c0e516179803b61aaaf3559ed1e4f336bda8cf98c21aadc67f8',
 'producer_manifest_sha256': '5179e869868633ff38c0a357039b8f223f5d9672a6453af6187d4d2bd025e3a1',
 'independent_manifest_path': 'validation/to-view-owner-review/delivery-manifest.json',
 'saved_entry_manifest_path': 'validation/to-view-owner-review/delivery-manifest.json',
 'functions': 614,
 'units': 12,
 'state_format': 'HTTOCP08',
 'backing_format': 'HTOTOCP8',
 'manifests': [['tools/tensor_to_copy/package-manifest.json',
                '5179e869868633ff38c0a357039b8f223f5d9672a6453af6187d4d2bd025e3a1',
                308],
               ['validation/tensor-to-copy-reference/package-manifest.json',
                '351209e5b68730161e7701fe6d1ca7042f73f4b3a8fe3c46447b06354c7559d7',
                41],
               ['validation/to-view-owner-review/delivery-manifest.json',
                'e4d87725ce87e9e6d8784c2290359a6b29a2577a3d38ac9e97f0f23b23220971',
                76]],
 'references': {'to54': {'path': 'validation/tensor-to-copy-reference/torch-tensor-to-copy.json',
                         'bytes': 2784319,
                         'sha256': '00bdbbe17709b0cb737161faf944eae873c09b274311dd4c74a28c251e20b6be',
                         'probe_path': 'tests/pytorch_tensor_to_copy_probe.py',
                         'probe_sha256': '07f09ea7e0ec6b8aa9195f9d059ea905753e86170f48ce6c6b23fb6d58923251',
                         'recipes': 54,
                         'conversion_calls': 186,
                         'provenance_checks': 11},
                'view4': {'path': 'validation/view-dropout-reference/torch-view-dropout.json',
                          'bytes': 607022,
                          'sha256': 'fb557ae6efd61f2cc8bdb47705732e6dace764cb6615b3b572b8e02fd99b1ad3',
                          'probe_path': 'tests/pytorch_view_dropout_probe.py',
                          'probe_sha256': 'ed0be6013977793914828c8e3be3cdeca88524b14fbc6b0a2aac422ca317c5d3',
                          'recipes': 4},
                'toggle4': {'path': 'validation/view-base-toggle-reference/torch-view-base-toggle.json',
                            'bytes': 3664697,
                            'sha256': '00bec77bc68af5cdc470f97612d1bd8bfd8519359b75b51d5c1ad49f82943290',
                            'probe_path': 'tests/pytorch_view_base_toggle_probe.py',
                            'probe_sha256': '9abb7c1c3ebb61e5a7f761e6720ce37bb5b214be7bc91a6ea83e5ecfb458823e',
                            'recipes': 4},
                'factory4': {'path': 'validation/view-factory-version-reference/torch-factory-version.json',
                             'bytes': 80363,
                             'sha256': '2b6f83e5ef4aa9de8832f70d19e28a382abb75a5b412f3934b11c8830db5d746',
                             'probe_path': 'tests/pytorch_factory_version_probe.py',
                             'probe_sha256': 'd9c0094fd28685def6252ca30cb67a0686dc2e8af3f158e85d2ada1bee992bc9',
                             'recipes': 4},
                'copy-bits4': {'path': 'validation/tensor-to-copy/reference/as-strided-copy4.json',
                               'bytes': 133222,
                               'sha256': '5c377c9b0f9d7ebee572c9bdd58bdb68d2dde6e312e040853527dab3e6827acc',
                               'probe_path': 'tests/pytorch_as_strided_copy_bits_probe.py',
                               'probe_sha256': 'e61ec8727551d513e7e1f7b442faac61098ea415d2b430fafcbe31350b083225',
                               'recipes': 4,
                               'reader_path': 'tests/tensor_to_copy_review/replay_as_strided_copy4.py',
                               'reader_sha256': 'aefc50e06944dc012f92e507e05b29309b5b32b2505cb027348dc802e3a328a9'},
                'reduction-bits4': {'path': 'validation/tensor-to-copy/reference/as-strided-reduction4.json',
                                    'bytes': 133284,
                                    'sha256': '6cb9f88cbcc5d079f6b7c97e5dfa92bfa468478f927b1f57dc13a81dc565f6fc',
                                    'probe_path': 'tests/pytorch_as_strided_reduction_bits_probe.py',
                                    'probe_sha256': 'cea71a50c31d181c8e6615ed18e12cf4b138ea13974257336f7fe7b85bf3850d',
                                    'recipes': 4,
                                    'reader_path': 'tests/tensor_to_copy_review/replay_as_strided_reduction4.py',
                                    'reader_sha256': '867487d046bb0748d93b0e2bffbfdee0ee259bf3d96dce84ad273a3af363dbf1'},
                'sum-lanes2': {'path': 'validation/tensor-to-copy/reference/as-strided-lanes2.json',
                               'bytes': 91856,
                               'sha256': 'e4ca312aef1dba177830e5e90f91dc01676fa015ca659a67b5a447628bb0689c',
                               'probe_path': 'tests/pytorch_as_strided_sum_lanes_probe.py',
                               'probe_sha256': '0c1ce7e828fc457189d74340951a94f60245139370f4ab6654e54bd8804d6cb8',
                               'recipes': 2,
                               'reader_path': 'tests/tensor_to_copy_review/replay_as_strided_lanes2.py',
                               'reader_sha256': 'cfe835462001b4ced5c05d139f0e265bca32503538e8716120213df89732b9c2'},
                'sum-branches6': {'path': 'validation/tensor-to-copy/reference/as-strided-branches6.json',
                                  'bytes': 1050769,
                                  'sha256': 'cf8e8dc4c6104fc3ea42d50972bc0728b3b591f1eaaf1f1c446dc67f9a8f5730',
                                  'probe_path': 'tests/pytorch_as_strided_sum_branches_probe.py',
                                  'probe_sha256': 'c976b3150e4d22648e5def8f3bcfb2e5a8ab7fb331f934ec8ab455b0c9a9996b',
                                  'recipes': 6,
                                  'reader_path': 'tests/tensor_to_copy_review/replay_as_strided_branches6.py',
                                  'reader_sha256': '424f5ded942c5d94315d3ec2711bcda40c9bef325dfc5e0bd9887aa2b510d2e0'},
                'mixed36': {'path': 'validation/mixed-integer-autograd/reference/torch-mixed-integer-autograd.json',
                            'sha256': '98f24165401f9aadfd86147113454104ae1c8e969dcc105a16b6cb83d605bdfb',
                            'bytes': 1081870,
                            'probe_path': 'validation/mixed-integer-autograd/reference/pytorch_mixed_integer_autograd_probe.py',
                            'probe_sha256': 'cf3017df475fcb64b4eca771476b8433588e8d2f379b3cc3cb05a9addbd3d165'},
                'Saved14': {'path': 'validation/saved-public-lifetime/fixed-reference-original.json',
                            'sha256': 'c80ee752f7d7fffe3e808eb4720de0a0447e146e1b20eadc4b219dbbff96c3b5',
                            'bytes': 488060,
                            'probe_path': 'tests/saved_public_lifetime_review/pytorch_saved_public_lifetime_probe.py',
                            'probe_sha256': 'eca3e273d0afbb3265fca7c73451950d011f41fa1670a7d695911776657296ae'}},
 'producer_steps': [{'name': 'cold-native-build',
                     'kind': 'cold',
                     'expected': {'units_reproduced_exactly': 12}},
                    {'name': 'fixed-to-copy54',
                     'kind': 'reference',
                     'role': 'to54',
                     'expected': {'cases': 54,
                                  'passed': 54,
                                  'failed': 0,
                                  'frames': 5588,
                                  'checks': 38344,
                                  'exact_words': 20118,
                                  'rollback_bytes': 20271696,
                                  'rng_bytes': 2335872,
                                  'conversion_calls': 186,
                                  'reference_indices': [0,
                                                        1,
                                                        2,
                                                        3,
                                                        4,
                                                        5,
                                                        6,
                                                        7,
                                                        8,
                                                        9,
                                                        10,
                                                        11,
                                                        12,
                                                        13,
                                                        14,
                                                        15,
                                                        16,
                                                        17,
                                                        18,
                                                        19,
                                                        20,
                                                        21,
                                                        22,
                                                        23,
                                                        24,
                                                        25,
                                                        26,
                                                        27,
                                                        28,
                                                        29,
                                                        30,
                                                        31,
                                                        32,
                                                        33,
                                                        34,
                                                        35,
                                                        36,
                                                        37,
                                                        38,
                                                        39,
                                                        40,
                                                        41,
                                                        42,
                                                        43,
                                                        44,
                                                        45,
                                                        46,
                                                        47,
                                                        48,
                                                        49,
                                                        50,
                                                        51,
                                                        52,
                                                        53]},
                     'reader_path': 'tests/tensor_to_copy_review/replay_actual54.py',
                     'reader_sha256': 'f3be80e1f5139931bf9a8a7b1b89e51ac77431d11ec7dcf5265dd551ae495c8e',
                     'observer_path': 'tests/tensor_to_copy_review/native_observer09.py',
                     'observer_sha256': '42b03aaccb5fc7df828ef9aa9953b15a8ff1d911fe4d644d990e60f0348571fc',
                     'report_schema': 'heptatorch.tensor-to-copy-actual54-replay.v1'},
                    {'name': 'fixed-view-dropout4',
                     'kind': 'reference',
                     'role': 'view4',
                     'expected': {'cases': 4,
                                  'passed': 4,
                                  'failed': 0,
                                  'frames': 2061,
                                  'checks': 11802,
                                  'exact_words': 444,
                                  'rollback_bytes': 22368768,
                                  'rng_bytes': 323584},
                     'reader_path': 'tests/tensor_to_copy_review/replay_view_dropout4_v2.py',
                     'reader_sha256': 'dc52bf90a70f98b50c28dc04e28672495b7a3e4f3e450922aa146a6fee532ec0',
                     'observer_path': 'tests/tensor_to_copy_review/native_observer09.py',
                     'observer_sha256': '42b03aaccb5fc7df828ef9aa9953b15a8ff1d911fe4d644d990e60f0348571fc',
                     'report_schema': 'heptatorch.real-view-dropout-history4-native.v1'},
                    {'name': 'fixed-base-toggle4',
                     'kind': 'reference',
                     'role': 'toggle4',
                     'expected': {'cases': 4,
                                  'passed': 4,
                                  'failed': 0,
                                  'frames': 5474,
                                  'checks': 77288,
                                  'exact_words': 4590,
                                  'rollback_bytes': 48931680,
                                  'rng_bytes': 910080,
                                  'snapshot_order_matches_reference': True},
                     'reader_path': 'tests/tensor_to_copy_review/replay_view_base_toggle4.py',
                     'reader_sha256': '44a0d3d62c7b46af998cadca30720b75a073c94d1a53fd0570523ee0eed6121e',
                     'observer_path': 'tests/tensor_to_copy_review/native_observer09.py',
                     'observer_sha256': '42b03aaccb5fc7df828ef9aa9953b15a8ff1d911fe4d644d990e60f0348571fc',
                     'report_schema': 'heptatorch.real-view-base-toggle4-native.v1'},
                    {'name': 'fixed-as-strided-copy4',
                     'kind': 'reference',
                     'role': 'copy-bits4',
                     'expected': {'cases': 4,
                                  'passed': 4,
                                  'failed': 0,
                                  'frames': 480,
                                  'checks': 2568,
                                  'exact_words': 384,
                                  'rollback_bytes': 0,
                                  'rng_bytes': 465152},
                     'reader_path': 'tests/tensor_to_copy_review/replay_as_strided_copy4.py',
                     'reader_sha256': 'aefc50e06944dc012f92e507e05b29309b5b32b2505cb027348dc802e3a328a9',
                     'observer_path': 'tests/tensor_to_copy_review/native_observer09.py',
                     'observer_sha256': '42b03aaccb5fc7df828ef9aa9953b15a8ff1d911fe4d644d990e60f0348571fc',
                     'report_schema': 'heptatorch.as-strided-copy-bits-native4.v1'},
                    {'name': 'fixed-as-strided-reduction4',
                     'kind': 'reference',
                     'role': 'reduction-bits4',
                     'expected': {'cases': 4,
                                  'passed': 4,
                                  'failed': 0,
                                  'frames': 480,
                                  'checks': 2568,
                                  'exact_words': 384,
                                  'rollback_bytes': 0,
                                  'rng_bytes': 465152},
                     'reader_path': 'tests/tensor_to_copy_review/replay_as_strided_reduction4.py',
                     'reader_sha256': '867487d046bb0748d93b0e2bffbfdee0ee259bf3d96dce84ad273a3af363dbf1',
                     'observer_path': 'tests/tensor_to_copy_review/native_observer09.py',
                     'observer_sha256': '42b03aaccb5fc7df828ef9aa9953b15a8ff1d911fe4d644d990e60f0348571fc',
                     'report_schema': 'heptatorch.as-strided-reduction-bits-native4.v1'},
                    {'name': 'fixed-as-strided-lanes2',
                     'kind': 'reference',
                     'role': 'sum-lanes2',
                     'expected': {'cases': 2,
                                  'passed': 2,
                                  'failed': 0,
                                  'frames': 240,
                                  'checks': 1284,
                                  'exact_words': 552,
                                  'rollback_bytes': 0,
                                  'rng_bytes': 232576},
                     'reader_path': 'tests/tensor_to_copy_review/replay_as_strided_lanes2.py',
                     'reader_sha256': 'cfe835462001b4ced5c05d139f0e265bca32503538e8716120213df89732b9c2',
                     'observer_path': 'tests/tensor_to_copy_review/native_observer09.py',
                     'observer_sha256': '42b03aaccb5fc7df828ef9aa9953b15a8ff1d911fe4d644d990e60f0348571fc',
                     'report_schema': 'heptatorch.as-strided-sum-lanes-native2.v1'},
                    {'name': 'fixed-as-strided-branches6',
                     'kind': 'reference',
                     'role': 'sum-branches6',
                     'expected': {'cases': 6,
                                  'passed': 6,
                                  'failed': 0,
                                  'frames': 720,
                                  'checks': 3852,
                                  'exact_words': 20572,
                                  'rollback_bytes': 0,
                                  'rng_bytes': 697728},
                     'reader_path': 'tests/tensor_to_copy_review/replay_as_strided_branches6.py',
                     'reader_sha256': '424f5ded942c5d94315d3ec2711bcda40c9bef325dfc5e0bd9887aa2b510d2e0',
                     'observer_path': 'tests/tensor_to_copy_review/native_observer09.py',
                     'observer_sha256': '42b03aaccb5fc7df828ef9aa9953b15a8ff1d911fe4d644d990e60f0348571fc',
                     'report_schema': 'heptatorch.as-strided-sum-branches-native6.v1'},
                    {'name': 'fixed-mixed36',
                     'kind': 'reference',
                     'role': 'mixed36',
                     'expected': {'cases': 36,
                                  'passed': 36,
                                  'failed': 0,
                                  'frames': 4728,
                                  'checks': 21477,
                                  'exact_words': 4670,
                                  'rollback_bytes': 58018992,
                                  'rng_bytes': 940416},
                     'reader_path': 'tests/tensor_to_copy_review/parent-regression09b/replay_actual36.py',
                     'reader_sha256': '1609db9c48727cc30ffdc60d70db4d6e1c8b319803dc44d63e6183a2534f8489',
                     'observer_path': 'tests/tensor_to_copy_review/native_observer09.py',
                     'observer_sha256': '42b03aaccb5fc7df828ef9aa9953b15a8ff1d911fe4d644d990e60f0348571fc',
                     'report_schema': 'heptatorch.mixed-integer-autograd-actual36-replay.v1'},
                    {'name': 'fixed-saved14',
                     'kind': 'reference',
                     'role': 'Saved14',
                     'expected': {'cases': 14,
                                  'passed': 14,
                                  'failed': 0,
                                  'frames': 2058,
                                  'checks': 8842,
                                  'exact_words': 2166,
                                  'rollback_bytes': 27261936,
                                  'rng_bytes': 480320},
                     'reader_path': 'tests/tensor_to_copy_review/parent-regression09b/replay_saved_public14.py',
                     'reader_sha256': 'e8866d6b3af142f29fdad67fab55fa2a9c06f293ba8efef24fb9c474328007ef',
                     'observer_path': 'tests/tensor_to_copy_review/native_observer09.py',
                     'observer_sha256': '42b03aaccb5fc7df828ef9aa9953b15a8ff1d911fe4d644d990e60f0348571fc',
                     'report_schema': 'heptatorch.saved-public-lifetime14-replay.v1'},
                    {'name': 'inherited-I64-eight', 'kind': 'integer', 'expected': {}},
                    {'name': 'legacy523-twelve',
                     'kind': 'legacy',
                     'expected': {'step_count': 12,
                                  'existing_native_units_verified': 12,
                                  'all_expected_statuses_and_counts_matched': True}}],
 'outer_expected': {'source-custody': {'status': 'SOURCE_AND_UNIT_CUSTODY_PASS',
                                       'schema': 'heptatorch.to-view-independent-source-audit09b.v1',
                                       'descriptor_bytes': 672,
                                       'descriptor_words': 84,
                                       'core_function_count': 614,
                                       'parent_functions': 556,
                                       'native_executed_here': False,
                                       'Torch_executed_here': False,
                                       'machine_code_certification': False,
                                       'inputs_unchanged': True},
                    'owner4-copied17': {'status': 'PASS',
                                        'schema': 'heptatorch.to-view-owner05.finite-roles.v1',
                                        'case_count': 4,
                                        'passed': 4,
                                        'frames_before_EOF': 68,
                                        'checks': 77,
                                        'copied_negative_cases': 17,
                                        'full_owner_rollback_bytes': 2097072,
                                        'inputs_after_unchanged': True,
                                        'native_executed': True,
                                        'Torch_executed_here': False,
                                        'observer_is_read_only': True,
                                        'copied_controls_mutate_target': False},
                    'resources6': {'status': 'PASS',
                                   'schema': 'heptatorch.to-copy-independent-resources05.v1',
                                   'case_count': 6,
                                   'passed': 6,
                                   'frames_before_EOF': 288,
                                   'checks': 329,
                                   'copied_negative_cases': 7,
                                   'full_owner_rollback_bytes': 48260848,
                                   'inputs_after_unchanged': True,
                                   'native_executed': True,
                                   'Torch_executed_here': False,
                                   'reference_sha256': '00bdbbe17709b0cb737161faf944eae873c09b274311dd4c74a28c251e20b6be'},
                    'owned-entry5': {'status': 'PASS',
                                     'schema': 'heptatorch.to-copy-owned-entry.v1',
                                     'case_count': 5,
                                     'passed': 5,
                                     'native_probe_processes': 45,
                                     'interactive_frames': 23,
                                     'complete_owner_bytes_compared': 28310352,
                                     'linked_modules': 13,
                                     'native_probe_compiled_this_invocation': True,
                                     'source_custody_after': True,
                                     'Torch_executed_here': False,
                                     'foreign_pointer_import': False,
                                     'owner_descriptor_memcpy': False,
                                     'compiler_sha256': 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f',
                                     'runtime_linker_sha256': '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7',
                                     'probe_source_sha256': '54f81a65652e2c2ba9eb3f04d608e89b71c92c5ad33ff7061e9a03b89f45f9ea',
                                     'probe_image_sha256': '7b4ed172c82d1c56406f6c6f164cf8a597575eff0ae77812786c3a4d95c3c51b'},
                    'Saved-entry6': {'status': 'PASS',
                                     'case_count': 6,
                                     'passed': 6,
                                     'native_probe_processes': 14,
                                     'interactive_frames': 17,
                                     'complete_owner_bytes_compared': 12582432,
                                     'linked_modules': 13,
                                     'native_probe_recompiled_this_invocation': False,
                                     'source_custody_after': True,
                                     'Torch_executed_here': False,
                                     'foreign_pointer_import': False,
                                     'owner_descriptor_memcpy': False,
                                     'EOF_exit_code': 0,
                                     'probe_source_sha256': '54f81a65652e2c2ba9eb3f04d608e89b71c92c5ad33ff7061e9a03b89f45f9ea',
                                     'probe_image_sha256': '7b4ed172c82d1c56406f6c6f164cf8a597575eff0ae77812786c3a4d95c3c51b'},
                    'typed-handoff2': {'status': 'PASS',
                                       'schema': 'heptatorch.to-copy-independent-typed-handoff.v1',
                                       'case_count': 2,
                                       'passed': 2,
                                       'frames_before_EOF': 22,
                                       'rollback_bytes': 2097072,
                                       'linked_modules': 24,
                                       'EOF_exit_code': 0,
                                       'source_custody_after': True,
                                       'Torch_executed_here': False,
                                       'raw_pointer_import': False,
                                       'owner_descriptor_memcpy': False,
                                       'controller_sha256': '4093b8a1a4ce1b19d6943f457649355c950cfba50743ac7afe8a8396437cc7ee',
                                       'compiler_sha256': 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f',
                                       'linker_sha256': 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16',
                                       'previous_source_sha256': '5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49',
                                       'previous_native_sha256': '69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6',
                                       'qualification_image_sha256': 'dc3338b9385e805ab84b29ff08ea56e06abf602139ad0d0b5eb2b50161c3d39b'}},
 'outer_baselines': {'source-custody': {'path': 'validation/to-view-owner-review/reports/source-custody.json',
                                        'sha256': '375194e6ed6ec07a709ffb313192cb7cd391a2d2ba049ad33c81ea300121a12f'},
                     'owner4-copied17': {'path': 'validation/to-view-owner-review/reports/owner4-copied17.json',
                                         'sha256': 'fea9a345c48ddd6530648187b076a2f4455f01fd152d951b4f7f9efff7a620b4'},
                     'resources6': {'path': 'validation/to-view-owner-review/reports/resources6.json',
                                    'sha256': '01b488758de1de830f881da2c404a552a37a2a705660c3b21dd5225c1cfdd4ed'},
                     'owned-entry5': {'path': 'validation/to-view-owner-review/reports/owned-entry5.json',
                                      'sha256': 'c23b027f11ef166ca2b8dc978fbe706e2c48a713d7bc1eb070fc0d7bac8c4f7e'},
                     'Saved-entry6': {'path': 'validation/to-view-owner-review/reports/Saved-entry6.json',
                                      'sha256': '84ba989c0146d1ffb6c149802677f924da9dd14d0e9d01a2845e2463c1946eb0'},
                     'typed-handoff2': {'path': 'validation/to-view-owner-review/reports/typed-handoff2.json',
                                        'sha256': '637a362fbe259539ba02878219729d2be836bb307980028039a43ea4c5fd0ed1'}},
 'scripts': {'source-custody': {'path': 'tests/to_view_owner_review/audit_source09b.py',
                                'sha256': '39cd35c25811c27e68a34eb126952188560bca7f49322eede791cad4361d6b76'},
             'owner4-copied17': {'path': 'tests/to_view_owner_review/test_view_roles.py',
                                 'sha256': '27a7248c186cb436cb26c628b20b6aadf9384264ecd46b245475b49b0beba52e'},
             'resources6': {'path': 'tests/to_view_owner_review/test_resources.py',
                            'sha256': '59d5e9b9166e67d2c4c2353e1d6fb68a0699712e9e0f4718260417ee1e205707'},
             'owned-entry5': {'path': 'tests/to_view_owner_review/test_owned_entry_view05.py',
                              'sha256': '4c0fa8b19751a58e795b4e9cae0846ea0ab1b2d3feb1fca5194e50b0400b35e6'},
             'Saved-entry6': {'path': 'tests/to_view_owner_review/test_saved_entry_to.py',
                              'sha256': 'f7cf3c2c3e5fdad6faa048247f09ad520c68cb193a0c03fd869d87366cc7fbf1'},
             'typed-handoff2': {'path': 'tests/to_view_owner_review/test_typed_handoff_view05.py',
                                'sha256': 'c7c03989461cb8cec296f0b8f330f7f04a4f9925d580f5e9b6ea4051dde3fc0c'}},
 'general_owner': 'tests/to_view_owner_review/owner_view05.py',
 'general_owner_sha256': '6d230e2df25e04e32f76fb10fd0936ec9d220b60acda37c40b52a45633372312',
 'extra_inputs': [{'path': 'tests/pytorch_oracle.py',
                   'sha256': 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae',
                   'bytes': 61783,
                   'mode': 420},
                  {'path': 'compatibility/pytorch-oracle-lock.json',
                   'sha256': 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944',
                   'bytes': 5018,
                   'mode': 420}],
 'historical_archives': [{'archive': 'validation/to-view-owner-review/raw-evidence.tar.xz',
                          'archive_bytes': 14813344,
                          'archive_sha256': '29f0ac8b44d0f452030b4914eae1dac368592e9941e852ff06c36ae9882162a9',
                          'manifest': 'validation/to-view-owner-review/evidence-manifest.json',
                          'manifest_sha256': 'c19ad4f8a76e4b897c35ed6fecc45f8238bc7e6a67b972fe1c0d3d2a414a4136',
                          'member_field': 'members',
                          'split_delivery': {'delivery_manifest_path': 'validation/to-view-owner-review/delivery-manifest.json',
                                             'delivery_manifest_sha256': 'e4d87725ce87e9e6d8784c2290359a6b29a2577a3d38ac9e97f0f23b23220971',
                                             'original_manifest_path': 'validation/to-view-owner-review/package-manifest.json',
                                             'original_manifest_sha256': 'f17bdec965f6f706911b57525af88033e0b3bfccf1cd742b36f10780b3387da5',
                                             'verifier_path': 'validation/to-view-owner-review/transport/verify_delivery.py',
                                             'verifier_sha256': '5d2db07ba036bf14a593e026c395a7bf168b694d7f38dab9b259edb58c1a221b'}}],
 'external_manifest_pins': [{'path': 'tools/mixed_integer_autograd/package-manifest.json',
                             'sha256': '8e8f775ea42fbec4898af318bc0f51be6dcb1510a1eff13b96e2e8f74cf6daca'},
                            {'path': 'tools/integer_tensors/package-manifest.json',
                             'sha256': 'ece14e766825fe0843dfe624fdab8a3b4ec407d27f7a0b5e155e2ee15ee77b71'},
                            {'path': 'tools/native_dropout/package-manifest.json',
                             'sha256': '3e087c9ddeae0e34f29120ee9e05632e3048aaf5d17b449330faaaed15633708'},
                            {'path': 'validation/int64-inplace-review/package-manifest.json',
                             'sha256': '60db4e5b20cb3f2ad83383fb4ed6c47a10741a5de9032a0b7db815197cd01e33'}]}
SOURCE = FINAL['source_sha256']
IMAGE = FINAL['native_sha256']
PARENT_SOURCE = '5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49'
PARENT_IMAGE = '69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6'
PARENT_PROFILE = '79ec4de7f4ffe1640e92eac91f9f52be9b0b10d1085c50c2175a36cde1712c15'
PARENT_AUTHOR = 'ebcdddd6f03513fcdea92d59667123925ce9c6c9999d843d2e544cbab19903e9'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
RUNTIME_LINKER = '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
FLAT = 'c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
REFERENCE_SUPPORT = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
PROBE = '54f81a65652e2c2ba9eb3f04d608e89b71c92c5ad33ff7061e9a03b89f45f9ea'
OUTER_STEPS = ('producer', 'source-custody', 'owner4-copied17', 'resources6',
               'owned-entry5', 'Saved-entry6', 'typed-handoff2')
REFERENCE_RECIPES = {'to54': 54, 'view4': 4, 'toggle4': 4, 'copy-bits4': 4,
    'reduction-bits4': 4, 'sum-lanes2': 2, 'sum-branches6': 6,
    'mixed36': 36, 'Saved14': 14, 'factory4': 4}
ACTUAL_NATIVE_ROLES = set(REFERENCE_RECIPES) - {'factory4'}
REFERENCE_EXECUTION_FLAGS = {'to54', 'view4', 'mixed36', 'Saved14'}
NATIVE_FILE = 'tensor-to-copy-session'




def require(value, message):
    if not value:
        raise AssertionError(message)

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, ('duplicate JSON key', key))
        result[key] = value
    return result

def loads(raw):
    return json.loads(raw, object_pairs_hook=unique)

def decode(path):
    return loads(Path(path).read_bytes())

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def relative(root, name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and '..' not in path.parts
            and str(path) == name and name != '.', ('unsafe relative path', name))
    return root / name

def fields(value, expected, label):
    for key, wanted in expected.items():
        require(key in value and type(value[key]) is type(wanted) and value[key] == wanted,
                ('exact field', label, key, value.get(key), wanted))

class Custody:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None, size=None, mode=None):
        original = Path(path)
        require(not original.is_symlink(), ('input symlink', str(original)))
        path = original.resolve()
        require(path.is_file(), ('missing input', str(path)))
        value = {'bytes': path.stat().st_size,
                 'mode': stat.S_IMODE(path.stat().st_mode), 'sha256': digest(path)}
        require(expected is None or value['sha256'] == expected, ('input SHA', str(path)))
        require(size is None or value['bytes'] == size, ('input size', str(path)))
        require(mode is None or value['mode'] == mode, ('input mode', str(path)))
        require(path not in self.files or self.files[path] == value, ('input changed', str(path)))
        self.files[path] = value
        return value

    def read(self, path, expected=None):
        self.bind(path, expected)
        return decode(path)

    def finish(self):
        for path, value in list(self.files.items()):
            self.bind(path, value['sha256'], value['bytes'], value['mode'])

def manifest(repo, name, pin, count, custody):
    value = custody.read(relative(repo, name), pin)
    rows = value['files']
    rows = [dict(row, path=key) for key, row in rows.items()] if isinstance(rows, dict) else rows
    require(len(rows) == count == len({row['path'] for row in rows}), ('manifest members', name))
    for row in rows:
        custody.bind(relative(repo, row['path']), row['sha256'], row['bytes'], row.get('mode'))
    return {'path': name, 'sha256': pin, 'members': count,
            'declared_modes': sum('mode' in row for row in rows),
            'all_declared_bytes_modes_lengths_verified': True,
            'all_actual_modes_captured_before_execution': True}

def open_raw(path_or_bytes):
    return io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).open('rb')

def archive_members(path_or_bytes, wanted):
    found = {}
    with open_raw(path_or_bytes) as raw, lzma.LZMAFile(raw, 'rb') as source, \
            tarfile.open(fileobj=source, mode='r|') as tar:
        for member in tar:
            if member.name in wanted:
                require(member.name not in found and member.isfile(), 'unique readable archive member')
                with tar.extractfile(member) as stream:
                    found[member.name] = stream.read()
    require(set(found) == set(wanted), ('archive reads complete', set(wanted) - set(found)))
    return found

def verify_archive(path_or_bytes, files, recurse=True):
    """Stream-check exact members and each nested tar's complete single XZ EOF."""
    require(isinstance(files, dict) and files, 'nonempty declared archive inventory')
    decoder = lzma.LZMADecompressor()
    tar_hash, tar_bytes = hashlib.sha256(), 0
    with open_raw(path_or_bytes) as source:
        while not decoder.eof:
            block = source.read(1 << 20) if decoder.needs_input else b''
            require(block or not decoder.needs_input, 'complete XZ EOF')
            raw = decoder.decompress(block, max_length=1 << 20)
            tar_hash.update(raw)
            tar_bytes += len(raw)
        require(not decoder.unused_data and not source.read(1), 'trailing or concatenated XZ data')
    seen = set()
    with open_raw(path_or_bytes) as raw, lzma.LZMAFile(raw, 'rb') as source, \
            tarfile.open(fileobj=source, mode='r|') as tar:
        for member in tar:
            relative(Path('.'), member.name)
            require(member.isfile() and not member.issym() and not member.islnk(), 'regular evidence member')
            require(member.name in files and member.name not in seen, ('unique declared member', member.name))
            pin = files[member.name]
            require(member.size == pin['bytes'] and member.mode == pin['mode'], ('member metadata', member.name))
            with tar.extractfile(member) as stream:
                require(hashlib.file_digest(stream, 'sha256').hexdigest() == pin['sha256'], ('member SHA', member.name))
            seen.add(member.name)
    require(seen == set(files), 'complete archive member set')
    nested = []
    if recurse:
        for name in sorted(n for n in files if n.endswith('.tar.xz')):
            manifest_name = name[:-7] + '.manifest.json'
            require(manifest_name in files, ('nested archive has exact manifest', name))
            pair = archive_members(path_or_bytes, {name, manifest_name})
            receipt = loads(pair[manifest_name])
            fields(receipt, {'archive': PurePosixPath(name).name,
                            'archive_sha256': files[name]['sha256'],
                            'archive_bytes': files[name]['bytes']}, 'nested archive binding')
            require(receipt['member_count'] == len(receipt['members']) and
                    receipt['original_bytes'] == sum(x['bytes'] for x in receipt['members'].values()),
                    'nested declared count and length')
            checked = verify_archive(pair[name], receipt['members'])
            nested.append(dict(checked, path=name, manifest_sha256=files[manifest_name]['sha256'],
                               archive_sha256=files[name]['sha256']))
    return {'members': len(seen), 'decompressed_tar_bytes': tar_bytes,
            'decompressed_tar_sha256': tar_hash.hexdigest(), 'single_XZ_stream_EOF': True,
            'all_member_bytes_modes_sizes_verified': True, 'nested_archives': nested}

def bind_report_inputs(value, custody, out):
    for key in ('inputs_sha256', 'own_inputs_sha256', 'inputs', 'checked_inputs'):
        entries = value.get(key, {})
        if not isinstance(entries, dict):
            continue
        for name, pin in entries.items():
            path = Path(name)
            if not path.is_absolute():
                continue  # Named roles are checked by their original frozen caller.
            resolved = path.resolve()
            require(resolved in custody.files or out in resolved.parents,
                    ('external input was not captured before execution', key, name))
            if isinstance(pin, str):
                custody.bind(path, pin)
            else:
                require(isinstance(pin, dict) and 'sha256' in pin, ('input digest schema', key, name))
                custody.bind(path, pin['sha256'], pin.get('bytes'), pin.get('mode'))

def child_report(root, row, custody, out, expected_status, counts, path_key='report'):
    require(row['exit_code'] == 0, ('original child process exit', row['name']))
    status = row.get('status', row.get('report_status'))
    require(status == expected_status, ('original child bounded status', row['name'], status))
    for suffix in ('stdout', 'stderr'):
        path = root / (row['name'] + '.' + suffix)
        custody.bind(path, row.get(suffix + '_sha256'))
    child = custody.read(relative(root, row[path_key]), row['report_sha256'])
    fields(child, dict(counts, status=expected_status), row['name'])
    bind_report_inputs(child, custody, out)
    return child

def complete_cases(value, frozen, count, label):
    require(type(value['cases']) is list and len(value['cases']) == count and
            [x['name'] for x in value['cases']] == [x['name'] for x in frozen['cases']] and
            len({x['name'] for x in value['cases']}) == count and
            all(x['status'] == 'PASS' for x in value['cases']), ('complete original case plan', label))

def archive_evidence(out):
    excluded = {'report.json', 'evidence-manifest.json', 'raw-evidence.tar.xz'}
    files = {}
    for path in sorted(out.rglob('*')):
        require(not path.is_symlink(), ('evidence symlink', str(path)))
        if path.is_file() and path.relative_to(out).as_posix() not in excluded:
            files[path.relative_to(out).as_posix()] = {'bytes': path.stat().st_size,
                'mode': stat.S_IMODE(path.stat().st_mode), 'sha256': digest(path)}
    archive = out / 'raw-evidence.tar.xz'
    with lzma.open(archive, 'wb', preset=3) as stream, tarfile.open(fileobj=stream, mode='w|') as tar:
        for name, pin in files.items():
            original = relative(out, name)
            require(digest(original) == pin['sha256'] and original.stat().st_size == pin['bytes'] and
                    stat.S_IMODE(original.stat().st_mode) == pin['mode'], 'archive input unchanged')
            member = tarfile.TarInfo(name)
            member.size, member.mode, member.mtime = pin['bytes'], pin['mode'], 0
            with original.open('rb') as source:
                tar.addfile(member, source)
    verified = verify_archive(archive, files, recurse=False)
    value = dict(verified, schema='heptatorch.tensor-to-copy-fresh-raw-evidence.v1', status='VERIFIED',
                 source_sha256=SOURCE, native_sha256=IMAGE, archive_sha256=digest(archive),
                 archive_bytes=archive.stat().st_size, uncompressed_bytes=sum(v['bytes'] for v in files.values()),
                 files=files, self_reference_exclusions=sorted(excluded),
                 scope='Every current output file except the three root self-references; nested producer archives independently verified.')
    dump(out / 'evidence-manifest.json', value)
    return {key: item for key, item in value.items() if key != 'files'} | {
        'archive': archive.name, 'manifest': 'evidence-manifest.json',
        'manifest_sha256': digest(out / 'evidence-manifest.json')}

def contract_ready():
    require(FINAL['status'] == 'FROZEN',
            'Draft refuses native execution: final producer/source/manifest/step pins are pending.')
    for name in ('source_sha256', 'native_sha256', 'profile_sha256',
                 'producer_manifest_sha256', 'general_owner_sha256'):
        value = FINAL[name]
        require(isinstance(value, str) and len(value) == 64 and
                all(c in '0123456789abcdef' for c in value), ('complete final pin', name))
    require(type(FINAL['units']) is int and FINAL['units'] > 0 and
            type(FINAL['functions']) is int and FINAL['functions'] > 0, 'final actual unit/function counts')
    require(FINAL['state_format'] and FINAL['backing_format'], 'final owner tags')
    require(FINAL['manifests'] and len({x[0] for x in FINAL['manifests']}) == len(FINAL['manifests']),
            'all final source manifests, no duplicates')
    manifest_names = {x[0] for x in FINAL['manifests']}
    require(FINAL['independent_manifest_path'] and FINAL['saved_entry_manifest_path'] and
            {'tools/tensor_to_copy/package-manifest.json',
             'validation/tensor-to-copy-reference/package-manifest.json',
             FINAL['independent_manifest_path'], FINAL['saved_entry_manifest_path']} <= manifest_names,
            'producer, original54, final independent and Saved-entry source manifests')
    require(('validation/tensor-to-copy-reference/package-manifest.json',
             '351209e5b68730161e7701fe6d1ca7042f73f4b3a8fe3c46447b06354c7559d7', 41)
            in tuple(tuple(x) for x in FINAL['manifests']), 'original54 complete exact manifest')
    require(set(FINAL['references']) == set(REFERENCE_RECIPES), 'all exact actual references including diagnostic factory')
    steps = FINAL['producer_steps']
    require(steps and len({x['name'] for x in steps}) == len(steps), 'nonempty exact producer step sequence')
    require(all(isinstance(x.get('expected'), dict) and
                Path(x['name']).name == x['name'] and x['name'] not in ('.', '..')
                for x in steps), 'explicit safe named producer field contracts')
    require(steps[0]['kind'] == 'cold' and sum(x['kind'] == 'cold' for x in steps) == 1, 'one genuine first cold build')
    require(sum(x['kind'] == 'integer' for x in steps) == 1 and
            sum(x['kind'] == 'legacy' for x in steps) == 1, 'both inherited aggregate gates')
    require([x['kind'] for x in steps].count('reference') == len(ACTUAL_NATIVE_ROLES) and
            {x['role'] for x in steps if x['kind'] == 'reference'} == ACTUAL_NATIVE_ROLES,
            'each required actual native reference exactly once')
    require(all(x['kind'] in ('cold', 'reference', 'integer', 'legacy') for x in steps), 'no unknown producer gate')
    for step in steps:
        if step['kind'] == 'reference':
            require(all(isinstance(step.get(key), str) and step[key] for key in
                        ('reader_path', 'reader_sha256', 'observer_path', 'observer_sha256', 'report_schema')),
                    ('explicit original reader contract', step['name']))
    require(set(FINAL['outer_expected']) == set(OUTER_STEPS[1:]) and
            set(FINAL['outer_baselines']) == set(OUTER_STEPS[1:]) and
            set(FINAL['scripts']) == set(OUTER_STEPS[1:]), 'six independent final gate contracts')
    require(all(isinstance(v, dict) and v for v in FINAL['outer_expected'].values()), 'no empty expected-field contracts')


def runtime_identity(value):
    fields(value, {'source_sha256': SOURCE, 'native_sha256': IMAGE}, 'current Tensor.to runtime')


def build_identity(directory, profile, custody, parent=False, fresh_root=None):
    value = custody.read(directory / 'build.json')
    fields(value, {'status': 'PASS', 'link_exit': 0}, 'native build')
    for key in ('source_sha256', 'native_sha256', 'functions', 'blocks', 'instructions',
                'descriptor_bytes', 'state_format', 'backing_format', 'units'):
        require(value[key] == profile[key], ('exact native profile', key))
    fields(value, {'compiler_sha256': COMPILER, 'linker_sha256': LINKER}, 'native tools')
    count = 11 if parent else FINAL['units']
    require(len(value['units']) == count == len({row['name'] for row in value['units']}),
            'all actual independent production units')
    custody.bind(directory / 'runtime.hepta', PARENT_SOURCE if parent else SOURCE)
    custody.bind(directory / ('tensor-mixed-autograd-session' if parent else NATIVE_FILE),
                 PARENT_IMAGE if parent else IMAGE)
    for row in value['units']:
        require(Path(row['name']).name == row['name'] and row['compiler_exit'] == 0,
                'actual module compilation')
        custody.bind(directory / (row['name'] + '.hepta'), row['source_sha256'])
        custody.bind(directory / (row['name'] + '.o'), row['object_sha256'])
    if fresh_root is not None:
        require(not parent, 'fresh-input validation applies only to the new build')
        bind_report_inputs(value, custody, fresh_root)
    return {'build_report_sha256': digest(directory / 'build.json'),
            'source_sha256': value['source_sha256'], 'native_sha256': value['native_sha256'],
            'actual_units_verified': count}


def actual_reference_fields(value, role, probe_sha256):
    count = REFERENCE_RECIPES[role]
    fields(value, {'status': 'OBSERVED', 'cases': count, 'probe_error_count': 0,
        'source_files_unchanged': True, 'probe_sha256': probe_sha256,
        'torch_executed': True, 'native_executed': False}, role)
    require(type(value['rows']) is list and len(value['rows']) == count and
            [row['index'] for row in value['rows']] == list(range(count)) and
            all(row['status'] == 'OBSERVED' for row in value['rows']), ('complete actual reference rows', role))
    if role == 'factory4':
        require([(row['numel'], row['deterministic_fill']) for row in value['rows']] ==
                [(0, True), (6, True), (0, False), (6, False)], 'original four factory configuration rows')
    else:
        require(len({row['name'] for row in value['rows']}) == count, ('unique original reference names', role))
    provenance = value['oracle_provenance']
    fields(provenance, {'status': 'PASS'}, role + ' actual provenance')
    require(set(provenance['checks']) == set(CHECKS) and
            all(provenance['checks'][key] is True for key in CHECKS), 'all eleven exact reference checks')
    require(REFERENCE_SUPPORT in value['inputs_sha256'].values(), ('original support input executed', role))
    states = value['rng_states' if role in ('mixed36', 'Saved14') else 'states']
    require(isinstance(states, dict) and states, ('nonempty complete reference RNG states', role))
    for pin, state in states.items():
        raw = bytes.fromhex(state['hex'])
        require(len(raw) == state['bytes'] == 5056 and hashlib.sha256(raw).hexdigest() == pin,
                ('complete original reference RNG bytes', role, pin))
    if role == 'to54':
        fields(value, {'conversion_variants_observed': 186}, role)
    if role == 'toggle4':
        fields(value, {'setup_rejected_count': 0}, role)
    if role in ('factory4', 'copy-bits4', 'reduction-bits4', 'sum-lanes2', 'sum-branches6'):
        fields(value, {'environment_restored': True}, role)
        require(value['environment_before'] == value['environment_after'], ('exact environment restoration', role))
        fields(value['environment_before'], {'deterministic_algorithms': True,
                                            'deterministic_fill': True}, 'fixed factory/reduction profile')
    if role == 'factory4':
        fields(value, {'operation_totals': {'attempted': 28, 'succeeded': 28, 'raised': 0}}, role)
    if role == 'sum-branches6':
        fields(value, {'cpu_capability': 'AVX512'}, 'actual CPU dispatch capability, separate from provenance')


def reference_identity(repo, custody):
    support_path = repo / 'tests/pytorch_oracle.py'
    custody.bind(support_path, REFERENCE_SUPPORT)
    support = ast.parse(support_path.read_text())
    require(sum(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
                ast.dump(node.value, include_attributes=False) == ast.dump(
                    ast.parse('torch.set_num_threads(1)').body[0].value, include_attributes=False)
                for node in support.body) == 1, 'pinned support actually sets one thread at module scope')
    result = []
    for role, count in REFERENCE_RECIPES.items():
        spec = FINAL['references'][role]
        value = custody.read(relative(repo, spec['path']), spec['sha256'])
        custody.bind(relative(repo, spec['probe_path']), spec['probe_sha256'])
        actual_reference_fields(value, role, spec['probe_sha256'])
        fields(value, spec.get('expected_fields', {}), role)
        if role == 'to54':
            fields(value, {'conversion_variants_observed': 186}, role)
            require(spec['sha256'] == '00bdbbe17709b0cb737161faf944eae873c09b274311dd4c74a28c251e20b6be',
                    'unchanged original54 actual bytes')
        result.append({'role': role, 'reference': spec['path'], 'sha256': spec['sha256'],
            'probe_sha256': spec['probe_sha256'], 'cases': count,
            'all_eleven_original_provenance_checks_true': True,
            'diagnostic_only': role == 'factory4', 'live_Torch_executed_this_invocation': False,
            'single_thread_evidence': 'Original executed ef8 support sets torch.set_num_threads(1); this is not an extra threads getter observation.',
            'support_sha256': REFERENCE_SUPPORT,
            'AVX512_required_and_directly_observed': role == 'sum-branches6'})
    return result


def native_reference_fields(child, role, reference, ref, spec):
    count = REFERENCE_RECIPES[role]
    fields(child, {'schema': spec['report_schema'], 'status': 'PASS', 'cases': count,
        'passed': count, 'failed': 0, 'reference_sha256': ref['sha256'],
        'probe_sha256': ref['probe_sha256'], 'all_inputs_unchanged': True}, role)
    runtime_identity(child)
    if role in REFERENCE_EXECUTION_FLAGS:
        fields(child, {'native_executed': True, 'Torch_executed_here': False}, role)
    else:
        # These original readers do not declare those two fields. Real execution
        # is checked below from the launched reader and original native journals.
        require('native_executed' not in child and 'Torch_executed_here' not in child,
                ('preserve original absent execution fields', role))
        fields(child, {'reader_sha256': spec['reader_sha256'],
                       'observer_sha256': spec['observer_sha256']}, role)
    require(set(child['provenance_checks']) == set(CHECKS) and
            all(child['provenance_checks'][key] is True for key in CHECKS), role + ' original eleven checks')
    rows = child['rows']
    require(type(rows) is list and len(rows) == count and
            [(r['index'], r['name']) for r in rows] == [(r['index'], r['name']) for r in reference['rows']] and
            all(r['status'] == 'PASS' and r['differences'] == [] and r.get('host_errors', []) == [] for r in rows),
            ('complete native reference row plan', role))
    for key in ('frames', 'checks', 'exact_words', 'rollback_bytes', 'rng_bytes'):
        require(type(child[key]) is int and child[key] >= 0 and
                sum(r[key] for r in rows) == child[key], ('native reference exact row total', role, key))
    require(child['frames'] > 0 and child['checks'] > 0, ('actual nonempty native execution', role))
    if role == 'sum-branches6':
        fields(child['fixed_configuration'], {'cpu_capability': 'AVX512',
               'deterministic_algorithms': True, 'fill_uninitialized_memory': True}, role)
        require(REFERENCE_SUPPORT in child['fixed_configuration']['single_thread_source'], 'exact original support single-thread attribution')


def reference_native_journals(child, role, archive, receipt):
    members = receipt['members']
    wanted = {f"case-{row['index']:02d}/raw/transport.json" for row in child['rows']}
    actual = {name for name in members if re.fullmatch(r'case-\d+/raw/transport\.json', name)}
    require(actual == wanted, ('all and only actual native reference journals', role))
    raw_files = archive_members(archive, wanted)
    results = []
    for row in child['rows']:
        prefix = f"case-{row['index']:02d}/raw/"
        name = prefix + 'transport.json'
        raw = raw_files[name]
        require(hashlib.sha256(raw).hexdigest() == members[name]['sha256'] and len(raw) == members[name]['bytes'], 'native journal exact archive bytes')
        journal = loads(raw)
        fields(journal, {'frames': row['frames'], 'executable_sha256': IMAGE,
                         'exit_code': 0, 'responses_before_stdin_eof': True}, role + ' real native EOF')
        require(type(journal['pid']) is int and journal['pid'] > 0 and
                len(journal['rows']) == journal['frames'] and
                [r['frame'] for r in journal['rows']] == list(range(1, journal['frames'] + 1)) and
                all(r['pid'] == journal['pid'] and r['stdin_open'] is True and
                    type(r['response_bytes']) is int and r['response_bytes'] >= 32 for r in journal['rows']),
                ('all original interactive native responses before EOF', role, row['index']))
        for suffix in ('requests.frames.bin', 'responses.frames.bin'):
            require(prefix + suffix in members and members[prefix + suffix]['bytes'] > 0,
                    ('complete original native byte stream', role, row['index'], suffix))
        results.append({'index': row['index'], 'name': row['name'], 'frames': journal['frames'],
                        'native_exit_code': 0, 'transport_sha256': members[name]['sha256']})
    return {'cases': len(results), 'frames': sum(x['frames'] for x in results), 'rows': results,
            'source': 'Original archive case raw/transport.json and complete byte streams; row.transport may be null.',
            'new_native_execution_by_this_verifier': False}


def bind_reference_invocation(root, repo, row, child, spec, ref, profile, custody):
    script = relative(repo, spec['reader_path']).resolve()
    observer = relative(repo, spec['observer_path']).resolve()
    require(script in custody.files and observer in custody.files, 'reader and observer captured before producer execution')
    custody.bind(script, spec['reader_sha256'])
    custody.bind(observer, spec['observer_sha256'])
    argv = row['command']
    positions = [i for i, v in enumerate(argv) if v == str(script)]
    require(len(positions) == 1 and positions[0] > 0 and argv[0] == sys.executable and
            '-B' in argv[1:positions[0]] and all(v in ('-S', '-B') for v in argv[1:positions[0]]),
            'actual exact source-bound Python reader command')
    expected_paths = {'--binary': root/'native'/NATIVE_FILE, '--source': root/'native/runtime.hepta',
                      '--build-report': root/'native/build.json', '--observer': observer,
                      '--reference': relative(repo, ref['path']), '--output': root/row['name']}
    for flag, path in expected_paths.items():
        require(argv.count(flag) == 1 and Path(argv[argv.index(flag)+1]).resolve() == path.resolve(),
                ('reader command actual current input', row['name'], flag))
    pins = {str(Path(name).resolve()): pin for name, pin in child['inputs_sha256'].items()}
    required = [(script, spec['reader_sha256']), (observer, spec['observer_sha256']),
                (root/'native/runtime.hepta', SOURCE), (root/'native'/NATIVE_FILE, IMAGE),
                (root/'native/build.json', digest(root/'native/build.json')),
                (relative(repo, ref['path']), ref['sha256'])]
    for unit in profile['units']:
        required.extend((root/'native'/(unit['name']+suffix), unit[key]) for suffix, key in
                        (('.hepta', 'source_sha256'), ('.o', 'object_sha256')))
    for path, pin in required:
        require(pins.get(str(path.resolve())) == pin, ('all source, image and actual units bound by reader', row['name'], str(path)))


def verify_named_inherited(root, children, custody, out):
    integer_name = next(x['name'] for x in FINAL['producer_steps'] if x['kind'] == 'integer')
    legacy_name = next(x['name'] for x in FINAL['producer_steps'] if x['kind'] == 'legacy')
    integer = children[integer_name]
    require(tuple(row['name'] for row in integer['steps']) == tuple(x[0] for x in INTEGER_GATES), 'original eight I64 gates')
    for row, (name, counts) in zip(integer['steps'], INTEGER_GATES):
        current = dict(row, report=name + '/report.json')
        child_report(root / integer_name, current, custody, out, 'PASS', counts)
        if name != 'inplace40':
            require(row['expected_counts'] == counts, ('unchanged I64 expected counts', name))
    legacy = children[legacy_name]
    fields(legacy, {'step_count': 12, 'existing_native_units_verified': FINAL['units'],
        'all_expected_statuses_and_counts_matched': True, 'cold_build_executed_by_this_driver': False}, 'legacy twelve')
    require(tuple(row['name'] for row in legacy['steps']) == tuple(x[0] for x in LEGACY_GATES), 'legacy twelve order')
    for row, (name, status, counts) in zip(legacy['steps'], LEGACY_GATES):
        child = child_report(root / legacy_name, row, custody, out, status, counts)
        if name == 'actual-retained12':
            require(child['inputs']['observations'] == 'ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3',
                    'original retained12 actual reference')
            require(set(child['reference_provenance_checks']) == set(CHECKS) and
                    all(child['reference_provenance_checks'][key] is True for key in CHECKS), 'retained12 eleven checks')
        if name == 'inherited16':
            require(tuple(x['name'] for x in child['steps']) == INHERITED_NAMES, 'all sixteen original suite identities')
            fields(child, {'native_build_reproduced_exactly': False, 'existing_native_build_receipt_verified': True}, 'inherited16 scope')
            for inner, (expected_status, expected_counts) in zip(child['steps'], INHERITED_EXPECTED):
                child_report(root / legacy_name / 'inherited16', inner, custody, out,
                             expected_status, expected_counts, path_key='report_path')
    return {'original_eight_I64_and_twelve_legacy_gates_verified': True,
            'original_sixteen_inherited_reports_and_bounded_statuses_preserved': True}


def producer_identity(root, repo, profile, custody, out):
    value = custody.read(root / 'report.json')
    runtime_identity(value)
    fields(value, {'schema': 'heptatorch.tensor-to-copy-portable-qualification.v1',
        'status': 'PASS', 'step_count': len(FINAL['producer_steps']), 'profile_sha256': FINAL['profile_sha256'],
        'package_manifest_sha256': FINAL['producer_manifest_sha256'],
        'native_cold_rebuild_started': True, 'native_cold_rebuild_completed': True,
        'pinned_Torch_executed_here': False, 'all_package_members_unchanged': True,
        'all_external_package_members_unchanged': True,
        'all_named_gate_statuses_and_counts_matched': True}, 'fresh producer')
    require(tuple(row['name'] for row in value['steps']) == tuple(x['name'] for x in FINAL['producer_steps']),
            'final exact producer gate order')
    require(value['fixed_reference'] == profile['fixed_reference'] and
            value['additional_references'] == profile['additional_references'] and
            value['fixed_configuration'] == profile['fixed_configuration'] and
            value['external_manifest_pins'] == profile['external_manifests'] == FINAL['external_manifest_pins'],
            'exact original source, reference and configuration plans')
    fields(value, {'factory_reference_is_diagnostic_only': True}, 'factory is not a new native factory API')
    archives, children, reference_journals = [], {}, []
    for row, spec in zip(value['steps'], FINAL['producer_steps']):
        fields(row, {'process_started': True, 'process_completed': True, 'exit_code': 0}, row['name'])
        child = child_report(root, row, custody, out, 'PASS', spec['expected'])
        runtime_identity(child)
        children[row['name']] = child
        if spec['kind'] == 'cold':
            fields(child, {'schema': 'heptatorch.tensor-to-copy-cold-rebuild.v1',
                'native_compilation_executed': True, 'native_link_executed': True,
                'prebuilt_new_image_used': False, 'units_reproduced_exactly': FINAL['units'],
                'all_package_members_unchanged': True, 'parent_image_sha256': PARENT_IMAGE,
                'package_manifest_sha256': FINAL['producer_manifest_sha256']}, 'genuine current cold build')
            custody.bind(root / 'native/build.json', child['build_receipt_sha256'])
            for suffix in ('stdout', 'stderr'):
                custody.bind(root / ('native/rebuild-builder.' + suffix))
            argv = custody.read(root / 'native/rebuild-builder.argv.json')
            require('--output' in argv and Path(argv[argv.index('--output') + 1]).resolve() == root / 'native',
                    'actual cold builder output is this invocation producer/native')
        else:
            pin = row['archive']
            archive = relative(root / 'evidence', pin['archive'])
            custody.bind(archive, pin['archive_sha256'], pin['archive_bytes'])
            receipt = custody.read(root / 'evidence' / (row['name'] + '.manifest.json'))
            fields(receipt, dict(pin), 'producer archive receipt')
            require(receipt['member_count'] == len(receipt['members']) and
                    receipt['original_bytes'] == sum(x['bytes'] for x in receipt['members'].values()),
                    'producer archive complete declared inventory')
            checked = verify_archive(archive, receipt['members'])
            archives.append(dict(checked, gate=row['name'], archive_sha256=pin['archive_sha256']))
            fields(child, {'all_inputs_unchanged': True}, row['name'])
        if spec['kind'] == 'reference':
            ref = FINAL['references'][spec['role']]
            reference = custody.read(relative(repo, ref['path']), ref['sha256'])
            native_reference_fields(child, spec['role'], reference, ref, spec)
            bind_reference_invocation(root, repo, row, child, spec, ref, profile, custody)
            reference_journals.append(dict(reference_native_journals(child, spec['role'], archive, receipt),
                gate=row['name'], role=spec['role'], reader_sha256=spec['reader_sha256'],
                native_executed_this_producer_invocation=True))
    inherited = verify_named_inherited(root, children, custody, out)
    return {'report_sha256': digest(root / 'report.json'), 'steps': len(FINAL['producer_steps']),
            'actual_cold_build_verified': True, 'raw_archives': archives,
            'original_native_reference_journals': reference_journals, **inherited}


def read_single_gzip(path, custody):
    custody.bind(path)
    decoder = zlib.decompressobj(31)
    raw = decoder.decompress(path.read_bytes()) + decoder.flush()
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
            ('complete single gzip observation', str(path)))
    return raw


def pure_owner_controls(root, value, frozen, expected_cases, negative_count, observer, custody):
    complete_cases(value, frozen, expected_cases, root.name)
    negatives = 0
    snapshots = 0
    for case, baseline in zip(value['cases'], frozen['cases']):
        require(case['EOF_exit_code'] == 0, ('real child EOF', case['name']))
        case_root = relative(root, case['name'])
        require(case['frames_before_EOF'] > 0 and case['snapshots'], 'nonempty actual role execution')
        expected_plan = [(x['name'], x['rule'], x['status']) for x in baseline['negative_controls']]
        plan = [(x['name'], x['rule'], x['status']) for x in case['negative_controls']]
        require(plan == expected_plan, ('exact copied-refusal plan', case['name']))
        for row in case['negative_controls']:
            raw = read_single_gzip(relative(case_root, row['file']), custody)
            require(hashlib.sha256(raw).hexdigest() == row['input_sha256'], 'current copied-state bytes')
            copied = loads(raw)
            try:
                observer.verify_snapshot(copied)
            except AssertionError as error:
                require(error.args and isinstance(error.args[0], dict) and
                        error.args[0].get('observer_rule') == row['rule'], 'same precise copied-state refusal')
            else:
                raise AssertionError('copied-state corruption admitted')
            require(row['status'] == 'PASS_EXPECTED_REJECTION', 'copied control bounded status')
            negatives += 1
        for row in case['snapshots']:
            raw = read_single_gzip(relative(case_root, row['snapshot_file']), custody)
            require(hashlib.sha256(raw).hexdigest() == row['snapshot_sha256'], 'actual positive snapshot bytes')
            current = loads(raw)
            runtime_identity(current)
            require(observer.verify_snapshot(current) == row['result'], 'exact positive snapshot pure result')
            owner_row = row['owner']
            owner_path = relative(case_root, owner_row['path'])
            custody.bind(owner_path, owner_row['archive_sha256'])
            decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
            original = decoder.decompress(owner_path.read_bytes())
            require(decoder.eof and not decoder.unused_data and len(original) == owner_row['bytes'] and
                    hashlib.sha256(original).hexdigest() == owner_row['sha256'], 'complete exact raw native owner')
            require(observer.decode_owner(original, current['pointer']) == current, 'snapshot decodes from full actual owner')
            snapshots += 1
        require(case['final_pages_only']['all_remaining_storage_and_holds_are_actual_descriptor_pages'] is True,
                'final resource reclamation')
    require(negatives == negative_count == value['copied_negative_cases'], 'all copied controls without missing cases')
    for total, per_case in (('frames_before_EOF', 'frames_before_EOF'), ('checks', 'checks'),
                            ('full_owner_rollback_bytes', 'rollback_bytes')):
        require(sum(row[per_case] for row in value['cases']) == value[total], ('exact case total', total))
    return {'snapshots_verified_from_original_owner_bytes': snapshots,
            'exact_copied_refusals_rechecked': negatives, 'new_native_executions': 0,
            'scope': 'Pure JSON/raw-byte recheck of this invocation actual cases, not another API qualification.'}


def frozen_report(repo, name, custody):
    spec = FINAL['outer_baselines'][name]
    value = custody.read(relative(repo, spec['path']), spec['sha256'])
    runtime_identity(value)
    return value


def source_audit_complete(value, frozen, profile):
    fields(value, {'status': 'SOURCE_AND_UNIT_CUSTODY_PASS', 'descriptor_bytes': 672,
        'descriptor_words': 84, 'core_function_count': FINAL['functions'],
        'parent_source_sha256': PARENT_SOURCE, 'parent_native_sha256': PARENT_IMAGE,
        'parent_functions': 556, 'native_executed_here': False, 'Torch_executed_here': False,
        'machine_code_certification': False, 'inputs_unchanged': True}, 'independent source audit')
    require(len(value['units']) == FINAL['units'] and len(value['parent_units']) == 11,
            'all current and parent units in source audit')
    require(len(value['core_to_units_exact_body_mapping']) == FINAL['functions'], 'all exact core-body mappings')
    for key in ('core_to_units_exact_body_mapping', 'changed_parent_functions', 'added_from_parent',
                'changed_previous_functions', 'added_from_previous', 'preserved_callee_aliases',
                'observer_reversals', 'anchors', 'reference54', 'toggle4', 'factory4',
                'unchanged_parent_functions', 'previous_source_sha256', 'previous_functions',
                'unchanged_previous_functions', 'observer_sha256', 'entry_offset_inverse_exact',
                'typed_offset_inverse_exact', 'entry_first_host_index_failure_preserved_sha256',
                'native_and_numeric_expectations_changed_by_offset_adapters',
                'frozen05_to_current_source_delta'):
        require(value[key] == frozen[key], ('same frozen source-audit fact', key))
    fields(value['frozen05_to_current_source_delta'], {'source_sha256': SOURCE, 'native_sha256': IMAGE,
        'previous_source_sha256': 'c5ff5821aa7a70bacf89e76ae47abc016ff9a30e7a596abe1e4d67be8a6bf350',
        'current_functions': FINAL['functions'], 'previous_functions': 593,
        'numerical_and_capacity_and_error_and_lifecycle_expectations_changed': False}, 'explicit original05 source relationship')
    require(set(value['changed_parent_functions']) == set(profile['changed_parent_functions']) and
            set(value['added_from_parent']) == set(profile['added_functions']), 'actual producer body delta agrees')


def historical_archives(repo, custody):
    result = []
    for spec in FINAL['historical_archives']:
        transport = None
        if 'split_delivery' in spec:
            split = spec['split_delivery']
            verifier = relative(repo, split['verifier_path'])
            custody.bind(verifier, split['verifier_sha256'])
            custody.bind(relative(repo, split['delivery_manifest_path']), split['delivery_manifest_sha256'])
            custody.bind(relative(repo, split['original_manifest_path']), split['original_manifest_sha256'])
            # The physical manifest was checked before this call. The pinned
            # read-only verifier checks all seventy unchanged original files,
            # the original logical71 manifest, ordered parts and original modes.
            loaded = importlib.util.spec_from_file_location('to09b_immutable_delivery_verifier', verifier)
            module = importlib.util.module_from_spec(loaded)
            loaded.loader.exec_module(module)
            transport, archive = module.verify(repo, split['delivery_manifest_sha256'])
            fields(transport, {'status': 'PASS_TRANSPORT_ONLY', 'whole_archive_installed': False,
                'original_manifest_sha256': split['original_manifest_sha256'],
                'reconstructed_archive_sha256': spec['archive_sha256'],
                'reconstructed_archive_bytes': spec['archive_bytes'], 'part_count': 2,
                'original_payload_count': 71, 'unchanged_installed_original_payloads': 70,
                'all_inputs_unchanged': True, 'native_executed': False, 'Torch_executed': False,
                'old_reports_or_results_reclassified': False, 'original_bytes_or_modes_changed': False},
                'exact original split delivery')
            require(type(archive) is bytes and len(archive) == spec['archive_bytes'] and
                    hashlib.sha256(archive).hexdigest() == spec['archive_sha256'], 'whole original archive in memory')
            require(not relative(repo, spec['archive']).exists(), 'no fictitious whole archive installation')
        else:
            archive = relative(repo, spec['archive'])
            custody.bind(archive, spec['archive_sha256'])
        receipt = custody.read(relative(repo, spec['manifest']), spec['manifest_sha256'])
        files = receipt[spec['member_field']]
        checked = verify_archive(archive, files)
        result.append({'logical_original_archive': spec['archive'],
                       'whole_archive_installed': transport is None,
                       'split_delivery_verification': transport,
                       'archive_sha256': spec['archive_sha256'],
                       'manifest_sha256': spec['manifest_sha256'], **checked})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'parent-build', 'upstream', 'compiler', 'linker', 'flat-compiler', 'stack-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    repo, parent, stack, out = args.repo.resolve(), args.parent_build.resolve(), args.stack_build.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    custody = Custody()
    report = {'schema': 'heptatorch.tensor-to-copy-native-integration.v1', 'status': 'RUNNING',
        'source_sha256': SOURCE, 'native_sha256': IMAGE, 'producer_mode': 'FRESH_COLD_REQUIRED',
        'producer_executed_this_invocation': False, 'producer_completed_this_invocation': False,
        'producer_reused': False, 'Torch_executed_here': False, 'commands': [],
        'whole_pytorch_gate': 'HOLD', 'full_pytorch_rewrite_complete': False,
        'pytorch_api_denominator': 2590, 'closed_schema_count': 0,
        'counting_boundary': 'Named suites overlap. No sum is a unique upstream test or schema count.'}
    dump(out / 'invocation.json', {'argv': sys.argv, 'python_executable': sys.executable,
        'source_path': str(Path(__file__).resolve()), 'source_sha256': digest(Path(__file__)),
        'paths': {name: str(value.resolve()) for name, value in vars(args).items()}})

    def save():
        dump(out / 'report.json', report)

    def execute(name, script, arguments, expected, expected_status='PASS'):
        receipt = out / name / 'report.json'
        require(not (out / name).exists(), 'fresh child output required')
        custody.bind(script)
        argv = [sys.executable, '-S', '-B', str(script), *map(str, arguments), '--output', str(out / name)]
        row = {'name': name, 'argv': argv, 'status': 'STARTING', 'process_started': False, 'process_completed': False}
        report['commands'].append(row)
        dump(out / (name + '.argv.json'), argv)
        save()
        try:
            with (out / (name + '.stdout')).open('wb') as so, (out / (name + '.stderr')).open('wb') as se:
                process = subprocess.Popen(argv, cwd=repo, stdout=so, stderr=se)
                row.update(status='RUNNING', process_started=True, pid=process.pid)
                if name == 'producer':
                    report['producer_executed_this_invocation'] = True
                save()
                try:
                    process.wait(timeout=7200)
                except BaseException:
                    process.kill()
                    process.wait()
                    row.update(exit_code=process.returncode, process_reaped_after_interrupt=True)
                    raise
                row.update(process_completed=True, exit_code=process.returncode)
                if name == 'producer':
                    report['producer_completed_this_invocation'] = True
        finally:
            for suffix in ('stdout', 'stderr'):
                path = out / (name + '.' + suffix)
                if path.is_file():
                    row[suffix] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
            save()
        value = decode(receipt) if receipt.is_file() else {}
        row.update(status=value.get('status', 'NO_REPORT'))
        if receipt.is_file():
            row.update(report=receipt.relative_to(out).as_posix(), report_sha256=digest(receipt))
        save()
        require(process.returncode == 0 and value.get('status') == expected_status, ('actual gate failure', name))
        custody.bind(receipt, row['report_sha256'])
        runtime_identity(value)
        fields(value, expected, name)
        bind_report_inputs(value, custody, out)
        return value

    save()
    try:
        custody.bind(Path(__file__))
        contract_ready()
        report['manifests'] = [manifest(repo, *row, custody) for row in PARENT_MANIFESTS + tuple(FINAL['manifests'])]
        # Original54 installs by its original manifest. Other reference packets
        # preserve original modes inside their archives and bind installed 0644
        # copies through separate final manifests; no original mode is relabelled.
        profile = custody.read(repo / 'tools/tensor_to_copy/profile.json', FINAL['profile_sha256'])
        parent_profile = custody.read(repo / 'tools/mixed_integer_autograd/profile.json', PARENT_PROFILE)
        runtime_identity(profile)
        fields(profile, {'descriptor_bytes': 672, 'functions': FINAL['functions'],
                         'state_format': FINAL['state_format'], 'backing_format': FINAL['backing_format'],
                         'parent_source_sha256': PARENT_SOURCE, 'parent_native_sha256': PARENT_IMAGE}, 'current native profile')
        fields(parent_profile, {'source_sha256': PARENT_SOURCE, 'native_sha256': PARENT_IMAGE}, 'mixed10 parent')
        custody.bind(args.compiler, COMPILER)
        custody.bind(args.linker, LINKER)
        custody.bind(args.flat_compiler, FLAT)
        require(args.compiler.resolve() == (stack / 'native/stack-module-compiler').resolve() and
                args.linker.resolve() == (stack / 'native/stack-linker-session').resolve(), 'explicit same canonical stack build')
        custody.bind(stack / 'native/stack-linker-runtime', RUNTIME_LINKER)
        report['parent_build'] = build_identity(parent, parent_profile, custody, parent=True)
        custody.bind(parent / 'overlay-authoring.py', PARENT_AUTHOR)
        author = custody.read(repo / 'tools/owned_tensors/profile.json', '69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242')
        custody.bind(repo / 'tools/owned_tensors/authoring.py', '2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6')
        for name, pin in author['upstream_authoring_sources'].items():
            custody.bind(relative(args.upstream.resolve(), name), pin)
        for name in ('tests/retained_nonleaf_review/protocol.py', 'tests/retained_resource_review/protocol.py'):
            custody.bind(repo / name, PROTOCOL)
            custody.bind((repo / name).with_name('owner_client.py'), CLIENT)
        for spec in FINAL['extra_inputs']:
            custody.bind(relative(repo, spec['path']), spec['sha256'], spec.get('bytes'), spec.get('mode'))
        for name, spec in FINAL['scripts'].items():
            custody.bind(relative(repo, spec['path']), spec['sha256'])
            frozen_report(repo, name, custody)
        custody.bind(relative(repo, FINAL['general_owner']), FINAL['general_owner_sha256'])
        report['historical_archives_verified_without_reclassifying_results'] = historical_archives(repo, custody)
        report['fixed_actual_references'] = reference_identity(repo, custody)
        report['inputs_captured_before_execution'] = len(custody.files)
        dump(out / 'inputs-before-execution.json', {str(path): value for path, value in custody.files.items()})
        execute('producer', repo / 'tools/tensor_to_copy/qualify.py', [
            '--parent', parent, '--repository', repo, '--upstream', args.upstream.resolve(),
            '--compiler', args.compiler.resolve(), '--linker', args.linker.resolve(),
            '--flat-compiler', args.flat_compiler.resolve()], {'step_count': len(FINAL['producer_steps'])})
        native = out / 'producer/native'
        report['new_build'] = build_identity(native, profile, custody, fresh_root=out)
        report['producer_admission'] = producer_identity(out / 'producer', repo, profile, custody, out)
        protocol = repo / 'tests/retained_resource_review/protocol.py'
        ref = lambda role: relative(repo, FINAL['references'][role]['path'])
        script = lambda name: relative(repo, FINAL['scripts'][name]['path'])
        expected = lambda name: FINAL['outer_expected'][name]
        source = execute('source-custody', script('source-custody'), [
            '--native-build', native, '--parent-build', parent, '--reference-package', repo,
            '--toggle-reference', ref('toggle4'), '--factory-reference', ref('factory4'),
            '--stack-build', stack], expected('source-custody'), expected_status='SOURCE_AND_UNIT_CUSTODY_PASS')
        source_audit_complete(source, frozen_report(repo, 'source-custody', custody), profile)
        observer_path = relative(repo, FINAL['general_owner'])
        spec = importlib.util.spec_from_file_location('qualified_to672_pure_owner', observer_path)
        observer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(observer)
        require(observer.SOURCE == SOURCE and observer.IMAGE == IMAGE and observer.DESCRIPTOR == 672,
                'exact final pure owner module')
        common = ['--native-build', native, '--protocol', protocol, '--reference', ref('to54')]
        owner = execute('owner4-copied17', script('owner4-copied17'), common + [
            '--toggle-reference', ref('toggle4'), '--factory-reference', ref('factory4')], expected('owner4-copied17'))
        report['owner_pure_recheck'] = pure_owner_controls(out / 'owner4-copied17', owner,
            frozen_report(repo, 'owner4-copied17', custody), 4, 17, observer, custody)
        resource = execute('resources6', script('resources6'), common, expected('resources6'))
        report['resource_pure_recheck'] = pure_owner_controls(out / 'resources6', resource,
            frozen_report(repo, 'resources6', custody), 6, 7, observer, custody)
        tests = script('owned-entry5').parent
        entry = execute('owned-entry5', script('owned-entry5'), common + [
            '--stack-build', stack, '--probe-source', tests / 'entry_probe.hepta'], expected('owned-entry5'))
        complete_cases(entry, frozen_report(repo, 'owned-entry5', custody), 5, 'entry')
        require(all(x['EOF_exit_code'] == 0 for x in entry['cases']) and
                entry['native_probe_compiled_this_invocation'] is True and
                entry['linked_modules'] == FINAL['units'] + 1, 'new actual entry5 probe and EOF')
        saved = execute('Saved-entry6', script('Saved-entry6'), [
            '--build', native, '--probe-build', out / 'owned-entry5', '--protocol', protocol], expected('Saved-entry6'))
        complete_cases(saved, frozen_report(repo, 'Saved-entry6', custody), 6, 'Saved entry')
        require(saved['real_previous_compile_and_link_receipt_sha256'] == digest(out / 'owned-entry5/report.json') and
                saved['probe_source_sha256'] == entry['probe_source_sha256'] == PROBE and
                saved['probe_image_sha256'] == entry['probe_image_sha256'] and
                saved['linked_modules'] == entry['linked_modules'] == FINAL['units'] + 1 and
                saved['native_probe_recompiled_this_invocation'] is False and saved['EOF_exit_code'] == 0,
                '107 controls consume this invocation freshly compiled entry5 probe')
        handoff = execute('typed-handoff2', script('typed-handoff2'), common + [
            '--parent-build', parent, '--stack-build', stack,
            '--controller', script('typed-handoff2').parent / 'state_handoff.hepta'], expected('typed-handoff2'))
        complete_cases(handoff, frozen_report(repo, 'typed-handoff2', custody), 2, 'typed handoff')
        require(handoff['linked_modules'] == 11 + FINAL['units'] + 1 and
                len(handoff['units']) == 11 + FINAL['units'] and
                len(handoff['frames']) == handoff['frames_before_EOF'] and handoff['EOF_exit_code'] == 0,
                'both real owned handoff directions and actual current+parent modules')
        require(tuple(row['name'] for row in report['commands']) == OUTER_STEPS, 'all seven actual outer commands')
        require(all(x['process_started'] and x['process_completed'] and x['exit_code'] == 0
                    for x in report['commands']), 'all seven actual complete child processes')
        report['separate_required_CI_gates'] = {
            'live32': 'NOT_EXECUTED_BY_NATIVE_COORDINATOR',
            'complete_accounted35_and_copied18_and_tombstone': 'NOT_EXECUTED_BY_NATIVE_COORDINATOR',
            'no_inference_from_old_profile_or_inherited_suites': True}
        report['counts'] = {'producer_top_level_steps': len(FINAL['producer_steps']),
            'actual_reference_recipes_by_role': REFERENCE_RECIPES,
            'factory_reference_is_diagnostic_not_native_factory_qualification': True,
            'inherited_I64_named_gates': 8, 'legacy_behavior_gates': 12,
            'inherited_named_suites_with_original_OPENs': 16,
            'source_core_functions': FINAL['functions'], 'actual_compiled_units': FINAL['units'],
            'native_owner_cases': 4, 'copied_new_role_refusals': 17,
            'native_resource_controls': 6, 'native_entry_controls': 5,
            'native_Saved_entry_controls': 6, 'native_typed_directions': 2}
        custody.finish()
        dump(out / 'checked-inputs-before-archive.json', {str(path): value for path, value in custody.files.items()})
        report['complete_current_execution_evidence'] = archive_evidence(out)
        custody.finish()
        report.update(status='PASS', command_count=len(report['commands']), all_inputs_unchanged=True,
                      checked_inputs={str(path): value for path, value in custody.files.items()})
        save()
        print(json.dumps({'status': 'PASS', 'report_sha256': digest(out / 'report.json'),
                          'commands_executed': len(report['commands']), 'source_sha256': SOURCE, 'native_sha256': IMAGE}))
        return 0
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc(),
                      successful_suite_completion_claimed=False)
        save()
        try:
            report['partial_failed_execution_evidence'] = archive_evidence(out)
        except BaseException as archive_error:
            report['archive_error'] = repr(archive_error)
        report['checked_inputs'] = {str(path): value for path, value in custody.files.items()}
        save()
        print(json.dumps({'status': 'FAIL', 'error': repr(error), 'report': str(out / 'report.json')}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())