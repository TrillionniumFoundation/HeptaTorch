# Recovered direct differentiable view: R2

This is the distinct recovered-source R2 package. The R1 package in tools/differentiable_view_recovered and its original native14 failure history are unchanged. The original 15 source files and source-manifest bytes are preserved here; installed copies explicitly map source mode0600 to Git mode0644.

R2 has been compiled with the pinned native compiler/linker into 12 units. profile.json binds the actual source, image, all unit sources/objects, full614 function maps and original build report. This installer only read and verified those artifacts. Runtime qualification is recorded separately; this package currently says BUILT_NOT_RUNTIME_QUALIFIED.

## Rebuild

Run from the repository root using the independently rebuilt09b parent:

    python3 -S -B tools/tensor_to_native_recovered/rebuild.py --compiler "$MODULE_COMPILER" --linker "$SESSION_LINKER" --output build/tensor-to-native-recovered
    python3 -B tools/differentiable_view_recovered_v2/build.py --previous build/tensor-to-native-recovered --repository . --upstream "$HEPTA_CHECKOUT" --compactor tools/native_dropout/compact_functions.py --compiler "$MODULE_COMPILER" --linker "$SESSION_LINKER" --output build/differentiable-view-r2

Each output directory must be new. The parent recipe is the explicit new recovered09b recipe, not a claim that the unavailable5179/a6f Cloud packages were recovered. The child authoring needs the fixed Hepta checkout and repository authoring inputs checked by ParentEmitter. No Cloud path or previous child cache is used as a build input.

The compiler is b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f and the session linker is aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16. The parent is source4f8f73f463f681b5eb29423d71a35a0b42d807fd92bf976bbf76c1970fbcee33 / image53fb43b869858f020c0b936a65f00f3b14b97d9e37038a74c357e1da471fe083.

## Source and behavior boundary

R2 retains all614 functions. Against R1, only four state-tag bodies and to_view_refresh change;609 other bodies are byte-identical. The constructor is the same934c source as R1. The local refresh correction moves a validated weak retained-gradient hook to the current view Node. Captured AccumulateGrad strong roots and600/208 counts remain unchanged. The exact old-function inverse and fixed official variable.cpp are included.

The generated directory contains actual HIA fragments, full runtime and all12 unit sources. Their original object and image hashes are in profile.json and the unmodified build receipt; binary files are not presented as a new build by this installation.

The full project remains HOLD with zero closed API schemas. A successful native build does not replace the separate fixed14 replay or wider inherited qualification.
