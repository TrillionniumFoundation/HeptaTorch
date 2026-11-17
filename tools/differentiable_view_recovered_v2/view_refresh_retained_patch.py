"""Direct-view R2: native retained-hook move on lazy ViewInfo refresh only.

The original constructor and every general history helper remain unchanged.
Only the weak retained-hook pair materialized by history_capture is moved:
old nonleaf snapshot608/624 is cleared and current608's disabled bit is removed.
Captured AccumulateGrad (old op0) bypasses this path. Low accumulator bits and
current624, all strong roots, Tensor/ViewInfo identity and edge geometry stay.
"""
from authoring_support import once, sha

PARENT_FUNCTION_SHA256 = '8cf3006514e33aed882769e653d9b587c0cf98b898b91dd203a73415d32d36f1'
OLD_BRANCH = "  branch v202 b23(v184) b25(v184,v185,v186,v187,v190,v191)\n".encode()
NEW_BRANCH = "  branch v202 b23(v184) b28(v184,v185,v186,v187,v190,v191,v200)\n".encode()
APPENDED_BLOCKS = "block b28(v253:mutbuf,v254:u64,v255:u64,v256:u64,v257:u64,v258:u64,v259:u64)\n  v260:u64 = call(v253,v259) {\"function\":\"dptr\"}\n  v261:u64 = const() {\"value\":40}\n  v262:u64 = int.add.wrap(v260,v261)\n  v263:u64 = call(v253,v262) {\"function\":\"getw\"}\n  v264:u64 = const() {\"value\":0}\n  v265:bool = compare.eq(v263,v264)\n  branch v265 b25(v253,v254,v255,v256,v257,v258) b29(v253,v254,v255,v256,v257,v258,v260)\nblock b29(v266:mutbuf,v267:u64,v268:u64,v269:u64,v270:u64,v271:u64,v272:u64)\n  v273:u64 = const() {\"value\":608}\n  v274:u64 = int.add.wrap(v272,v273)\n  v275:u64 = call(v266,v274) {\"function\":\"getw\"}\n  v276:u64 = const() {\"value\":624}\n  v277:u64 = int.add.wrap(v272,v276)\n  v278:u64 = call(v266,v277) {\"function\":\"getw\"}\n  v279:u64 = const() {\"value\":0}\n  v280:bool = compare.eq(v275,v279)\n  branch v280 b30(v266,v267,v268,v269,v270,v271,v278) b31(v266,v267,v268,v269,v270,v271,v272,v275,v278)\nblock b30(v281:mutbuf,v282:u64,v283:u64,v284:u64,v285:u64,v286:u64,v287:u64)\n  v288:u64 = const() {\"value\":0}\n  v289:bool = compare.eq(v287,v288)\n  branch v289 b25(v281,v282,v283,v284,v285,v286) b35(v281)\nblock b31(v290:mutbuf,v291:u64,v292:u64,v293:u64,v294:u64,v295:u64,v296:u64,v297:u64,v298:u64)\n  v299:u64 = const() {\"value\":65536}\n  v300:u64 = bit.or(v291,v299)\n  v301:bool = compare.eq(v297,v300)\n  branch v301 b32(v290,v291,v292,v293,v294,v295,v296,v298) b35(v290)\nblock b32(v302:mutbuf,v303:u64,v304:u64,v305:u64,v306:u64,v307:u64,v308:u64,v309:u64)\n  v310:u64 = const() {\"value\":320}\n  v311:u64 = int.add.wrap(v304,v310)\n  v312:u64 = call(v302,v311) {\"function\":\"getw\"}\n  v313:bool = compare.eq(v309,v312)\n  branch v313 b33(v302,v303,v304,v305,v306,v307,v308) b35(v302)\nblock b33(v314:mutbuf,v315:u64,v316:u64,v317:u64,v318:u64,v319:u64,v320:u64)\n  v321:u64 = const() {\"value\":608}\n  v322:u64 = int.add.wrap(v316,v321)\n  v323:u64 = call(v314,v322) {\"function\":\"getw\"}\n  v324:u64 = const() {\"value\":65535}\n  v325:u64 = bit.and(v323,v324)\n  v326:u64 = const() {\"value\":131072}\n  v327:u64 = bit.or(v325,v326)\n  v328:bool = compare.eq(v323,v327)\n  branch v328 b34(v314,v315,v316,v317,v318,v319,v320,v322,v325) b35(v314)\nblock b34(v329:mutbuf,v330:u64,v331:u64,v332:u64,v333:u64,v334:u64,v335:u64,v336:u64,v337:u64)\n  v338:u64 = const() {\"value\":0}\n  v339:u64 = const() {\"value\":608}\n  v340:u64 = int.add.wrap(v335,v339)\n  v341:u64 = call(v329,v340,v338) {\"function\":\"putw\"}\n  v342:u64 = const() {\"value\":624}\n  v343:u64 = int.add.wrap(v335,v342)\n  v344:u64 = call(v329,v343,v338) {\"function\":\"putw\"}\n  v345:u64 = call(v329,v336,v337) {\"function\":\"putw\"}\n  jump b25(v329,v330,v331,v332,v333,v334)\nblock b35(v346:mutbuf)\n  v347:u64 = const() {\"value\":9}\n  v348:u64 = call(v346,v347) {\"function\":\"terror\"}\n  ret(v348)\n".encode()


def apply_refresh(body):
    assert sha(body) == PARENT_FUNCTION_SHA256
    assert body.endswith(b'end\n') and b'block b28(' not in body
    revised = once(body, OLD_BRANCH, NEW_BRANCH)
    revised = revised[:-4] + APPENDED_BLOCKS + b'end\n'
    inverse = once(revised, APPENDED_BLOCKS, b'')
    inverse = once(inverse, NEW_BRANCH, OLD_BRANCH)
    assert inverse == body
    return revised, {
        'function': 'to_view_refresh',
        'before_sha256': sha(body), 'after_sha256': sha(revised),
        'exact_inverse': True, 'old_blocks_retained_except_capture_success_branch': True,
        'new_blocks': ['b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35'],
        'changed_persistent_offsets': [608, 624],
        'weak_previous_snapshot_hook_cleared': True,
        'current_accumulator_low_bits_preserved': True,
        'current_accumulator_incarnation_preserved': True,
        'old_op0_accumulator_bypasses_hook_move': True,
        'strong_tensor_refcounts_600_208_changed': False,
        'general_history_capture_changed': False,
        'upstream_variable_cpp_sha256': 'ca2f1e573307069dee2fdff70ba81f161f48817cd278ad5464b9809cf23ab2c6',
        'fixed_reference_sha256': '2270d5ba925bcf91c56db94fac663695e69079c74edbf80aff8fa086e7c81665'
    }
