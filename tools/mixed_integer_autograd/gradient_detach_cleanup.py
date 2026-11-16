"""Release owned Saved Tensor roots after the inherited detach admission guard.

Parent author file SHA256: 4a1881ad98f79e9807818a956af5a00f787dc46333b3a8f1a78d146d95fc3b72.
Removing the single cleanup call re-emits the exact parent HIA body.
"""

def gradient_detach_inplace(w: mutbuf, g: u64) -> u64:
    p = dptr(w, g)
    if getw(w, p + 40) != 0:
        if getw(w, p + 408) != 0 or getw(w, p + 208) != getw(w, p + 312) + getw(w, p + 400):
            return terror(w, 5)
        if dtype_mask_release(w, g) == 0:
            return 0
        j = 0
        while j < gradient_parent_count(w, g):
            if graph_release_edge(w, graph_parent_at(w, g, j), graph_edge_enabled(w, g, j)) == 0:
                return 0
            j = j + 1
        saved_release(w, getw(w, p + 280))
        saved_release(w, getw(w, p + 288))
        saved_release(w, getw(w, p + 296))
        hold_release(w, getw(w, p + 256))
        hold_release(w, getw(w, p + 264))
        hold_release(w, getw(w, p + 272))
        putw(w, p + 40, 0)
        putw(w, p + 48, 0)
        putw(w, p + 56, 0)
        putw(w, p + 72, 0)
        putw(w, p + 80, 0)
        putw(w, p + 160, 0)
        putw(w, p + 184, 0)
        putw(w, p + 256, 0)
        putw(w, p + 264, 0)
        putw(w, p + 272, 0)
        putw(w, p + 280, 0)
        putw(w, p + 288, 0)
        putw(w, p + 296, 0)
        putw(w, p + 328, 0)
        putw(w, p + 344, 0)
        putw(w, p + 352, 0)
    putw(w, p + 32, 0)
    return choose(getw(w, 0) == 0, 1, 0)
