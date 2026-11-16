def command(s:readbuf, o:mutbuf, w:mutbuf, cp:u64, at:u64, end:u64) -> u64:
    code = getr(s, cp)
    a = getr(s, cp + 8)
    b = getr(s, cp + 16)
    c = getr(s, cp + 24)
    result = 0
    payload = 0
    recognized = 0
    if code == 1:
        recognized = 1
        putw(w, 2048, c)
        putw(w, 2056, getr(s, cp + 32))
        putw(w, 2064, getr(s, cp + 40))
        putw(w, 2072, getr(s, cp + 48))
        result = tnew(w, a, 2048, b)
        if getw(w, 0) == 0:
            dp = dptr(w, result)
            n = getw(w, dp + 8)
            source = getr(s, cp + 56)
            if source < end or source > length(s):
                terror(w, 100)
            else:
                if n > (length(s) - source) // 4 or source % 4 != 0:
                    terror(w, 100)
                else:
                    i = 0
                    while i < n:
                        tset(w, result, i, read4(s, source + 4 * i))
                        i = i + 1
    if code == 2:
        recognized = 1
        result = binary(w, a, b, c)
    if code == 3:
        recognized = 1
        result = unary(w, a, b)
    if code == 4:
        recognized = 1
        result = reduce(w, a, b)
    if code == 5:
        recognized = 1
        result = mm(w, a, b)
    if code == 6:
        recognized = 1
        result = transpose(w, a, b, c)
    if code == 7:
        recognized = 1
        result = backward(w, a)
    if code == 8:
        recognized = 1
        result = zero_grad(w, a)
    if code == 9:
        recognized = 1
        result = sgd(w, a, b)
    if code == 10 or code == 11:
        recognized = 1
        ok = valid(w, a)
        dp = dptr(w, a)
        if ok != 0:
            if code == 11:
                if getw(w, dp + 40) != 0 or getw(w, dp + 32) == 0:
                    terror(w, 5)
            n = getw(w, dp + 8)
            if n > (length(o) - at - 32) // 4:
                terror(w, 3)
            else:
                i = 0
                while i < n:
                    value = 0
                    if code == 10:
                        value = tget(w, a, i)
                    else:
                        value = tgrad(w, a, i)
                    put4(o, at + 32 + i * 4, value)
                    i = i + 1
                result = n
                payload = n * 4
    if code == 12:
        recognized = 1
        result = tset(w, a, b, c)
    if code == 13:
        recognized = 1
        ok = valid(w, a)
        dp = dptr(w, a)
        if ok != 0:
            if length(o) - at - 32 < 104:
                terror(w, 3)
            else:
                putw(o, at + 32, getw(w, dp))
                putw(o, at + 40, getw(w, dp + 8))
                putw(o, at + 48, getw(w, dp + 32))
                i = 0
                while i < 4:
                    putw(o, at + 56 + i * 8, getw(w, dp + 96 + i * 8))
                    putw(o, at + 88 + i * 8, getw(w, dp + 128 + i * 8))
                    i = i + 1
                putw(o, at + 120, getw(w, dp + 64))
                putw(o, at + 128, tversion(w, a))
                result = a
                payload = 104
    if code == 14:
        recognized = 1
        result = scalar(w, a, b, c)
    if code == 15:
        recognized = 1
        result = linear(w, a, b, c)
    if code == 16:
        recognized = 1
        result = mse(w, a, b)
    if code == 17:
        recognized = 1
        putw(w, 2048, c)
        putw(w, 2056, getr(s, cp + 32))
        putw(w, 2064, getr(s, cp + 40))
        putw(w, 2072, getr(s, cp + 48))
        result = view(w, a, b, 2048)
    if code == 18:
        recognized = 1
        result = clone(w, a)
    if code == 19:
        recognized = 1
        result = detach(w, a)
    if code == 20:
        recognized = 1
        result = reduce_axis(w, a, b, c, getr(s, cp + 32))
    if code == 21:
        recognized = 1
        result = softmax(w, a, b, c)
    if code == 22:
        recognized = 1
        result = conv2d(w,a,b,c,getr(s,cp+32),getr(s,cp+40),getr(s,cp+48))
    if code == 23:
        recognized = 1
        result = pool2d(w,a,b,c,getr(s,cp+32),getr(s,cp+40),getr(s,cp+48))
    if code == 24 or code == 25:
        recognized = 1
        result = optimizer_command(s,w,cp,end)
    if code == 26:
        recognized = 1
        result = layernorm(w,a,b,c,getr(s,cp+32))
    if code == 27:
        recognized = 1
        result = cross_entropy_probs(w,a,b)
    if code == 28:
        recognized = 1
        result = batchnorm_eval(w,a,b,c,getr(s,cp+32),getr(s,cp+40),getr(s,cp+48))
    if code == 29:
        recognized = 1
        result = backward_seed(w,a,b)
    if code == 30:
        recognized = 1
        result = index_select_command(s,w,cp,end)
    if code == 31:
        recognized = 1
        result = gather_command(s,w,cp,end)
    if code == 32:
        recognized = 1
        result = matmul(w,a,b)
    if code == 33:
        recognized = 1
        result = graph_reset(w,a)
    if code == 34 or code == 35:
        recognized = 1
        result = concat_command(s,w,cp,end)
    if code == 36 or code == 37:
        recognized = 1
        result = indexed_add_command(s,w,cp,end)
    if recognized == 0:
        terror(w, 4)
    status = getw(w, 0)
    if status != 0:
        payload = 0
        result = 0
    putw(o, at, code)
    putw(o, at + 8, status)
    putw(o, at + 16, result)
    putw(o, at + 24, payload)
    return at + 32 + payload
