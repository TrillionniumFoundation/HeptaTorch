def wrapped_f64_to_f32(value: u64) -> u64:
    sign=(value>>32)&2147483648
    exponent=(value>>52)&2047
    fraction=value&4503599627370495
    if exponent==2047:
        if fraction==0:
            return sign|2139095040
        payload=(fraction>>29)|4194304
        return sign|2139095040|(payload&8388607)
    if exponent<873:
        return sign
    if exponent>1150:
        return sign|2139095040
    mantissa=fraction|4503599627370496
    shift=29
    if exponent<897:
        shift=926-exponent
    rounded=mantissa>>shift
    remainder=mantissa&((1<<shift)-1)
    half=1<<(shift-1)
    if remainder>half or (remainder==half and (rounded&1)!=0):
        rounded=rounded+1
    if exponent<897:
        return sign|rounded
    output_exponent=exponent-896
    if rounded==16777216:
        rounded=rounded>>1
        output_exponent=output_exponent+1
    if output_exponent>=255:
        return sign|2139095040
    return sign|(output_exponent<<23)|(rounded&8388607)

def wrapped_mul_f32(left: u64, right: u64) -> u64:
    a=left&4294967295
    b=right&4294967295
    if (a&2147483647)>2139095040:
        return a|4194304
    if (b&2147483647)>2139095040:
        return b|4194304
    if ((a&2147483647)==2139095040 and (b&2147483647)==0) or ((b&2147483647)==2139095040 and (a&2147483647)==0):
        return 4290772992
    return mulf(a,b)
