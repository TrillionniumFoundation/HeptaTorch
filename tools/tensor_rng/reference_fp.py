"""Exact signed-dyadic binary32 oracle, qualification only.

The arithmetic is independent of HIA bit-emulation functions. This is the
integer reference used by the frozen actual-PyTorch uniform probe a27756cc.
"""
def parts(word):
    exponent=(word>>23)&255;mantissa=word&0x7fffff
    assert exponent!=255
    if exponent:mantissa|=1<<23
    return (-mantissa if word>>31 else mantissa),exponent-150 if exponent else -149

def add(a,b):
    exponent=min(a[1],b[1])
    return (a[0]<<(a[1]-exponent))+(b[0]<<(b[1]-exponent)),exponent

def rne(pair):
    number,exponent=pair;sign=0
    if number<0:sign=1<<31;number=-number
    if not number:return sign
    top=number.bit_length()-1+exponent
    if top>127:return sign|0x7f800000
    quantum=max(top-23,-149);shift=quantum-exponent
    if shift>0:
        mantissa,remainder=divmod(number,1<<shift);half=1<<(shift-1)
        if remainder>half or remainder==half and mantissa&1:mantissa+=1
    else:mantissa=number<<(-shift)
    if mantissa>=1<<24:mantissa>>=1;quantum+=1
    if mantissa<1<<23:return sign|mantissa
    exp=quantum+150
    if exp>=255:return sign|0x7f800000
    return sign|(exp<<23)|(mantissa&0x7fffff)

def uniform(raw,lo,hi):
    left,right=parts(lo),parts(hi)
    width=parts(rne(add(right,(-left[0],left[1]))))
    unit=parts(rne((raw&0xffffff,-24)))
    product=unit[0]*width[0],unit[1]+width[1]
    value=rne(add(product,left))
    return lo if value==hi or (value&0x7fffffff)==(hi&0x7fffffff)==0 else value

def axis_order(shape,strides):
    order=list(reversed(range(len(shape))))
    for i in range(1,len(order)):
        anchor=i
        for j in range(i-1,-1,-1):
            left,right=order[j],order[anchor];a,b=strides[left],strides[right]
            if not a or not b:continue
            if a>b or a==b and shape[left]>shape[right]:
                order[j],order[anchor]=order[anchor],order[j];anchor=j
            elif a<b:break
    return order

def physical_indices(shape,strides,offset):
    n=1
    for d in shape:n*=d
    order=axis_order(shape,strides)
    for i in range(n):
        at=offset
        for axis in order:
            at+=(i%shape[axis])*strides[axis];i//=shape[axis]
        yield at
