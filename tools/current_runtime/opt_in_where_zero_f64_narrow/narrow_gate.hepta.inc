block rank0_f64_f32_convert(ncs:mutbuf,ncm:u64,ncx:u64,ncy:u64,nco:u64,ncw:u64,ncn:u64,nci:u64,ncword:u64,ncmode:bool,ncbool:bool,ncsrc:u64,nctgt:u64,ncsel:u64,ncsp:u64)
  nc_f64:u64 = const() {"value":4}
  nc_zero:u64 = const() {"value":0}
  nc_from:bool = compare.eq(ncsrc,nc_f64)
  nc_into:bool = compare.eq(nctgt,nc_zero)
  nc_pair:bool = bool.and(nc_from,nc_into)
  branch nc_pair rank0_f64_f32_check(ncs,ncm,ncx,ncy,nco,ncw,ncn,nci,ncword,ncmode,ncsel) i64_reduced_convert(ncs,ncm,ncx,ncy,nco,ncw,ncn,nci,ncword,ncmode,ncbool,ncsrc,nctgt,ncsel,ncsp)
block rank0_f64_f32_check(nqs:mutbuf,nqm:u64,nqx:u64,nqy:u64,nqo:u64,nqw:u64,nqn:u64,nqi:u64,nqword:u64,nqmode:bool,nqsel:u64)
  nq_desc:u64 = call(nqs,nqsel) {"function":"dptr"}
  nq_zero:u64 = const() {"value":0}
  nq_one:u64 = const() {"value":1}
  nq_eight:u64 = const() {"value":8}
  nq_rank:u64 = call(nqs,nq_desc) {"function":"getw"}
  nq_ca:u64 = int.add.wrap(nq_desc,nq_eight)
  nq_count:u64 = call(nqs,nq_ca) {"function":"getw"}
  nq_r0:bool = compare.eq(nq_rank,nq_zero)
  nq_c1:bool = compare.eq(nq_count,nq_one)
  nq_admit:bool = bool.and(nq_r0,nq_c1)
  branch nq_admit rank0_f64_f32_classify(nqs,nqm,nqx,nqy,nqo,nqw,nqn,nqi,nqword,nqmode) bad(nqs)
block rank0_f64_f32_classify(nks:mutbuf,nkm:u64,nkx:u64,nky:u64,nko:u64,nkw:u64,nkn:u64,nki:u64,nkword:u64,nkmode:bool)
  nk_52:u64 = const() {"value":52}
  nk_emask:u64 = const() {"value":2047}
  nk_fmask:u64 = const() {"value":4503599627370495}
  nk_zero:u64 = const() {"value":0}
  nk_up:u64 = int.shr.checked(nkword,nk_52)
  nk_exp:u64 = bit.and(nk_up,nk_emask)
  nk_frac:u64 = bit.and(nkword,nk_fmask)
  nk_special:bool = compare.eq(nk_exp,nk_emask)
  nk_nz:bool = compare.ne(nk_frac,nk_zero)
  nk_nan:bool = bool.and(nk_special,nk_nz)
  branch nk_nan rank0_f64_f32_nan(nks,nkm,nkx,nky,nko,nkw,nkn,nki,nkword,nkmode,nk_frac) rank0_f64_f32_rne(nks,nkm,nkx,nky,nko,nkw,nkn,nki,nkword,nkmode)
block rank0_f64_f32_nan(nas:mutbuf,nam:u64,nax:u64,nay:u64,nao:u64,naw:u64,nan_count:u64,nai:u64,naword:u64,namode:bool,nafrac:u64)
  na_32:u64 = const() {"value":32}
  na_29:u64 = const() {"value":29}
  na_signmask:u64 = const() {"value":2147483648}
  na_quiet:u64 = const() {"value":2143289344}
  na_signup:u64 = int.shr.checked(naword,na_32)
  na_sign:u64 = bit.and(na_signup,na_signmask)
  na_payload:u64 = int.shr.checked(nafrac,na_29)
  na_quiet_signed:u64 = bit.or(na_sign,na_quiet)
  na_bits:u64 = bit.or(na_quiet_signed,na_payload)
  jump store(nas,nam,nax,nay,nao,naw,nan_count,nai,na_bits,namode)
block rank0_f64_f32_rne(nrs:mutbuf,nrm:u64,nrx:u64,nry:u64,nro:u64,nrw:u64,nrn:u64,nri:u64,nrword:u64,nrmode:bool)
  nr_bits:u64 = call(nrword) {"function":"narrow64"}
  jump store(nrs,nrm,nrx,nry,nro,nrw,nrn,nri,nr_bits,nrmode)
