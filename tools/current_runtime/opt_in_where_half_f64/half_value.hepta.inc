block half_f64_convert(hcs:mutbuf,hcm:u64,hcx:u64,hcy:u64,hco:u64,hcw:u64,hcn:u64,hci:u64,hcword:u64,hcmode:bool,hcbool:bool,hcsrc:u64,hctgt:u64,hcselected:u64,hcsp:u64)
  hfcs:u64 = const() {"value":5}
  hbcs:u64 = const() {"value":6}
  hdct:u64 = const() {"value":4}
  hcfrom_f:bool = compare.eq(hcsrc,hfcs)
  hcfrom_b:bool = compare.eq(hcsrc,hbcs)
  hcinto_d:bool = compare.eq(hctgt,hdct)
  hcfok:bool = bool.and(hcfrom_f,hcinto_d)
  hcbok:bool = bool.and(hcfrom_b,hcinto_d)
  branch hcfok f16_to_f64(hcs,hcm,hcx,hcy,hco,hcw,hcn,hci,hcword,hcmode) half_f64_bf(hcs,hcm,hcx,hcy,hco,hcw,hcn,hci,hcword,hcmode,hcbool,hcsrc,hctgt,hcselected,hcsp,hcbok)
block half_f64_bf(hbs:mutbuf,hbm:u64,hbx:u64,hby:u64,hbo:u64,hbw:u64,hbn:u64,hbi:u64,hbword:u64,hbmode:bool,hbbool:bool,hbsrc:u64,hbtgt:u64,hbsel:u64,hbsp:u64,hbok:bool)
  branch hbok bf16_to_f64(hbs,hbm,hbx,hby,hbo,hbw,hbn,hbi,hbword,hbmode) bf_pair_convert(hbs,hbm,hbx,hby,hbo,hbw,hbn,hbi,hbword,hbmode,hbbool,hbsrc,hbtgt,hbsel,hbsp)
block f16_to_f64(hfs:mutbuf,hfm:u64,hfx:u64,hfy:u64,hfo:u64,hfw:u64,hfn:u64,hfi:u64,hfword:u64,hfmode:bool)
  hfword32:u64 = call(hfword) {"function":"widen_f16"}
  jump f32_to_f64(hfs,hfm,hfx,hfy,hfo,hfw,hfn,hfi,hfword32,hfmode)
block bf16_to_f64(hds:mutbuf,hdm:u64,hdx:u64,hdy:u64,hdo:u64,hdw:u64,hdn:u64,hdi:u64,hdword:u64,hdmode:bool)
  hdword32:u64 = call(hdword) {"function":"half_widen_bf16"}
  jump f32_to_f64(hds,hdm,hdx,hdy,hdo,hdw,hdn,hdi,hdword32,hdmode)
