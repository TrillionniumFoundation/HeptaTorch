block f16_pair_convert(tcs:mutbuf,tcm:u64,tcx:u64,tcy:u64,tco:u64,tcw:u64,tcn:u64,tci:u64,tcword:u64,tcmode:bool,tcbool:bool,tcsrc:u64,tctgt:u64,tcsnode:u64,tcsp:u64)
  f16_tc:u64 = const() {"value":5}
  f32_tc:u64 = const() {"value":0}
  from_f16:bool = compare.eq(tcsrc,f16_tc)
  into_f32:bool = compare.eq(tctgt,f32_tc)
  f16_match:bool = bool.and(from_f16,into_f32)
  branch f16_match f16_to_f32(tcs,tcm,tcx,tcy,tco,tcw,tcn,tci,tcword,tcmode,tcsnode,tcsp) convert_bool(tcs,tcm,tcx,tcy,tco,tcw,tcn,tci,tcword,tcmode,tcbool,tctgt)
block f16_to_f32(fcs:mutbuf,fcm:u64,fcx:u64,fcy:u64,fco:u64,fcw:u64,fcn:u64,fci:u64,fcword:u64,fcmode:bool,fcsnode:u64,fcsp:u64)
  selected_desc:u64 = call(fcs,fcsnode) {"function":"dptr"}
  selected_count_off:u64 = const() {"value":8}
  selected_count_pos:u64 = int.add.wrap(selected_desc,selected_count_off)
  original_n:u64 = call(fcs,selected_count_pos) {"function":"getw"}
  grain_limit:u64 = const() {"value":32768}
  in_profile:bool = compare.lt(original_n,grain_limit)
  in_source:bool = compare.lt(fcsp,original_n)
  prove_source:bool = bool.and(in_profile,in_source)
  branch prove_source f16_profile_store(fcs,fcm,fcx,fcy,fco,fcw,fcn,fci,fcword,fcmode,fcsp,original_n) bad(fcs)
block f16_profile_store(zqs:mutbuf,zqm:u64,zqx:u64,zqy:u64,zqo:u64,zqw:u64,zqn:u64,zqi:u64,zqword:u64,zqmode:bool,zqsource:u64,zqcount:u64)
  f32_word:u64 = call(zqword,zqsource,zqcount) {"function":"where_f16_fbgemm_profile_word"}
  jump store(zqs,zqm,zqx,zqy,zqo,zqw,zqn,zqi,f32_word,zqmode)
