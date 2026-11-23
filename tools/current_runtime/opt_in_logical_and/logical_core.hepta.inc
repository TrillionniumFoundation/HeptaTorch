fn la_error(lerr_s:mutbuf,lerr_status:u64) -> (u64) effects(read,write,trap) entry(la_fail)
block la_fail(lerr_s:mutbuf,lerr_status:u64)
  lerr_set:u64 = call(lerr_s,lerr_status) {"function":"terror"}
  ret(lerr_set)
end
fn la_canonical(lcan_s:mutbuf,lcan_desc:u64,lcan_rank:u64,lcan_count:u64) -> (u64) effects(read,write,trap) entry(la_canon)
block la_canon(lcan_s:mutbuf,lcan_desc:u64,lcan_rank:u64,lcan_count:u64)
  lcan_one:u64 = const() {"value":1}
  jump la_axis(lcan_s,lcan_desc,lcan_rank,lcan_one,lcan_count)
block la_axis(lax_s:mutbuf,lax_desc:u64,lax_rank:u64,lax_prod:u64,lax_count:u64)
  lax_z:u64 = const() {"value":0}
  lax_more:bool = compare.gt(lax_rank,lax_z)
  branch lax_more la_axis_read(lax_s,lax_desc,lax_rank,lax_prod,lax_count,lax_z) la_check_prod(lax_prod,lax_count)
block la_axis_read(lar_s:mutbuf,lar_desc:u64,lar_rank:u64,lar_prod:u64,lar_count:u64,lar_z:u64)
  lar_one:u64 = const() {"value":1}
  lar_eight:u64 = const() {"value":8}
  lar_shape:u64 = const() {"value":96}
  lar_stride:u64 = const() {"value":128}
  lar_cap:u64 = const() {"value":262144}
  lar_prev:u64 = int.sub.wrap(lar_rank,lar_one)
  lar_base:u64 = int.mul.wrap(lar_prev,lar_eight)
  lar_sd:u64 = int.add.wrap(lar_desc,lar_shape)
  lar_td:u64 = int.add.wrap(lar_desc,lar_stride)
  lar_sa:u64 = int.add.wrap(lar_sd,lar_base)
  lar_ta:u64 = int.add.wrap(lar_td,lar_base)
  lar_dim:u64 = call(lar_s,lar_sa) {"function":"getw"}
  lar_st:u64 = call(lar_s,lar_ta) {"function":"getw"}
  lar_nonempty:bool = compare.gt(lar_dim,lar_z)
  lar_small:bool = compare.le(lar_dim,lar_cap)
  lar_exact:bool = compare.eq(lar_st,lar_prod)
  lar_bounded:bool = bool.and(lar_nonempty,lar_small)
  lar_ok:bool = bool.and(lar_bounded,lar_exact)
  branch lar_ok la_axis_product(lar_s,lar_desc,lar_prev,lar_prod,lar_count,lar_dim,lar_cap) la_canon_bad()
block la_axis_product(lap_s:mutbuf,lap_desc:u64,lap_prev:u64,lap_prod:u64,lap_count:u64,lap_dim:u64,lap_cap:u64)
  lap_new:u64 = int.mul.wrap(lap_prod,lap_dim)
  lap_ok:bool = compare.le(lap_new,lap_cap)
  branch lap_ok la_axis(lap_s,lap_desc,lap_prev,lap_new,lap_count) la_canon_bad()
block la_check_prod(lac_prod:u64,lac_count:u64)
  lac_one:u64 = const() {"value":1}
  lac_z:u64 = const() {"value":0}
  lac_same:bool = compare.eq(lac_prod,lac_count)
  lac_reply:u64 = select(lac_same,lac_one,lac_z)
  ret(lac_reply)
block la_canon_bad()
  lacb_z:u64 = const() {"value":0}
  ret(lacb_z)
end
fn la_tensor(la_s:mutbuf,la_x:u64,la_y:u64) -> (u64) effects(read,write,trap) entry(la_admit)
block la_admit(la_s:mutbuf,la_x:u64,la_y:u64)
  la_xt:u64 = call(la_s,la_x) {"function":"tensor_dtype"}
  la_yt:u64 = call(la_s,la_y) {"function":"tensor_dtype"}
  la_private:u64 = const() {"value":3}
  la_last:u64 = const() {"value":6}
  la_same:bool = compare.eq(la_xt,la_yt)
  la_public:bool = compare.ne(la_xt,la_private)
  la_bounded:bool = compare.le(la_xt,la_last)
  la_tag_ok:bool = bool.and(la_public,la_bounded)
  la_ok:bool = bool.and(la_tag_ok,la_same)
  branch la_ok la_capacity(la_s,la_x,la_y,la_xt) la_bad(la_s)
block la_capacity(lca_s:mutbuf,lca_x:u64,lca_y:u64,lca_t:u64)
  lca_cap:u64 = const() {"value":262144}
  lca_eight:u64 = const() {"value":8}
  lca_xd:u64 = call(lca_s,lca_x) {"function":"dptr"}
  lca_yd:u64 = call(lca_s,lca_y) {"function":"dptr"}
  lca_xa:u64 = int.add.wrap(lca_xd,lca_eight)
  lca_ya:u64 = int.add.wrap(lca_yd,lca_eight)
  lca_n:u64 = call(lca_s,lca_xa) {"function":"getw"}
  lca_m:u64 = call(lca_s,lca_ya) {"function":"getw"}
  lca_xsmall:bool = compare.le(lca_n,lca_cap)
  lca_ysmall:bool = compare.le(lca_m,lca_cap)
  lca_ok:bool = bool.and(lca_xsmall,lca_ysmall)
  branch lca_ok la_inputs(lca_s,lca_x,lca_y,lca_t,lca_xd,lca_yd,lca_n,lca_m) la_too_big(lca_s)
block la_inputs(li_s:mutbuf,li_x:u64,li_y:u64,li_t:u64,li_xd:u64,li_yd:u64,li_n:u64,li_m:u64)
  li_z:u64 = const() {"value":0}
  li_w:u64 = call(li_t) {"function":"dtype_width"}
  li_xok:u64 = call(li_s,li_x,li_w) {"function":"la_ordinary"}
  li_yok:u64 = call(li_s,li_y,li_w) {"function":"la_ordinary"}
  li_both:u64 = bit.and(li_xok,li_yok)
  li_owned:bool = compare.ne(li_both,li_z)
  li_equal:bool = compare.eq(li_n,li_m)
  li_nonempty:bool = compare.gt(li_n,li_z)
  li_count_ok:bool = bool.and(li_equal,li_nonempty)
  li_ok:bool = bool.and(li_owned,li_count_ok)
  branch li_ok la_geometry(li_s,li_x,li_y,li_t,li_xd,li_yd,li_n) la_bad(li_s)
block la_geometry(lg_s:mutbuf,lg_x:u64,lg_y:u64,lg_t:u64,lg_xd:u64,lg_yd:u64,lg_n:u64)
  lg_z:u64 = const() {"value":0}
  lg_xr:u64 = call(lg_s,lg_xd) {"function":"getw"}
  lg_yr:u64 = call(lg_s,lg_yd) {"function":"getw"}
  lg_same:u64 = call(lg_s,lg_x,lg_y) {"function":"incoming_same_shape"}
  lg_a:u64 = call(lg_s,lg_xd,lg_xr,lg_n) {"function":"la_canonical"}
  lg_b:u64 = call(lg_s,lg_yd,lg_yr,lg_n) {"function":"la_canonical"}
  lg_both:u64 = bit.and(lg_a,lg_b)
  lg_all:u64 = bit.and(lg_both,lg_same)
  lg_ok:bool = compare.ne(lg_all,lg_z)
  branch lg_ok la_allocate(lg_s,lg_x,lg_y,lg_t,lg_xd,lg_xr,lg_n) la_bad(lg_s)
block la_allocate(llo_s:mutbuf,llo_x:u64,llo_y:u64,llo_t:u64,llo_xd:u64,llo_rank:u64,llo_n:u64)
  llo_z:u64 = const() {"value":0}
  llo_one:u64 = const() {"value":1}
  llo_b:u64 = const() {"value":96}
  llo_src:u64 = int.add.wrap(llo_xd,llo_b)
  llo_out:u64 = call(llo_s,llo_rank,llo_src,llo_z,llo_one,llo_one) {"function":"typed_descriptor"}
  llo_ok:bool = compare.ne(llo_out,llo_z)
  branch llo_ok la_output(llo_s,llo_x,llo_y,llo_t,llo_out,llo_n) la_zero()
block la_output(lot_s:mutbuf,lot_x:u64,lot_y:u64,lot_t:u64,lot_out:u64,lot_n:u64)
  lot_z:u64 = const() {"value":0}
  lot_one:u64 = const() {"value":1}
  lot_live:u64 = call(lot_s,lot_out,lot_one) {"function":"la_ordinary"}
  lot_type:u64 = call(lot_s,lot_out) {"function":"tensor_dtype"}
  lot_same:u64 = call(lot_s,lot_x,lot_out) {"function":"incoming_same_shape"}
  lot_tags:bool = compare.eq(lot_type,lot_one)
  lot_root:bool = compare.ne(lot_live,lot_z)
  lot_geometry:bool = compare.ne(lot_same,lot_z)
  lot_owned:bool = bool.and(lot_tags,lot_root)
  lot_ok:bool = bool.and(lot_geometry,lot_owned)
  branch lot_ok la_output_count(lot_s,lot_x,lot_y,lot_t,lot_out,lot_n) la_bad(lot_s)
block la_output_count(loc_s:mutbuf,loc_x:u64,loc_y:u64,loc_t:u64,loc_out:u64,loc_n:u64)
  loc_z:u64 = const() {"value":0}
  loc_eight:u64 = const() {"value":8}
  loc_d:u64 = call(loc_s,loc_out) {"function":"dptr"}
  loc_r:u64 = call(loc_s,loc_d) {"function":"getw"}
  loc_caddr:u64 = int.add.wrap(loc_d,loc_eight)
  loc_count:u64 = call(loc_s,loc_caddr) {"function":"getw"}
  loc_same:bool = compare.eq(loc_count,loc_n)
  loc_can:u64 = call(loc_s,loc_d,loc_r,loc_count) {"function":"la_canonical"}
  loc_canon:bool = compare.ne(loc_can,loc_z)
  loc_ok:bool = bool.and(loc_same,loc_canon)
  branch loc_ok la_loop(loc_s,loc_x,loc_y,loc_t,loc_out,loc_n,loc_z) la_bad(loc_s)
block la_loop(llp_s:mutbuf,llp_x:u64,llp_y:u64,llp_t:u64,llp_out:u64,llp_n:u64,llp_i:u64)
  llp_more:bool = compare.lt(llp_i,llp_n)
  branch llp_more la_read(llp_s,llp_x,llp_y,llp_t,llp_out,llp_n,llp_i) la_done(llp_out)
block la_read(lrd_s:mutbuf,lrd_x:u64,lrd_y:u64,lrd_t:u64,lrd_out:u64,lrd_n:u64,lrd_i:u64)
  lrd_z:u64 = const() {"value":0}
  lrd_one:u64 = const() {"value":1}
  lrd_a:u64 = call(lrd_s,lrd_x,lrd_i) {"function":"tget"}
  lrd_b:u64 = call(lrd_s,lrd_y,lrd_i) {"function":"tget"}
  lrd_error:u64 = call(lrd_s,lrd_z) {"function":"getw"}
  lrd_ok:bool = compare.eq(lrd_error,lrd_z)
  lrd_is_bool:bool = compare.eq(lrd_t,lrd_one)
  lrd_left:bool = compare.le(lrd_a,lrd_one)
  lrd_right:bool = compare.le(lrd_b,lrd_one)
  lrd_both:bool = bool.and(lrd_left,lrd_right)
  lrd_raw_ok:bool = select(lrd_is_bool,lrd_both,lrd_ok)
  lrd_ready:bool = bool.and(lrd_ok,lrd_raw_ok)
  branch lrd_ready la_store(lrd_s,lrd_x,lrd_y,lrd_t,lrd_out,lrd_n,lrd_i,lrd_a,lrd_b) la_read_failure(lrd_s,lrd_error)
block la_store(lst_s:mutbuf,lst_x:u64,lst_y:u64,lst_t:u64,lst_out:u64,lst_n:u64,lst_i:u64,lst_a:u64,lst_b:u64)
  lst_z:u64 = const() {"value":0}
  lst_one:u64 = const() {"value":1}
  lst_l:u64 = call(lst_a,lst_t) {"function":"logical_truth"}
  lst_r:u64 = call(lst_b,lst_t) {"function":"logical_truth"}
  lst_bits:u64 = bit.and(lst_l,lst_r)
  lst_at:u64 = call(lst_s,lst_out,lst_i) {"function":"taddr"}
  lst_put:u64 = call(lst_s,lst_at,lst_one,lst_bits) {"function":"typed_store"}
  lst_ready:bool = compare.ne(lst_put,lst_z)
  lst_next:u64 = int.add.wrap(lst_i,lst_one)
  branch lst_ready la_loop(lst_s,lst_x,lst_y,lst_t,lst_out,lst_n,lst_next) la_zero()
block la_read_failure(lrf_s:mutbuf,lrf_error:u64)
  lrf_z:u64 = const() {"value":0}
  lrf_is_clear:bool = compare.eq(lrf_error,lrf_z)
  branch lrf_is_clear la_bad(lrf_s) la_zero()
block la_done(ldone_out:u64)
  ret(ldone_out)
block la_zero()
  lzero:u64 = const() {"value":0}
  ret(lzero)
block la_too_big(ltb_s:mutbuf)
  ltb_code:u64 = const() {"value":3}
  ltb_set:u64 = call(ltb_s,ltb_code) {"function":"la_error"}
  ret(ltb_set)
block la_bad(lbad_s:mutbuf)
  lbad_code:u64 = const() {"value":7}
  lbad_set:u64 = call(lbad_s,lbad_code) {"function":"la_error"}
  ret(lbad_set)
end
