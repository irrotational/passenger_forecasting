/*There may be some cabins missing but these will be added in when we add capacity*/
sel sd.*,
capd.CBN_CD,
capd.CAPACITY,
capdh.CAPACITY as CAPACITY_LY,
fc.CAPACITY as FLOWN_CAPACITY,
fc.op_flown_pax,
fp.pax as flown_pax,
zeroifnull(hd.bus_PSJs) as HELD_BUS_PSJs,
zeroifnull(hd.PSJs) as HELD_PSJs,
zeroifnull(hd.bus_PSJs) - zeroifnull(hdlw4.bus_PSJs) as HELD_BUS_PSJs_L4W,
zeroifnull(hd.PSJs) - zeroifnull(hdlw4.PSJs) as HELD_PSJs_L4W,
zeroifnull(hd.bus_PSJs) - zeroifnull(hdlw.bus_PSJs) as HELD_BUS_PSJs_LW,
zeroifnull(hd.PSJs) - zeroifnull(hdlw.PSJs) as HELD_PSJs_LW,
case when capdh.CAPACITY is null Then null Else zeroifnull(hdhist.bus_PSJs) END as HELD_BUS_PSJs_HIST,
case when capdh.CAPACITY is null Then null Else zeroifnull(hdhist.PSJs) END  as HELD_PSJs_HIST,
case when capdh.CAPACITY is null Then null Else zeroifnull(hdhist.bus_PSJs) - zeroifnull(hdlw4h.bus_PSJs) END as HELD_BUS_PSJs_L4W_HIST,
case when capdh.CAPACITY is null Then null Else zeroifnull(hdhist.PSJs) - zeroifnull(hdlw4h.PSJs) END as HELD_PSJs_L4W_HIST,
case when capdh.CAPACITY is null Then null Else zeroifnull(hdhist.bus_PSJs) - zeroifnull(hdlwh.bus_PSJs) END as HELD_BUS_PSJs_LW_HIST,
case when capdh.CAPACITY is null Then null Else zeroifnull(hdhist.PSJs) - zeroifnull(hdlwh.PSJs) END as HELD_PSJs_LW_HIST

/*Pull schedule data from sandbox generated from loops in Python code*/
from ldb_sbox_OR.TG_SCHEDULE_Data_for_ML sd

/*Pull capacity data from sandbox generated from loops in Python code*/
JOIN ldb_sbox_OR.TG_CAPACITY_Data_for_ML capd
on sd.snapshot_dt between capd.effective_dt and capd.expiry_dt
and sd.LOCAL_DEP_DT = capd.LOCAL_UPL_DT
and sd.OPG_FLT_NO = capd.OPG_FLT_NO
and sd.DSG_STN_CD = capd.DSG_STN_CD
and sd.UPL_STN_CD = capd.UPL_STN_CD

/*Retrieve flown capacity - also check filter out flights that ended up being cancelled*/
JOIN ldb_sbox_or.TG_FLOWN_CAP_DATA_FOR_ML fc
on fc.gmt_flt_dt = sd.gmt_flt_dt
and fc.opg_flt_no = sd.opg_flt_no
and fc.upl_stn_cd = sd.upl_stn_cd
and fc.dsg_stn_cd = sd.dsg_stn_cd
and capd.cbn_cd = fc.cbn_cd

/*Retrieve flown pax from DOC_COUPON*/
LEFT JOIN ldb_sbox_or.TG_FLOWN_PAX_DATA_FOR_ML fp
on fp.LOCAL_UPL_DT = sd.gmt_flt_dt
and fp.opg_flt_no = sd.opg_flt_no
and fp.upl_stn_cd = sd.upl_stn_cd
and fp.dsg_stn_cd = sd.dsg_stn_cd
and capd.cbn_cd = fp.cbn_cd

/*Pull capacity data for last year from sandbox generated from loops in Python code*/
/*Left join as there may not be an equivalent flight TY*/
LEFT JOIN ldb_sbox_OR.TG_CAPACITY_Data_for_ML capdh
/*Equiv snapshot in 2019*/
on sd.snapshot_dt between capdh.effective_dt and capdh.expiry_dt
and sd.local_dep_dt = capdh.LOCAL_UPL_DT
and sd.OPG_FLT_NO = capdh.OPG_FLT_NO
and sd.DSG_STN_CD = capdh.DSG_STN_CD
and sd.UPL_STN_CD = capdh.UPL_STN_CD
and capd.CBN_CD = capdh.CBN_CD

/*Cains from HD will be duplicated when capacity is added*/
/*Left join as there may be 0 bookings*/
LEFT JOIN ldb_sbox_OR.TG_Held_Data_for_ML hd
on sd.snapshot_dt = hd.snapshot_dt
and sd.LOCAL_DEP_DT = hd.LOCAL_UPL_DT
and sd.OPG_FLT_NO = hd.OPG_FLT_NO
and sd.DSG_STN_CD = hd.DSG_STN_CD
and sd.UPL_STN_CD = hd.UPL_STN_CD
and capd.CBN_CD = hd.CBN_CD

/*Join to this time last year*/
/*Left join as there may be 0 bookings*/
LEFT JOIN ldb_sbox_OR.TG_Held_Data_for_ML hdhist
on sd.snapshot_dt = hdhist.snapshot_dt
and sd.local_dep_dt = hdhist.LOCAL_UPL_DT
and sd.OPG_FLT_NO = hdhist.OPG_FLT_NO
and sd.DSG_STN_CD = hdhist.DSG_STN_CD
and sd.UPL_STN_CD = hdhist.UPL_STN_CD
/*Join on cabin code when both are in held tables otherwise join on anyway creating a new row*/
and capd.CBN_CD = hdhist.CBN_CD

/*Join to held bookings to get the number of bookings x weeks ago*/
LEFT JOIN ldb_sbox_OR.TG_Held_Data_for_ML hdlw
on sd.snapshot_dt = hdlw.snapshot_dt + 7
and sd.LOCAL_DEP_DT = hdlw.LOCAL_UPL_DT
and sd.OPG_FLT_NO = hdlw.OPG_FLT_NO
and sd.DSG_STN_CD = hdlw.DSG_STN_CD
and sd.UPL_STN_CD = hdlw.UPL_STN_CD
and capd.CBN_CD = hdlw.CBN_CD

/*Join to held bookings to get the number of bookings x weeks ago*/
LEFT JOIN ldb_sbox_OR.TG_Held_Data_for_ML hdlw4
on sd.snapshot_dt = hdlw4.snapshot_dt + 28
and sd.LOCAL_DEP_DT = hdlw4.LOCAL_UPL_DT
and sd.OPG_FLT_NO = hdlw4.OPG_FLT_NO
and sd.DSG_STN_CD = hdlw4.DSG_STN_CD
and sd.UPL_STN_CD = hdlw4.UPL_STN_CD
and capd.CBN_CD = hdlw4.CBN_CD

/*Join to held bookings to get the number of bookings x weeks ago hist*/
LEFT JOIN ldb_sbox_OR.TG_Held_Data_for_ML hdlwh
on sd.snapshot_dt = hdlwh.snapshot_dt + 7
and sd.local_dep_dt = hdlwh.LOCAL_UPL_DT
and sd.OPG_FLT_NO = hdlwh.OPG_FLT_NO
and sd.DSG_STN_CD = hdlwh.DSG_STN_CD
and sd.UPL_STN_CD = hdlwh.UPL_STN_CD
and capd.CBN_CD = hdlwh.CBN_CD

/*Join to held bookings to get the number of bookings x weeks ago hist*/
LEFT JOIN ldb_sbox_OR.TG_Held_Data_for_ML hdlw4h
on sd.snapshot_dt = hdlw4h.snapshot_dt + 28
and sd.local_dep_dt = hdlw4h.LOCAL_UPL_DT
and sd.OPG_FLT_NO = hdlw4h.OPG_FLT_NO
and sd.DSG_STN_CD = hdlw4h.DSG_STN_CD
and sd.UPL_STN_CD = hdlw4h.UPL_STN_CD
and capd.CBN_CD = hdlw4h.CBN_CD
order by 1,2,7,8

/* Restrict to short haul only. */
WHERE
HAUL_CD = 'S'

