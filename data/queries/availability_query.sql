SELECT
fk.inventory_flt_no as OPG_FLT_NO,
fi.LOCAL_FLT_DT as LOCAL_FLT_DT,
d.chkpoint as DTD,
fi.SEATS_AVL_QTY as SEATS_AVL



FROM flight_inventory fi



JOIN flight_keys fk
on fk.flt_seq_no = fi.flt_seq_no



JOIN days_to_departure d
on fi.local_flt_dt - d.chkpoint between fi.eff_dt and fi.exp_dt - 1
and d.chkpoint between 0 and 90




WHERE
fk.inventory_carrier_cd = 'BA'
/* Change date range etc as appropriate. */
and fi.local_flt_dt between date '2018-03-01' and date '2020-03-31'
and fi.seg_or_leg_cd = 'S'
and fk.sub_cls_cd = 7 and fk.cls_cd = 'X'
/*SH LHR+LGW flight numbers*/
and (fk.inventory_flt_no between 300 and 1999 or fk.inventory_flt_no between 2500 and 2999)

AND

/* To save space, you can restrict DTD range and only use DTDs that are multiples of 7.
   I've commented this out for now. */
/*
DTD <= 42 AND DTD >= 7
AND
(DTD MOD 7)
*/



order by LOCAL_FLT_DT, OPG_FLT_NO, DTD