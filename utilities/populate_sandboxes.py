import os
import datetime as dt
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cadspy import *

pd.options.display.max_rows = 1000

########################################################################################################

""" This script populates the sandboxes with appropriate data, based upon a specified
    train/test period (date range). The script checks which data is already in the sandboxes,
    and only adds data where it is required so as to be maximally efficient. Therefore, if the
    sandboxes are already populated with the appropriate data, then this script may do nothing at all. """

###############################################################################################################
# Parse arguments

parser=argparse.ArgumentParser()
parser.add_argument('train_date_range',type=str,nargs=2,help="Train set date range in 'YYYY-mm-dd' format as two space-separated values, e.g. '2018-06-01 2018-06-01'")
parser.add_argument('test_date_range',type=str,nargs=2,help="Test set date range in 'YYYY-mm-dd' format as two space-separated values, e.g. '2018-06-01 2018-06-01'")
parser.add_argument('-bkg_periods',type=int,nargs=2,default=[1,4],help="Recent booking periods (booking 'lookback') - Creates last x weeks' worth of booking variables (i.e. the LW and L4W variables). If you wanted to add, say, a 'last 6 weeks' variable (L6W) too, then this would be [1,4,6] etc.")
parser.add_argument('-fcst_DTD',type=int,nargs=2,default=[7,42],help="fcst DTD is the range of DTDs that you want to forecast for, e.g. '-fcst_DTD 7 42'")
args=parser.parse_args()

train_date_range=args.train_date_range
test_date_range=args.test_date_range
bkg_periods=args.bkg_periods
fcst_DTD=args.fcst_DTD

# Periods to train model on.
# Dates should be in Teradata format i.e. 'yyyy-mm-dd'
train_prd = [ train_date_range[0],train_date_range[1] ]
test_prd = [ test_date_range[0],test_date_range[1] ]

###############################################################################################################

def get_snap_dates(flight_date):
    """ Function that calculates all of the required snapshot dates associated
    with a given provided flight date. This includes both the recent snapshots
    leading up to the flight, and the 'historic' snapshots from 1 year previous.
    The function returns a list of datetime objects, which are all of the required snapshot dates.
    
    The supplied argument 'flight_date' should be a datetime object (departure date of flight).

    The earliest required date for the 'recent' snapshots is:

        flight_date - dt.timedelta(days=fcst_DTD[1]+max(bkg_periods)*7)

    And the latest required date for the 'recent' snapshots is:

        flight_date - dt.timedelta(days=fcst_DTD[0])

    We then fill all of the days in-between these two extremal dates, which gives us all
    of the required 'recent' snapshot dates. We then do a similar analysis for the historic data;
    the earliest required date for the 'historic' snapshots is: 

        flight_date - dt.timedelta(days=365) - dt.timedelta(days=fcst_DTD[1]+max(bkg_periods)*7)
    
    And finally, the latest required date for the 'historic' snapshots is simply:

        flight_date - dt.timedelta(days=365)

    Since for the historic data, we have the luxury of watching all the way up to the actual flight date. """

    # Process the 'recent' snapshopts first
    earliest_snap_date_recent = pd.to_datetime(flight_date,format='%Y-%m-%d') - dt.timedelta(days=fcst_DTD[1]+max(bkg_periods)*7)
    latest_snap_date_recent = pd.to_datetime(flight_date,format='%Y-%m-%d') - dt.timedelta(days=fcst_DTD[0])

    # Now process the 'historic' snapshopts
    earliest_snap_date_historic = pd.to_datetime(flight_date,format='%Y-%m-%d') - dt.timedelta(days=365) - dt.timedelta(days=fcst_DTD[1]+max(bkg_periods)*7)
    latest_snap_date_historic = pd.to_datetime(flight_date,format='%Y-%m-%d') - dt.timedelta(days=365)

    all_recent_dates = pd.date_range(earliest_snap_date_recent,latest_snap_date_recent)
    all_historic_dates = pd.date_range(earliest_snap_date_historic,latest_snap_date_historic)

    all_relevant_dates = all_recent_dates.union(all_historic_dates)

    return all_relevant_dates

# Get all the flight dates in the requested range
all_flight_dates = pd.date_range(train_prd[0],test_prd[1])

# Now calculate all the relevant snapshots for these flight dates
all_snaps = []
for date in all_flight_dates:
    for snap in get_snap_dates(date):
        all_snaps.append(snap)

# There will be duplicates, so get the unique set
all_snaps = pd.unique(all_snaps)

# The list 'all_snaps' now contains all relevant snapshots that need to be considered.
# In other words, if every snapshot in 'all_snaps' is added to the sandboxes, then the dataset should be complete.

# ### 2. ICW Data Pulls

# ### Quite a few queries here and the model pulls back more than just train and test periods to create base data
#     1. Held bookings at multiple snapshots in time
#     2. Flown data
#     3. More to come - including competitor and RevMan demand information

icw = DatabaseConnection()

# Download all the snapshot/flight date combinations from the table to see what we need to add.
# As discussed above, each snapshot is associated with an earliest departure date 'min(local_upl_dt)',
# and a latest departure date 'max(local_upl_dt)', which are dependent upon 'fcst_DTD' and 'bkg_periods'.

# The following query function finds the *currently held* minimum and maximum dates, and dumps them into a DataFrame
# ('dcheck') with three columns: 'SNAPSHOT_DT', 'min_lcl_upl_dt', and 'max_lcl_upl_dt' - Which are the (currently held)
# snapshot date, and the earliest and latest (respectively) departure dates associated with that snapshot.
# These will then be compared against what *should* be in the sandbox for each departure date, and data will be added if required.

def dcheck_query(tbl):
    return """
    select distinct
    snapshot_dt,
    min(local_upl_dt) as min_lcl_upl_dt,
    max(local_upl_dt) as max_lcl_upl_dt

    from {tbl_nm}

    group by 1
    """.format(tbl_nm = tbl)

dcheck = icw.queryToDataframe(dcheck_query('ldb_sbox_OR.TG_Held_Data_for_ML'))
dcheck.SNAPSHOT_DT = pd.to_datetime(dcheck['SNAPSHOT_DT'])
dcheck = dcheck.set_index('SNAPSHOT_DT')

# Define a 'dummy date' to make sure we don't pull anything in the TD (use Jan 1st 2000)
dont_pull = pd.to_datetime('2000-01-01')

# Now, populate the sandboxes: Compares min/max dates for each snapshot in the 'dcheck' DataFrame, and adds data if required
for snap in all_snaps:   
    # Start date should always be the snapshot date plus the minimum DTD we're interested in
    sdt = snap + dt.timedelta(days=fcst_DTD[0]-7) # JACK: Have added -7 so that we can look at the historic 'forward week' (the 'HIST_vFW' variable)
    # End date is always just the max DTD we're interested in - there will be some un-needed information pulled for the 
    # extreme snapshots but this should add minimal extra time, so leave in
    edt = snap + dt.timedelta(days=fcst_DTD[1]+max(bkg_periods)*7)
    if snap in dcheck.index:
        min_cur_dt = dcheck.loc[snap]['min_lcl_upl_dt']
        max_cur_dt = dcheck.loc[snap]['max_lcl_upl_dt']
        if  min_cur_dt > sdt:
            sdt1 = sdt
            edt1 = min_cur_dt - dt.timedelta(days=1)
        else:
            sdt1 = dont_pull
            edt1 = dont_pull
        if  max_cur_dt < edt:
            sdt2 = max_cur_dt + dt.timedelta(days=1)
            edt2 = edt
        else:
            sdt2 = dont_pull
            edt2 = dont_pull
    else: 
        sdt1 = sdt
        edt1 = edt
        sdt2 = dont_pull
        edt2 = dont_pull
    # Convert dates to string for teradata    
    sdt1 = sdt1.strftime("'%Y-%m-%d'")
    edt1 = edt1.strftime("'%Y-%m-%d'")
    sdt2 = sdt2.strftime("'%Y-%m-%d'")
    edt2 = edt2.strftime("'%Y-%m-%d'")
    snap = snap.strftime("'%Y-%m-%d'")
    

    Held_Data = """
    INSERT INTO ldb_sbox_OR.TG_Held_Data_for_ML 
     select
    fbss.snapshot_dt,
    fbss.leg_local_uplift_dt as LOCAL_UPL_DT,
    fbss.leg_operating_flt_no as OPG_FLT_NO,
    fbss.leg_uplift_stn_cd as upl_stn_cd,
    fbss.leg_discharge_stn_cd as dsg_stn_cd,
    fbss.BKG_CABIN_CD AS CBN_CD,
    SUM(CASE WHEN ahb.BUS_LEIS_SCORE > 0.5
                Then ZEROIFNULL(FBSS.REV_GROSS_VLU) - ZEROIFNULL(FBSS.REV_SPC_VLU)  - ZEROIFNULL(FBSS.REV_COMMISSION_VLU) + ZEROIFNULL(FBSS.YQ_TAX_VLU)
                 Else 0 END) as BUS_REV_OTX,
    SUM(CASE WHEN ahb.BUS_LEIS_SCORE le 0.5 or ahb.BUS_LEIS_SCORE is null
                Then ZEROIFNULL(FBSS.REV_GROSS_VLU) - ZEROIFNULL(FBSS.REV_SPC_VLU)  - ZEROIFNULL(FBSS.REV_COMMISSION_VLU) + ZEROIFNULL(FBSS.YQ_TAX_VLU)
                 Else 0 END) as LEIS_REV_OTX,
    SUM(ZEROIFNULL(FBSS.REV_GROSS_VLU) - ZEROIFNULL(FBSS.REV_SPC_VLU)  - ZEROIFNULL(FBSS.REV_COMMISSION_VLU) + ZEROIFNULL(FBSS.YQ_TAX_VLU)) as NET_REV_OTX,
    SUM(CASE WHEN ahb.BUS_LEIS_SCORE > 0.5 THEN pax_qty ELSE 0 END ) as BUS_PSJs,
    SUM(CASE WHEN ahb.BUS_LEIS_SCORE le 0.5 or ahb.BUS_LEIS_SCORE is null THEN pax_qty ELSE 0 END) as LEIS_PSJs,
    SUM(pax_qty) as PSJs

    from FBSS_DAILY_LEG fbss

    left join ah_booking ahb
    on ahb.bkg_order_id = fbss.bkg_order_id 

    where snapshot_dt = date {snapshot}
    /*Relevant snapshots only*/
    and leg_operating_airline_cd = 'BA'
    and (fbss.leg_local_uplift_dt between date {sdate1} and date {edate1}
        or fbss.leg_local_uplift_dt between date {sdate2} and date {edate2})

    group by 1,2,3,4,5,6
    """.format(sdate1 = sdt1, edate1 = edt1, sdate2 = sdt2, edate2 = edt2,  snapshot = snap)
    print('SNAP date:',snap)
    print('PRINT sdt1:',sdt1)
    print('PRINT edt1:',edt1)
    print('PRINT sdt2:',sdt2)
    print('PRINT edt2:',edt2)
    print('\n')
    icw.executeSQL(Held_Data, getResults=False)

# Get capacity and schedule data at various snapshots

# Pull data capacity with effective and expiry dates so we can join onto held
sdt = min(all_snaps)
sdt = sdt.strftime("'%Y-%m-%d'")
edt = max(all_snaps)
edt = edt.strftime("'%Y-%m-%d'")

def drop_table(tbl):
    drop_command = "DROP TABLE {tbl_nm}".format(tbl_nm = tbl)
    return drop_command

icw.executeSQL(drop_table('ldb_sbox_OR.TG_CAPACITY_Data_for_ML'), getResults=False)

capacity_query = """
CREATE TABLE ldb_sbox_OR.TG_CAPACITY_Data_for_ML AS (
SELECT
 SSFC.SCHED_DEP_DT as LOCAL_UPL_DT
,SSFC.OPERATING_FLT_NO as OPG_FLT_NO 
,SSFC.DEP_STN_CD as UPL_STN_CD
,SSFC.ARR_STN_CD as DSG_STN_CD
,SSFC.CABIN_CD as CBN_CD
,ssfc.effective_dt
,ssfc.expiry_dt
,SSFC.CAPACITY_QTY as CAPACITY

FROM SCHEDULE_SALES_FLT_CAPACITY SSFC

/*Join to AHUB to look at BA inventory controlled flights only*/
JOIN AH_FLIGHT_LEG ah
on ah.mktg_aln_cd = 'BA'
and ah.mktg_flt_no = ssfc.operating_flt_no
and ah.local_upl_dt = ssfc.sched_dep_dt
and ah.upl_stn_cd = ssfc.dep_stn_cd
and ah.dsg_stn_cd = ssfc.arr_stn_cd
and ah.OR_INV_CNTRL_IND = 1

WHERE SSFC.OPERATING_AIRLINE_CD  = 'BA'
/* Add in qualify statement to remove one flights where there are two flights on the same snapshot with same flt_dt, leg and flt_no*/
Qualify row_number() over (partition by LOCAL_UPL_DT,OPG_FLT_NO,UPL_STN_CD,DSG_STN_CD,CBN_CD,effective_dt,expiry_dt order by operating_sfx_cd) = 1
and SSFC.SCHED_DEP_DT between {sdate} and {edate}
and SSFC.EXPIRY_DT ge {sdate}
and SSFC.EFFECTIVE_DT le {edate}
AND SSFC.operating_sfx_cd in ('','Z')

) with data unique primary index (LOCAL_UPL_DT,OPG_FLT_NO,UPL_STN_CD,DSG_STN_CD,CBN_CD,effective_dt,expiry_dt)
""".format(sdate = sdt, edate = edt)

icw.executeSQL(capacity_query, getResults=False)

# ### Now pull schedule data for the train and test period at all relevant DTD - once again loop and insert to SBOX

for snap in all_snaps:
    # Start date should always be today plus the minimum DTD we're interested in 
    # Then convert to string using strftime
    sdt = snap + dt.timedelta(days=fcst_DTD[0])
    sdt = sdt.strftime("'%Y-%m-%d'")
    # End date is always just the max DTD we're interested in - there will be some uneeded information pulled for the 
    # extreme snapshots but should add minimal extra time so leave in
    # Then convert to string using strftime
    edt = snap + dt.timedelta(days=fcst_DTD[1])
    edt = edt.strftime("'%Y-%m-%d'")
    
    # Finally convert the snapshot into a string
    snap = snap.strftime("'%Y-%m-%d'")

    Schedule_query = """
    
    INSERT INTO ldb_sbox_OR.TG_SCHEDULE_Data_for_ML 
    SELECT
     date {snapshot} as Snapshot_dt
    ,SSFL.LOCAL_FLT_DT
    ,SSFL.SCHED_DEP_DT
    ,SSFL.GMT_FLT_DT
    ,SSFL.SCHED_DEP_TM 
    ,td_day_of_week(SSFL.SCHED_DEP_DT) as DOW_NO
    ,SSFL.OPERATING_FLT_NO as FLT_NO 
    ,SSFL.DEP_STN_CD as UPL_STN_CD
    ,SSFL.ARR_STN_CD as DSG_STN_CD
    ,haul_cd
    ,rrsp.stn_1_cd as HUB_STN_CD
    ,rrsp.stn_2_cd as OER_STN_CD

    FROM  SCHEDULE_SALES_FLT_LEG SSFL

    LEFT JOIN ref_ras_station_pair RRSP
    ON RRSP.DEP_STN_CD = SSFL.DEP_STN_CD
    AND RRSP.ARR_STN_CD = SSFL.ARR_STN_CD

    LEFT JOIN ref_station_haul RSH
    ON RSH.UPL_STN_CD = stn_1_cd
    AND RSH.DSG_STN_CD = stn_2_cd

    /*Join to AHUB to look at BA inventory controlled flights only*/
    JOIN AH_FLIGHT_LEG ah
    on ah.mktg_aln_cd = 'BA'
    and ah.mktg_flt_no = ssfl.operating_flt_no
    and ah.local_upl_dt = ssfl.sched_dep_dt
    and ah.upl_stn_cd = ssfl.dep_stn_cd
    and ah.dsg_stn_cd = ssfl.arr_stn_cd
    and ah.OR_INV_CNTRL_IND = 1

    WHERE SSFL.SCHED_DEP_DT between date {sdate} and date {edate}
    and SSFL.OPERATING_AIRLINE_CD  = 'BA'
    and date {snapshot} between ssfl.effective_dt and ssfl.expiry_dt
    /*Exclude non-commercial flights*/
    and SSFL.SERVICE_TYP = 'J'
    and SSFL.operating_sfx_cd in ('','Z')

    group by 1,2,3,4,5,6,7,8,9,10,11,12
    """.format(sdate = sdt, edate = edt, snapshot = snap)
    print('SNAP=',snap)
    print('START=',sdt)
    print('END=',edt)
    print('\n')
    icw.executeSQL(Schedule_query, getResults=False)

# ### Finally need flown passengers and capacity (and check flight wasn't cancelled)

# Flown passengers from DOC_COUPON
# Pull data capacity with effective and expiry dates so we can join onto held

# This query pulls flown passenger and revenue numbers from DOC_COUPON (tickets)
flown_pax_query = """
INSERT INTO ldb_sbox_OR.TG_FLOWN_PAX_Data_for_ML
select 
 UPL_DT as LOCAL_UPL_DT
,opg_aln_cd
,OPG_FLT_NO
,upl_stn_cd
,dsg_stn_cd
,OPG_CBN_CD as CBN_CD
,sum(GRS_REV_VLU - SPC_VLU + YQ_TAX_VLU) as net_yq_rev
,count(*) as pax

from doc_coupon dc

JOIN DOC_SALE DS
ON    DS.ALN_NO=DC.ALN_NO     
AND    DS.PRIME_DOC_id=DC.PRIME_DOC_id     
and ds.doc_typ_cd = 'TICKET'

where opg_aln_cd='BA'
and dc.REV_PAX_IND = 'Y' 
and CPN_USAGE_CD in ('T')
and DC.UPL_DT between {sdate} and {edate}

group by 1,2,3,4,5,6
""".format(sdate = sdt, edate = edt)

# This query shows capacity at departure and also checks if the flight actually departed
# Also take pax as sense check and in case we want to match operational pax
flown_cap_query = """
INSERT INTO ldb_sbox_OR.TG_FLOWN_CAP_Data_for_ML
select 
 FLT_DT as GMT_FLT_DT
,FLT_NO as OPG_FLT_NO
,DEP_STN_CD as UPL_stn_cd
,ARR_STN_CD as DSG_stn_cd
,CBN_CD
,AVL_SEAT_QTY as CAPACITY
,PAX_QTY as op_flown_pax

from TOFLGL tl

where aln_des_cd='BA'
and FLT_DT between {sdate} and {edate}
and aln_co_no = 1
and BUSN_TYP = 'PAX'
and cbn_cd in ('C','M')

""".format(sdate = sdt, edate = edt)

icw.executeSQL(flown_pax_query, getResults=False)
icw.executeSQL(flown_cap_query, getResults=False)

