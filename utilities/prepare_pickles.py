import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import datetime as dt
import argparse

CODE_START_TIME = time.time()

###############################################################################################################
# Set Pandas options, globally useful stuff, etc

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100000
pd.options.mode.chained_assignment = None

cwd=os.getcwd()

# Path separator (different for Linux & Windows)
pthsep = os.path.sep

###############################################################################################################

""" This script reads in one or more .csv files, each of which should contain data in the same format
    as those datsets in the '\data\csvs' folder. Note that availability data ('availability.txt') and
    competitor data ('competitor_data.txt') are contained in separate files to the snapshot data. Again,
    if you add data to these files, the added data should be compliant with the format of those files. This
    shouldn't be too difficult, as I've supplied the exact SQL queries used to get the availability and 
    competitor data in the '\data\queries' folder.

    If you're pickling data for the purposes of forecasting, then obviously you won't have a value for the
    'flown_pax' and 'op_flown_pax' columns as these flights haven't happened yet. This script will still expect
    these columns to be present however (so as to have a standardised input format). In this case, just fill the
    'flown_pax' and 'op_flown_pax' columns with any values (e.g. 999) - This is completely safe as, at the
    forecasting stage (i.e. using the 'prepare_forcasts.py' script), these columns are ignored completely.

    Once this script has read=in the .csv file(s), it merges, manipulates and cleans the data. If successful, it
    will save one file ('CLEANED.pkl') to the current working directory, which contains the cleaned dataset, ready
    to be fed into the 'construct_train_test_pickles.py' script (please also read the documentation in that script). """

###############################################################################################################
# Parse arguments

parser=argparse.ArgumentParser()
parser.add_argument('csv_files',type=str,nargs='+',help="One or more csv files, located in the current directory. Supply as a space-separated list.")
parser.add_argument('-sampling_frequency',type=int,default=7,help="""How often to sample DTDs for flights. For example, assuming that your dataset's snapshots
                                                                     go up to a week out from departure day, if you specify '-sampling_frequency 7',
                                                                     the code will use [7,14,21,28,35]. If you specify '-sampling_frequency 3', the code
                                                                     will use [7,10,13,16,19,21,24,...] etc. By default, the code uses a sampling frequency
                                                                     of 7 (i.e. use only weekly snapshots).""")
args=parser.parse_args()

csv_files=args.csv_files # This is a list
sampling_frequency=args.sampling_frequency

###############################################################################################################
# Reading-in CSVs

# Loop over supplied csv file(s)
dfs = []
for file in csv_files:
    df = pd.read_csv(cwd+pthsep+file,delimiter=',',low_memory=False,na_values=['?'])
    dfs.append(df)

# Concatenate all the DataFrames together & sort by 'Snapshot_dt','LOCAL_DEP_DT','OPG_FLT_NO'
df_main = pd.concat(dfs)
df_main.sort_values(by=['Snapshot_dt','LOCAL_DEP_DT','OPG_FLT_NO'],inplace=True)

# Drop NaNs
df_main.dropna(inplace=True)

###############################################################################################################
# Add a 'Days to Departure' column (DTD), which is a datetime timedelta object.
# Also convert all datelike/timelike columns ('SCHED_DEP_TM' etc) to datetime objects

# NOTE: In future, you may wish to just add-in the DTD at the SQL query stage, i.e. something
#       like 'SELECT (LOCAL_DEP_DT-Snapshot_dt) AS DTD'. This will remove the need for the next
#       few lines of code.

dt_format = '%d/%m/%Y' # Most columns have this format
sched_dep_tm_format = '%d/%m/%Y %H:%M:%S' # This particular column also has time in addition to date
df_main['SCHED_DEP_TM'] = pd.to_datetime(df_main['SCHED_DEP_TM'],format=sched_dep_tm_format)
df_main['GMT_FLT_DT'] = pd.to_datetime(df_main['GMT_FLT_DT'],format=dt_format)
df_main['LOCAL_FLT_DT'] = pd.to_datetime(df_main['LOCAL_FLT_DT'],format=dt_format)
df_main['LOCAL_DEP_DT'] = pd.to_datetime(df_main['LOCAL_DEP_DT'],format=dt_format)
df_main['Snapshot_dt'] = pd.to_datetime(df_main['Snapshot_dt'],format=dt_format)

df_main['DTD'] = df_main['LOCAL_DEP_DT'] - df_main['Snapshot_dt']

# Now cast the DTD column to type int (requires a special
# approach, as it's currently a timedelta object)
df_main['DTD'] = ( df_main['DTD'] / np.timedelta64(1,'D') ).astype('int16')

####################################################################
# Remove some fight numbers

# For some reason, some long haul flights are classed as SH (e.g. Tel Aviv, which is flown with an A350)
# You should only include flight nos. 300-1999 and 2500-2999 in your data, which will exclude LH flight numbers
# As a result of this filter, df should only contain 'M' or 'C' cabin codes (CBN_CD). If such flight numbers have
# already been excluded from the dataset, then this line will do nothing at all.

df_main = df_main.loc[ ( (df_main['OPG_FLT_NO'] >= 300) & (df_main['OPG_FLT_NO'] < 2000) ) | ( (df_main['OPG_FLT_NO'] < 3000) & (df_main['OPG_FLT_NO'] >= 2500) ) ]

####################################################################
# Read-in historical data (i.e. the actual passengers flown on the same flight last year)
# We added this in later on, so it's contained in a separate file.

# NOTE: If somebody edits this code in the future, it might be better to combine all the 'auxiliary'
# data that we added in late-on (historical data, competitor data, availability data, etc) into
# a single query and thus a single csv file. Then, the following join could be removed, as the
# historical data would already be in the DataFrame by now.

#try:
dirname = os.path.dirname(__file__)
df_hist = pd.read_csv(dirname+pthsep+".."+pthsep+"data"+pthsep+"csvs"+pthsep+"hist_flown.txt",delimiter=',',na_values=['?'])
#except:
#    print("ERROR: Couldn't find the file 'hist_flown.txt' in the 'csvs' folder. Please check that it is there.")
#    exit()
    
# Rename columns to be consistent with the main data before we attempt a join
df_hist.rename(columns={'LOCAL_UPL_DT':'LOCAL_DEP_DT','upl_stn_cd':'UPL_STN_CD'},inplace=True)
# Convert departure dates and historic dates to datetime objects
df_hist['LOCAL_DEP_DT'] = pd.to_datetime(df_hist['LOCAL_DEP_DT'],format=dt_format)
df_hist['LOCAL_UPL_DT_HIST'] = pd.to_datetime(df_hist['LOCAL_UPL_DT_HIST'],format=dt_format)
# Add the historical flight data to the main dataset in df via an inner join
df_main = df_main.merge(df_hist, how='left', on=['LOCAL_DEP_DT','OPG_FLT_NO','CBN_CD','UPL_STN_CD'])

####################################################################
# Join on macro_group_nm information, which classifies each
# destination into the relevant category ('Business', 'Multi', etc)

# Read-in file containing definitions for macro_group_names
try:
    dirname = os.path.dirname(__file__)
    df_macro_group_nms = pd.read_csv(dirname+pthsep+".."+pthsep+"data"+pthsep+"csvs"+pthsep+"sh_route_groups.txt",delimiter='\t',na_values=['?'])
except:
    print("ERROR: Couldn't find the file 'sh_route_groups.txt' in the 'csvs' folder. Please check that it is there.")
    exit()
    
# Drop first column, which just indexes each value with a number
df_macro_group_nms = df_macro_group_nms.iloc[: , 1:]
# Rename column to be consistent with the main data before we attempt a join
df_macro_group_nms.rename(columns={'dest':'DSG_STN_CD'},inplace=True)

df_main = df_main.merge(df_macro_group_nms,how='left',on=['DSG_STN_CD'])

####################################################################
# Join on competitor data

try:
    dirname = os.path.dirname(__file__)
    df_comp = pd.read_csv(dirname+pthsep+".."+pthsep+"data"+pthsep+"csvs"+pthsep+"competitor_data.txt",delimiter=',',na_values=['?'])
except:
    print("ERROR: Couldn't find the file 'competitor_data.txt' in the 'csvs' folder. Please check that it is there.")
    exit()
    
# Cast column as datetime object
sched_dep_tm_format = '%d/%m/%Y %H:%M:%S' # This column also has time in addition to date
# Rename column to be consistent with the main data before we attempt a join
df_comp.rename(columns={'LOCAL_DEP_DT':'SCHED_DEP_TM'},inplace=True)
df_comp['SCHED_DEP_TM'] = pd.to_datetime(df_comp['SCHED_DEP_TM'],format=sched_dep_tm_format)

df_main = df_main.merge(df_comp,how='left',on=['SCHED_DEP_TM','OPG_FLT_NO','UPL_STN_CD','DSG_STN_CD','DTD'])

####################################################################
# Join on availability data

try:
    dirname = os.path.dirname(__file__)
    df_avail = pd.read_csv(dirname+pthsep+".."+pthsep+"data"+pthsep+"csvs"+pthsep+"availability_data.txt",delimiter=',',na_values=['?'])
except:
    print("ERROR: Couldn't find the file 'availability_data.txt' in the 'csvs' folder. Please check that it is there.")
    exit()

# Cast column as datetime object
df_avail['LOCAL_FLT_DT'] = pd.to_datetime(df_avail['LOCAL_FLT_DT'],format='%d/%m/%Y')

df_main = df_main.merge(df_avail,how='left',on=['OPG_FLT_NO','LOCAL_FLT_DT','DTD'])

###############################################################################################################
# Cast types explicitly to save memory and disk space
# (for example, most variables are fine as 16-bit integers - no need to use int32, which is the default)

# NOTE: If you want to go extreme, you could probably even get away with using unsigned ints for some of
#       the variables (e.g. 'HELD_PSJs','DOW_NO','flown_pax' etc). Just be careful that some variables,
#       e.g. all of the 'v' variables ('vL4W','vFW_HIST' etc) can be negative and must be signed ints.

# Drop NaNs
df_main.dropna(inplace=True)

# Strings
df_main['UPL_STN_CD'] = df_main['UPL_STN_CD'].astype(str)
df_main['DSG_STN_CD'] = df_main['DSG_STN_CD'].astype(str)
df_main['HAUL_CD'] = df_main['HAUL_CD'].astype(str)
df_main['HUB_STN_CD'] = df_main['HUB_STN_CD'].astype(str)
df_main['OER_STN_CD'] = df_main['OER_STN_CD'].astype(str)
df_main['CBN_CD'] = df_main['CBN_CD'].astype(str)

# Ints
df_main['DOW_NO'] = df_main['DOW_NO'].astype('int16')
df_main['CAPACITY'] = df_main['CAPACITY'].astype('int16')
df_main['CAPACITY_LY'] = df_main['CAPACITY_LY'].astype('int16')
df_main['op_flown_pax'] = df_main['op_flown_pax'].astype('int16')
df_main['flown_pax'] = df_main['flown_pax'].astype('int16')
df_main['OPG_FLT_NO'] = df_main['OPG_FLT_NO'].astype('int16')
df_main['HELD_PSJs'] = df_main['HELD_PSJs'].astype('int16')
df_main['PSJs_LY'] = df_main['PSJs_LY'].astype('int16')
df_main['HELD_BUS_PSJs'] = df_main['HELD_BUS_PSJs'].astype('int16')
df_main['HELD_BUS_PSJs_vL4W'] = df_main['HELD_BUS_PSJs_vL4W'].astype('int16')
df_main['HELD_PSJs_vL4W'] = df_main['HELD_PSJs_vL4W'].astype('int16')
df_main['HELD_BUS_PSJs_vLW'] = df_main['HELD_BUS_PSJs_vLW'].astype('int16')
df_main['HELD_PSJs_vLW'] = df_main['HELD_PSJs_vLW'].astype('int16')
df_main['HELD_BUS_PSJs_vFW_HIST'] = df_main['HELD_BUS_PSJs_vFW_HIST'].astype('int16')
df_main['HELD_PSJs_vFW_HIST'] = df_main['HELD_PSJs_vFW_HIST'].astype('int16')
df_main['HELD_BUS_PSJs_HIST'] = df_main['HELD_BUS_PSJs_HIST'].astype('int16')
df_main['HELD_PSJs_HIST'] = df_main['HELD_PSJs_HIST'].astype('int16')
df_main['HELD_BUS_PSJs_vL4W_HIST'] = df_main['HELD_BUS_PSJs_vL4W_HIST'].astype('int16')
df_main['HELD_PSJs_vL4W_HIST'] = df_main['HELD_PSJs_vL4W_HIST'].astype('int16')
df_main['HELD_BUS_PSJs_vLW_HIST'] = df_main['HELD_BUS_PSJs_vLW_HIST'].astype('int16')
df_main['HELD_PSJs_vLW_HIST'] = df_main['HELD_PSJs_vLW_HIST'].astype('int16')

####################################################################
# Downsample DTDs as requested

smallest_DTD = min( df_main['DTD'].values )

df_main = df_main.loc[ ( (df_main['DTD'] - smallest_DTD) % sampling_frequency == 0 ) ]

####################################################################
# Finally, write to pickle

df_main.to_pickle(cwd+pthsep+"CLEANED.pkl")

print("Wrote cleaned dataset to 'CLEANED.pkl'")

###############################################################################################################
# Print timing info to terminal

CODE_END_TIME = time.time()
CODE_RUN_TIME = CODE_END_TIME - CODE_START_TIME

print( "RUN TIME: Took %f seconds to clean, manipulate & pickle supplied dataset(s)" % (CODE_RUN_TIME) )

