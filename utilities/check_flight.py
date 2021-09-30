import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime as dt
import argparse
import pickle

###############################################################################################################

""" This script pulls details for a particular specified flight number and flight date, and plots
    the data associated with that flight. Useful for checking basic info. If you want to plot more
    variables than are currently plotted, then editing the bottom of this script should be easy. """

###############################################################################################################
# Parse arguments

parser=argparse.ArgumentParser()
parser.add_argument('pickled_dataframe',type=str,help="The dataset within which the flight is located (should be a pickled DataFrame).")
parser.add_argument('-flight_number',type=int,default=None,help="Flight number of the flight to check. If this is left unspecified, will pick a flight at random.")
parser.add_argument('-departure_date',type=str,default=None,help="Departure date, in the format: 'YYYY-MM-DD' . If this is left unspecified, will pick a flight at random.")
args=parser.parse_args()

pickled_dataframe=args.pickled_dataframe
flight_number=args.flight_number
departure_date=args.departure_date

###############################################################################################################
# Set Pandas options, globally useful stuff, etc

pd.options.mode.chained_assignment = None

cwd=os.getcwd()

# Path separator (different for Linux & Windows)
pthsep = os.path.sep

###############################################################################################################
# Read in pickled dataset, parse the supplied departure date

df = pd.read_pickle(cwd+pthsep+pickled_dataframe)

if (flight_number and departure_date): # If user has supplied a particular flight_number and departure_date
    local_flt_date = pd.to_datetime(departure_date,format='%Y/%m/%d')
    
else: # Otherwise, let's pick one at random (default behaviour)
    unique_flight_nums = pd.unique(df['OPG_FLT_NO'])
    flights_dict = {}
    for flt_num in unique_flight_nums:
        unique_dates = pd.unique( df.loc[ df['OPG_FLT_NO']==flt_num ]['LOCAL_FLT_DT'] )
        flights_dict[flt_num] = unique_dates
        
    rand_flt_idx = int( len(unique_flight_nums) * np.random.random() )
    flight_number = unique_flight_nums[rand_flt_idx]
    rand_date_idx = int( len(flights_dict[flight_number]) * np.random.random() )
    local_flt_date = flights_dict[flight_number][rand_date_idx]

# So from here onwards, we have a 'flight_number' (int) and a 'local_flt_date' (datetime object)

date_as_string = pd.to_datetime(local_flt_date).strftime('%Y-%m-%d')

print( "Plotting data for OPG_FLT_NO %d leaving on departure date %s" % (flight_number,date_as_string) )

###############################################################################################################
# Get flight info & plot

# Filter by flight number & departure date
new_df = df[ ( (df['OPG_FLT_NO'] == flight_number) & (df['LOCAL_FLT_DT'] == local_flt_date) ) ]

# Order by DTD
new_df.sort_values('DTD',inplace=True)

if ( len(new_df) == 0 ):
    print( "ERROR: Did not find data matching OPG_FLT_NO=%d leaving on %s in the file '%s'" % (flight_number,departure_date,pickled_dataframe) )
    print("""This is either because the flight genuinely isn't in the dataset, or it was removed at the cleaning stage due to missing data etc
             (quite a few flights are removed from the cleaned dataset for this reason).""")
    print( "Try another flight and/or date - Alternatively, double check that your dataset in '%s' is indeed what you want" % (pickled_dataframe) )
    exit()

# Get flight details (departure location, destination, etc)
departure_location = pd.unique( new_df['UPL_STN_CD'] )[0]
arrival_location = pd.unique( new_df['DSG_STN_CD'] )[0]

# Split into cabin codes (only 'M' and 'C' for SH)
m_cabin = new_df[new_df['CBN_CD'] == 'M']
m_cabin['HELD_PSJs'] = pd.to_numeric(m_cabin['HELD_PSJs'].values,errors='coerce')
c_cabin = new_df[new_df['CBN_CD'] == 'C']
c_cabin['HELD_PSJs'] = pd.to_numeric(c_cabin['HELD_PSJs'].values,errors='coerce')
# Add a column with total number of currently held passengers, i.e. summed over all cabin codes.
# It doesn't matter whether this is added to the 'm_cabin' or 'c_cabin' dataframe (it's the same for both)
# Frustratingly, this fails if the flight is missing one or more DTDs - So let the user know...
try:
    m_cabin['total_held_pax'] = [ m_cabin['HELD_PSJs'].values[i] + x for i,x in enumerate(c_cabin['HELD_PSJs'].values) ]
except:
    pass

# Plot held passengers for each cabin
ax = m_cabin.plot(x='DTD',y='HELD_PSJs',label='HELD_PSJs_m_cabin',marker='.',ms=8) # add all subsequent plots to this ax
c_cabin.plot(x='DTD',y='HELD_PSJs',label='HELD_PSJs_c_cabin',ax=ax,linestyle='dashed',marker='.',ms=8)
# Plot passengers held over Last 4 Weeks for each cabin
c_cabin.plot(x='DTD',y='HELD_PSJs_vL4W',label='HELD_PSJs_vL4W_c_cabin',ax=ax,linestyle='dashed',marker='x',ms=4)
m_cabin.plot(x='DTD',y='HELD_PSJs_vL4W',label='HELD_PSJs_vL4W_m_cabin',ax=ax,marker='x',ms=4)

# Plot historic passengers held Vs next week ('HELD_BUS_PSJs_vFW_HIST') for each cabin
c_cabin.plot(x='DTD',y='HELD_BUS_PSJs_vFW_HIST',label='HELD_BUS_PSJs_vFW_HIST_c_cabin',ax=ax,linestyle='dashed',marker='x',ms=4)
m_cabin.plot(x='DTD',y='HELD_BUS_PSJs_vFW_HIST',label='HELD_BUS_PSJs_vFW_HIST_m_cabin',ax=ax,marker='x',ms=4)

# Plot capacity & flown capacity for each cabin
c_cabin.plot(x='DTD',y='CAPACITY',label='capacity_c_cabin',ax=ax,linestyle='dashed',marker='s',ms=4)
m_cabin.plot(x='DTD',y='CAPACITY',label='capacity_m_cabin',ax=ax,marker='s',ms=4)
c_cabin.plot(x='DTD',y='FLOWN_CAPACITY',label='flown_capacity_c_cabin',ax=ax,linestyle='dashed',marker='.',ms=8)
m_cabin.plot(x='DTD',y='FLOWN_CAPACITY',label='flown_capacity_m_cabin',ax=ax,marker='.',ms=8)

# Add a (flat) line to show the final number of PAX flown
m_cabin.plot(x='DTD',y='flown_pax',label='final_flown_pax',ax=ax,marker='x',ms=4)

# Plot total held PAX at each DTD
try:
    m_cabin.plot(x='DTD',y='total_held_pax',label='total_held_pax',ax=ax,marker='x',ms=4)
except:
    print("This particular flight appears to be missing one or more DTDs!")
    print("(if passed to the 'construct_train_test_pickles.py' script, this flight would be filtered out)")

# Plot pax flown last year (flat line)
m_cabin.plot(x='DTD',y='PSJs_LY',label='PSJs_LY',ax=ax,marker='s',ms=4)

# Plot availability
m_cabin.plot(x='DTD',y='SEATS_AVL',label='SEATS_AVL',ax=ax,marker='s',ms=4)

ax.set_ylabel('Passengers',fontsize=12)
plt.legend(bbox_to_anchor=(1.100,1.000), loc="upper left")

# Now, define a twinned y-axis for the competitor & BA price, which obviously have units of 'price' and not PAX
ax_price = ax.twinx()

# Plot BA & Competitor Price
m_cabin.plot(x='DTD',y='MIN_BA_PRICE',label='MIN_BA_PRICE',ax=ax_price,marker='.',ms=8,color='lime')
m_cabin.plot(x='DTD',y='MIN_COMP_PRICE',label='MIN_COMPETITOR_PRICE',ax=ax_price,linestyle='dashed',marker='.',ms=8,color='k')

ax_price.set_ylabel('Price of Flight',fontsize=12)
plt.legend(bbox_to_anchor=(1.100,0.625), loc="upper left")

ax.invert_xaxis() # We want to count down the DTD (i.e. smaller DTD at right of plot)
plt.title('OPG_FLT_NO=%d, %s to %s \n LOCAL_FLT_DT=%s' % (flight_number,departure_location,arrival_location,date_as_string))
ax.set_xlabel('Days to Departure',fontsize=12)
plt.xticks( pd.unique(df['DTD']) )
plt.tight_layout()
plt.show()
