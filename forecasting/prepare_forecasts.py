import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import pearsonr
import argparse
import time
import pickle
import json

CODE_START_TIME = time.time()

###############################################################################################################

""" Reads-in a pickled dataset (that is; pickled by the 'prepare_pickles.py' script) and cleans + reshapes
    the data, ready for use with the 'make_forecasts.py' script. """

###############################################################################################################
# Parse arguments

parser=argparse.ArgumentParser()
parser.add_argument('full_dataset',type=str,help="The full dataset (a Pandas DataFrame as a serialized pickle) to be forecasted for.")
parser.add_argument('-forecast_DTD',type=int,default=None,help="""Specify the DTD at which the model should make a prediction. For example, when the 'prepare_pickles.py' script
                                                                            was run, the user will have supplied a DTD  'sampling_frequency' (e.g. '-sampling_frequency 3'), for which the
                                                                            default value is 7 (i.e. use every weekly snapshot). In this case, the pickled dataset produced by 'prepare_pickles.py'
                                                                            will, for each flight, contain DTD values like [7,14,21,28,35,...] . If 'forecast_DTD' is set as 21 (i.e. '-forecast_DTD 21'),
                                                                            then the model will be trained on all datapoints at and before 21 days from departure (in this case [21,28,35,...]).
                                                                            The code will throw an error if you specify a 'forecast_DTD' that is incompatible with the dataset - For example, if your
                                                                            dataset contains only [7,14,21,28,35,...] DTDs, and you specify '-forecast_DTD 8', an error will be thrown.""")

args=parser.parse_args()

full_dataset=args.full_dataset
forecast_DTD=args.forecast_DTD

###############################################################################################################
# Set Pandas options, globally useful stuff, etc

pd.options.mode.chained_assignment = None

cwd=os.getcwd()

# Path separator (different for Linux & Windows)
pthsep = os.path.sep

###############################################################################################################
# Read in pickled dataset, handle NaNs, one-hot encode some stuff, define some useful quantities

try:
    df = pd.read_pickle(cwd+pthsep+full_dataset)

except:
    print( "ERROR: Either could not find the pickle %s, or it is not a valid pickled dataframe." % (full_dataset) )
    print( "To fix, ensure that a valid pickled dataframe produced by 'prepare_pickles.py' exists in the cwd." )
    exit()

# Drop the 'flown_pax' column, which will contain all NULL or zero
# values, as the flight hasn't taken place yet...
df.drop(columns=['flown_pax'],inplace=True)

# One-hot encode the 'macro_group_nm' variable
df = pd.get_dummies(df,columns=['macro_group_nm'],drop_first=True)

# Sort by departure date, and then flight number (will look much better when we write to file)
df.sort_values(by=['LOCAL_DEP_DT','OPG_FLT_NO'],inplace=True)

# Get a list of unique flight numbers
flight_numbers = list( pd.unique(df['OPG_FLT_NO']) )
unique_flights_count = len(flight_numbers)

# Figure out DTDs

unique_DTDs = list( pd.unique(df['DTD']) )
unique_DTDs.sort()

if forecast_DTD: # Then the user has specified a DTD at which a forecast must be made, so just check that their input is sensible...
    if ( forecast_DTD in unique_DTDs ):
        use_these_DTDs = [ x for x in unique_DTDs if x >= forecast_DTD ]
    else:
        print( "ERROR: You specified to forecast at DTD=%d, but your dataset appears to only contain these DTDs: %s" % (forecast_DTD,unique_DTDs) )
        print("To fix, either specify a valid 'forecast_DTD' argument (i.e. one that's in the list above) or generate a new dataset with suitable DTDs.")
        exit()

num_DTDs_used = len(use_these_DTDs)

input_length = num_DTDs_used * 2 # The number of unique input 'observations' for each flight (one for every DTD and cabin)
num_output_variables = 1 # Here, we're summing over all cabins to get total_pax, so there's just one output for each flight ('flown_pax')

# The variables that change with both cabin and timestep
time_depn_and_cabin_depn_vars = ['HELD_BUS_PSJs','HELD_PSJs',
                                 'HELD_BUS_PSJs_vL4W','HELD_PSJs_vL4W','HELD_BUS_PSJs_vLW',
                                 'HELD_PSJs_vLW','HELD_BUS_PSJs_vFW_HIST','HELD_PSJs_vFW_HIST',
                                 'HELD_BUS_PSJs_HIST','HELD_PSJs_HIST','HELD_BUS_PSJs_vL4W_HIST',
                                 'HELD_PSJs_vL4W_HIST','HELD_BUS_PSJs_vLW_HIST','HELD_PSJs_vLW_HIST']
num_time_depn_and_cabin_depn_vars = len(time_depn_and_cabin_depn_vars)

# The variables that can change at each timestep, but do *not* depend on cabin
time_depn_not_cabin_depn_vars = ['MIN_COMP_PRICE','MIN_BA_PRICE','SEATS_AVL']
num_non_cabin_time_dep_vars = len(time_depn_not_cabin_depn_vars)

# The 'constant' variables (i.e. variables that are the same at every timestep & cabin)
constant_vars = ['DOW_NO','YEAR','MONTH','DAY','PSJs_LY_c','PSJs_LY_m','capacity_c','capacity_m',
                 'macro_group_nm1','macro_group_nm2','macro_group_nm3','macro_group_nm4']
num_cnst_vars = len(constant_vars)

# The total overall length of a single input vector
input_vector_total_length = input_length * num_time_depn_and_cabin_depn_vars
input_vector_total_length += num_DTDs_used * num_non_cabin_time_dep_vars
input_vector_total_length += num_cnst_vars

# The width of the time-dependent, but *non-cabin dependent* data
time_depn_not_cabin_depn_width = num_non_cabin_time_dep_vars * num_DTDs_used

# The width of a single time-dependent cabin-level 'observation' (i.e. a particular DTD and cabin)
observation_width = int( (input_vector_total_length - num_cnst_vars - time_depn_not_cabin_depn_width) / input_length )

###############################################################################################################
# Construct pickled dataset to forecast

def get_forecast_data():

    """ Function that returns a prepared X array ready for forecasting (that is; ready for use with
        the 'make_forecasts.py' script) from the dataset (df). This function is based upon the
        'get_train_test_data' function in the 'utilities\construct_train_test_pickles.py' script. """

    flight_numbers_dates_capacity = [] # Record these; will write to file later

    Xs = []
    count = 0
    while ( len(flight_numbers) > 100 ):
        # Pick a flight, and it remove from the list of flights todo
        this_flight = flight_numbers.pop(count)
        # Get a DataFrame containing data just for this specific flight number
        data_this_flight = df.loc[ df['OPG_FLT_NO'] == this_flight ]
        # Now, get all the unique departure dates associated with this flight number
        unique_dates_for_this_flight = pd.unique( data_this_flight['LOCAL_DEP_DT'] )
        unique_dates_count = len(unique_dates_for_this_flight)

        # Loop over unique departure dates
        for date in unique_dates_for_this_flight:

            date_used = date
            
            # Extract the data for the specific flight that departs on 'date_used'
            data = data_this_flight.loc[ data_this_flight['LOCAL_DEP_DT'] == date_used ]

            # Remove any incompatible DTDs            
            data = data.loc[ data['DTD'].isin(use_these_DTDs) ]

            # Discard any data that does not have the correct shape.
            # This happens relatively rarely, and usually means a DTD or cabin was missing for that flight
            if ( len(data) != input_length ):
                continue

            # Force 'M' cabin data to appear before 'C' cabin data so as to get a consistent input structure
            data.sort_values(by=['DTD','CBN_CD'],inplace=True)

            # CONSTANT VALUES (for a given flight, these are the same for every cabin and timestep)
            dow_no = data['DOW_NO'].values[0]
            capacity_c,capacity_m = min( pd.unique(data['CAPACITY'].values) ) , max( pd.unique(data['CAPACITY'].values) )
            year,month,day = pd.to_datetime(date_used).year , pd.to_datetime(date_used).month , pd.to_datetime(date_used).day
            PSJs_LY_both_cabins = pd.unique(data['PSJs_LY'].values)
            PSJs_LY_c,PSJs_LY_m = min(PSJs_LY_both_cabins),max(PSJs_LY_both_cabins) # 'C' always has less passengers

            # Competitor Data & Availability (for a given flight, these are the same for every cabin, but *can* change between each timestep)
            # Extract these values and remove from the DataFrame (we'll add them to the start of the input vector later)
            # Keep only the even elements of the list (or odd - doesn't matter - but data is duplicated for 'M' and 'C' cabins
            # so just need to extract a unique list of values - I'm choosing to keep even)
            
            comp_prices = list( data.pop('MIN_COMP_PRICE') )[::2]
            ba_prices = list( data.pop('MIN_BA_PRICE') )[::2]
            availability = list( data.pop('SEATS_AVL') )[::2]

            # Store the one-hot-encoded 'macro_group_name' information, then drop it from data
            macro_group_name_dummies = []
            labels = data.columns
            for label in labels:
                if ( 'macro_group_nm' in label ):
                    macro_group_name_dummies.append( data.iloc[0][label] ) # We use .iloc[0], but any will do as they're all the same
                    data = data.drop(columns=[label])

            # Drop all remaining un-needed crap
            data = data.drop(columns=['HAUL_CD','HUB_STN_CD','OER_STN_CD','FLOWN_CAPACITY','op_flown_pax','dsg_stn_cd','DSG_STN_CD','UPL_STN_CD'])
            data = data.drop(columns=['DTD','CBN_CD','OPG_FLT_NO','PSJs_LY','DOW_NO','CAPACITY','CAPACITY_LY'])
            data = data.drop(columns=['LOCAL_DEP_DT','SCHED_DEP_TM','GMT_FLT_DT','LOCAL_FLT_DT','Snapshot_dt','LOCAL_UPL_DT_HIST'])
            
            # Separate input and output variables for the model (X,y pairs)
            data_X = data

            # Appending the time-dependent-but-not-cabin-dependent
            # data, and the 'constant' data
            
            # First, flatten data_X and cast to type list so we can append
            data_X = list( data_X.values.flatten() )
            # Now, append the data that changes with DTD, but not with cabin (e.g. min_price, availability, etc)
            for x in list(comp_prices):
                data_X.append(x)
            for x in list(ba_prices):
                data_X.append(x)
            for x in list(availability):
                data_X.append(x)
            # Finally, append the 'constant' data (which for a given flight, is same for all DTD and cabins)
            data_X.append(dow_no)
            data_X.append(year)
            data_X.append(month)
            data_X.append(day)
            data_X.append(PSJs_LY_c)
            data_X.append(PSJs_LY_m)
            data_X.append(capacity_c)
            data_X.append(capacity_m)
            for group_name in macro_group_name_dummies:
                data_X.append(group_name)
            # Recast to numpy array, and flatten
            data_X = np.array(data_X).flatten()

            Xs.append(data_X)

            # Pad day & month with zeros if necessary, before writing to file
            day = str(day)
            month = str(month)
            if len(day) == 1:
                day = "0"+day
            if len(month) == 1:
                month = "0"+month
            year = str(year)

            total_capacity = str(capacity_c + capacity_m)

            date_used = "%s-%s-%s" % (year,month,day)

            flight_numbers_dates_capacity.append( (this_flight,date_used,total_capacity) )

        count += 1

    return np.array(Xs),flight_numbers_dates_capacity


X,flight_numbers_dates_capacity = get_forecast_data()
    
# Save 'X_forecast.pkl' as pickle to the current directory
pickle.dump( X, open(cwd+pthsep+"X_forecast.pkl",'wb') )

print("Wrote forecast-ready pickle to 'X_forecast.pkl'")

# Write the names of the flight numbers & departure dates to file
flt_info_file = open("flt_info_for_forecast.txt",'w')
flt_info_file.write("OPG_FLT_NO,LOCAL_FLT_DT,TOTAL_CAPACITY\n") # Header
for flight_num,date,capacity in flight_numbers_dates_capacity:
    flt_info_file.write( "%s,%s,%s\n" % (flight_num,date,capacity) )
flt_info_file.close()

print("Created the file 'flt_info_for_forecast.txt', which will contain the forecast predictions after you run 'make_forecasts.py'")

# Print shape of data to terminal
print('\n')
print('DATA SHAPE after processing:')
print('X_forecast has shape: ',X.shape)

###############################################################################################################
# Print timing info to terminal

CODE_END_TIME = time.time()
CODE_RUN_TIME = CODE_END_TIME - CODE_START_TIME

print( "RUN TIME: Took %f seconds to prepare dataset for forecasting" % (CODE_RUN_TIME) )
