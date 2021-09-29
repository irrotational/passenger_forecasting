import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime as dt
import argparse
import time
import pickle

###############################################################################################################

CODE_START_TIME = time.time()

###############################################################################################################

""" This script takes a pickled dataframe as input and produces
    model-agnostic train and test sets ready for training. """

###############################################################################################################
# Parse arguments

parser=argparse.ArgumentParser()
parser.add_argument('full_dataset',type=str,help="""The full dataset (a Pandas DataFrame as a serialized pickle) from which the train/test data will be drawn. This should be a .pkl
                                                    file produced by the 'prepare_pickles.py' script, and should be in the current working directory.""")
parser.add_argument('-train_size',type=int,default=250,help="Will randomly choose this many individual flights (i.e. a particular FLT_NO, DEP_DATE pair) to be in the training set.")
parser.add_argument('-test_size',type=int,default=50,help="Will randomly choose this many individual flights (i.e. a particular FLT_NO, DEP_DATE pair) to be in the test set.")
parser.add_argument('-train_date_range',type=str,nargs=2,default=None,help="""Demand that training data be taken only from this date range. Obviously, this date range must lie within
                                                                              the data in the supplied dataset, else the code will throw an error. Specify in 'YYYY-mm-dd' format as two
                                                                              space-separated values, i.e. '-train_date_range <start_date> <stop_date>' . Default behaviour is to just
                                                                              use the first (i.e. earliest) 75%% of dates in the supplied dataset as training samples. Unless you have a
                                                                              good reason to, you may as well just use the default behaviour.""")
parser.add_argument('-test_date_range',type=str,nargs=2,default=None,help="""Demand that testing data be taken only from this date range. Obviously, this date range must lie within
                                                                             the data in the supplied dataset, else the code will throw an error. Specify in 'YYYY-mm-dd' format as two
                                                                             space-separated values, i.e. '-test_date_range <start_date> <stop_date>' . Default behaviour is to just
                                                                             use the last (i.e. latest) 25%% of dates as testing samples. Unless you have a good reason to, you may as
                                                                             well just use the default behaviour.""")
parser.add_argument('-forecast_DTD',type=int,default=None,help="""Specify the DTD at which the model should make a prediction. For example, when the 'prepare_pickles.py' script
                                                                            was run, the user will have supplied a DTD  'sampling_frequency' (e.g. '-sampling_frequency 3'), for which the
                                                                            default value is 7 (i.e. use every weekly snapshot). In this case, the pickled dataset produced by 'prepare_pickles.py'
                                                                            will, for each flight, contain DTD values like [7,14,21,28,35,...] . If 'forecast_DTD' is set as 21 (i.e. '-forecast_DTD 21'),
                                                                            then the model will be trained on all datapoints at and before 21 days from departure (in this case [21,28,35,...]).
                                                                            The code will throw an error if you specify a 'forecast_DTD' that is incompatible with the dataset - For example, if your
                                                                            dataset contains only [7,14,21,28,35,...] DTDs, and you specify '-forecast_DTD 8', an error will be thrown.""")
args=parser.parse_args()

full_dataset=args.full_dataset
train_size=args.train_size
test_size=args.test_size
train_date_range=args.train_date_range
test_date_range=args.test_date_range
forecast_DTD=args.forecast_DTD

###############################################################################################################
# Set Pandas options, globally useful stuff, etc

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100000
pd.options.mode.chained_assignment = None

cwd=os.getcwd()

# Path separator (different for Linux & Windows)
pthsep = os.path.sep

###############################################################################################################
# Read in pickled dataset (produced by 'ml_project.py'), handle NaNs,
# one-hot encode some stuff, define some useful quantities

try:
    df = pd.read_pickle(cwd+pthsep+full_dataset)

    # Remove NaNs if there are somehow any left...
    df.dropna(inplace=True)

    # One-hot encode the 'macro_group_nm' variable
    df = pd.get_dummies(df,columns=['macro_group_nm'],drop_first=True)

    # Get a list of unique flight numbers that we will randomly draw from
    flight_numbers = pd.unique(df['OPG_FLT_NO'])
    unique_flights_count = len(flight_numbers)

except:
    print( "ERROR: Either could not find the pickle %s, or it is not a valid pickled dataframe." % (full_dataset) )
    print( "To fix, ensure that a valid pickled dataframe produced by 'ml_project.py' exists in the cwd." )
    exit()

# Figure out which date ranges for train/test that the user wishes to use

if ( train_date_range and test_date_range ): # Then the user has supplied these dates already, so don't do anything
    pass
else: # Otherwise, we'll default to testing in the last (i.e. most-recent) 25% of values and use the first 75% for training
    test_fraction = 0.25
    train_fraction = 1 - test_fraction
    all_dates = pd.unique( df['LOCAL_FLT_DT'].sort_values() ) # Unique, ordered list of available dates
    num_dates_total = len(all_dates)
    train_date_range = [ all_dates[0] , all_dates[int(num_dates_total*train_fraction)-1] ]
    test_date_range = [ all_dates[int(num_dates_total*train_fraction)] , all_dates[-1] ]

# Figure out which DTDs the user wishes to use

unique_DTDs = list( pd.unique(df['DTD']) )
unique_DTDs.sort()

if forecast_DTD: # Then the user has specified a DTD at which a forecast must be made, so just check that their input is sensible...
    if ( forecast_DTD in unique_DTDs ):
        use_these_DTDs = [ x for x in unique_DTDs if x >= forecast_DTD ]
    else:
        print( "ERROR: You specified to forecast at DTD=%d, but your dataset appears to only contain these DTDs: %s" % (forecast_DTD,unique_DTDs) )
        print("To fix, either specify a valid 'forecast_DTD' argument (i.e. one that's in the list above) or generate a new dataset with suitable DTDs.")
        exit()
else: # Otherwise, default to using all possible available DTDs for each flight
    use_these_DTDs = unique_DTDs
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
# Construct train/test set

def get_train_test_data(size,only_these_DTDs=None,date_range=None,return_flight_numbers=False):

    """ Function that returns 'size' (X,y) pairs from the dataset (df) as two numpy
        ndarrays of shape (size,input_vector_total_length) and (size,1) respectively
        (where 'input_vector_total_length' is defined above).

        The actual flight numbers used, and dates associated with that flight number, are chosen at random.
        'flown_pax' is the variable to be predicted.
        
        Several variables in df that are either not used or redundant are dropped.

        If the 'date_range' argument is passed (which should be a list of dates in 'YYYY-mm-dd' format, e.g.
        ['2018-06-01','2019-01-01']), then only data found in this date range will be used. Otherwise, all data
        irrespective of date will be used indiscriminately.

        If the boolean 'return_flight_numbers' is passed as True, then the flight numbers of each chosen flight in
        the dataset are recorded and returned as a list. This can be useful to determine which flight numbers are in
        the train and test sets, because the flight number itself ('OPG_FLT_NO') is dropped from the model input. """

    if date_range:
        start_date = pd.to_datetime( date_range[0],format='%Y/%m/%d' )
        end_date = pd.to_datetime( date_range[1],format='%Y/%m/%d' )

    flight_numbers_used = []
    Xs,ys = [],[]
    
    while (len(Xs) < size):
        # Pick a random flight number
        idx = int( np.random.random() * unique_flights_count )
        this_flight = flight_numbers[idx]
        # Get a DataFrame containing data just for this specific flight number
        data_this_flight = df.loc[ df['OPG_FLT_NO'] == this_flight ]
        # Now, get all the unique departure dates associated with this flight number
        unique_dates_for_this_flight = pd.unique( data_this_flight['LOCAL_DEP_DT'] )
        # If a date_range was specified, keep only the flights that lie in the requested range
        if date_range:
            unique_dates_for_this_flight = [ date for date in unique_dates_for_this_flight if ( (date > start_date) and (date < end_date) ) ]
            if ( len(unique_dates_for_this_flight) == 0 ):
                continue
        unique_dates_count = len(unique_dates_for_this_flight)
        # Now, pick a random date
        idx = int( np.random.random() * unique_dates_count )
        date_used = unique_dates_for_this_flight[idx]
        
        # Extract the data for the specific flight that departs on 'date_used'
        data = data_this_flight.loc[ data_this_flight['LOCAL_DEP_DT'] == date_used ]

        # If requested to use only specific DTDs ('only_these_DTDs'), then remove any incompatible DTDs
        if only_these_DTDs:
            data = data.loc[ data['DTD'].isin(only_these_DTDs) ]

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
        # 'flown_pax' is to be predicted; the other variables are all inputs
        data_X,data_y = data.drop(columns=['flown_pax']) , pd.unique(data['flown_pax'])
        # Combine cabin codes to get total pax
        data_y = np.array( [data_y.sum()] )

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
        ys.append(data_y)

        flight_numbers_used.append(this_flight)

    return np.array(Xs),np.array(ys),flight_numbers_used


X_train,y_train,train_flight_numbers_used = get_train_test_data(train_size, only_these_DTDs=use_these_DTDs, date_range=train_date_range, return_flight_numbers=True)
    
X_test,y_test,test_flight_numbers_used = get_train_test_data(test_size, only_these_DTDs=use_these_DTDs, date_range=test_date_range, return_flight_numbers=True)
    
# Save 'X_train.pkl', 'y_train.pkl', 'X_test.pkl' and 'y_test.pkl' as pickles to the current directory
pickle.dump( X_train, open(cwd+pthsep+"X_train.pkl",'wb') )
pickle.dump( y_train, open(cwd+pthsep+"y_train.pkl",'wb') )
pickle.dump( X_test, open(cwd+pthsep+"X_test.pkl",'wb') )
pickle.dump( y_test, open(cwd+pthsep+"y_test.pkl",'wb') )

print("Wrote 'X_train.pkl', 'y_train.pkl', 'X_test.pkl' and 'y_test.pkl' to the current directory.")

# Print shapes of data to terminal
print('\n')
print('DATA SHAPES after processing:')
print('X_train has shape: ',X_train.shape)
print('y_train has shape: ',y_train.shape)
print('X_test has shape: ',X_test.shape)
print('y_test has shape: ',y_test.shape)
print('\n')

###############################################################################################################
# Print timing info to terminal

CODE_END_TIME = time.time()
CODE_RUN_TIME = CODE_END_TIME - CODE_START_TIME

print( "RUN TIME: Took %f seconds to prepare train and test sets" % (CODE_RUN_TIME) )

