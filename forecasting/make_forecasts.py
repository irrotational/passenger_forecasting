import sys
import os
import pickle
import argparse
import numpy as np
import pandas as pd

cwd=os.getcwd()

# Path separator (different for Linux & Windows)
pthsep = os.path.sep

###############################################################################################################

""" This script uses a provided pickled trained model ('model file') to make forecasts on data for which the
    passengers flown ('pax_flown') is unknown. The data should first have been processed by 'prepare_forecasts.py'
    (see the documentation for that script), which should have produced two files: 'X_forecast.pkl' and 'flt_info_for_forecast.txt' .

    This script will append two new columns to the 'flt_info_for_forecast.txt' file:

        - 'PREDICTED_FLOWN_PAX' (predicted passengers flown on that flight)
        - 'PREDICTED_SF' (predicted seat factor for that flight) """

###############################################################################################################
# Parse arguments and set model type

parser=argparse.ArgumentParser()
parser.add_argument('model_file',type=str,help="Name of trained model saved in .pkl format (for the case of an sklearn model) or .json format (for a keras NeuralNet). Should be in the current directory.")
args=parser.parse_args()

model_file=args.model_file

# Load model from .pkl file (in the case of sklearn models)
if ".pkl" in model_file:
    try:
        with open(model_file,'rb') as file:
            regressor = pickle.load(file)
    except:
        print( "ERROR: Could not find '%s' and/or it is not a suitable .pkl file." % (model_file) )
        print("""For sklearn models, you should be supplying a saved model produced by the 'sklearn_models.py' script in the 'models' directory,
                 with the '-save_model_to_file' flag having been passed (which should have saved the model in .pkl format).""")
        exit()
elif ".json" in model_file: # For the case of a keras NeuralNet model, load model from .json and weights from .h5
    from keras.models import model_from_json
    try:
        json_file = open(model_file,'r')
        loaded_model_json = json_file.read()
        json_file.close()
        regressor = model_from_json(loaded_model_json)
        # Load weights
        weights_file = "NeuralNet_WEIGHTS.h5"
        regressor.load_weights(weights_file)
        print( "Loaded NeuralNet model from %s and weights from %s" % (model_file,weights_file) )
    except:
        print( "ERROR: Could not find model '%s' and/or weights in %s." % (model_file,weights_file) )
        print("""For a NeuralNet, you should be supplying a saved model produced by the 'NeuralNets.py' script in the 'models' directory,
                 with the '-save_model_to_file' flag having been passed (which should have saved the model in .json format). The model
                 weights should also be in the current directory in a file called 'NeuralNet_WEIGHTS.h5'.""")

else:
    print("ERROR: Model type not recognised.")
    print("For sklearn models, this script expects a saved model with '.pkl' in the filename (e.g. 'LinearRegresion.pkl'), produced by the 'sklearn_models.py' script.")
    print("For a neural network, this script expects a saved model with '.json' in the filename (e.g. 'NeuralNet_MODEL.json'), produced by the 'NeuralNets.py' script.")
    exit()

###############################################################################################################
# Read-in dataset to forecast

try:
    X = pickle.load( open(cwd+pthsep+"X_forecast.pkl",'rb') )
except:
    print("ERROR: Could not find 'X_forecast.pkl' in the current directory, which should have been produced automatically by the 'prepare_forecasts.py' script.")
    print("To fix, run 'prepare_forecasts.py' on a suitable dataset, which will generate this file.")
    exit()

# Read-in file containing respective flight numbers, departure dates and capacities
try:
    flt_info = pd.read_csv( open(cwd+pthsep+"flt_info_for_forecast.txt",'r') )
except:
    print("ERROR: Could not find 'flt_info_for_forecast.txt' in the current directory, which should have been produced automatically by the 'prepare_forecasts.py' script.")
    print("To fix, run 'prepare_forecasts.py' on a suitable dataset, which will generate this file.")
    exit()

###############################################################################################################
# Catch a potentially common error where the model was trained on data of a different shape to that supplied here.
# This is likely to have happened if the user passed a different 'sampling_frequency' to 'prepare_pickles.py' when
# preparing the train/test and forecasting data (they should, of course, be consistent).

# If this is not caught here, then a ValueError will be raised when calling the .predict method below

# The shape of the input data we've been given
data_input_shape = X.shape[-1]

# Now, find the shape of the input that the model is expecting...

if ".pkl" in model_file: # Case of sklearn models
    if "LinearRegression" in model_file: # LinearRegression, Ridge & LinearSVR cases (the 'coef_' attribute gives the expected input shape)
        regressor_input_shape = regressor.coef_.shape[-1]
    else: # RandomForest case (here, the 'n_features_' attribute gives the expected input shape)
        regressor_input_shape = regressor.n_features_

elif ".json" in model_file: # Case of keras neural network
    # The expected input shape is given by the input shape to the first layer
    regressor_input_shape = regressor.layers[0].input_shape[-1]

# If they don't match, let the user know and exit

if ( regressor_input_shape != data_input_shape ):
    print( """ERROR: The model in '%s' is expecting input vectors with shape %d, but your data contains
              input vectors with shape %d. This has probably happened because you passed a different
              'sampling_frequency' to the 'prepare_pickles.py' and 'prepare_forecasts.py' scripts. These
              need to be the same, as the architecture of the model is different for different sampling
              frequencies (because the length of the input vector changes). Please ensure that these are
              consistent and then try again.""" % (model_file,regressor_input_shape,data_input_shape) )
    exit()

###############################################################################################################
# Predict using loaded-in model

print( "Using model from file: '%s'" % (model_file) )

preds = regressor.predict(X)

# Cast predictions to integer
preds = [ int(x) for x in preds ]

# Add the predictions as a column to the 'flt_info' DataFrame
flt_info['PREDICTED_FLOWN_PAX'] = preds

# Add Seat Factor (SF) info, which can be determined by using the 'TOTAL_CAPACITY' column
flt_info['PREDICTED_SF'] = flt_info['PREDICTED_FLOWN_PAX'] / flt_info['TOTAL_CAPACITY']

# Finally, write to .csv (using 6 dp precision for seat factor)
flt_info.to_csv(cwd+pthsep+"flt_info_for_forecast.txt",index=False,float_format="%.6f")

print("SUCCESS: Wrote forecasts to 'flt_info_for_forecast.txt'")

