import sys
import os
import pickle
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

CODE_START_TIME = time.time()

###############################################################################################################
# Globally useful stuff

cwd=os.getcwd()

# Path separator (different for Linux & Windows)
pthsep = os.path.sep

# Import my 'scoring_and_statistics' module from 'passenger_forecasting\utilities'
dirname = os.path.dirname(__file__)
sys.path.append(dirname+pthsep+".."+pthsep+"utilities")
import scoring_and_statistics as scst

###############################################################################################################
# Parse arguments

parser=argparse.ArgumentParser()
parser.add_argument('-epochs',type=int,default=300,help="Number of epochs for neural network training.")
parser.add_argument('-hidden_layers',type=int,nargs='+',default=[48],help="Specify the hidden layers of the neural network. E.g. '-hidden_layers 48 24' will create a 48x24x1 neural net (the output length is 1).")
parser.add_argument('-save_model_to_file',action='store_true',help="If supplied, will save the trained model to .json format and weights to .h5 format in the cwd.")
args=parser.parse_args()

epochs=args.epochs
hidden_layers=args.hidden_layers
save_model_to_file=args.save_model_to_file

###############################################################################################################
# Set-up model

# IF YOU'RE GETTING MODULE NOT FOUND ERRORS: Try replacing 'from tensorflow.keras' with 'from keras'
# If this still doesn't work, then double-check your tensorflow and/or keras installation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError,MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

regressor = Sequential()

# Define custom Adam optimizer so can control learning rate (LR), etc
my_adam = Adam(learning_rate=5E-5)

# Specify the hidden layers
for layer in hidden_layers:
    regressor.add( Dense(layer,activation='relu') )

# Output layer
regressor.add( Dense(1,activation='relu') )

# Now that all layers have been added, compile model w/ keras
regressor.compile( loss=MeanSquaredError(),optimizer=my_adam,metrics=[MeanSquaredError(),MeanAbsoluteError()] )

###############################################################################################################
# Read-in train & test sets

try:
    X_train = pickle.load( open(cwd+pthsep+"X_train.pkl",'rb') )
    y_train = pickle.load( open(cwd+pthsep+"y_train.pkl",'rb') )
    X_test = pickle.load( open(cwd+pthsep+"X_test.pkl",'rb') )
    y_test = pickle.load( open(cwd+pthsep+"y_test.pkl",'rb') )
except:
    print("ERROR: Could not find one or more of 'X_train.pkl', 'y_train.pkl', 'X_test.pkl', 'y_test.pkl' in the current directory.")
    print("To fix, run 'construct_train_test_pickles.py', which will generate these files.")
    exit()

###############################################################################################################
# Loss metrics, etc

print("Using model: NeuralNet")

history_callback = regressor.fit( X_train,y_train,validation_data=(X_test,y_test),epochs=epochs ) # .fit method returns a history callback object

logged_train_MAE = history_callback.history['mean_absolute_error']
logged_test_MAE = history_callback.history['val_mean_absolute_error']

plt.plot(logged_train_MAE,label='Train Set')
plt.plot(logged_test_MAE,label='Test Set')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()

print("\n")
print("METRIC LABELS:")
print(regressor.metrics_names)

# Train set RMSE and MAE
print("NN train set metrics:")
print( regressor.evaluate(X_train,y_train) )

# Test set RMSE and MAE
print("NN test set metrics:")
print( regressor.evaluate(X_test,y_test) )

# Some randomly chosen predictions from test set.
# NOTE: To suppress this printout, set 'print_each_prediction' to False
num_preds = 15
y_pred,avg_RMSE,avg_MAE = scst.make_random_predictions(regressor,X_test,y_test,num_preds,print_each_prediction=True)

# Save model as .json and weights as .h5 in the cwd, if requested.
# You'll need to do this if you want to make forecasts with the model...
if save_model_to_file:
    # Serialize model to JSON
    model_json = regressor.to_json()
    with open("NeuralNet_MODEL.json","w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    regressor.save_weights("NeuralNet_WEIGHTS.h5")
    print("Saved model to file 'NeuralNet_MODEL.json' and weights to 'NeuralNet_WEIGHTS.h5'")

###############################################################################################################
# Print timing info to terminal

CODE_END_TIME = time.time()
CODE_RUN_TIME = CODE_END_TIME - CODE_START_TIME

print( "RUN TIME: Took %f seconds to train and test model" % (CODE_RUN_TIME) )
