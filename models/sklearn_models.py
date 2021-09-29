import sys
import os
import pickle
import argparse
import numpy as np
import time

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
# Parse arguments and set model type

parser=argparse.ArgumentParser()
parser.add_argument('model_name',type=str,default='LinearRegression',help="Name of model to be used. Choose from 'LinearRegression', 'Ridge', 'LinearSVR', 'RandomForest'")
parser.add_argument('-save_model_to_file',action='store_true',help="If supplied, will save the trained model to a .pkl file in the cwd.")
args=parser.parse_args()

model_name=args.model_name
save_model_to_file=args.save_model_to_file

if model_name == 'LinearRegression':
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression(fit_intercept=True,normalize=True)
    
elif model_name == 'Ridge':
    from sklearn.linear_model import Ridge
    regressor = Ridge(fit_intercept=True,normalize=True)
    
elif model_name == 'LinearSVR':
    from sklearn.svm import LinearSVR
    regressor = LinearSVR()
    
elif model_name == 'RandomForest':
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor()

else:
    print("ERROR: Model name not recognised - Must be one of: 'LinearRegression', 'Ridge', 'LinearSVR', 'RandomForest'")
    exit()

###############################################################################################################
# Read-in train & test sets

try:
    X_train = pickle.load( open(cwd+pthsep+"X_train.pkl",'rb') )
    y_train = pickle.load( open(cwd+pthsep+"y_train.pkl",'rb') )
    X_test = pickle.load( open(cwd+pthsep+"X_test.pkl",'rb') )
    y_test = pickle.load( open(cwd+pthsep+"y_test.pkl",'rb') )
except:
    print("ERROR: Could not find one or more of 'X_train.pkl', 'y_train.pkl', 'X_test.pkl', 'y_test.pkl' in the current directory.")
    print("To fix, run 'construct_traintest_pickles.py', which will generate these files.")
    exit()

###############################################################################################################
# Fit model & predict

print( "Using model: %s" % (model_name) )

regressor.fit(X_train,y_train)

print( 'R^2 on train set = ',regressor.score(X_train,y_train) )
print( 'R^2 on test set = ',regressor.score(X_test,y_test) )

# Train set RMSE, MAE and stdev
y_pred = regressor.predict(X_train)
avg_RMSE,avg_MAE = scst.rmse_and_mae_score(y_pred,y_train)
print('%s avg. train set RMSE = %f' % (model_name,avg_RMSE) )
print('%s avg. train set MAE = %f' % (model_name,avg_MAE) )
print('%s stdev of train set predictions = %f' % (model_name,np.std(y_pred)) )
print('Actual stdev of train set = %f' % (np.std(y_train)) )

# Test set RMSE, MAE and stdev
y_pred = regressor.predict(X_test)
avg_RMSE,avg_MAE = scst.rmse_and_mae_score(y_pred,y_test)
print('%s avg. test set RMSE = %f' % (model_name,avg_RMSE) )
print('%s avg. test set MAE = %f' % (model_name,avg_MAE) )
print('%s stdev of test set predictions = %f' % (model_name,np.std(y_pred)) )
print('Actual stdev of test set = %f' % (np.std(y_test)) )

# Some randomly chosen predictions from test set
num_preds = 15
y_pred,avg_RMSE,avg_MAE = scst.make_random_predictions(regressor,X_test,y_test,num_preds,print_each_prediction=True)

# Save model as .pkl in the cwd, if requested
# You'll need to do this if you want to make forecasts with the model...
if save_model_to_file:
    pkl_model_name = "%s.pkl" % (model_name)
    with open(pkl_model_name,'wb') as file:
        pickle.dump(regressor,file)
        print( "Saved trained model to file '%s.pkl'" % (model_name) )

###############################################################################################################
# Print timing info to terminal

CODE_END_TIME = time.time()
CODE_RUN_TIME = CODE_END_TIME - CODE_START_TIME

print( "RUN TIME: Took %f seconds to train and test model" % (CODE_RUN_TIME) )

