import numpy as np

###############################################################################################################

""" This contains scoring functions, utility functions, etc, and is designed to be used
    as a *module* (i.e. imported by other scripts). """

###############################################################################################################
# Globally useful stuff

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

# A function to calculate the total length of an input vector, given the DTDs used ( passed as a list, e.g. [7,14,21] )
def get_vector_lengths(DTDs_used):

    num_DTDs_used = len(DTDs_used)

    input_length = num_DTDs_used * 2 # The number of input 'observations' for each flight (one for every DTD and cabin)

    # The total overall length of a single input vector
    input_vector_total_length = input_length * num_time_depn_and_cabin_depn_vars
    input_vector_total_length += num_DTDs_used * num_non_cabin_time_dep_vars
    input_vector_total_length += num_cnst_vars

    # The width of the time-dependent, but *non-cabin dependent* data
    time_depn_not_cabin_depn_width = num_non_cabin_time_dep_vars * num_DTDs_used

    # The width of a single time-dependent cabin-level 'observation' (i.e. a particular DTD and cabin)
    observation_width = int( (input_vector_total_length - num_cnst_vars - time_depn_not_cabin_depn_width) / input_length )

    return input_vector_total_length,time_depn_not_cabin_depn_width,observation_width

###############################################################################################################

def print_input_vector_description(X,y,DTDs_used):
    """ Prints a human-readable description of the input vectors to terminal.
        Useful for common sense checks, understanding the data, etc. """

    num_DTDs_used = len(DTDs_used)

    # DTD and cabin-dependent variables
    print("The following variables can change at every timestep, and are different for each cabin:")
    print(time_depn_and_cabin_depn_vars)
    for idx,DTD in enumerate(DTDs_used):
        start = 2 * observation_width * idx
        print('Cabin C (%d DTD):' % (DTD))
        print( X[start:start+num_time_depn_and_cabin_depn_vars] )
        print('Cabin M (%d DTD):' % (DTD))
        print( X[start+num_time_depn_and_cabin_depn_vars:start+2*num_time_depn_and_cabin_depn_vars] )

    # The non-cabin dependent, but time-dependent, data starts at this index:
    p = len(DTDs_used) * 2*num_time_depn_and_cabin_depn_vars

    # MIN_COMP_PRICE, MIN_BA_PRICE and SEATS_AVL
    print("The following variables can change at every timestep, but do not depend on cabin:")
    print(time_depn_not_cabin_depn_vars)
    print('MIN_COMP_PRICE:',use_these_DTDs)
    print( X[p:p+num_DTDs_used] )
    print('MIN_BA_PRICE:',use_these_DTDs)
    print( X[p+num_DTDs_used:p+2*num_DTDs_used] )
    print('SEATS_AVL:',use_these_DTDs)
    print( X[p+2*num_DTDs_used:p+3*num_DTDs_used] )

    # The 'constant' variables start at this index:
    q = p + num_non_cabin_time_dep_vars*len(use_these_DTDs)

    # Constant vars
    print("The following variables are always constant for a given flight (i.e. do not depend on timestep or cabin):")
    print(constant_vars)
    print('Constant variables:')
    print( X[q:] )

    # Output
    print('Target Output:')
    print(y)

###############################################################################################################

def calculate_pearson(X,y,DTDs_used,show_plots=False):
    """ Function to calculate cross input-input (X-X) Pearson correlation and
        input-output (X-y) Pearson correlation, given X and y data. """

    # Construct a list such that the index of the list gives the name of the variable.
    # This will be used to format the terminal output.
    variable_reference_list = []
    for DTD in DTDs_used:
        for j in range(2):
            for label in time_depn_and_cabin_depn_vars:
                variable_reference_list.append(label)   
    for label in time_depn_not_cabin_depn_vars:
        variable_reference_list.append(label)
    for label in constant_vars:
        variable_reference_list.append(label)

    # First, the cross input-input (X-X) terms (this gives a 2D matrix of values)

    input_cross_pearson_plot = []
    # Loop over all pairs of variables
    for i in range( len(X[0]) ):
        inputs_pearson = []
        for j in range( len(X[0]) ):
            var1 = []
            var2 = []
            for idx,x in enumerate(X):
                var1.append(x[i])
                var2.append(x[j])
            pearson_val = pearsonr(var1,var2)[0]
            inputs_pearson.append( pearson_val )
            if ( (abs(pearson_val) > 0.90) and (i!=j) and (i>j) ):
                print( "Input variables %d (%s) and %d (%s) are strongly correlated, with Pearson Correlation: %f" %
                       (i,variable_reference_list[i],j,variable_reference_list[j],pearson_val) )
        input_cross_pearson_plot.append( inputs_pearson )

    input_cross_pearson_plot = np.array(input_cross_pearson_plot)
    input_cross_pearson_plot = input_cross_pearson_plot.reshape(len(X[0]),len(X[0]))
    if show_plots:
        plt.imshow(input_cross_pearson_plot)
        plt.show()

    print('\n')

    # Now, the input-output (X-y) terms (this gives a 1D output of values)

    output_pearson_plot = []
    # Loop over all variables
    for i in range( len(X[0]) ):
        var1 = []
        ys = []
        for idx,x in enumerate(X):
            var1.append(x[i])
            ys.append(y[idx][0])
        pearson_val = pearsonr(var1,ys)[0]
        output_pearson_plot.append( pearson_val )
        if ( (abs(pearson_val) > 0.75) ):
            print( "Input variable %d (%s) is strongly correlated with the output, with Pearson Correlation: %f" %
                   (i,variable_reference_list[i],pearson_val) )

    if show_plots:
        plt.plot(output_pearson_plot)
        plt.xlabel('Variable Index')
        plt.ylabel('Pearson Correlation')
        plt.show()

    print('\n')

###############################################################################################################

def rmse_and_mae_score(y_predicted,y_actual):
    """ Returns the average RMSE and MAE given a provided set of
        predicted values (y_predicted) and actual values (y_actual). """
    
    avg_RMSE = 0
    avg_MAE = 0
    num_preds = len(y_predicted)
    for idx,prediction in enumerate(y_predicted):
        pred,actual = int(prediction),int(y_actual[idx]) # Snap to int
        SE = (pred - actual)**2
        avg_RMSE += SE
        AE = abs(pred - actual)
        avg_MAE += AE
    avg_RMSE = np.sqrt(avg_RMSE / num_preds)
    avg_MAE = avg_MAE / num_preds
    return avg_RMSE,avg_MAE

###############################################################################################################

def make_random_predictions(model,X_dataset,y_dataset,num_preds,print_each_prediction=False):
    """ Returns 'num_preds' randomly chosen predictions using 'model'
        for the predictions, along with the MAE and RMSE. The X (y) data
        is drawn from 'X_dataset' ('y_dataset') respectively. It is assumed
        that 'model' has a .predict method compliant with the sklearn API.

        If the boolean 'print_each_prediction' is True, then every prediction
        (along with its actual value) is printed to terminal. """
    
    # Pick some random datapoints from supplied dataset
    X_for_predict,y_for_predict = [],[]
    data_size = len(X_dataset)
    for i in range(num_preds):
        idx = int(np.random.random() * data_size)
        X_for_predict.append( X_dataset[idx] )
        y_for_predict.append( y_dataset[idx] )
    X_for_predict = np.array(X_for_predict)
    y_for_predict = np.array(y_for_predict)
        
    # Predict
    y_pred = model.predict(X_for_predict)

    if print_each_prediction:
        # repr returns a string representation like 'ModelName(various model info)',
        # so just take the first bit before the bracket to get the name of the model
        model_name = repr(model).split('(')[0]
        print( "Here are %d randomly chosen predictions for %s..." % (num_preds,model_name) )
        for idx,prediction in enumerate(y_pred):
            pred,actual = int(prediction),int(y_for_predict[idx]) # Snap to int
            print( "y_pred= %f, y_actual=%f" % (pred,actual) ) # Maybe add FLT_NO, date, UPL_STN, DSG_STN info in future?
    # MAE and RMSE
    avg_RMSE,avg_MAE = rmse_and_mae_score(y_pred,y_for_predict)
    print( "On these %d randomly chosen predictions, %s achieved a MAE of %f and a RMSE of %f" % (num_preds,model_name,avg_MAE,avg_RMSE) )

    return (y_pred,avg_RMSE,avg_MAE)

