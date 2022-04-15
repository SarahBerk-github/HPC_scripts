# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:18:02 2022

@author: Sarah_Berk
"""

# Code for all models
# Code for fitting MLR, RFR, GAM and GPR and saving the models/ metrics 

# Import required packages
import numpy as np
import pandas as pd
import pickle

#use grid search to find hyperparameters
from sklearn.model_selection import GridSearchCV                  #for cross validation
#from sklearn.feature_selection import RFE                        #for selecting features for the linear reg
#from sklearn.model_selection import cross_val_score              #for cross validation
#from sklearn.model_selection import KFold                             
from sklearn.preprocessing import StandardScaler                  #for normalising the data
from sklearn.metrics import r2_score                              #metrics for assessing model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#for timing how long it takes to fit model
from datetime import datetime

#Load up the data
CITY_COUNTRY_lat_lon = pd.read_excel('CITY_COUNTRY_lat_lon.xlsx', index_col=None) # city info

with open('aqua_all_monthly_data_df2.pkl', 'rb') as f:                            # variables
    all_monthly_data_df = pickle.load(f)

#URBAN - RURAL differences
#add evi difference variable
all_monthly_data_df['evi_diff'] = all_monthly_data_df['urb_mean_evi'] - all_monthly_data_df['rur_mean_evi'] 
#add in climate difference variables
all_monthly_data_df['evap_fract_diff'] = all_monthly_data_df['urban_evap_fract'] - all_monthly_data_df['rural_evap_fract'] 
all_monthly_data_df['rh_diff'] = all_monthly_data_df['urban_rh'] - all_monthly_data_df['rural_rh']
all_monthly_data_df['tp_diff'] = all_monthly_data_df['urban_tp'] - all_monthly_data_df['rural_tp']
all_monthly_data_df['ssr_diff'] = all_monthly_data_df['urban_ssr'] - all_monthly_data_df['rural_ssr']

#add in the log_area_x_ef variable 
#Define the ef function
def logarea_ef(x_ef, threshold_ef, m, c):
    x, ef = x_ef
    y = m*(x * (ef - threshold_ef)) + c
    return y
#set values of the area function parameters
threshold_ef = 0.469 
m = 6.661
c = 2.543
x_ef = np.log10(all_monthly_data_df.Area.values.astype(float)), all_monthly_data_df.annual_rural_ef

all_monthly_data_df['log_area_x_rur_ef'] = logarea_ef(x_ef, threshold_ef, m, c)

#split into training and test data
#define the test cities to remove
test_cities = ['Sikasso', 'Anapolis', 'Campo_Grande', 'Bobo_Dioulasso', 'Diwaniyah', 'Al_Obeid', 'Akola', 'Potiskum','Cascavel']
#define the overpass time 
overpass_time = '13:30'
#define the predictor and target values for training and test data
#just ssr and ef
predictor_variables = ['evap_fract_diff', 'rural_evap_fract','ssr_diff','rural_ssr','rur_mean_evi', 'evi_diff', 
                         'log_area_x_rur_ef', 'Eccentricity']
target_variable = 'a0'
#target_variable = 'method_2_SUHI'
#target_variable = 'footprint_area'

#clean the data - first remove columns which are not the target or predictor variables, then remove nans
variables = predictor_variables.copy()
variables.append(target_variable)
variables.append('City')
variables.append('Overpass')
all_monthly_data_df2 = all_monthly_data_df[variables]
all_monthly_data_df2 = all_monthly_data_df2.dropna()

#create training and test datasets
training_data = all_monthly_data_df2[((~all_monthly_data_df2['City'].isin(test_cities)) 
                                    & (all_monthly_data_df2['Overpass'] == overpass_time))].copy()
test_data = all_monthly_data_df2[((all_monthly_data_df2['City'].isin(test_cities))
                                  & (all_monthly_data_df2['Overpass'] == overpass_time))].copy()

print('Train_percent', 100* len(training_data)/len(all_monthly_data_df2[all_monthly_data_df2['Overpass'] == overpass_time]))
print('Test_percent', 100* len(test_data)/len(all_monthly_data_df2[all_monthly_data_df2['Overpass'] == overpass_time]))
print('Total Datapoints',len(all_monthly_data_df2[all_monthly_data_df2['Overpass'] == overpass_time]))

#split the data into training and test
X_train = training_data[predictor_variables]  #predictors
y_train = training_data[target_variable]      #target

X_test = test_data[predictor_variables]       #predictors
y_test = test_data[target_variable]           #target

#get the groups
groups_train = training_data.City
groups_test = test_data.City

#create normalised datasets
#the test dataset is normalised using the normalisation parameters from the training data
scaler = StandardScaler()
X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns) #fit and transform
X_test_norm = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns) #transform only
y_train_norm = pd.DataFrame(scaler.fit_transform(y_train.values.reshape(-1,1)), columns = [target_variable]) #fit and transform
y_test_norm = pd.DataFrame(scaler.transform(y_test.values.reshape(-1, 1)), columns = [target_variable]) #transform only

#create lists for the model metrics to be stored 
model_name_list = [] 
train_r2_list = []
test_r2_list = []
train_rmse_list = []
test_rmse_list = []
train_mae_list = []
test_mae_list = []

#function to append the model metrics
def add_metrics(model, y_train, y_test, y_train_pred, y_test_pred):
    model_name_list.append(model)
    train_rmse_list.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_rmse_list.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    train_r2_list.append(r2_score(y_train, y_train_pred))
    test_r2_list.append(r2_score(y_test, y_test_pred))
    train_mae_list.append(mean_absolute_error(y_train, y_train_pred))
    test_mae_list.append(mean_absolute_error(y_test, y_test_pred))

###################################
####### FIT THE MLR MODEL #########
###################################

#Import and create linear regression object
from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()

#Fit the model
print("Fitting the MLR...")
startTime = datetime.now()                                      # start timer for MLR fit
lin_reg.fit(X_train_norm, y_train)
print('MLR fit time:', datetime.now() - startTime)              # print how long it took to fit model

#Generate r square, RMSE and MAE
y_train_pred = lin_reg.predict(X_train_norm)                    # predict the training 
y_test_pred = lin_reg.predict(X_test_norm)                      # predict the test
add_metrics('MLR', y_train, y_test, y_train_pred, y_test_pred)  # generate the stats & save

#Save the model
filename = 'mlr_{}.sav'.format(target_variable)
pickle.dump(lin_reg, open(filename, 'wb'))

###################################
######### FIT THE RFR  ############
###################################

#Import and create rfr object
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()

#fit the model - first carry out cross validation
print("Fitting the RFR...")
startTime = datetime.now()                                            #start timer for RFR fit
folds = 5                                                             #define the number of folds (K)
#specify the hyperparameters to test
#num_features_for_split = total_input_features/3
hyper_params = {'n_estimators':[5,10,20,40,60,80,100]                 #the number of trees
                ,'max_depth':[10,20,30,40,50,60,70,80,90,100,None]    #the tree depth
                ,'min_samples_leaf':[1,2,4,15]                        #min number of samples required at each leaf node
                ,'min_samples_split':[2,5,10,15] }                    #min number of samples required to split a node

#use cross validation to determine the hyperparameter values
rfr_model_cv = GridSearchCV(estimator = forest_reg, 
                        param_grid = hyper_params, 
                        cv = folds, 
                        #verbose = 1,                                  #verbose- how detailed output is   
                        return_train_score=True) 
#fit the model
rfr_model_cv.fit(X_train_norm, y_train)
print('RFR fit time:', datetime.now() - startTime)                    #print how long it took to fit model

#Generate r square, RMSE and MAE
y_train_pred = rfr_model_cv.predict(X_train_norm)                     #predict the training 
y_test_pred = rfr_model_cv.predict(X_test_norm)                       #predict the test
add_metrics('RFR', y_train, y_test, y_train_pred, y_test_pred)        #generate the stats & save

#save the model
filename = 'RFR_{}.sav'.format(target_variable)
pickle.dump(rfr_model_cv, open(filename, 'wb'))

###################################
######### FIT THE GAM  ############
###################################

#import and create linear GAM object
from pygam import LinearGAM   
gam = LinearGAM().fit(X_train_norm, y_train)

#fit the model
lam = np.array([0.6,10,100])                                    #hyperparameter values to try
lams = [lam] * 8                                                #there are 8 splines (for 8 variables)
print("Fitting the GAM...")
startTime = datetime.now()                                      #start timer for GAM fit
gam_reg = gam.gridsearch(X_train_norm, y_train, lam = lams)
print('GAM fit time:', datetime.now() - startTime)              #print how long it took to fit model 

#Generate r square, RMSE and MAE
y_train_pred = gam_reg.predict(X_train_norm)                    #predict the training 
y_test_pred = gam_reg.predict(X_test_norm)                      #predict the test
add_metrics('GAM', y_train, y_test, y_train_pred, y_test_pred)  #generate the stats & save

#save the model
filename = 'gam_{}.sav'.format(target_variable)
pickle.dump(gam_reg, open(filename, 'wb'))

###################################
######### FIT THE GPR  ############
###################################

#import the GPR and kernels to try
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF               #import the kernels for GPR
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern

gpr = GaussianProcessRegressor()                               #define the model
RBF_kernel = RBF()                                             #define the kernels
Dot_Product_kernel = DotProduct()
Matern_kernel = Matern()

#fit the model - first carry out cross validation
print("Fitting the GPR...")
startTime = datetime.now()                                     #start timer for GPR fit
folds = 5                                                      #define the number of folds (K)                                                                   #define the number of folds (K)
#specify the hyperparameters to test
hyper_params = [{'kernel':[RBF_kernel]                         #using the RBF kernel
                ,'alpha':[1e-2, 1e-3, 1e-5, 1e-10]             #e-10 is the default alpha (value added to cov matrix diagonal)
                ,'n_restarts_optimizer':[0,20,50]              #number of restarts for the optimizer to find kernels parameters    
            } ,{'kernel':[Dot_Product_kernel]                  #using the dot prod kernel
                ,'alpha':[1e-2, 1e-3, 1e-5, 1e-10]    
                ,'n_restarts_optimizer':[0,20,50] 
            }  ,{'kernel':[Matern_kernel]                      #using the Matern kernel
                ,'alpha':[1e-2, 1e-3, 1e-5, 1e-10]    
                ,'n_restarts_optimizer':[0,20,50] 
            }]                 

#default value for max_iter = 1000
gpr_model_cv = GridSearchCV(estimator = gpr, 
                        param_grid = hyper_params, 
                        cv = folds, 
                        return_train_score=True) 

#fit the model
gpr_model_cv.fit(X_train_norm, y_train)  
print('GPR fit time:', datetime.now() - startTime)              #print how long it took to fit model

#Generate r square, RMSE and MAE
y_train_pred = gpr_model_cv.predict(X_train_norm)               #predict the training 
y_test_pred = gpr_model_cv.predict(X_test_norm)                 #predict the test
add_metrics('GPR', y_train, y_test, y_train_pred, y_test_pred)  #generate the stats & save

#save the model
filename = 'GPR_{}.sav'.format(target_variable)
pickle.dump(gpr_model_cv, open(filename, 'wb'))

####################### END OF MODEL FITS ##########################

#Put the metrics into a df and save as pickle
d = {'Model': model_name_list, 'Train_r2':train_r2_list,'Test_r2':test_r2_list, 
     'Train_RMSE':train_rmse_list,'Test_RMSE':test_rmse_list, 'Train_MAE': train_mae_list,
     'Test_MAE': test_mae_list}
metrics_df = pd.DataFrame(data = d)
with open('metrics_dataframe.pkl', 'wb') as f:
    pickle.dump(metrics_df, f)