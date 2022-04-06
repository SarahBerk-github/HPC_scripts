# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:18:02 2022

@author: Sarah_Berk
"""

# GAM code
# Code for fitting the linear GAM and saving the model 

# Import required packages
import numpy as np
import pandas as pd
import pickle

#use grid search to find hyperparameters (pg 73 Geron) RFR
#from sklearn.model_selection import GridSearchCV                      #for cross validation
#from sklearn.feature_selection import RFE                             #for selecting features for the linear reg
#from sklearn.model_selection import cross_val_score                   #for cross validation
#from sklearn.model_selection import KFold                            
#from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import StandardScaler                      #for normalising the data
#from sklearn.metrics import r2_score                                  #metrics for assessing model
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_squared_error

#for timing how long it takes to fit model
from datetime import datetime

# Load up the data

#read in the city info table
CITY_COUNTRY_lat_lon = pd.read_excel('CITY_COUNTRY_lat_lon.xlsx', index_col=None)

#read the table with all variables in as pickle
with open('aqua_all_monthly_data_df2.pkl', 'rb') as f:
    all_monthly_data_df = pickle.load(f)

# remove years after 2015 until have sorted the area data for 2016-2020
all_monthly_data_df = all_monthly_data_df[all_monthly_data_df.year <= 2015]

# URBAN - RURAL differences
# add evi difference variable
all_monthly_data_df['evi_diff'] = all_monthly_data_df['urb_mean_evi'] - all_monthly_data_df['rur_mean_evi'] 
# add in climate difference variables
all_monthly_data_df['evap_fract_diff'] = all_monthly_data_df['urban_evap_fract'] - all_monthly_data_df['rural_evap_fract'] 
all_monthly_data_df['rh_diff'] = all_monthly_data_df['urban_rh'] - all_monthly_data_df['rural_rh']
all_monthly_data_df['tp_diff'] = all_monthly_data_df['urban_tp'] - all_monthly_data_df['rural_tp']
all_monthly_data_df['ssr_diff'] = all_monthly_data_df['urban_ssr'] - all_monthly_data_df['rural_ssr']

# add in the log_area_x_ef variable 
# Define the ef function
def logarea_ef(x_ef, threshold_ef, m, c):
    x, ef = x_ef
    y = m*(x * (ef - threshold_ef)) + c
    return y
# set values of the area function parameters
threshold_ef = 0.498 
m = 6.622
c = 2.797
x_ef = np.log10(all_monthly_data_df.Area.values.astype(float)), all_monthly_data_df.annual_rural_ef

all_monthly_data_df['log_area_x_rur_ef'] = logarea_ef(x_ef, threshold_ef, m, c)

#split into training and test data

#define the test cities to remove
test_cities = ['Sikasso', 'Anapolis', 'Campo_Grande', 'Bobo_Dioulasso', 'Diwaniyah', 'Al_Obeid', 'Akola', 'Potiskum','Cascavel']

#define the overpass time 
overpass_time = '13:30'

#define the predictor and target values for training and test data

# just ssr and ef
predictor_variables = ['evap_fract_diff', 'rural_evap_fract','ssr_diff','rural_ssr','rur_mean_evi', 'evi_diff', 
                         'log_area_x_rur_ef', 'Eccentricity']


target_variable = 'a0'
#target_variable = 'method_2_SUHI'
#target_variable = 'quantile_a0'

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

# fit the linear GAM
from pygam import LinearGAM   #import GAM
gam = LinearGAM().fit(X_train_norm, y_train)

lam = np.array([0.6,10,100]) # hyperparameter values to try
lams = [lam] * 8 # there are 8 splines (for 8 variables)
print("Fitting the GAM...")
# time how long it takes to fit the GAM
startTime = datetime.now()
gam_reg = gam.gridsearch(X_train_norm, y_train, lam = lams)

#save the model
filename = 'finalised_model_gam_a0.sav'
pickle.dump(gam_reg, open(filename, 'wb'))

# print how long it took to fit model 
print(datetime.now() - startTime)