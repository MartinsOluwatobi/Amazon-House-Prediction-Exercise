import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor, HuberRegressor, Ridge, Lasso
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import f_regression, SelectKBest
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
import statsmodels.robust.norms 
from sklearn.pipeline import Pipeline 

#import data 
path = r"C:\Users\marti\OneDrive\Documents\Melbourne_Housing.csv" 
data = pd.read_csv(path, na_values=['missing','inf'])

#Drop duplicate data point
data= data.drop_duplicates()

#The only single missing row value in Distance column was filled using the mean distance of the houses in it region group
data['Distance'] = data['Distance'].fillna(value = data.groupby(['Regionname'])['Distance'].transform('mean'))

#date datatype was coversion and data extraction
data['Date'] =  pd.to_datetime(data['Date'], format ='%d-%m-%Y')
data['Age'] = 2025 - data['Date'].dt.year
data.drop(['Date'], axis = 1, inplace= True)

"""I observed that majority of the Houses have the same number of rooms as their bedroom. Therefore, I used rooms to fill the na value in Bedroom For scenario where Bedroom was noted to be zero, I filled that with the value of rooms on the same column. For scenarion where bedroom is greater than rooms, the value of the bedroom is replaced with the value of the rooms """

data['Bedroom']= data['Bedroom'].fillna(value = data['Rooms']) 
empty_bedroom = data['Bedroom']==0 
data.loc[empty_bedroom,'Bedroom'] = data.loc[empty_bedroom,'Rooms'] 
condition = data['Bedroom'] > data['Rooms'] 
data.loc[condition, 'Rooms'] = data.loc[condition, 'Bedroom']

"""During EDA, the distribution of Bathroom, Car and Landsize columns were evaluated and the right centre of tendency was used to fill the na values for each of the columns"""

data['Bathroom'] = data['Bathroom'].fillna(value= data.groupby(['Rooms','Type'])['Bathroom'].transform('mean').round(0)) 
data['Car'] = data['Car'].fillna(data['Car'].median()) 
data['Landsize'] = data['Landsize'].fillna(data['Landsize'].median())
data['Postcode'] = data['Postcode'].fillna(data['Postcode'].mode()[0])

# The percentage of the nan values for BuildingArea and YearrBuilt were considered to be too much and I decided to drop the columns

data= data.drop(['BuildingArea','YearBuilt'],axis = 1) 
data= data.dropna(subset= ['Landsize'])

#Handling Outlied datapoint by filling them with the median value of their respective column

numerical_data = data.select_dtypes(include = 'number')
for i in numerical_data.columns: 
    if i == 'Price':
        continue
    Q3= data[i].quantile(0.75) 
    Q1= data[i].quantile(0.25) 
    IQR = Q3 - Q1 
    lower_limit = Q1 - 1.5 * IQR 
    higher_limit = Q3 + 1.5*IQR 
    outliers_conditions= (data[i] < lower_limit) | (data[i] > higher_limit)
    data.loc[outliers_conditions,i]= data[i].median()

# Scaling numerical data 
def ScaleNumericalFeatures(Feature_dataframe):
    scalar= StandardScaler()
    scaled = scalar.fit_transform(Feature_dataframe)
    scaled_feature = pd.DataFrame(scaled, columns= Feature_dataframe.columns, index = Feature_dataframe.index)
    for i in scaled_feature.columns:
        if i == 'Price':
            continue
        data[i] = scaled_feature[i]
    return

ScaleNumericalFeatures(numerical_data)

# OneHotEncoding Categorical data  
categorical_data = data.select_dtypes(include = ['string','object']).columns.to_list()
encoder = ColumnTransformer(transformers = [('cat', OneHotEncoder(sparse_output= False, handle_unknown = 'ignore'), categorical_data)],remainder= 'passthrough')
encoded = encoder.fit_transform(data)
encoded_array = pd.DataFrame(encoded, columns = encoder.get_feature_names_out())
encoded_array.reset_index(drop= True , inplace = True)

# Changing the columns name after encoder altered it 
data_columns=data.select_dtypes(['number']).columns.to_list() 
for i in range(len(data.select_dtypes(['number']).columns.to_list())):
    old_name= 'remainder__' + data_columns[i]
    new_name = data_columns[i]
    encoded_array.rename(columns= {old_name: new_name},inplace= True)

# feature selection
target = encoded_array['Price']
feature = encoded_array.drop(['Price'],axis = 1)
x_train,x_test,y_train,y_test = train_test_split(feature,target, random_state=42, test_size= 0.2)
test_model = LinearRegression(); rmse_list = []
for k in range(1,len(feature.columns)+1):
    selector = SelectKBest(f_regression,k = k)
    selector.fit(x_train,y_train)

    x_train_selected = selector.transform(x_train)
    x_test_selected = selector.transform(x_test)

    test_model.fit(x_train_selected,y_train)
    y_pred = test_model.predict(x_test_selected)
    test_root_mean_square_error = np.sqrt(mean_squared_error(y_test,y_pred))
    
    rmse_list.append(test_root_mean_square_error)

min_rmse =min(rmse_list)
k =  rmse_list.index(min_rmse) + 1

selector = SelectKBest(f_regression, k = k)
selector.fit(x_train,y_train)

mask = selector.get_support()
Features = x_train.columns[mask]

# Training of model 

model = LinearRegression()
x_train, x_test , y_train, y_test = train_test_split(encoded_array[Features], encoded_array['Price'], random_state=42, test_size= 0.2)
model.fit(x_train, y_train)
pred_val = model.predict(x_test)
root_mean_square_error = np.sqrt(mean_squared_error(y_test,pred_val))
r2_score_ = r2_score(y_test,pred_val)

print(f'The root mean square error using Linear Regression is {root_mean_square_error}')
print(f'The r squared of the Linear Regression is {r2_score_}')

# After doing residual plot analysis, heteroscadasticity was confirmed. Therefore, I proceeded to weighted linear regression as a solution for the heteroscadasticity 

x_train_with_constant = sm.add_constant( x_train)
ols_model = sm.OLS(y_train,x_train_with_constant).fit()
ols_residual = ols_model.resid 
weights_wls = 1/(ols_residual**2 + 1e-8)
wls_model = sm.WLS(y_train,x_train_with_constant, weights= weights_wls).fit()
x_test_with_constant = sm.add_constant(x_test)
wls_pred_val =wls_model.predict(x_test_with_constant)

wls_root_mean_square_error = np.sqrt(mean_squared_error(y_test,wls_pred_val))
wls_r2_score = r2_score(y_test,wls_pred_val)

print(f'The root mean square error of weighted linear regression is {wls_root_mean_square_error}')
print(f'The r squared of weighted linear regression is {wls_r2_score}')


models = [("Random Sampling Algorithm", RANSACRegressor(residual_threshold=5.0, random_state=42)),('TheilSenRegression',TheilSenRegressor(random_state= 42)),('Huber Regression', HuberRegressor()),("Lasso Regression",Lasso(random_state=42)),('Ridge Regression', Ridge(random_state=42))]

for name, model in models:
    selected_model = model
    selected_model.fit(x_train,y_train)
    selected_model_pred_val = selected_model.predict(x_test)
    selected_model_rmse = np.sqrt(mean_squared_error(y_test,selected_model_pred_val))
    selected_model_r2 = r2_score(y_test,selected_model_pred_val)
    print(f'The rmse of {name} is {selected_model_rmse} \nThe r squared score of {name} is {selected_model_r2}')
