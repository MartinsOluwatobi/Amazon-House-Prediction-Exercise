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
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class AgeCalculator(BaseEstimator,TransformerMixin):
    def fit (self,x,y= None):
        return self
    
    def transform(self,x):
        x_copy = x.copy()
        x_copy['Date']= pd.to_datetime(x_copy['Date'], format = '%d-%m-%Y')
        x_copy['Age']= 2025 - x_copy['Date'].dt.year 
        x_copy.drop('Date', axis = 1, inplace = True)
        return x_copy
    
class BedroomImputer(BaseEstimator,TransformerMixin):
    def fit (self,x,y= None):
        return self
    def transform (self,x):
        x_copy = x.copy()
        x_copy['Bedroom']= x_copy['Bedroom'].fillna(value = x_copy['Rooms']) 
        empty_bedroom = x_copy['Bedroom']==0 
        x_copy.loc[empty_bedroom,'Bedroom'] = x_copy.loc[empty_bedroom,'Rooms'] 
        condition = x_copy['Bedroom'] > x_copy['Rooms'] 
        x_copy.loc[condition, 'Rooms'] = x_copy.loc[condition, 'Bedroom']
        return x_copy
    
class IQROutlierReplacer(BaseEstimator,TransformerMixin):
    def __init__(self, numerical_data):
        self.numerical_data = numerical_data
        self.limit_ = {}

    def fit(self,x,y=None):
        x_num = x.loc[:,self.numerical_data]
        for col in self.numerical_data:
            Q3= x_num[col].quantile(0.75) 
            Q1= x_num[col].quantile(0.25) 
            IQR = Q3 - Q1 
            lower_limit = Q1 - 1.5 * IQR 
            higher_limit = Q3 + 1.5*IQR 
            median_val = x_num[col].median()
            self.limit_[col]= (lower_limit,higher_limit,median_val)
        return self 

    def transform (self,x):
        x_copy = x.copy()
        for col in self.numerical_data:
            if col in self.limit_ and col in x_copy.columns:
               low_limit, up_limiit, median = self.limit_[col]
               outliers_conditions= (x_copy[col] < low_limit) | (x_copy[col] > up_limiit)
               x_copy.loc[outliers_conditions,col]= median
        return x_copy

class NumericalScaler(BaseEstimator,TransformerMixin):

    def __init__(self,numerical_data):
        self.numerical_data = numerical_data
        self.param_ = {}

    def fit(self,x,y =None):
        x_num = x.copy()
        for col in self.numerical_data: 
            col_mean = x_num[col].mean()
            col_std = x_num[col].std()
            self.param_[col]= (col_mean,col_std)
        return self 
    
    def transform ( self,x,y=None):
        x_num = x.copy()
        for col in self.numerical_data: 
            mean, std = self.param_[col]
            x_num[col]= (x_num[col]- mean)/std
        return x_num

class NumericalSimpleImputer (BaseEstimator,TransformerMixin):
    
    def __init__ (self,numerical_data,strategy):
        self.numerical_data = numerical_data
        self.replace_values = {}
        self.strategy = strategy
    def fit(self,x,y=None):
        x_num = x.copy()
        for col in self.numerical_data:
            if self.strategy == 'median':
               col_input_value = x_num[col].median()
               self.replace_values[col]= col_input_value
            elif self.strategy == 'mean':
               col_input_value = x_num[col].mean()
               self.replace_values[col]= col_input_value
            elif self.strategy == 'most_frequent':
                col_input_value = x_num[col].mode()
                self.replace_values[col] = col_input_value
        return self 

    def transform (self,x,y=None):
        x_num = x.copy()
        for col in self.numerical_data:
            fill_value = self.replace_values[col]
            x_num[col]=x_num[col].fillna(fill_value)
        return x_num
    


#import data 
path = r"C:\Users\marti\OneDrive\Documents\Melbourne_Housing.csv" 
data = pd.read_csv(path, na_values=['missing','inf'])

#Drop duplicate data point
data= data.drop_duplicates()

data= data.drop(['BuildingArea','YearBuilt'],axis = 1) 
data= data.dropna(subset= ['Landsize'])
numerical_data = data.select_dtypes(include= 'number').columns.to_list()
to_remove_feature = ['Postcode','Price']
numerical_data = [ var for var in numerical_data if var not in to_remove_feature]
numerical_data.append('Age')
categorical_data = data.select_dtypes(include=['string','object']).columns.to_list()
categorical_data.remove('Date')
categorical_data.append('Postcode')
#data split 
x_train,x_test,y_train,y_test = train_test_split (data.drop(['Price'], axis = 1), data['Price'], random_state= 42, test_size= 0.2)
basic_preprocessor = ColumnTransformer(transformers=[('num',Pipeline([('nan handling',NumericalSimpleImputer(numerical_data,strategy= 'median')),('Outlier handling',IQROutlierReplacer(numerical_data=numerical_data)),('scaling', NumericalScaler(numerical_data))]),
                                                                      numerical_data),
                                                                      ('cat',Pipeline([('nan handling_2',SimpleImputer(strategy= 'most_frequent')),('encoding', OneHotEncoder(handle_unknown='ignore'))]),categorical_data)],remainder= 'passthrough',sparse_threshold= 0)

k = list(range(1,833,5))
rmse_values = {}
for k_value in k:
        full_preprocessing = Pipeline([('age calculator',AgeCalculator()),('Bedroom Imputer',BedroomImputer()),('preprocessor',basic_preprocessor),('Feature Selection', SelectKBest(f_regression,k=k_value)),('regressor',LinearRegression())])
        full_preprocessing.fit(x_train,y_train)
        y_pred = full_preprocessing.predict(x_test)
        rmse= np.sqrt(mean_squared_error(y_test,y_pred))
        rmse_values[k_value] = rmse
least_rmse = min(rmse_values.values())
for key in rmse_values:
    if rmse_values[key] == least_rmse:
       k_value = key
       break


models = [('Linear Regression', LinearRegression()),("Random Sampling Algorithm", RANSACRegressor(residual_threshold=5.0, random_state=42)),('TheilSenRegression',TheilSenRegressor(random_state= 42)),('Huber Regression', HuberRegressor()),("Lasso Regression",Lasso(random_state=42)),('Ridge Regression', Ridge(random_state=42))]

for name, model in models: 
    full_preprocessing = Pipeline([('age calculator',AgeCalculator()),('Bedroom Imputer',BedroomImputer()),('preprocessor',basic_preprocessor),('Feature Selection', SelectKBest(f_regression,k=k_value)),('regressor',model)])
    full_preprocessing.fit(x_train,y_train)
    y_pred = full_preprocessing.predict(x_test)
    r2_score_model = r2_score(y_test,y_pred)
    rmse_model = np.sqrt(mean_squared_error(y_test,y_pred))
    residual = y_test - y_pred
    plt.scatter(y_pred,residual)
    plt.xlabel('Actual value')
    plt.ylabel('Residual')
    plt.title (f'Residual plot of {name} model')
    plt.show()
    print(f'The r squared value of the test data using {name} is {r2_score_model}\nThe root mean squared error of the test data using {name} is {rmse_model}\n')

