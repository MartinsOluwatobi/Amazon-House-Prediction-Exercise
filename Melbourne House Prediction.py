import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

path = r"C:\Users\marti\OneDrive\Documents\Melbourne_Housing.csv"
data = pd.read_csv(path, na_values=['missing','inf'])

data = data.dropna(subset= ['Distance'])
data['Date'] =  pd.to_datetime(data['Date'], format ='%d-%m-%Y')
data= data.dropna(subset =['Distance'])
data['Bedroom']= data['Bedroom'].fillna(value = data['Rooms'])
empty_bedroom = data['Bedroom']==0
data.loc[empty_bedroom,'Bedroom'] = data.loc[empty_bedroom,'Rooms']
condition = data['Bedroom'] > data['Rooms']
data.loc[condition, 'Rooms'] = data.loc[condition, 'Bedroom']
data['Bathroom'] = data['Bathroom'].fillna(value= data.groupby(['Rooms','Type'])['Bathroom'].transform('mean').round(0))
data['Car'] = data['Car'].fillna(value= data.groupby(['Regionname','Type'])['Car'].transform('mean').round(0))
data['Landsize'] = data['Landsize'].fillna(value = data.groupby(['Regionname','Rooms'])['Landsize'].transform('mean').round(0))
data= data.drop(['BuildingArea','YearBuilt'],axis = 1)
data= data.dropna(subset= ['Landsize'])
Outlied_data, Outlied_dataframe = [], pd.DataFrame()
numerical_data = data.select_dtypes(include = 'number')
for i in numerical_data.columns[:-1]:
    Q3= data[i].quantile(0.75)
    Q1= data[i].quantile(0.25)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5*IQR
    higher_limit = Q3 + 1.5*IQR
    outliers = data[(data[i] < lower_limit) | (data[i] > higher_limit)]
    if outliers.shape[0] > 0: 
       Outlied_data.append(i)
       Outlied_dataframe = pd.concat([Outlied_dataframe, outliers])
Outlied_dataframe.drop_duplicates()
data = data.drop(Outlied_dataframe.index)

correlation_score = numerical_data.corr()['Price'][:-1]
features= []
for i in range(len(correlation_score)):
    if abs(correlation_score[i])> 0.1:
       features.append(numerical_data.columns[i])
target = data['Price']

def ScaleNumericalFeatures(Feature_dataframe):
    scalar= StandardScaler()
    scaled = scalar.fit_transform(Feature_dataframe)
    scaled_feature = pd.DataFrame(scaled, columns= Feature_dataframe.columns)
    return scaled_feature

features = ScaleNumericalFeatures(data[features])
x_train,x_test,y_train,y_test = train_test_split(features,target, random_state = 42, test_size= 0.2)
model = LinearRegression()
model.fit(x_train,y_train)

pred_val = model.predict(x_test)
print(np.sqrt(mean_squared_error(pred_val,y_test)))