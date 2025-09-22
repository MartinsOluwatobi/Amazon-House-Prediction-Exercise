import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import re
import statsmodels as sm 

housing_data = pd.read_csv(r"C:\Users\marti\Downloads\American_Housing_Data_20231209.csv")

# catching any nan values in the dataset 
columns_with_nan =[]
for column in housing_data.columns:
    if housing_data[column].isna().any():
        columns_with_nan.append(column)

#Median Household column has 2 nan values that couldn't be replaced with either mode or media and I have decided to drop it
nan_index = housing_data[housing_data["Median Household Income"].isna()].index
housing_data =housing_data.drop(index = nan_index)

#Drop duplicate by row, target encode categorical variable 
housing_data = housing_data.drop_duplicates()
column_datatype = []
for i in housing_data.columns:
    if housing_data[i].dtype == "object":
        column_datatype.append(i)

def get_streetname_address(address):
    for suffix in ['OAKS','PT','KY','CLOSE','BOULEVARD','STREET','WALK','JUNEWAY','BROADWAY','ST','LANE','ARCH','ARC','DRIVE','RUN','ROAD','TRL', 'AVE','BLVD', 'PL', 'RD','LN','LOOP','PKWY','HILL OVAL','PARK','CONCOURSE','CT','TER','AVENUE','EXPY','WAY','DR','TPKE','CRES','SQ','CIR','CONVENT','UNIT','TURNPIKE','ALY','RITTENHOUSE']:
        pattern = rf'\b((?:\w+\s){{0,3}}){suffix}\b(?:\s\w+)?'
        match = re.search(pattern, address)
        if match:
            words = match.group(1).strip().split()
            filtered = [word for word in words if len(word) > 1 and not word.isdigit()]
            first_address = " ".join(filtered)
            return f"{first_address} {suffix}"      
    cleaned = remove_digit_from_address(address)
    filt = cleaned.split()
    if "APT" in filt:
        cleaned = " ".join(filt[:filt.index('APT')])
    return cleaned

def remove_digit_from_address(address):
    words = address.strip().split()
    filtered_word= [word for word in words if not word.isdigit()]
    return ' '.join(filtered_word)


housing_data["street"] = housing_data["Address"].apply(get_streetname_address)

#Simplifying words for likely matching before encoding
housing_data["street"] = housing_data["street"].str.replace(r'\bBOULEVARD\b$', 'BLVD', case=False, regex=True)
housing_data["street"] = housing_data["street"].str.replace(r'\bAVENUE\b$', 'AVE', case=False, regex=True)
housing_data["street"] = housing_data["street"].str.replace(r'\bSTREET\b$', 'ST', case=False, regex=True)
housing_data["street"] = housing_data["street"].str.replace(r'\bPLACE\b$', 'PL', case=False, regex=True)

housing_data.drop("Address", axis=1, inplace = True)

### housing["street"] is too sparsed. Therefore, it will be dropped to avoid curse of dimensionality 

housing_data.drop(['street'], axis = 1 , inplace = True) 

#Creating disctint address by merging State, City and County
housing_data['location'] = housing_data.apply(lambda row: f"{row['City']} {row['State']} {row['County']}", axis=1)
#Target enconding using mean of location 
housing_data['City_encoded'] = housing_data.groupby('location')['Price'].transform('mean')

#check for linearlity between target and potential features 

features = ['Zip Code','Beds','Baths','Living Space','Zip Code Population','Zip Code Density','Longitude','City_encoded']
plt.figure( figsize=(8,20))
for num in range(1, (len(features)+1)):
    plt.subplot(len(features),1,num)
    plt.plot(housing_data['Price'],housing_data[features[num-1]])
plt.show()

#checking for outliers using boxplot
 
plt.figure(figsize=(20,8))
for num in range(1, len(features)+1):
    plt.subplot(1,len(features),num)
    plt.boxplot(housing_data[features[num-1]])
    plt.title(features[num-1])
plt.show()

#dropping outlier 

outlied_feature = ['Beds','Baths','Living Space','Zip Code Population','Zip Code Density']
def drop_outlier (data, features):
    for feature in features: 
        IQR = data[feature].quantile(0.75) - data[feature].quantile(0.25)
        high_limit = data[feature].quantile(0.75) + IQR*1.5
        low_limit = data[feature].quantile(0.25) - IQR*1.5
        data.drop(data[data[feature]> high_limit].index, inplace = True)
        data.drop(data[data[feature]< low_limit].index, inplace = True)
    return data 

cleaned_housing_data = drop_outlier(housing_data, outlied_feature)

# Feature Selection and Model training to check the best feature for OLS model 

def best_feature(x_train,y_train):
    r2_score_values =[]; rmse_values = []
    for k in range(1,len(x_train.columns)+1):
        selector = SelectKBest(f_regression,k=k)
        selector.fit(x_train,y_train)

        selected_x_train= selector.transform(x_train)
        selected_x_test = selector.transform(x_test)

        model = LinearRegression()
        model.fit(selected_x_train,y_train)
        kbest_pred = model.predict(selected_x_test)
        rmse_score_kbest = round(np.sqrt(mean_squared_error(y_test, kbest_pred)),3)
        r2_score_kbest = r2_score (y_test,kbest_pred)
        r2_score_values.append(r2_score_kbest)
        rmse_values.append(rmse_score_kbest)
    rmse_percentage_change = [0]
    for i in range(1,len(rmse_values)):
        change = (rmse_values[i+1] - rmse_values[i])/rmse_values[i] * 100
        if abs(change) < 5:
           return i + 1 
    return len(x_train.columns)
     

# OLS Regression 

scalar = StandardScaler()
x = cleaned_housing_data[features]
y = cleaned_housing_data['Price']
scaled_features = scalar.fit_transform(x)
scaled_features =pd.DataFrame(scaled_features, columns = features)
x_train,x_test,y_train,y_test = train_test_split(scaled_features,y,test_size= 0.2, random_state=42)
model = LinearRegression()
k_best_value = best_feature(x_train, y_train)

selector = SelectKBest(f_regression,k = k_best_value)
selector.fit(x_train,y_train)

selected_x_train = selector.transform(x_train)
selected_x_test = selector.transform(x_test)

model.fit(selected_x_train, y_train)
pred_val = model.predict(selected_x_test)

r2_score_value = r2_score(y_test, pred_val)
rmse_value = np.sqrt(mean_squared_error(y_test, pred_val))
print(f'The r sqaured and root mean square of the model is {r2_score_value} and {rmse_value} respectively')
