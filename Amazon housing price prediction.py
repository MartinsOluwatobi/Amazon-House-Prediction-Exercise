import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re

housing_data = pd.read_csv(r"C:\Users\marti\Downloads\American_Housing_Data_20231209.csv")

# catching any nan values in the dataset 
columns_with_nan =[]
for column in housing_data.columns:
    if housing_data[column].isna().any():
        columns_with_nan.append(column)

#Median Household column has 2 nan values that couldn't be replaced with either mode or media and I have decided to drop it
nan_index = housing_data[housing_data["Median Household Income"].isna()].index
housing_data =housing_data.drop(index = nan_index)

#Drop duplicate by row, one-hot encode categorical variable 
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