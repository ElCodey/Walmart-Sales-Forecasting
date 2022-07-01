
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split 

def merge_and_drop(df, features, stores):
    df = pd.merge(left=df, right=features, how='left', on=['Store', 'Date'])
    df = pd.merge(df, stores, how='left', on='Store')
    df = df.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHoliday_y'], axis=1)
    return df

def dummy_and_one_hot(df):
    df= pd.get_dummies(df, columns=['Store', 'Dept', 'Type'])
    df['IsHoliday_x'] = df['IsHoliday_x'].replace({True: 1, False:0})
    return df

def extract_dates(train):
    train['Date'] = pd.to_datetime(train['Date'])
    train['Month'] = train['Date'].dt.month
    train['Week'] = train['Date'].dt.isocalendar().week
    train['Day'] = train['Date'].dt.dayofyear
    train['Week_till_xmas'] = 51 - train['Date'].dt.isocalendar().week
    train['Year'] = train['Date'].dt.year
  
    return train

def split_x_y(df):
    df_X = df.drop('Weekly_Sales', axis=1)
    df_y = df['Weekly_Sales']
    return df_X, df_y

def complete_clean(df, features, stores):
    df = merge_and_drop(df, features, stores)
    df = dummy_and_one_hot(df)
    df = extract_dates(df)
    
    return df

def transform_scale(df_X):
    scaler = StandardScaler()
    col_names = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Month', 'Week', 'Day', 'Week_till_xmas', 'Year']
    features = df_X[col_names]
    ct = ColumnTransformer([
        ('name', StandardScaler(), ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size','Month', 'Week', 'Day', 'Week_till_xmas', 'Year' ])
    ], remainder='passthrough')
    col_names = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Month', 'Week', 'Day', 'Week_till_xmas', 'Year']
    features = df_X[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    df_X[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Month', 'Week', 'Day', 'Week_till_xmas', 'Year']] = features
    return df_X

def split_train(X, y):
    X_train, y_train, X_dev, y_dev = train_test_split(X, y, train_size=0.9, random_state=1)
    return  X_train, y_train, X_dev, y_dev

X = pd.read_csv('X_train')
y = pd.read_csv('y_train')
X_train, y_train, X_dev, y_dev = split_train(X, y)
X_train.to_csv('X_train2')
y_train.to_csv('y_train2')
X_dev.to_csv('X_dev')
y_dev.to_csv('y_dev')