import pandas as pd
df=pd.read_csv('titanic.csv')

#Display The First Few Rows Of The Dataset
print(df.head())

#Summary Statistics
print(df.describe())

#Information About The Dataset
print(df.info())

#Check For Missing Values
print(df.isnull().sum())

#Drop Rows With Missing Valueds and Place It In A New Variable
df_cleaned=df.dropna()



#Identify Duplicates
print(df.duplicated().sum())

#Remove Duplicates
df_no_duplicates=df.drop_duplicates()

#Convert 'Column1' to float
df['Survived'] = df['Survived'].astype(float)

#Display updated data types
print(df.dtypes)

#To convert categorical data from the column "Name" to numerical data
df_encode = pd.get_dummies(df, columns=['Name'])

#Using median calculations and IQR, outliers are identified and these data points should be removed
Q1 = df["PassengerId"].quantile(0.25)
Q3 = df["PassengerId"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[df["PassengerId"].between(lower_bound, upper_bound)]