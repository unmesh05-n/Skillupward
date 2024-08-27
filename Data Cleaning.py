import pandas as pd
import numpy as np
#Loading The Dataset , Here We Are Using Titanic Dataset
df=pd.read_csv('titanic.csv')
df.head()
df.duplicated()
df.info()
#Categorized Columns
cat_col=[col for col in df.columns if df[col].dtype=='object']
print('Categorical Columns: ',cat_col)
#Numerical Columns
num_col=[col for col in df.columns if df[col].dtype=='object']
print('Numerical Columns: ',num_col)
df[cat_col].nunique()


#Data Cleansing Starts

df['Ticket'].unique()[:50]

df1=df.drop(columns=['Name','Ticket'])
df1.shape

round((df1.isnull().sum()/df1.shape[0])*100,2)

df2 = df1.drop(columns='Cabin')
df2.dropna(subset=['Embarked'], axis=0, inplace=True)
df2.shape

# Mean imputation
df3 = df2.fillna(df2.Age.mean())
# Let's check the null values again
df3.isnull().sum()

import matplotlib.pyplot as plt

plt.boxplot(df3['Age'], vert=False)
plt.ylabel('Variable')
plt.xlabel('Age')
plt.title('Box Plot')
plt.show()

# calculate summary statistics
mean = df3['Age'].mean()
std  = df3['Age'].std()

# Calculate the lower and upper bounds
lower_bound = mean - std*2
upper_bound = mean + std*2

print('Lower Bound :',lower_bound)
print('Upper Bound :',upper_bound)

# Drop the outliers
df4 = df3[(df3['Age'] >= lower_bound) 
                & (df3['Age'] <= upper_bound)]

X = df3[['Pclass','Sex','Age', 'SibSp','Parch','Fare','Embarked']]
Y = df3['Survived']

from sklearn.preprocessing import MinMaxScaler

# initialising the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Numerical columns
num_col_ = [col for col in X.columns if X[col].dtype != 'object']
x1 = X
# learning the statistical parameters for each of the data and transforming
x1[num_col_] = scaler.fit_transform(x1[num_col_])
x1.head()
