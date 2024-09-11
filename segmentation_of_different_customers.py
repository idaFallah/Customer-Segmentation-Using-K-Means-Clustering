
# importing libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importing dataset
dataset = pd.read_csv('/content/CC GENERAL.csv')

# data exploration
dataset.head()

dataset.shape

dataset.info()

dataset.columns

# categorical columns
dataset.select_dtypes(include='object').columns

len(dataset.select_dtypes(include='object').columns)

# numerical columns
dataset.select_dtypes(include=['int64', 'float64']).columns

len(dataset.select_dtypes(include=['int64', 'float64']).columns)

# statistical summary
dataset.describe()

# dealing with missing values
dataset.isnull().values.any()

dataset.isnull().values.sum()

dataset.columns[(dataset.isnull().any())]

len(dataset.columns[(dataset.isnull().any())])

dataset['CREDIT_LIMIT'] = dataset['CREDIT_LIMIT'].fillna(dataset['CREDIT_LIMIT'].mean())
dataset['MINIMUM_PAYMENTS'] = dataset['MINIMUM_PAYMENTS'].fillna(dataset['MINIMUM_PAYMENTS'].mean())

dataset.isnull().values.any()

dataset.isnull().values.sum()

dataset.columns[(dataset.isnull().any())]

len(dataset.columns[(dataset.isnull().any())])

# encoding categorical data
dataset.select_dtypes(include='object').columns

dataset.head()

# since id is not important in ML calculations we just drop it instead of hot encoding it
dataset = dataset.drop(columns='CUST_ID')

dataset.head()

dataset.select_dtypes(include='object').columns

len(dataset.select_dtypes(include='object').columns)

# correlation matrix
corr = dataset.corr()

# heatmap
plt.figure(figsize=(16, 9))
ax = sns.heatmap(corr, annot=True, cmap='coolwarm')

# splitting the dataset
# in this dataset we only have independant variables, not dependant/ target variables(only a classification project, not prediction)

# feature scaling
dataset_beforeScaling = dataset

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset = sc.fit_transform(dataset)

dataset

# finding the optimal number of clusters by elbow method

from sklearn.cluster import KMeans

wcss = []
#Within Clusters Sum of Squares
for i in range(1, 20):
  kmeans = KMeans(n_clusters=i, init='k-means++')
  kmeans.fit(dataset)
  wcss.append(kmeans.inertia_)

plt.plot(range(1, 20), wcss, 'bx-')
plt.title('The elbow method')
plt.xlabel('Number f clusters')
plt.ylabel('WCSS')
plt.show()

# optimal number of clusters is kinda around 7-8 here, where the slope with sharp degree ends and the slope softens

# building the model
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8, init='k-means++', random_state=0)

y_kmeans = kmeans.fit_predict(dataset)
# creating a dependant variable by training the k_means on the dataset

y_kmeans

# getting the output

y_kmeans.shape

y_kmeans = y_kmeans.reshape(len(y_kmeans), 1)

y_kmeans.shape

b = np.concatenate((y_kmeans, dataset_beforeScaling), axis=1)

dataset_beforeScaling.columns

final_dataframe = pd.DataFrame(data=b, columns=['Cluster_Number', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
       'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
       'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
       'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
       'TENURE'])

final_dataframe.head()

final_dataframe.to_csv('Segmented_Customers')



