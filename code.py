# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# code starts here
df=pd.read_csv(path)

df.head()

df.isnull().sum()

#Drop null values
df.dropna(subset=["Description","CustomerID"],inplace=True)

df.isnull().sum()

#Only select data where country is United Kingdom
df=df[df["Country"]=="United Kingdom"]


#Create new column returns
df["Return"]=df["InvoiceNo"].str.contains('C')

#store the result in purchase
df["Purchase"]=np.where(df["Return"]==True,0,1)



# code ends here


# --------------
# create new dataframe customer





customers = pd.DataFrame({'CustomerID': df['CustomerID'].unique()},dtype=int)

# calculate the recency
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Recency'] = pd.to_datetime("2011-12-10") - (df['InvoiceDate'])

# remove the time factor
df.Recency = df.Recency.dt.days

# purchase equal to one   (ie. non-cancelled orders) 
temp = df[df['Purchase']==1]

# customers latest purchase day
recency=temp.groupby(by='CustomerID',as_index=False).min()


# Create a merge b/w customers(df) and recency(df) 
customers=customers.merge(recency[['CustomerID','Recency']],on='CustomerID')












# --------------
# code stars here

#Creating a temp df
temp_1=df[["CustomerID","InvoiceNo","Purchase"]]

#dropping duplicate rows
temp_1.drop_duplicates(subset=["InvoiceNo"],inplace=True)

#Calculate frequency of purchase via grouping
annual_invoice = temp_1.groupby(["CustomerID"],as_index=False).sum()

annual_invoice.rename(columns={"Purchase":"Frequency"},inplace=True)

#customers has customerid,recency,frequency
customers=customers.merge(annual_invoice,on="CustomerID")



# code ends here


# --------------
# code starts here

#Total spent on each item

df["Amount"]=df["Quantity"]*df["UnitPrice"]

annual_sales=df.groupby(["CustomerID"],as_index=False).sum()

annual_sales.rename(columns={"Amount":"monetary"},inplace=True)

customers=customers.merge(annual_sales[["CustomerID","monetary"]],on="CustomerID")

customers

# code ends here


# --------------
# code ends here

#Preprocessing 

#Replace negative monetary values with 0
customers["monetary"]=np.where(customers["monetary"]<0,0,customers["monetary"])

customers["Recency_log"]=np.log(customers["Recency"]+0.1)

customers["Frequency_log"]=np.log(customers["Frequency"]+0.1)

customers["Montary_log"]=np.log(customers["monetary"]+0.1)

customers

# code ends here


# --------------
# import packages
from sklearn.cluster import KMeans


# code starts here
dist=[]

#Elbow method for k-means
for i in range(1,10):
    km=KMeans(n_clusters=i, init='k-means++',max_iter=300,n_init=10,random_state=0)
    km.fit(customers)
    dist.append(km.inertia_)

#plottting k-values(no.of clusters) and WCSS
plt.figure(figsize=(10,10))
plt.plot(range(1,10),dist)

# code ends here


# --------------

# Code starts here

# initialize KMeans object
cluster = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

# create 'cluster' column
customers['cluster'] = cluster.fit_predict(customers.iloc[:,1:7])

# plot the cluster

customers.plot.scatter(x= 'Frequency_log', y= 'Montary_log', c='cluster', colormap='viridis')
plt.show()

# Code ends here


