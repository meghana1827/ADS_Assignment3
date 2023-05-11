#!/usr/bin/env python
# coding: utf-8

# In[20]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as plt
import seaborn as sns
import errors
import cluster_tools
from sklearn.cluster import KMeans


# ## Clustering (K-Means)

# In[21]:


country_codes = ['IND', 'IRQ', 'GBR', 'ISR']
ind1=["SP.POP.GROW"]  # Population Growth
ind2=["EN.ATM.CO2E.KT"]  # Energy Use


# In[22]:


my_df1  = wb.data.DataFrame(ind1, country_codes, mrv=50).T   # Read Population Growth data and transpose it
my_df1=my_df1.fillna(my_df1.mean())    # clean missing values
my_df1.head()


# In[23]:


my_df2  = wb.data.DataFrame(ind2, country_codes, mrv=50).T   # Read Energy used data and transpose it
my_df2=my_df2.fillna(my_df2.mean())    # clean missing values
my_df2.head()


# In[24]:


plt.figure(figsize=(9,4))    # plot figure size  
plt.title('Population Growth (Countries)')    # plot title
plt.plot(my_df1[my_df1.columns[0]],"mv-",label=my_df1.columns[0])    # plot line graph for 'IND' for Population Growth 
plt.plot(my_df1[my_df1.columns[1]],"bX-",label=my_df1.columns[1])    # plot line graph for 'IRQ' for Population Growth 
plt.plot(my_df1[my_df1.columns[2]],"go-",label=my_df1.columns[2])    # plot line graph for 'GBR' for Population Growth 
plt.plot(my_df1[my_df1.columns[3]],"yo-",label=my_df1.columns[3])    # plot line graph for 'ISR' for Population Growth 
plt.xlabel("Year")    # x-label
plt.xticks(rotation=90)  # rotating x-ticks (label)
plt.ylabel("Population Growth")     # y-label
plt.legend(loc="best")    # set legend to beft position 
plt.grid()   # set grid
plt.show()  # show plot


# In[25]:


plt.figure(figsize=(9,4))    # plot figure size      
plt.title('Energy Use (Countries)')    # plot title
plt.plot(my_df2[my_df2.columns[0]],"mv-",label=my_df2.columns[0])    # plot line graph for 'IND' for  Energy Use
plt.plot(my_df2[my_df2.columns[1]],"bX-",label=my_df2.columns[1])    # plot line graph for 'IRQ' for  Energy Use
plt.plot(my_df1[my_df1.columns[2]],"go-",label=my_df1.columns[2])    # plot line graph for 'GBR' for  Energy Use
plt.plot(my_df1[my_df2.columns[3]],"yD-",label=my_df2.columns[3])    # plot line graph for 'ISR' for Energy Use
plt.xlabel("Year")    # x-label
plt.xticks(rotation=90)  # rotating x-ticks (label)
plt.ylabel("Energy Use")     # y-label
plt.legend(loc="best")    # set legend to beft position 
plt.grid()   # set grid
plt.show()  # show plot


# In[26]:


plt.figure(figsize=(5,3))    # plot figure size    
plt.title('Countries by Average Population Growth')    # plot title
# bar chart for showing Average Population Growth
plt.bar(my_df1.columns,[my_df1.iloc[:,0].mean(),my_df1.iloc[:,1].mean(),my_df1.iloc[:,2].mean(),my_df1.iloc[:,3].mean()],color=["g","r","b"])
plt.grid()   # set grid
plt.show()  # show plot


# In[27]:


plt.figure(figsize=(5,3))    # plot figure size    
plt.title('Countries by Average Energy Use')    # plot title
# bar chart for showing Average Energy Use
plt.bar(my_df2.columns,[my_df2.iloc[:,0].mean(),my_df2.iloc[:,1].mean(),my_df2.iloc[:,2].mean(),my_df2.iloc[:,3].mean()],color=["g","r","b"])
plt.grid()   # set grid
plt.show()  # show plot


# In[28]:



def country_heatmap(dt):
    '''
    Function:
         country_heatmap()
    Description:
         Function generates a heatmap for the correlation matrix of the input dataframe using the map_corr() function from the cluster_tools module.
    Parameters:
        -dt: pandas DataFrame object, the input dataframe for which the heatmap needs to be generated.
    Returns:
        -None. The function generates heatmap plot using matplotlib.
    '''
    cluster_tools.map_corr(dt)

popeudfs = [my_df1, my_df2]
country_heatmap(popeudfs[0])
country_heatmap(popeudfs[1])

print(country_heatmap.__doc__)


# In[29]:


def normalization(df):
    """
    Function: normalization

    Description:
    This function applies data normalization using the `scaler()` function from the `cluster_tools` module to the input dataframe.

    Parameters:
    - df: pandas DataFrame object, the input dataframe to be normalized.

    Returns:
    - normdt[0]: pandas DataFrame object, the normalized dataframe.
    - normdt[1]: float, the minimum value used for normalization.
    - normdt[2]: float, the maximum value used for normalization.
    """
    normdt = cluster_tools.scaler(df)
    return normdt[0], normdt[1], normdt[2]

# Example usage
popeudfs = [my_df1, my_df2]
nrmdfs = []
valmn = []
valmx = []
for d in range(len(popeudfs)):
    nrmrs = normalization(popeudfs[d])
    nrmdfs.append(nrmrs[0])
    valmn.append(nrmrs[1])
    valmx.append(nrmrs[2])
print(nrmdfs[0].head(), "\n")
print(nrmdfs[1].head())

print(normalization.__doc__)
# ## K-Means Clustering

# In[30]:


value_elbow = []
setclus=10
for i in range(1, setclus):
    kmdl = KMeans(n_clusters=i, init='k-means++', max_iter=400,  random_state=20)   # apply k-means for clusters from 1-10
    kmdl.fit(nrmdfs[0])   # training model
    value_elbow.append(kmdl.inertia_)    # store inetrtia by cluster value
plt.figure(figsize=(6,3))    # plot figure size
plt.title('Determining Best Cluster using Elbow Curve')    # plot title
plt.plot(range(1, setclus), value_elbow,"b--")    # plot inertias 
plt.plot(range(1, setclus), value_elbow,"Dg")    # plot inertias 
plt.xlabel('Number of clusters')    # x-label
plt.ylabel('Inertia')     # y-label
plt.grid()   # set grid
plt.show()  # show plot


# In[31]:


optkmdl = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=400, random_state=20)  # apply k-means wirh final clsuter value
kmodel = optkmdl.fit(nrmdfs[1])   # train model
cls=[] 
for i in optkmdl.labels_:
    if i==0:
        cls.append(my_df1.columns[0])
    elif i==1:
        cls.append(my_df1.columns[1])
df=pd.DataFrame(nrmdfs[1],columns=my_df1.columns)
plt.figure(figsize=(7,4))    # plot figure size
plt.title('Cluster Visualization for K-Means')    # plot title
# scatter plot to show data
sns.scatterplot(data=df, x=my_df1.columns[0], y=my_df1.columns[1], hue=cls,palette="magma")
# scatter plot to show cluster centre
plt.scatter(optkmdl.cluster_centers_[:,0], optkmdl.cluster_centers_[:,1], marker="D", c="b", s=80, label="centroids")
plt.grid()   # set grid
plt.legend()   # set legent ot best position
plt.show()  # show plot


# ## Curve Fitting

# In[32]:


get_ipython().system('pip install lmfit')


# In[33]:


from scipy.optimize import curve_fit
from lmfit import Model


# In[34]:


def func(x, amp, cen, wid):   # design associative fucntion for the curve fitting
    return amp * np.exp(-(x-cen)**2 / wid)    # calculate y value


# In[35]:


nvals= np.random.normal(nrmdfs[1].values)

y = func(nvals[:,1], 1.9, 0.4, 1.22) + np.random.normal(0, 0.2, nvals.shape[0])   # get y value

init_vals = [1, 0, 1]    # initiate values
best_vals, covar = curve_fit(func, nvals[:,1], y, p0=init_vals,maxfev = 7000)   # apply curve fit
gmodel = Model(func)    # apply lmfit


# In[36]:


result = gmodel.fit(y, x=nvals[:,1], amp=5, cen=5, wid=1)   # training gmodel and obtain result

print(result.fit_report())   # print report for the result

plt.figure(figsize=(8,4))   # plot figure size
plt.title('View of Data Points')    # plot title
plt.plot(nvals[:,1],"mv",label="Data")   # plot data points
plt.legend()   # set legent ot best position
plt.grid()   # set grid
plt.show()  # show plot


# In[37]:


plt.figure(figsize=(8,4))   # plot figure size
plt.title('Curve Fitting')    # plot title
plt.plot(result.init_fit, 'b--', label='initial fit')   #plot initial fitted model
plt.legend()   # set legent ot best position
plt.grid()   # set grid
plt.show()  # show plot


# In[38]:


plt.figure(figsize=(8,4))   # plot figure size
plt.title('Best Curve Fitting')    # plot title
plt.plot(result.best_fit, 'g-', label='best fit')     #plot best fitted model
plt.legend()   # set legent ot best position
plt.grid()   # set grid
plt.show()  # show plot


# In[ ]:




