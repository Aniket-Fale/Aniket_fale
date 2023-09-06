#!/usr/bin/env python
# coding: utf-8

#  <font size="6"><strong><center>Project  - Health Care<center></strong></font>

# **Problem statement:**
# Cardiovascular diseases are the leading cause of death globally. It is therefore necessary to identify the causes and develop a system to predict heart attacks in an effective manner. The data below has the information about the factors that might have an impact on cardiovascular health. 
# 
# Dataset description:
# 
# |Variable|Description|
# | --- | --- |
# |Age|Age in years|
# |Sex|1 = male; 0 = female|
# |cp|Chest pain type|
# |trestbps|Resting blood pressure (in mm Hg on admission to the hospital)|
# |chol|Serum cholesterol in mg/dl|
# |fbs|Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)|
# |restecg|Resting electrocardiographic results|
# |thalach|Maximum heart rate achieved|
# |exang|Exercise induced angina (1 = yes; 0 = no)|
# |oldpeak|ST depression induced by exercise relative to rest|
# |slope|Slope of the peak exercise ST segment|
# |ca|Number of major vessels (0-3) colored by fluoroscopy|
# |thal|3 = normal; 6 = fixed defect; 7 = reversible defect|
# |Target|1 or 0|
# 
# 
# 
# **Task to be performed:**
# 
# 1. Preliminary analysis:
#        a. Perform preliminary data inspection and report the findings on the structure of the data, missing values, duplicates, etc.
#        b. Based on these findings, remove duplicates (if any) and treat missing values using an appropriate strategy.
# 2. Prepare a report about the data explaining the distribution of the disease and the related factors using the steps listed below:
#        a. Get a preliminary statistical summary of the data and explore the measures of central tendencies and spread of the data.
#        b. Identify the data variables which are categorical and describe and explore these variables using the appropriate tools, such as count plot.
#        c. Study the occurrence of CVD across the Age category.
#        d. Study the composition of all patients with respect to the Sex category.
#        e. Study if one can detect heart attacks based on anomalies in the resting blood pressure (trestbps) of a patient.
#        f. Describe the relationship between cholesterol levels and a target variable.
#        g. State what relationship exists between peak exercising and the occurrence of a heart attack.
#        h. Check if thalassemia is a major cause of CVD.
#        i. List how the other factors determine the occurrence of CVD.
#        j. Use a pair plot to understand the relationship between all the given variables.
# 3. Build a baseline model to predict the risk of a heart attack using a logistic regression and random forest and explore the results while using correlation analysis and logistic regression (leveraging standard error and p-values from statsmodels) for feature selection.
# 

# In[1]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# In[2]:


hcare = pd.read_excel("1645792390_cep1_dataset.xlsx")


# In[3]:


hcare.head()


# In[4]:


hcare.tail()


# In[5]:


hcare.shape


# In[6]:


hcare.info()


# In[7]:


hcare.dtypes


# In[8]:


# Checking for missing values
hcare.isnull().sum(axis = 0)


# In[9]:


hcare.describe()


# We can see that the scale of each feature column is different and varied.

# In[10]:


# For visualizations
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[11]:


# Histogram of the Heart Dataset

fig = plt.figure(figsize = (40,30))
hcare.hist(ax = fig.gca());


# *From the above histogram plots, we can see that the features are skewed and not normally distributed. Also, the scales are different between one and another.*
# 
# <b><font size="3">Understanding the Data</font></b>

# *Let us observe the creelation between different features with help of a heat mat.*

# In[12]:


# Creating a correlation heatmap
sns.heatmap(hcare.corr(),annot=True, cmap='terrain', linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(20,20)
plt.show()


# *From the above HeatMap, we can see that cp and thalach are the features with highest positive correlation whereas exang, oldpeak and ca are negatively correlated.While other features do not hold much correlation with the response variable "target".*

# <b>Outlier Detection</b>
# 
# *Since the dataset is not large, we cannot discard the outliers. We will treat the outliers as potential observations.*

# In[13]:


# Boxplots
fig_dims = (15,8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.boxplot(data=hcare, ax=ax);


# <b>Handling Imbalance</b>
# 
# Imbalance in a dataset leads to inaccuracy and high precision, recall scores. There are certain resampling techniques such as undersampling and oversampling to handle these issues.
# 
# Considering our dataset, the response variable target has two outcomes "Patients with Heart Disease" and "Patients without Heart Disease". Let us now observe their distribution in the dataset.

# In[14]:


hcare["target"].value_counts()


# From the above chart, we can conclude even when the distribution is not exactly 50:50, but still the data is good enough to use on machine learning algorithms and to predict standard metrics like Accuracy and AUC scores. So, we do not need to resample this dataset.

# <b>Train-Test Split</b>

# Let us distribute the data into **training** and **test** datasets using the **train_test_split()** function.

# In[15]:


X = hcare.drop("target",axis=1)
y = hcare["target"]


# <b>Logistic Regression</b>

# In[16]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,stratify=y,random_state=7)


# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[19]:


pred = lr.predict(X_test)


# In[20]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[21]:


# Accuracy on Test data
accuracy_score(y_test, pred)


# In[22]:


# Accuracy on Train data
accuracy_score(y_train, lr.predict(X_train))


# <b>Building a predictive system</b>

# In[23]:


import warnings
in_data = (57,0,0,140,241,0,1,123,1,0.2,1,0,3)

# Changing the input data into a numpy array
in_data_as_numpy_array = np.array(in_data)

# Reshaping the numpy array as we predict it
in_data_reshape = in_data_as_numpy_array.reshape(1,-1)
pred = lr.predict(in_data_reshape)
print(pred)

if(pred[0] == 0):
    print('The person does not have heart disease.')
else:
    print('The person has heart disease.')


# 
