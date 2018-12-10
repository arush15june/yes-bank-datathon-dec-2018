#!/usr/bin/env python
# coding: utf-8

# # Importing libraries.

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit


# # Importing data.

# #### Original data.

# In[6]:


train_data = pd.read_csv('./Yes_Bank_Training.csv')
test_data = pd.read_csv('./Yes_Bank_Test.csv')
train_df = train_data
test_df = test_data


# #### Processed data.

# In[7]:


pre_num = pd.read_csv("./preprocessed_numerical.csv")
print("Numerical train data.\nColumns: ",len(pre_num.columns))
print(pre_num.columns)
pre_num = pre_num.drop(['serial_number'], axis = 1)
print("Dropped 'serial_number'")

print(pre_num.shape)
print("\n")

pre_cat = pd.read_csv("./preprocessed_categorical.csv")
print("Categorical train data.\nColumns: ",len(pre_cat.columns))
print(pre_cat.columns)
pre_cat = pre_cat.drop(['date'], axis = 1)
print("Dropped 'date'")

print(pre_cat.shape)
print("\n")

pre_num_test = pd.read_csv("./preprocessed_numerical_test.csv")
print("Numerical test data.\nColumns: ",len(pre_num_test.columns))
print(pre_num_test.columns)
pre_num_test = pre_num_test.drop(['serial_number'], axis = 1)
print("Dropped 'serial_number'")

print(pre_num_test.shape)
print("\n")

pre_cat_test = pd.read_csv("./preprocessed_categorical_test.csv")
print("Categorical test data.\nColumns: ",len(pre_cat_test.columns))
print(pre_cat_test.columns)
pre_cat_test = pre_cat_test.drop(['date'], axis = 1)
print("Dropped 'date'")

print(pre_cat_test.shape)


# #### Labels.

# In[8]:


labels = train_data['outcome']


# In[9]:


labels.shape


# # Cleaning data.

# In[10]:


print(train_data.columns)


# ## Encoding data.

# #### Constructing encoder.

# In[11]:


le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
le5 = LabelEncoder()
le6 = LabelEncoder()
le7 = LabelEncoder()
lel = LabelEncoder()


# #### Joining numerical and categorical data together into a single dataframe.

# In[12]:


pre_encoded = pre_num.join(pre_cat)
pre_encoded_test = pre_num_test.join(pre_cat_test)


# In[13]:


pre_encoded.shape


# #### Encoding data.

# In[14]:


le1.fit(pre_encoded['job_description'])
le2.fit(pre_encoded['marital_status'])
le3.fit(pre_encoded['education_details'])
le4.fit(pre_encoded['has_default'])
le5.fit(pre_encoded['housing_status'])
le6.fit(pre_encoded['previous_loan'])
le7.fit(pre_encoded['phone_type'])

encoded = pre_encoded.copy()
encoded_test = pre_encoded_test.copy()


# In[15]:


le1_name_mapping = dict(zip(le1.classes_, le1.transform(le1.classes_)))
print(le1_name_mapping, "\n")

le2_name_mapping = dict(zip(le2.classes_, le2.transform(le2.classes_)))
print(le2_name_mapping, "\n")

le3_name_mapping = dict(zip(le3.classes_, le3.transform(le3.classes_)))
print(le3_name_mapping, "\n")

le4_name_mapping = dict(zip(le4.classes_, le4.transform(le4.classes_)))
print(le4_name_mapping, "\n")

le5_name_mapping = dict(zip(le5.classes_, le5.transform(le5.classes_)))
print(le5_name_mapping, "\n")

le6_name_mapping = dict(zip(le6.classes_, le6.transform(le6.classes_)))
print(le6_name_mapping, "\n")

le7_name_mapping = dict(zip(le7.classes_, le7.transform(le7.classes_)))
print(le7_name_mapping, "\n")


# In[16]:


encoded['job_description'] = le1.transform(encoded['job_description'])
encoded['marital_status'] = le2.transform(encoded['marital_status'])
encoded['education_details'] = le3.transform(encoded['education_details'])
encoded['has_default'] = le4.transform(encoded['has_default'])
encoded['housing_status'] = le5.transform(encoded['housing_status'])
encoded['previous_loan'] = le6.transform(encoded['previous_loan'])
encoded['phone_type'] = le7.transform(encoded['phone_type'])


# In[17]:


encoded_test['job_description'] = le1.transform(encoded_test['job_description'])
encoded_test['marital_status'] = le2.transform(encoded_test['marital_status'])
encoded_test['education_details'] = le3.transform(encoded_test['education_details'])
encoded_test['has_default'] = le4.transform(encoded_test['has_default'])
encoded_test['housing_status'] = le5.transform(encoded_test['housing_status'])
encoded_test['previous_loan'] = le6.transform(encoded_test['previous_loan'])
encoded_test['phone_type'] = le7.transform(encoded_test['phone_type'])


# #### Encoding labels.

# In[18]:


lel.fit(labels)


# In[19]:


lel_name_mapping = dict(zip(lel.classes_, lel.transform(lel.classes_)))
print(lel_name_mapping)


# In[20]:


lelab = pd.DataFrame(lel.transform(labels), columns = ['outcome'])


# ## Upsampling.
# 
# Since the data provided was biased towards class "no", we have increased the class "yes" using upsampling so as to reduce Undercoverage bias in the data distribution.

# In[21]:


upsample = encoded.join(lelab)


# In[22]:


df_majority = upsample[upsample.outcome==0]
df_minority = upsample[upsample.outcome==1]


# In[23]:


df_minority_upsampled = resample(df_minority, replace = True, n_samples = df_majority.shape[0], random_state = 42)


# In[24]:


df_upsampled = pd.concat([df_majority, df_minority_upsampled])


# In[25]:


df_upsampled.outcome.value_counts()


# In[26]:


Y = df_upsampled['outcome']
X = df_upsampled.drop(['outcome'], axis = 1)


# ## Data split.

# In[27]:


sss = StratifiedShuffleSplit(test_size=0.3)


# In[28]:


for train_ix, test_ix in sss.split(X = X.values, y = Y.values):
    x_train, y_train = X.values[train_ix], Y.values[train_ix]
    x_val, y_val = X.values[test_ix], Y.values[test_ix]


#  

#  

# # Train.

# ## Classifier.

# In[29]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


rnd_forest = RandomForestClassifier(n_estimators=500 , bootstrap=True, n_jobs=-1, max_features=8)
rnd_forest.fit(X=x_train, y=y_train)
rnd_forest.score(X=x_val, y=y_val), rnd_forest.score(X=x_train, y=y_train)


# In[ ]:




