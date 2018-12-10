#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('./Yes_Bank_Training.csv')


# In[3]:


data.shape


# In[4]:


data.describe()


# In[5]:


data.head()


# In[6]:


categorical_data = data[["job_description", "marital_status", "education_details", "has_default", "housing_status", "previous_loan", "phone_type", "date"]]
numerical_data = data.drop(["job_description", "marital_status", "education_details", "has_default", "housing_status", "previous_loan", "phone_type", "month_of_year", "date", "poutcome_of_campaign","outcome", "days_passed", "previous_contact"], axis =1)


# In[7]:


numerical_data.head()


# In[8]:


categorical_data.head()


# ## Outlier removal
# 
# Here we are removing outliers from the training as well as testing data. Outliers are those values which are
# 
# * greater than μ + 2.σ
# * less than μ - 2.σ

# In[9]:


main_data = numerical_data.copy(deep=True)


# #### for age_in_years

# In[10]:


[i for i in zip(*np.unique(main_data["age_in_years"], return_counts = True))]


# In[11]:


mean, std = np.mean(main_data["age_in_years"]), np.std(main_data["age_in_years"])


# In[12]:


min_val , max_val = int(mean- 2 * std), int(mean + 2 *std)


# In[13]:


min_val, max_val, mean


# In[14]:


np.random.randint(min_val, max_val)


# In[15]:


def age_in_years(val):
    if val < 21 or val > 60:
        return np.random.randint(21, 60)
    else:
        return val


# In[16]:


main_data["age_in_years"] = main_data["age_in_years"].apply(age_in_years)


# In[17]:


main_data.head()


# In[18]:


[i for i in zip(*np.unique(main_data["age_in_years"], return_counts = True))]


# #### for balance_in_account

# In[19]:


mean, std = main_data["balance_in_account"].mean(), main_data["balance_in_account"].std()


# In[20]:


mean, std


# In[21]:


min_val, max_val = mean - 2 * std, mean + 2 * std


# In[22]:


int(min_val), int(max_val)


# In[23]:


def balance_in_account(val):
    if val < -4628 or val > 7215:
        if val <= -4628:
            return int(-4628)
        else:
            return int(7215)
    else:
        return val


# In[24]:


main_data["balance_in_account"] = main_data["balance_in_account"].apply(balance_in_account)


# In[25]:


main_data["balance_in_account"].values


# In[26]:


main_data["balance_in_account"].mean(), main_data["balance_in_account"].std()


# In[27]:


main_data.head()


# #### for campaign_contacts

# In[28]:


[i for i in zip(*np.unique(main_data["campaign_contacts"], return_counts = True))]


# In[29]:


mean, std = main_data["campaign_contacts"].mean(), main_data["campaign_contacts"].std()


# In[30]:


min_val, max_val = 0 , int(mean + 2 * std)


# In[31]:


min_val, max_val


# In[32]:


def campaign_contacts(val):
    if val < 0 or val > 10:
        if val <= 0:
            return int(0)
        else:
            return int(10)
    else:
        return val


# In[33]:


main_data["campaign_contacts"] = main_data["campaign_contacts"].apply(campaign_contacts)


# In[34]:


[i for i in zip(*np.unique(main_data["campaign_contacts"], return_counts = True))]


# In[35]:


main_data.head()


# In[36]:


main_data.head()


# #### for call_duration

# In[37]:


main_data["call_duration"] = main_data["call_duration"]//60 ## in mins


# In[38]:


[i for i in zip(*np.unique(main_data["call_duration"], return_counts = True))]


# In[39]:


mean, std = main_data["call_duration"].mean(), main_data["call_duration"].std()


# In[40]:


min_val, max_val = 0 , int(mean + 2 * std)


# In[41]:


min_val, max_val


# In[42]:


np.sum(np.unique(main_data[main_data["call_duration"] > 12]["call_duration"], return_counts = True)[1])


# In[43]:


def call_duration(val):
    if val > 12:
        return 12
    else:
        return val


# In[44]:


main_data["call_duration"] = main_data["call_duration"].apply(call_duration)


# In[45]:


[i for i in zip(*np.unique(main_data["call_duration"], return_counts = True))]


# In[46]:


main_data.head()


# #### Saving processed data

# In[47]:


main_data.to_csv(path_or_buf='./preprocessed_numerical.csv', index=False)


# # CATEGORICAL
# 
# Here we fill the unknowns with the best feature value which we calculate using the features that are most influential towards the feature being filled.
# Here we do an l2 norm to calculate the distance between 'the numerical features of the unknown being replaced' and the numerical features of all the other features that are related to it. 
# The feature value with the least l2 norm will be used to replace the unknown.

# In[50]:


main_data = categorical_data.copy(deep=True)


# #### for job_description

# In[51]:


main_data.head()


# In[52]:


[i for i in zip(*np.unique(main_data["job_description"], return_counts = True))]


# Here we choose "age_in_years" and "balance_in_account" to be most influential with "job_description"

# In[53]:


jobs = np.unique(main_data["job_description"])
error = {}
for job in jobs:
    unknown_data, known_data = numerical_data[main_data["job_description"] == "unknown"][["age_in_years","balance_in_account"]], numerical_data[main_data["job_description"] == job][["age_in_years","balance_in_account"]]
    unknown_mean, known_mean = unknown_data.mean(axis=0), known_data.mean(axis=0)
    error[job] = np.mean(np.square(unknown_mean.values - known_mean.values))


# In[54]:


sorted(error.items(), key=lambda x: x[1])


# Here "management" has the least l2 norm, thus we will change unknowns in "job_description" with "management"

# In[56]:


def job_description(val):
    if val == "unknown":
        return "management"
    else:
        return val


# In[57]:


main_data["job_description"] = main_data["job_description"].apply(job_description)


# In[58]:


[i for i in zip(*np.unique(main_data["job_description"], return_counts = True))]


# In[59]:


main_data.head()


# #### education_details

# In[60]:


[i for i in zip(*np.unique(main_data["education_details"], return_counts = True))]


# Here we choose "age_in_years" and "balance_in_account" to be most influential with "education_details"

# In[61]:


edus = np.unique(main_data["education_details"])
error = {}
for edu in edus:
    unknown_data, known_data = numerical_data[main_data["education_details"] == "unknown"][["age_in_years","balance_in_account"]], numerical_data[main_data["education_details"] == edu][["age_in_years","balance_in_account"]]
    unknown_mean, known_mean = unknown_data.mean(axis=0), known_data.mean(axis=0)
    error[edu] = np.mean(np.square(unknown_mean.values - known_mean.values))

sorted(error.items(), key=lambda x: x[1])


# So, we choose "tertiary" to replace with unknowns.

# In[63]:


def education_details(val):
    if val == "unknown":
        return "tertiary"
    else:
        return val


# In[64]:


main_data["education_details"] = main_data["education_details"].apply(education_details)


# In[65]:


main_data.head()


# In[66]:


[i for i in zip(*np.unique(main_data["education_details"], return_counts = True))]


# #### phone_type

# In[67]:


[i for i in zip(*np.unique(main_data["phone_type"], return_counts = True))]


# Here we choose "call_duration" to be most influential with "phone_type"

# In[68]:


phones = np.unique(main_data["phone_type"])
error = {}
for phone in phones:
    unknown_data, known_data = numerical_data[main_data["phone_type"] == "unknown"]["call_duration"], numerical_data[main_data["phone_type"] == phone]["call_duration"]
    unknown_mean, known_mean = unknown_data.mean(axis=0), known_data.mean(axis=0)
    error[phone] = np.mean(np.sqrt(np.square(unknown_mean - known_mean)))

sorted(error.items(), key=lambda x: x[1])


# hence it will be changed to cellular

# In[70]:


def phone_type(val):
    if val == "unknown":
        return 'cellular'
    else:
        return val


# In[71]:


main_data["phone_type"] = main_data["phone_type"].apply(phone_type)


# In[72]:


[i for i in zip(*np.unique(main_data["phone_type"], return_counts = True))]


# In[73]:


main_data.head()


# #### date

# In[74]:


[i for i in zip(*np.unique(main_data["date"], return_counts = True))]


# In[75]:


main_data["date"].describe()


# Here we **categorize** feature "date", using inter-quartile range.

# In[76]:


q1, q2, q3 = 9, 18, 23


# In[77]:


def date(val):
    if val < 9:
        return "start"
    elif val < 18:
        return "mid-1"
    elif val< 23:
        return "mid-2"
    else:
        return "end"


# In[78]:


main_data["date"] = main_data["date"].apply(date)


# In[79]:


[i for i in zip(*np.unique(main_data["date"], return_counts = True))]


# In[80]:


main_data.head()


# In[81]:


main_data.to_csv(path_or_buf='./preprocessed_categorical.csv', index=False)


#  

#  

# # TEST DATA

# In[82]:


test_data = pd.read_csv('./Yes_Bank_Test.csv')


# In[83]:


categorical_data = test_data[["job_description", "marital_status", "education_details", "has_default", "housing_status", "previous_loan", "phone_type", "date"]]
numerical_data = test_data.drop(["job_description", "marital_status", "education_details", "has_default", "housing_status", "previous_loan", "phone_type", "month_of_year", "date", "poutcome_of_campaign", "days_passed", "previous_contact"], axis =1)


# ### Numerical

# In[84]:


main_data = numerical_data.copy(deep=True)


# In[85]:


main_data["age_in_years"] = main_data["age_in_years"].apply(age_in_years)
main_data["balance_in_account"] = main_data["balance_in_account"].apply(balance_in_account)
main_data["campaign_contacts"] = main_data["campaign_contacts"].apply(campaign_contacts)
main_data["call_duration"] = main_data["call_duration"]//60 ## in mins
main_data["call_duration"] = main_data["call_duration"].apply(call_duration)


# In[86]:


main_data.head()


# In[87]:


main_data.to_csv(path_or_buf='./preprocessed_numerical_test.csv', index=False)


# ### Catergorical

# In[88]:


main_data = categorical_data.copy(deep=True)


# In[89]:


main_data["job_description"] = main_data["job_description"].apply(job_description)
main_data["education_details"] = main_data["education_details"].apply(education_details)
main_data["phone_type"] = main_data["phone_type"].apply(phone_type)
main_data["date"] = main_data["date"].apply(date)


# In[90]:


main_data.head()


# In[91]:


main_data.to_csv(path_or_buf='./preprocessed_categorical_test.csv', index=False)


# In[92]:


for col in main_data.columns:
    print([i for i in zip(*np.unique(main_data[col], return_counts = True))])


# ## END
