#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[81]:


teams = pd.read_csv("teams.csv")


# In[3]:


pwd


# In[83]:


teams


# In[84]:


teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]


# In[85]:


teams


# In[86]:


teams.corr()["medals"]


# In[14]:


import seaborn as sns


# In[16]:


sns.lmplot(x="athletes", y="medals" , data=team, fit_reg=True, ci=None)


# In[19]:


sns.lmplot(x="age",y="medals",data=team, fit_reg=True, ci=None)


# In[21]:


team.plot.hist(y="medals")


# In[87]:


teams[teams.isnull().any(axis=1)].head(20)


# In[88]:


teams = teams.dropna()


# In[89]:


teams.shape


# In[92]:


train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()


# In[93]:


train.shape


# In[94]:


from sklearn. linear_model import LinearRegression

reg = LinearRegression()


# In[95]:


predicators = ["athletes", "prev_medals"]


# In[101]:


reg.fit(train[predicators], train["medals"])


# In[100]:


reg.fit(train[predicators], train["medals"])


# In[103]:


predictions = reg.predict(test[predicators])


# In[104]:


predictions.shape


# In[106]:


test["predictions"] = predictions


# In[107]:


predictions


# In[108]:


from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(test["medals"], test["predictions"])


# In[109]:


error


# In[110]:


teams.describe()["medals"]


# In[115]:


test[test["team"]=="USA"]


# In[116]:


test[test["team"]=="IND"]


# In[117]:


errors = (test["medals"] - test["predictions"]).abs()


# In[118]:


errors


# In[123]:


errors_by_teams = errors.groupby(test["team"]).mean()


# In[126]:


errors_by_teams


# In[127]:


error_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio =  error_by_team / medals_by_team 


# In[128]:


import numpy as np
error_ratio = error_ratio[np.isfinite(error_ratio)]


# In[129]:


error_ratio.plot.hist()


# In[130]:


error_ratio.plot.hist()


# In[ ]:




