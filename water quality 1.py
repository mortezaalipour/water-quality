#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pycaret

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/water_potability.csv")
data.head()


# In[2]:


data = data.dropna()
#remove the NAN
data.isnull().sum()
#check the NAN rows


# In[3]:


plt.figure(figsize=(15, 10))
sns.countplot(data.Potability)
plt.title("Distribution of Unsafe and Safe Water")
plt.show()


# In[4]:


import plotly.express as px
data = data
figure = px.histogram(data, x = "ph", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: PH")
figure.show()


# In[5]:


figure = px.histogram(data, x = "Hardness", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Hardness")
figure.show()


# In[6]:


figure = px.histogram(data, x = "Solids", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Solids")
figure.show()


# In[7]:


figure = px.histogram(data, x = "Chloramines", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Chloramines")
figure.show()


# In[8]:


figure = px.histogram(data, x = "Sulfate", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Sulfate")
figure.show()


# In[9]:


figure = px.histogram(data, x = "Conductivity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Conductivity")
figure.show()


# In[10]:


figure = px.histogram(data, x = "Organic_carbon", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Organic Carbon")
figure.show()


# In[11]:


figure = px.histogram(data, x = "Trihalomethanes", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Trihalomethanes")
figure.show()


# In[12]:


figure = px.histogram(data, x = "Turbidity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Turbidity")
figure.show()


# In[13]:


correlation = data.corr()
correlation["ph"].sort_values(ascending=False)


# In[ ]:


from pycaret.classification import *
clf = setup(data, target = "Potability", silent = True, session_id = 786)
compare_models()


# In[ ]:


get_ipython().system('pip install pycaret')
get_ipython().system('pip install --upgrade pip')


# In[ ]:


model = create_model("rf")
predict = predict_model(model, data=data)
predict.head()


# In[ ]:




