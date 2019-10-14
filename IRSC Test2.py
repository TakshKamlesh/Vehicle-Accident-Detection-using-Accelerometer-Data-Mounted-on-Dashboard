#!/usr/bin/env python
# coding: utf-8

# In[77]:


from keras.models import load_model
import numpy as np


# In[78]:


print("Enter Value of X component of Acceleration")
x = input()
print("Enter Value of Y component of Acceleration")
y = input()


# In[79]:


model = load_model('Desktop/IRSC/Vehicle-Accident-Detection-using-Accelerometer-Data-Mounted-on-Dashboard-master/curr_model.h5')


# In[80]:


preds = model.predict(np.array([[x,y]]))


# In[81]:


ans = {0 : 'Car Crash', 1 : 'Phone has Fallen from Dashboard'}


# In[89]:


print("*")
print("*")
print("*")
print(ans[np.argmax(preds)])
print("*")
print("*")
print("*")


# In[83]:


Y = {0:np.array([1,0]), 1:np.array([0,1])}


# # BONUS TASK 

# In[90]:


#TRAINS FOR EVERY INPUT RECEIVED ON CMD W.R.T TO THE PREDICTED OUTPUT 

model.train_on_batch(x=np.array([[x,y]]),y=np.array([Y[np.argmax(preds)]]))
                     


# In[88]:


#NEW TRAINED MODEL

model.save('curr_model.h5')


# In[ ]:




