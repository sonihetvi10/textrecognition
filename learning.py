#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
'''
from keras.models import load_model
import utils
import tensorflow as tf
import keras as ks
'''


# In[3]:


#image read
i=cv2.imread('C:\Users\MY\Desktop\wally.jpg')
img=cv2.resize(i,(1024,1024))
plt.imshow(img)


# In[4]:


#image original
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img1)


# In[5]:


#image conversion
im1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(im1)
print(im1)


# In[6]:


#ret,t_img=cv2.threshold(im1,130,256,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#plt.imshow(t_img)


# In[7]:


#cv2.imwrite('output/oy-no-blur-thresh.jpg', t_img)
img_m = cv2.medianBlur(img, 5)
plt.imshow(img_m)
#plt.imshow(cv2.cvtColor(img_m,cv2.COLOR_GRAY2RGB))
#print(im1)


# In[8]:


#canny edge edge detection
e = cv2.Canny(img_m,100,200)
plt.imshow(cv2.cvtColor(e,cv2.COLOR_GRAY2RGB))


# In[9]:


#dilation operation
kernel = np.ones((5,5),np.uint8)
d= cv2.dilate(e,kernel,iterations =1)
plt.imshow(cv2.cvtColor(d,cv2.COLOR_GRAY2RGB))


# In[10]:


#morphological opening
o = cv2.morphologyEx(d, cv2.MORPH_OPEN, kernel)
plt.imshow(cv2.cvtColor(o,cv2.COLOR_GRAY2RGB))


# In[11]:



#binary threshold
ret,t_img=cv2.threshold(d,130,256,cv2.THRESH_BINARY_INV)
plt.imshow(cv2.cvtColor(t_img,cv2.COLOR_GRAY2RGB))



# In[12]:



#text detection
# find contours
# cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
cv2MajorVersion = cv2.__version__.split(".")[0]
# check for contours on thresh
if int(cv2MajorVersion) >= 4:
    ctrs, hier = cv2.findContours(t_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
else:
    ctrs,hier = cv2.findContours(t_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])




for i, ctr in enumerate(sorted_ctrs):
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = t_img[y:y + h, x:x + w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(t_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if w>20 and h>20:
        cv2.imwrite('C:\Users\My\Desktop\output\{}.jpg'.format(i), roi)
        boxes=np.array(roi)
        print(boxes)
        
        


# In[13]:


plt.imshow(cv2.cvtColor(t_img,cv2.COLOR_GRAY2RGB))


# In[14]:


character_boxes = []  # find all the character containing boxes 
for i in range(0,len(boxes)):
    x,y,w,h = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
    cropped = t_img [y: y+h, x:x+w]
    predict = utils.is_text(cropped)
    if predict > 0.95:
        character_boxes.append(boxes[i])
        character_boxes = np.array(character_boxes)






