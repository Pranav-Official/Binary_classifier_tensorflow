#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[18]:


import numpy as np


# In[3]:


import tensorflow as tf
import os


# In[15]:


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[16]:


from tensorflow.keras.preprocessing import image


# In[5]:


data_path = # path to your data set


# In[36]:


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.95):
            print("\nReached 95.0% accuracy so cancelling training!")
            self.model.stop_training = True


# In[6]:


def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
    
    model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])

    return model3#


# In[43]:


data_set = ImageDataGenerator(rescale= 1/255,       # ImageGenerator with augmentation
                              rotation_range=40,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest')

data_gen = data_set.flow_from_directory(
        data_path,  
        target_size=(150, 150),  
        batch_size=128,
        class_mode='binary')


# In[47]:


model =  create_model()


# In[37]:


callbacks = myCallback()


# In[48]:


history = model.fit(data_gen,epochs = 8,steps_per_epoch = 8,callbacks = [callbacks], verbose = 1)


# In[39]:


test_dir =  # path to you test dir
test_list = os.listdir(test_dir)
print(test_list)


# In[45]:


for fl in test_list:
    path =  os.path.join(test_dir, fl)
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0]>0.5:
        print(fl + " is a human")
    else:
        print(fl + " is a horse")
 


# In[46]:


import matplotlib.pyplot as plt
acc = history.history['acc']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


# In[ ]:




