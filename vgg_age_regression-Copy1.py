#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from data_generator.batch_generator import BatchGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from models import AlexNet, LeNet
from keras.applications.vgg16 import VGG16
from keras import backend as K
K.set_image_data_format('channels_last')

from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D, UpSampling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Model

import tensorflow as tf


# Image-treat-1: sem data augmentation, com normalização e equalização
# 
# Image-treat-2: com data augmentation, com normalização e equalização
# 
# Image-treat-3: com data augmentation, com normalização e com equalização

# In[2]:


#config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allocator_type = 'BFC'
#config.gpu_options.per_process_gpu_memory_fraction = 0.9


# In[3]:


approach = 'abordagem6' 
activation = 'relu'
net = 'vgg16'

if net == 'alexnet':
    model = AlexNet
elif net =='lenet':
    model = LeNet
elif net == 'vgg16':
    model = VGG16 
csvlogger_name = 'callbacks/'+net +'/age/history-regression-' + approach + '-' + activation + '.csv'
checkpoint_filename = 'callbacks/'+net+'/age/class-weights-' + approach + '-' + activation + '.{epoch:02d}-{val_loss:.2f}.hdf5'
csvlogger_name, checkpoint_filename


# In[4]:


df = pd.read_csv('dataset/csv/imdb_csv/imdb_age_regression_train_split_47950-70-10-20.csv')


# In[5]:


cols = list(df.columns[1:])
in_format = list(df.columns)
cols, in_format


# In[6]:


train_dataset = BatchGenerator(box_output_format=cols)
validation_dataset = BatchGenerator(box_output_format=cols)

train_dataset. parse_csv(labels_filename='dataset/csv/imdb_csv/imdb_age_regression_train_split_47950-70-10-20.csv', 
                        images_dir='dataset/imdb-hand-crop',
                        input_format=in_format)

validation_dataset.parse_csv(labels_filename='dataset/csv/imdb_csv/imdb_age_regression_val_split_47950-70-10-20.csv', 
                             images_dir='dataset/imdb-hand-crop',
                             input_format=in_format)


# In[7]:


img_height, img_width, img_depth = (224,224,3)

epochs = 1000

train_batch_size = 64
shuffle = True
ssd_train = False

validation_batch_size = 32


# In[8]:


train_generator = train_dataset.generate(batch_size=train_batch_size,
                                         shuffle=shuffle,
                                         ssd_train=ssd_train,
                                         random_rotation=20,
                                         translate=(0.2, 0.2),
                                         scale=(0.8, 1.2),
                                         flip=0.5,
                                         divide_by_stddev=255,
                                         returns={'processed_labels'},
                                         resize=(img_height, img_width))

validation_generator = validation_dataset.generate(batch_size=validation_batch_size,
                                                   shuffle=shuffle,
                                                   ssd_train=ssd_train,
                                                   divide_by_stddev=255,
                                                   returns={'processed_labels'},
                                                   resize=(img_height, img_width))

print("Number of images in the dataset:", train_dataset.get_n_samples())
print("Number of images in the dataset:", validation_dataset.get_n_samples())


# In[9]:


steps_per_epoch = train_dataset.get_n_samples()/train_batch_size
validation_steps = validation_dataset.get_n_samples()/validation_batch_size


# In[10]:


base_model = VGG16(include_top=True, weights=None, input_tensor=None, 
                  input_shape=(img_height, img_width, img_depth), 
                  pooling='avg')

base_model.summary()


# In[11]:


base_model.layers.pop()


# In[12]:


last = base_model.layers[-1].output

preds = Dense(1, activation='relu')(last)

model = Model(base_model.input, preds)


# In[13]:


model.summary()


# In[14]:


optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=True)


# In[15]:


csv_logger = CSVLogger(csvlogger_name, append=True, separator=',')

checkpoint = ModelCheckpoint(checkpoint_filename,
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=False,
                           period=1)

earlystopping = EarlyStopping(patience=30, mode='min')

#callbacks = [tensorboard, checkpoint]
callbacks=[checkpoint, csv_logger, earlystopping]


# In[16]:


model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])


# In[17]:


model.fit_generator(train_generator, epochs=epochs, 
                             steps_per_epoch=128, 
                             validation_data=validation_generator,
                             validation_steps=validation_steps,
                             callbacks=callbacks)


# In[ ]:




