
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from data_generator.batch_generator import BatchGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from models import AlexNet
from keras import backend as K
K.set_image_data_format('channels_last')


# In[2]:


df = pd.read_csv('/home/nicoli/github/alexnet/dataset/csv/imdb_csv/imdb_age_regression.csv')


# In[3]:


cols = list(df.columns[1:])
in_format = list(df.columns)


# In[4]:


train_dataset = BatchGenerator(box_output_format=cols)
validation_dataset = BatchGenerator(box_output_format=cols)

train_dataset.parse_csv(labels_filename='/home/nicoli/github/alexnet/dataset/csv/imdb_csv/imdb_age_regression_train_split_47950-70-10-20.csv', 
                        images_dir='/home/nicoli/github/alexnet/dataset/imdb-hand-crop',
                        input_format=in_format)

validation_dataset.parse_csv(labels_filename='/home/nicoli/github/alexnet/dataset/csv/imdb_csv/imdb_age_regression_train_split_47950-70-10-20.csv', 
                             images_dir='/home/nicoli/github/alexnet/dataset/imdb-hand-crop',
                             input_format=in_format)


# In[5]:


img_height, img_width, img_depth = (224,224,3)

epochs = 100

train_batch_size = 64
shuffle = True
ssd_train = False

validation_batch_size = 32


# In[6]:


train_generator = train_dataset.generate(batch_size=train_batch_size,
                                         shuffle=shuffle,
                                         ssd_train=ssd_train,
                                         flip=0.5,
                                         equalize=True,
                                         divide_by_stddev=255,
                                         returns={'processed_labels'},
                                         resize=(img_height, img_width))

validation_generator = validation_dataset.generate(batch_size=validation_batch_size,
                                                   shuffle=shuffle,
                                                   ssd_train=ssd_train,
                                                   flip=0.5,
                                                   equalize=True,
                                                   divide_by_stddev=255,
                                                   returns={'processed_labels'},
                                                   resize=(img_height, img_width))

print("Number of images in the dataset:", train_dataset.get_n_samples())
print("Number of images in the dataset:", validation_dataset.get_n_samples())


# In[7]:


steps_per_epoch = train_dataset.get_n_samples()/train_batch_size
validation_steps = validation_dataset.get_n_samples()/validation_batch_size


# In[7]:


alexnet = AlexNet(n_classes=1, img_width=img_width, img_depth=img_depth, img_height=img_height)#, weights_path='callbacks/alexnet/age/class-weights.05-15.74.hdf5')


# In[9]:


alexnet.model.summary()


# In[10]:


optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=True)


# In[11]:


csv_logger = CSVLogger('callbacks/alexnet/age/history-regression-fase2-2.csv', append=True, separator=',')

checkpoint = ModelCheckpoint(filepath='callbacks/alexnet/age/class-weights-reg-fase2-2.{epoch:02d}-{val_loss:.2f}.hdf5',
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=False,
                           period=1)
#callbacks = [tensorboard, checkpoint]
callbacks=[checkpoint, csv_logger]


# In[12]:


alexnet.model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])


# In[13]:


alexnet.model.fit_generator(train_generator, epochs=epochs, 
                             steps_per_epoch=steps_per_epoch, 
                             validation_data=validation_generator,
                             validation_steps=validation_steps,
                             callbacks=callbacks)

