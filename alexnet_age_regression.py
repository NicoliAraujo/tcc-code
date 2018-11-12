
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from data_generator.batch_generator import BatchGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from models import AlexNet, LeNet
from keras import backend as K
K.set_image_data_format('channels_last')


# Image-treat-1: sem data augmentation, com normalização e equalização
# 
# Image-treat-2: com data augmentation, com normalização e equalização

# In[2]:


approach = 'image-treat-2' 
activation = 'relu'

csvlogger_name = 'callbacks/lenet/age/history-regression-' + approach + '-' + activation + '.csv'
checkpoint_filename = 'callbacks/lenet/age/class-weights-' + approach + '-' + activation + '.{epoch:02d}-{val_mean_average_error:.2f}.hdf5'
csvlogger_name, checkpoint_filename


# In[3]:


df = pd.read_csv('/home/nicoli/github/alexnet/dataset/csv/imdb_csv/imdb_age_regression_train_split_47950-70-10-20.csv')


# In[4]:


cols = list(df.columns[1:])
in_format = list(df.columns)


# In[5]:


train_dataset = BatchGenerator(box_output_format=cols)
validation_dataset = BatchGenerator(box_output_format=cols)

train_dataset. parse_csv(labels_filename='/home/nicoli/github/alexnet/dataset/csv/imdb_csv/imdb_age_regression_train_split_47950-70-10-20.csv', 
                        images_dir='/home/nicoli/github/alexnet/dataset/imdb-hand-crop',
                        input_format=in_format)

validation_dataset.parse_csv(labels_filename='/home/nicoli/github/alexnet/dataset/csv/imdb_csv/imdb_age_regression_val_split_47950-70-10-20.csv', 
                             images_dir='/home/nicoli/github/alexnet/dataset/imdb-hand-crop',
                             input_format=in_format)


# In[6]:


img_height, img_width, img_depth = (224,224,3)

epochs = 100

train_batch_size = 64
shuffle = True
ssd_train = False

validation_batch_size = 32


# In[7]:


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


# In[8]:


steps_per_epoch = train_dataset.get_n_samples()/train_batch_size
validation_steps = validation_dataset.get_n_samples()/validation_batch_size


# In[9]:


model = LeNet(n_classes=1, img_width=img_width, img_depth=img_depth, img_height=img_height, activation=activation)


# In[10]:


model.summary()


# In[11]:


optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=True)


# In[12]:


csv_logger = CSVLogger(csvlogger_name, append=True, separator=',')

checkpoint = ModelCheckpoint(checkpoint_filename,
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=False,
                           period=1)

earlystopping = EarlyStopping(patience=5, mode='min')

#callbacks = [tensorboard, checkpoint]
callbacks=[checkpoint, csv_logger, earlystopping]


# In[13]:


model.compile(loss='mae', optimizer=optimizer, metrics=['mse'])


# In[14]:


model.fit_generator(train_generator, epochs=epochs, 
                             steps_per_epoch=steps_per_epoch, 
                             validation_data=validation_generator,
                             validation_steps=validation_steps,
                             callbacks=callbacks)

