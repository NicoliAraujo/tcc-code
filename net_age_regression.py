
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from data_generator.batch_generator import BatchGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from models import AlexNet, LeNet
from keras import backend as K
K.set_image_data_format('channels_last')


# Image-treat-1: sem data augmentation, com normalização e sem equalização
# 
# Image-treat-2: com data augmentation, com normalização e sem equalização

# Image-treat-3: com data augmentation, com normalização e com equalização

# Abordagem 4: com data augmentation, com normalização e com equalização, usando mae pra loss, batch de 128 --> pq eu observei muita alteração da direção do crescimento

# Abordagem 5: sem data augmentation, com normalização e sem equalização, usando mae pra loss, batch de 128, sem patience
#Abordagem lenet: com data augmentation, com normalização e com equalização, usando mae pra loss, batch de 128 --> pq eu observei muita alteração da direção do crescimento

# In[7]:


approach = 'abordagem-lenet1' 
activation = 'relu'
net = 'lenet'

if net == 'alexnet':
    model = AlexNet
elif net =='lenet':
    model = LeNet
csvlogger_name = '../alexnet/callbacks/'+net +'/age/history-regression-' + approach + '-' + activation + '.csv'
checkpoint_filename = '../alexnet/callbacks/'+net+'/age/class-weights-' + approach + '-' + activation + '.{epoch:02d}-{val_loss:.2f}.hdf5'
csvlogger_name, checkpoint_filename


# In[8]:


df = pd.read_csv('/home/nicoli/github/alexnet/dataset/csv/imdb_csv/imdb_age_regression_train_split_47950-70-10-20.csv')


# In[9]:


cols = list(df.columns[1:])
in_format = list(df.columns)


# In[10]:


train_dataset = BatchGenerator(box_output_format=cols)
validation_dataset = BatchGenerator(box_output_format=cols)

train_dataset. parse_csv(labels_filename='/home/nicoli/github/alexnet/dataset/csv/imdb_csv/imdb_age_regression_train_split_47950-70-10-20.csv', 
                        images_dir='/home/nicoli/github/alexnet/dataset/imdb-hand-crop',
                        input_format=in_format)

validation_dataset.parse_csv(labels_filename='/home/nicoli/github/alexnet/dataset/csv/imdb_csv/imdb_age_regression_val_split_47950-70-10-20.csv', 
                             images_dir='/home/nicoli/github/alexnet/dataset/imdb-hand-crop',
                             input_format=in_format)


# In[11]:


img_height, img_width, img_depth = (224,224,3)

epochs = 100

train_batch_size = 128
shuffle = True
ssd_train = False

validation_batch_size = 32


# In[12]:


train_generator = train_dataset.generate(batch_size=train_batch_size,
                                         shuffle=shuffle,
                                         ssd_train=ssd_train,
                                         random_rotation=20,
                                         translate=(0.2, 0.2),
                                         scale=(0.8, 1.2),
                                         flip=0.5,
                                         divide_by_stddev=255,
                                         equalize=True,
                                         returns={'processed_labels'},
                                         resize=(img_height, img_width))

validation_generator = validation_dataset.generate(batch_size=validation_batch_size,
                                                   shuffle=shuffle,
                                                   ssd_train=ssd_train,
                                                   divide_by_stddev=255,
                                                   equalize=True,
                                                   returns={'processed_labels'},
                                                   resize=(img_height, img_width))

print("Number of images in the dataset:", train_dataset.get_n_samples())
print("Number of images in the dataset:", validation_dataset.get_n_samples())


# In[13]:


steps_per_epoch = train_dataset.get_n_samples()/train_batch_size
validation_steps = validation_dataset.get_n_samples()/validation_batch_size


# In[14]:


model = model(n_classes=1, img_width=img_width, img_depth=img_depth, img_height=img_height, activation=activation)


# In[15]:


model.summary()


# In[16]:


optimizer = SGD(lr=0.01)


# In[17]:
patience=30

csv_logger = CSVLogger(csvlogger_name, append=True, separator=',')

checkpoint = ModelCheckpoint(checkpoint_filename,
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=False,
                           period=1)

earlystopping = EarlyStopping(patience=5, mode='min')
reduce_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience//5, min_lr=0.001)
#callbacks = [tensorboard, checkpoint]
callbacks=[checkpoint, csv_logger, reduce_on_plateau]#, earlystopping]


# In[18]:


model.compile(loss='mae', optimizer=optimizer, metrics=['mse'])


# In[ ]:


model.fit_generator(train_generator, epochs=epochs, 
                             steps_per_epoch=steps_per_epoch, 
                             validation_data=validation_generator,
                             validation_steps=validation_steps,
                             callbacks=callbacks)

