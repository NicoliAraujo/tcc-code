
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from data_generator.batch_generator import BatchGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint#, TensorBoard
import pickle
from models import AlexNet, LeNet
from keras import backend as K
from math import sqrt
import glob
K.set_image_data_format('channels_last')


# In[2]:


activations = ['relu', 'lrelu']
img_treats=['image-treat-1', 'image-treat-2', 'image-treat-3']
nets = ['lenet', 'alexnet']

for img_treat in img_treats:
    for net in nets:
        for activation in activations:

            # In[3]:


            test_dataset = BatchGenerator(box_output_format=['class_id'])
            test_dataset.parse_csv(labels_filename='/home/nicoli/github/alexnet/dataset/csv/imdb_csv/imdb_age_regression_test_split_47950-70-10-20.csv', 
                                    images_dir='/home/nicoli/github/alexnet/dataset/imdb-hand-crop/',
                                    input_format=['image_name', 'class_id'])


            # In[4]:


            img_height, img_width, img_depth = (224,224,3)

            epochs = 90

            train_batch_size = 64
            shuffle = True
            ssd_train = False

            validation_batch_size = 32


            # In[15]:

            test_generator = test_dataset.generate(batch_size=train_batch_size,
                                                   shuffle=shuffle,
                                                   ssd_train=ssd_train,
                                                   divide_by_stddev = 225,
                                                   #equalize=True,
                                                   returns={'processed_labels'},
                                                   resize=(img_height, img_width))


            # In[5]:


            print("Number of images in the dataset:", test_dataset.get_n_samples())


            # In[6]:


            steps = test_dataset.get_n_samples()/train_batch_size


            # In[7]:


            if net == 'alexnet':
                net_obj = AlexNet
            elif net == 'lenet':
                net_obj = LeNet
            weights_path = glob.glob('callbacks/' + net + '/age/consolidados/class-weights-' + img_treat + '-' + activation + '*')[0]
            model = net_obj(1, img_width, img_height, img_depth, activation, weights_path=weights_path)


            # In[7]:


            model.summary()
            #alexnet.model.load_weights('callbacks/alexnet/age/weights.24-1658.03.hdf5')


            # In[8]:


            optimizer = Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0000000001, amsgrad=True)


            # In[9]:


            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])


            # In[10]:


            stats = model.evaluate_generator(test_generator,steps=30)


            #sqrt(stats[0]), stats


            # In[11]:

            start = weights_path.find(activation)+len(activation)+1
            end = start + 2
            epoca = int(weights_path[start:end]) + 1
            
            
            try:
                df_results = pd.read_csv('graficos/results_teste.csv', index_col=0)

            except FileNotFoundError:
                df_results = pd.DataFrame({'Rede':[], 'Ativação':[], 'RMSE':[], 'MAE':[], 'FASE': [], 'Epoca': []})
            # In[12]:


            df_results = df_results.append({'Rede':net, 'Ativação':activation, 'RMSE': sqrt(stats[0]), 'MAE':stats[1], 'FASE': img_treat, 'Epoca' = epoca}, ignore_index=True)


            # In[13]:


            #df_results.to_csv('graficos/results_teste_' + img_treat + '.csv', index_col=0)
            df_results.to_csv('graficos/results_teste.csv')



