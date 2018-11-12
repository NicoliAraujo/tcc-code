
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# Image-treat-1: sem data augmentation, com normalização e equalização
# 
# Image-treat-2: com data augmentation, com normalização e equalização
# 
# Image-treat-3: com data augmentation, com normalização e com equalização

# In[26]:


activations = ['relu', 'lrelu']
img_treats=['image-treat-1', 'image-treat-2', 'image-treat-3']
nets = ['lenet', 'alexnet']


for img_treat in img_treats:
    for net in nets:
        for activation in activations:
            # In[27]:


            df_hist = pd.read_csv('callbacks/'+net+'/age/consolidados/history-regression-'+img_treat+ '-' + activation + '.csv').rename(
                columns={'epoch': 'Época', 'loss':'RMSE Treino', 'val_loss': 'RMSE Validação', 
                         'mean_absolute_error': 'MAE Treino', 'val_mean_absolute_error': 'MAE Validação'})
            df_hist['Época']+=1
            #df_hist_alex = df_hist[0:6]
            #df_hist_lenet = df_hist[6:]


            # In[28]:


            df_hist[['RMSE Treino', 'RMSE Validação']] = np.sqrt(df_hist[['RMSE Treino', 'RMSE Validação']])


            # In[29]:


            df_hist = df_hist.round(2)
            df_hist


            # In[30]:


            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("", ["0.5","0.1"])


            # In[31]:


            #df_hist.loc[2:].drop(['MAE Treino', 'MAE Validação'], axis=1)
            df_hist.drop(['MAE Treino', 'MAE Validação'], axis=1)


            # In[32]:


            ax1 = df_hist.drop(['MAE Treino', 'MAE Validação'], axis=1).plot(x='Época', colormap=cmap)
            ax1.set_ylabel("RMSE (anos)")
            ax1.get_figure().savefig('graficos/result-hist/fig-history-' + img_treat+ '-' + net + '-' + activation +'-rmse.png')

            
            plt.close('all')
            # In[33]:


            ax1 = df_hist.drop(['RMSE Treino', 'RMSE Validação'], axis=1).plot(x='Época', colormap=cmap)
            ax1.set_ylabel("MAE (anos)")
            ax1.get_figure().savefig('graficos/result-hist/fig-history-' + img_treat+ '-' + net + '-' + activation +'-mae.png')
            
            plt.close('all')
