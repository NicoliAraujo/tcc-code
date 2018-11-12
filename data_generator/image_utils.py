'''
Created on 13 de mar de 2018

@author: nicoli
'''
import imageio.core
import cv2
import numpy as np

from data_generator.dataset_generator import *

def get_means(path, imgpath):
    img = imageio.imread(path +imgpath)
        
    if len(img.shape)!=3:#if grayscale, change to rgbdf_train
        #print(type(img), imgpath)
        
        try:
            cvt_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)                   
            #plt.imshow(img)
        except:
            print(img.shape)
            cvt_img = img
    else:#if bgr, change to rgb
        try:
            cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            cvt_img = img
            print(imgpath)
    return cvt_img[:,:,0].mean(), cvt_img[:,:,1].mean(), cvt_img[:,:,2].mean()

def get_df_channel_mean(path, df):
    paths = df['image_name'].values
    channel_means = np.zeros((len(df),3))
    for i in range(len(paths)):
        if i % 1000==0:
            print(i)
        channel_means[i] = get_means(path, paths[i])
    return channel_means[:,0].mean(), channel_means[:,1].mean(), channel_means[:,2].mean()

if __name__ == '__main__':
    df_train = read_df('../dataset/imdb_csv/imdb_train_split_30000-70-10-20.csv', with_index=False)
    get_df_channel_mean('../dataset/' + df_train)