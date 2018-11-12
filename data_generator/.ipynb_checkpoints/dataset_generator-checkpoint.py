'''
Created on 12 de mar de 2018

@author: nicoli
'''
import scipy.io
import pandas as pd
import numpy as np
import datetime as dt
from multiprocessing import Process, Queue

def read_df(filename, with_index):
    if with_index: return pd.read_csv(filename, index_col=0)
    elif not with_index: return pd.read_csv(filename)
    
def read_data_from_mat(filename):
    return scipy.io.loadmat(filename)

def get_dataframe_from_dictionary(mat, key):
    '''
    mat: .mat file read from scipy
    key: name of the dataset (imdb or wiki)
    '''
    df = pd.DataFrame()
    names = list(mat[key].dtype.names)
    
    try:
        names.remove('celeb_names')
    except ValueError:
        print('Name not in list')
    for name in names:
        df[name] = mat[key][name][0][0][0,:]
    
    return df 

def check_null(df):
    for name in df.columns:
        print(name, ': ', df[df[name].isnull()].shape[0])
    
def fix_full_path(args, q):
    df, origin = args 
    for i in df.index:
        try:
            df.loc[i, 'full_path'] = origin + '/' + df.loc[i, 'full_path'][0]
        except:
            print(df.loc[i, 'full_path'])
            df.loc[i, 'full_path'] = np.NaN

    q.put(df)
    
def fix_full_path_no_thread(df, origin):
    fp = df['full_path'].values.reshape(len(df),1)
    a = []
    for i in list(df.index):
        try:
            a.append('imdb/'+fp[i][0][0])
        except IndexError:
            print(i)
    df.loc[:,'full_path'] = a
    return df


def remove_low_face_score(df, face_score_threshold):
    return df.drop(df[df['face_score']<face_score_threshold].index)


def remove_null_gender(df):
    df.drop(df[df['gender'].isnull()].index, inplace=True)
    df['gender'] = df['gender'].astype(int)
    return df

def set_age(df):
    df['age'] = df['photo_taken'] - df['yob']
    return df

def set_names(df_in, q):
    for i in range(df['name'].shape[0]):
        if type(df_in.loc[i,'name']) != 'numpy.str_':
            try:
                df_in.loc[i,'name'] = df_in.loc[i,'name'][0]
            except:
                df_in.loc[i, 'name'] = np.NaN
    q.put(df_in)

def set_dob_datetime(matlab_datenum):
    try:
        return dt.datetime.fromordinal(int(matlab_datenum)) + dt.timedelta(days=int(matlab_datenum)%1) - dt.timedelta(days = 366)
    except OverflowError:
        return np.NaN
    

def part_dfs(num_threads, df):
    df_list = []
    part = int(len(df)/num_threads)
    for i in range(num_threads-1):
        df_part = df[i*part:((i+1)*part)].copy()
        print(df_part.shape)
        df_list.append(df_part)
    df_part = df[(num_threads-1)*part:].copy()
    print(df_part.shape)
    df_list.append(df_part)
    return df_list

def set_face_locations(df):
    face_loc = df['face_location'].values.reshape(len(df),1)
    new_face_loc = np.zeros((len(df), 1, 4)) 
    for i in list(df.index):
        new_face_loc[i] = face_loc[i][0]
    df['xmin'] = new_face_loc[:, 0, 0]
    df['xmax'] = new_face_loc[:, 0, 2]
    df['ymin'] = new_face_loc[:, 0, 1]
    df['ymax'] = new_face_loc[:, 0, 3]
    
    return df

def set_labels_for_age(dataframe, task='regression'):
    dataframe_edited = dataframe.rename(columns={'full_path': 'image_name',
                               'age': 'class_id'})
    dataframe_edited['class_id'] = dataframe_edited['class_id'].astype(int)
    if task=='regression':
        dataframe_edited = dataframe_edited[['image_name', 'class_id']]
        return dataframe_edited
    else: 
        raise ValueError('task must be regression')
    
def set_labels_for_gender(df, task):
    dataframe_edited = df.rename(columns={'full_path': 'image_name',
                       'gender': 'class_id'})
    if task=='classification':
        return dataframe_edited[['image_name', 'class_id']]
    elif task=='detection':
        return dataframe_edited[['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']]
    else:
        raise ValueError('task must be either classification or detection')
    
def process(num_threads, function, args):
    q = Queue()
    
    p_list = []
    for thread in range(num_threads):
        p = Process(target=function, args = (args, q))
        p.start()
        p_list.append(p)
    
    results = []
    for i in range(num_threads):
        results.append(q.get(True))
    df = pd.concat(results)
    
    for p in p_list:
        p.join()
    
    return df

def split_data(df, train_split, val_split, test_split):
    total_size = df.shape[0]
    
    train_lim = round(total_size*train_split)
    val_lim = train_lim + round(total_size*val_split)
    test_lim = val_lim + round(total_size*test_split)
    
    train_df = df[:train_lim]
    val_df = df[train_lim:val_lim]
    test_df = df[val_lim:test_lim]
    
    return train_df, val_df, test_df

def set_gender_data_splits(df, new_size, train_split, val_split, test_split, class_list, target):
    nb_classes = len(class_list)
    train_df_list = []
    val_df_list = []
    test_df_list = []
    
    for class_id in class_list:
        df_class = df[df[target].values==class_id].reset_index(drop=True)
        df_class = sample_dataframe(df_class, int(new_size/nb_classes))
        train_class_df, val_class_df, test_class_df = split_data(df_class, train_split, val_split, test_split)
        train_df_list.append(train_class_df)
        val_df_list.append(val_class_df)
        test_df_list.append(test_class_df)
        
    train_df = pd.concat(train_df_list).reset_index(drop=True).sample(frac=1)
    val_df = pd.concat(val_df_list).reset_index(drop=True).sample(frac=1)
    test_df = pd.concat(test_df_list).reset_index(drop=True).sample(frac=1)
    
    return train_df, val_df, test_df

def set_age_data_splits(df, new_size, train_split, val_split, test_split, target):
    train_df, val_df, test_df = split_data(df, train_split, val_split, test_split)
    return train_df, val_df, test_df

def sample_dataframe(df, new_size):
    rand_index = np.random.randint(df.shape[0], size=new_size)
    new_df = df.loc[rand_index, :]
    new_df.reset_index(drop=True, inplace=True)
    return new_df
    
if __name__ == '__main__':
    df = read_df('../dataset/imdb/imdb.csv', True)
    print(set_gender_data_splits(df, 30000, 0.7, 0.1, 0.2, [0,1], 'class_id'))
