import keras

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout

from keras.layers.advanced_activations import LeakyReLU, ELU

from keras.layers.normalization import BatchNormalization
from keras import backend as K
import warnings

class Conv2Net:
    def __init__(self, width, height, depth, classes, weights_path=None):
        model=Sequential()
        
        model.add(Conv2D(20, (5, 5) ,padding='same', input_shape=(height, width, depth)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        model.add(Conv2D(50, (5, 5) , padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        if weights_path is not None:
            model.load_weights(weights_path)
            
        self.model = model

class Conv2NetRegressor:
    def __init__(self, width, height, depth, weights_path=None):
        model=Sequential()
        
        model.add(Conv2D(32, (3, 3) ,padding='same', input_shape=(height, width, depth)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
              
        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        
        model.add(Dense(1))
        model.add(Activation('relu'))
        
        if weights_path is not None:
            model.load_weights(weights_path)
            
        self.model = model
        
class CNN(keras.models.Sequential):
    def __init__(self, n_classes, img_width, img_height, img_depth, alpha=0.01, activation='relu', weights_path=None):
        super(CNN, self).__init__(name='mlp')
        
        self.n_classes = n_classes
        self.alpha = alpha
        
        self.activation = activation
        self.build_model(img_width, img_height, img_depth, weights_path)

    
    def add_activation(self):
        if self.activation=='relu':
            self.add(Activation('relu'))
        elif self.activation =='lrelu':
            self.add(LeakyReLU(alpha=self.alpha))
        elif self.activation=='elu':
            self.add(ELU(alpha=self.alpha))
    
    def build_model():
        pass
    
    
class LeNet(CNN):
        
    def build_model(self, img_width, img_height, img_depth, weights_path):
        if K.image_data_format()=='channels_first':
            input_shape=(img_depth, img_width,img_height)
        elif K.image_data_format()=='channels_last':
            input_shape=(img_width,img_height,img_depth)
                           
        self.add(Conv2D(6, (5,5), strides=(1,1), input_shape=input_shape, padding='valid', name='conv_1'))
        self.add_activation()
        self.add(MaxPooling2D(pool_size=(1,1), strides=(2,2), padding='valid'))

        self.add(Conv2D(16, (5,5), strides=(1,1), padding='valid', name='conv_2'))
        self.add_activation()
        self.add(MaxPooling2D(pool_size=(1,1), strides=(2,2), padding='valid'))

        self.add(Flatten())

        self.add(Dense(120))
        self.add_activation()

        self.add(Dense(84))
        self.add_activation()

        self.add(Dense(self.n_classes))

        if self.n_classes==1:
            warning = 'Considering a regression task with output function being ' + self.activation
            warnings.warn(warning)
            self.add_activation()
        else:
            self.add(Activation('softmax', name='softmax'))
            
        if weights_path:
            self.load_weights(weights_path, by_name=True)

class AlexNet(CNN):
        
    def build_model(self, img_width, img_height, img_depth, weights_path=None):
        if K.image_data_format()=='channels_first':
            input_shape=(img_depth, img_width,img_height)
        elif K.image_data_format()=='channels_last':
            input_shape=(img_width,img_height,img_depth)
        
        self.add(Conv2D(96, (11,11), strides=(4,4), input_shape=input_shape, padding='same', name='conv_1'))
        self.add_activation()
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

        self.add(Conv2D(256, (5,5), strides=(1,1), padding='same',  name='conv_2'))
        self.add_activation()
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='convpool_1'))

        self.add(Conv2D(384, (3,3), strides=(1,1), padding='same', name='conv_3'))
        self.add_activation()
        
        self.add(Conv2D(384, (3,3), strides=(1,1), padding='same', name='conv_4'))
        self.add_activation()
        
        self.add(Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_5'))
        self.add_activation()
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='convpool_5'))

        self.add(Flatten(name='flatten'))
        self.add(Dense(4096, name='dense_1'))
        self.add_activation()
        self.add(Dropout(0.5))
        
        self.add(Dense(4096, name='dense_2'))
        self.add_activation()
        self.add(Dropout(0.5))
        
        self.add(Dense(self.n_classes, name='dense_3'))
        
        if self.n_classes==1:
            warning = 'Considering a regression task with output function being ' + self.activation
            warnings.warn(warning)
            self.add_activation()
        else:
            self.add(Activation('softmax', name='softmax'))
            
        if weights_path:
            self.load_weights(weights_path, by_name=True)
