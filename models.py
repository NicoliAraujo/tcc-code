import keras

from keras.models import Sequential

from keras.models import Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D


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
        #self.input_shape = self.set_input_shape(img_width, img_height, img_depth)
        self.build_model(img_width, img_height, img_depth, weights_path)
        
    
    def set_input_shape(self, img_width, img_height, img_depth):
        if K.image_data_format()=='channels_first':
            self.my_input_shape = img_depth, img_width,img_height
        elif K.image_data_format()=='channels_last':
            self.my_input_shape = img_width, img_height, img_depth
            
    def add_activation(self):
        if self.activation=='relu':
            self.add(Activation('relu'))
        elif self.activation =='lrelu':
            self.add(LeakyReLU(alpha=self.alpha))
        elif self.activation=='elu':
            self.add(ELU(alpha=self.alpha))
    
    def build_model():
        pass
    
class SqueezeNet(CNN):
    def build_model(self, img_width, img_height, img_depth, weights_path):
        """ Keras Implementation of SqueezeNet(arXiv 1602.07360)

        @param nb_classes: total number of final categories

        Arguments:
        inputs -- shape of the input images (channel, cols, rows)

        """
        if K.image_data_format()=='channels_first':
            input_shape=(img_depth, img_width,img_height)
        elif K.image_data_format()=='channels_last':
            input_shape=(img_width,img_height,img_depth)
        
        #params nb_classes, inputs=(3, 224, 224)
        
        self.add(Conv2D(96, (7, 7), kernel_initializer='glorot_uniform', strides=(2, 2), padding='same', name='conv1', input_shape=input_shape))
        self.add_activation()
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1'))
        self.add(Conv2D(16, (1, 1), kernel_initializer='glorot_uniform',
            padding='same', name='fire2_squeeze'))
        self.add_activation()
        self.add(Conv2D(64, (1, 1), kernel_initializer='glorot_uniform',
            padding='same', name='fire2_expand1'))
        self.add_activation()
        self.add(Convolution2D(
            64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire2_expand2')(fire2_squeeze)
        self.add(Concatenate(axis=1)([fire2_expand1, fire2_expand2])

        self.add(Convolution2D(
            16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire3_squeeze')(merge2)
        self.add(Convolution2D(
            64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire3_expand1')(fire3_squeeze)
        self.add(Convolution2D(
            64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire3_expand2')(fire3_squeeze)
        self.add(Concatenate(axis=1)([fire3_expand1, fire3_expand2])

        self.add(Convolution2D(
            32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire4_squeeze')(merge3)
        fire4_expand1 = Convolution2D(
            128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire4_expand1')(fire4_squeeze)
        fire4_expand2 = Convolution2D(
            128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire4_expand2')(fire4_squeeze)
        merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
        maxpool4 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), name='maxpool4')(merge4)

        fire5_squeeze = Convolution2D(
            32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire5_squeeze')(maxpool4)
        fire5_expand1 = Convolution2D(
            128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire5_expand1')(fire5_squeeze)
        fire5_expand2 = Convolution2D(
            128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire5_expand2')(fire5_squeeze)
        merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])

        fire6_squeeze = Convolution2D(
            48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire6_squeeze')(merge5)
        fire6_expand1 = Convolution2D(
            192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire6_expand1')(fire6_squeeze)
        fire6_expand2 = Convolution2D(
            192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire6_expand2')(fire6_squeeze)
        merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])

        fire7_squeeze = Convolution2D(
            48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire7_squeeze')(merge6)
        fire7_expand1 = Convolution2D(
            192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire7_expand1')(fire7_squeeze)
        fire7_expand2 = Convolution2D(
            192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire7_expand2')(fire7_squeeze)
        merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])

        fire8_squeeze = Convolution2D(
            64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire8_squeeze')(merge7)
        fire8_expand1 = Convolution2D(
            256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire8_expand1')(fire8_squeeze)
        fire8_expand2 = Convolution2D(
            256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire8_expand2')(fire8_squeeze)
        merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])

        maxpool8 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), name='maxpool8')(merge8)
        fire9_squeeze = Convolution2D(
            64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire9_squeeze')(maxpool8)
        fire9_expand1 = Convolution2D(
            256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire9_expand1')(fire9_squeeze)
        fire9_expand2 = Convolution2D(
            256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire9_expand2')(fire9_squeeze)
        merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])

        fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
        conv10 = Convolution2D(
            self.n_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='valid', name='conv10')(fire9_dropout)

        global_avgpool10 = GlobalAveragePooling2D()(conv10)
        
        if self.n_classes==1:
            warning = 'Considering a regression task with output function being ' + self.activation
            warnings.warn(warning)
            last = Activation('relu', name='relu')(global_avgpool10)
        else:
            last = Activation("softmax", name='softmax')(global_avgpool10)

        return Model(inputs=input_shape, outputs=last)

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
