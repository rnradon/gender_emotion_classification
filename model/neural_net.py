import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,BatchNormalization,GlobalAveragePooling2D,Flatten
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint


def first_deep_cnn(num_classes, input_shape):
    model = Sequential()

    # layer 1
    model.add(Conv2D(32,(3,3),padding = 'same',activation = 'relu',input_shape = input_shape, name = 'c1'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2),name = 'mp1'))

    # layer 2
    model.add(Conv2D(64,(3,3),padding = 'same',activation = 'relu',name = 'c2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2),name = 'mp2'))

    # layer 3
    model.add(Conv2D(128,(3,3),padding = 'same',activation = 'relu',name = 'c3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2),name = 'mp3'))

    # layer 4
    model.add(Conv2D(256,(3,3),padding = 'same',activation = 'relu',name = 'c4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2),name = 'mp4'))
    # layer 5
    model.add(Conv2D(512,(3,3),padding = 'same',activation = 'relu',name = 'c5'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2),name = 'mp5'))

    # layer 6
    model.add(Conv2D(256,(3,3),padding = 'same',activation = 'relu',name = 'c6'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2),name = 'mp6'))

    # # layer 7
    # model.add(Conv2D(128,(3,3),padding = 'same',activation = 'relu',name = 'c7'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size = (2,2),name = 'mp7'))

    # # layer 8
    # model.add(Conv2D(64,(3,3),padding = 'same',activation = 'relu',name = 'c8'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size = (2,2),name = 'mp8'))

    # layer 9
    model.add(Conv2D(num_classes,(3,3),padding = 'same',activation = 'relu',name = 'c9'))
    model.add(Flatten())
    #model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax', name='predictions'))

    
    return model


