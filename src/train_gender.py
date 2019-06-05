import keras
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
import sys
sys.path.append("../models")
import neural_net

# hyper parameters
num_classes = 2
epochs = 100
batch_size =20
lr = 0.001

model = neural_net.first_deep_cnn(num_classes)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0, patience=6, mode='auto')
adam = optimizers.Adam(lr=lr)

model.compile(optimizer = adam,loss = "binary_crossentropy",metrics = ['accuracy'])

print(model.summary())