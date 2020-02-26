import tensorflow
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras. layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# бот
# 

# бот
#

#загрузка набора данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# подготовка данных к подачи на вход MNIST

batch_size, img_rows, img_cols = 64, 28, 28
train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)
X_train =X_train.astype("float32")
X_test= X_test.astype("float32")
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
# слои 
model = Sequential()
model.add(Convolution2D(32, 5, 5, border_mode = "same", input_shape= input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode= "same"))
model.add(Convolution2D(64, 5, 5, border_mode="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode= "same"))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#weights_file = 'weights.hdf5'
#chekpoint = ModelCheckpoint(weights_file, monitor = 'acc', mode = 'max', save_best_only = True, verbose = 1)
# тренировка
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch= 10, verbose=1, validation_data=(X_test, Y_test), callbacks = [chekpoint])
score = model.evaluate(X_test, Y_test, verbose=0)
print("Test score: %f" % score[0])
print("Test accuracy: %f" % score[1])
# сохранение модели
model.save('test.h5')


