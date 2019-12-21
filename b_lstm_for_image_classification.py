import keras 
from keras.datasets import mnist 
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional, GlobalMaxPooling1D, Lambda, Concatenate, Dense
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM
  from keras.layers import CuDNNGRU as GRU

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')/255
x_test  = x_train.astype('float32')/255.

import matplotlib.pyplot as plt
plt.imshow(x_train[5], cmap = "gray")

plt.imshow(np.rot90(x_train[5]), cmap = "gray")

x_rotated = np.zeros(shape = (x_train.shape[0], x_train.shape[2], x_train.shape[1]))
for i in range(len(x_train)):
    x_rotated[i, :, :] = np.rot90(x_train[i])

x_rotated[0].shape

plt.imshow(x_rotated[0], cmap='gray' )

M = 15
D = 28
input_ = Input(shape=(D, D))

rnn1 = Bidirectional(LSTM(M, return_sequences=True))
x1 = rnn1(input_) 
x1 = GlobalMaxPooling1D()(x1) 


rnn2 = Bidirectional(LSTM(M, return_sequences=True))


permutor = Lambda(lambda t: K.permute_dimensions(t, pattern=(0, 2, 1)))

x2 = permutor(input_)
x2 = rnn2(x2)
x2 = GlobalMaxPooling1D()(x2) 


concatenator = Concatenate(axis=1)
x = concatenator([x1, x2])


output = Dense(10, activation='softmax')(x)

model = Model(inputs=input_, outputs=output)

model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


print('Training model...')
r = model.fit(x_train, y_train, batch_size=1024, epochs=50, validation_split=0.33)


plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

model.save_weights("b-lstm-image-classification.h5")

