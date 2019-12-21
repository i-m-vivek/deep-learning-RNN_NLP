import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, GlobalMaxPool1D
from keras.layers import LSTM ,Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score

# Hyperparametrs 
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

from google.colab import drive
drive.mount('/content/drive')

# importing required modules 
from zipfile import ZipFile 

# specifying the zip file name 
file_name = "/content/drive/My Drive/glove-vector/glove.6B.zip"

# opening the zip file in READ mode 
with ZipFile(file_name, 'r') as zip: 
	# printing all the contents of the zip file 
	zip.printdir() 
	# extracting all the files 
	print('Extracting all the files now...') 
	zip.extractall() 
	print('Done!')

"""### A look at the glove vector file
**to** 0.68047 -0.039263 0.30186 -0.17792 0.42962 0.032246 -0.41376 0.13228 -0.29847 -0.085253 0.17118 0.22419 -0.10046 -0.43653 0.33418 0.67846 0.057204 -0.34448 -0.42785 -0.43275 0.55963 0.10032 0.18677 -0.26854 0.037334 -2.0932 0.22171 -0.39868 0.20912 -0.55725 3.8826 0.47466 -0.95658 -0.37788 0.20869 -0.32752 0.12751 0.088359 0.16351 -0.21634 -0.094375 0.018324 0.21048 -0.03088 -0.19722 0.082279 -0.09434 -0.073297 -0.064699 -0.26044
"""

# Load the pretraining word embedding 
print("Loading word embeddings .....")
# a word2vec dict to map from word to the vector representation 
word2vec = {}
# /content/glove.6B.100d.txt
with open(os.path.join('/content/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
    for line in f: 
        values  = line.split()
        word = values[0]
        vec = np.asarray(values[1: ], dtype = np.float32)
        word2vec[word] = vec

print(f"We have {len(word2vec)} word vector.")

print('Loading in comments...')
# /content/drive/My Drive/datasets/jigsaw-toxic-comment-classification-challenge
train = pd.read_csv("/content/drive/My Drive/datasets/jigsaw-toxic-comment-classification-challenge/train.csv/train.csv")
sentences = train["comment_text"].fillna("DUMMY_VALUE").values
possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = train[possible_labels].values

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

print("max sequence length:", max(len(s) for s in sequences))
print("min sequence length:", min(len(s) for s in sequences))
s = sorted(len(s) for s in sequences)
print("median sequence length:", s[len(s) // 2])

word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False
)

inputs = Input(shape = (MAX_SEQUENCE_LENGTH, ))
x = embedding_layer(inputs)
x = LSTM(10, activation='relu', return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
output = Dense(len(possible_labels), activation='sigmoid')(x)

model = Model(inputs, output)
model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

model.summary()

r = model.fit(
  data,
  targets,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=VALIDATION_SPLIT
)

#losses
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))

model.save_weights('rnn_weights-nlp.h5')

