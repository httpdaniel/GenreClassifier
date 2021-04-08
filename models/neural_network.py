# Import packages
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras.layers.merge import Concatenate
import matplotlib.pyplot as plt

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

df = pd.read_csv('../data/final_datasets/raw_nopop.csv')

X = df[['Lyrics', 'Lyric_Count', 'Character_Count', 'Noun_Count', 'Verb_Count', 'Adjective_Count', 'Adverb_Count',
        'TTR', 'Bigram_Score', 'Trigram_Score', 'Unigram_Score', 'Valence_Pos', 'Valence_Neg', 'Profanity_Count']]

Y = pd.get_dummies(df['Genre'])

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20, random_state=42)

text_xtrain = xtrain['Lyrics']
text_xtest = xtest['Lyrics']

meta_xtrain = xtrain.drop(['Lyrics'], axis=1)
meta_xtest = xtest.drop(['Lyrics'], axis=1)

input_1 = layers.Input(shape=[], dtype=tf.string)
input_2 = layers.Input(shape=(13,))

keras_layer = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4', trainable=True)(input_1)
dense_layer = tf.keras.layers.Dense(128, activation='relu')(keras_layer)

batch_layer1 = layers.BatchNormalization()(input_2)
dense_layer_1 = layers.Dense(128, activation='relu')(batch_layer1)

concat_layer = Concatenate()([dense_layer, dense_layer_1])
dropout_layer1 = layers.Dropout(0.4)(concat_layer)
dense_layer_3 = layers.Dense(64, activation='softmax')(dropout_layer1)
dropout_layer2 = layers.Dropout(0.4)(dense_layer_3)
output = layers.Dense(6, activation='sigmoid')(dropout_layer2)
model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(x=[text_xtrain, meta_xtrain], y=ytrain, epochs=10, verbose=1,
                    validation_data=([text_xtest, meta_xtest], ytest))

# plot loss and accuracy over epoch
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# evaluate predictions on training data
score = model.evaluate(x=[text_xtest, meta_xtest], y=ytest, verbose=1)
print("Test Accuracy:", score[1])
