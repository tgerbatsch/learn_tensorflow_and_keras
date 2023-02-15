# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:40:29 2023
SOURCE: https://medium.com/analytics-vidhya/understanding-embedding-layer-in-keras-bbe3ff1327ce

@author: user_upcnb00125
"""

from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Embedding,Dense


# Define 10 restaurant reviews
reviews =[
          'Never coming back!',
          'horrible service',
          'rude waitress',
          'cold food',
          'horrible food!',
          'awesome',
          'awesome services!',
          'rocks',
          'poor work',
          'couldn\'t have done better'
]#Define labels
labels = array([1,1,1,1,1,0,0,0,0,0])


Vocab_size = 50
encoded_reviews = [one_hot(d,Vocab_size) for d in reviews]
print(f'encoded reviews: {encoded_reviews}')


#encoded_reviews= [[18, 39, 17], [27, 27], [5, 19], [41, 29], [27, 29], [2], [2, 1], [49], [26, 9], [6, 9, 11, 21]]


max_length = 4
padded_reviews = pad_sequences(encoded_reviews,maxlen=max_length,padding='post')
print(padded_reviews)

"""padded_and_encodet_reviews =
[[18 39 17  0]
 [27 27  0  0]
 [ 5 19  0  0]
 [41 29  0  0]
 [27 29  0  0]
 [ 2  0  0  0]
 [ 2  1  0  0]
 [49  0  0  0]
 [26  9  0  0]
 [ 6  9 11 21]]
"""

model = Sequential()
#output_dim = 8 ... die Dimension des Embedding-Vektors
embedding_layer = Embedding(input_dim=Vocab_size, output_dim=8, input_length=max_length)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
print(model.summary())

model.fit(padded_reviews,labels,epochs=100,verbose=0)

print(embedding_layer.get_weights()[0].shape)