import tensorflow as tf
import numpy as np      
from cypher import Cypher
from tensorflow.keras.metrics import Precision, Recall
from model import cracker 
from aux import data_loader

cyphers = list()
keys = list()

dataset = tf.data.Dataset.from_generator(data_loader,  
                                         output_types=(tf.float32, tf.float32)
                                         )

cyphers = np.array(cyphers).astype(np.float32)
keys = np.array(keys)


model = cracker(100)
model.compile(optimizer="Adam", 
              loss="binary_crossentropy",
              metrics=["accuracy"]
              )


for cyphers, keys in dataset.batch(batch_size).take(1).repeat(epochs):
    model.fit(cyphers, keys, epochs=10 )
