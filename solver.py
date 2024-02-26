import tensorflow as tf
import numpy as np      
from cypher import Cypher
from tensorflow.keras.metrics import Precision, Recall
from model import cracker 
from aux import data_loader


dataset = tf.data.Dataset.from_generator(data_loader,  
                                         output_types=(tf.float32, tf.float32)
                                         )

model = cracker(1000)
model.compile(optimizer="Adam", 
               loss="binary_crossentropy",
              metrics=["accuracy"]
              )


for cyphers, keys in dataset.batch(4).take(1).repeat(50):
    keys = tf.cast(keys, tf.int32)
    keys = tf.one_hot(keys, 26)
    model.fit(cyphers, keys, epochs=1)


c = Cypher(1000)
test_set = [Cypher(1000) for i in range(10)]

x = np.array([c.encoded_cypher_text for c in test_set])


pred = model.predict(x)

percents = list()
for idx, y_hat in enumerate(pred):
    #y_hat = np.argmax(y_hat, axis=1)
    y = np.array(test_set[idx].key)
    results = y_hat == y
    percents.append(float(results[results == True].size) / 26)
    breakpoint()
percents = np.array(percents)
