from pathlib import Path
from glob import glob
from random import shuffle, randint
import numpy as np

data_dir = Path(r'/Users/work/Data/Cypher')

abc = 'abcdefghijklmnopqrstuvwxyz'
class Cypher():
    def __init__(self, size):
         
        self.plain_text = self.load_text()
        slice_idx = randint(0, len(self.plain_text) - size)

        self.plain_text = self.plain_text[slice_idx : slice_idx + size]
        self.key = self.generate_key()
        self.cypher_text = self.encrypt(self.key, self.plain_text)

        #self.encoded_plain_text = self.encode(self.plain_text) 
        self.encoded_cypher_text = self.encode(self.cypher_text) 


    @staticmethod 
    def load_text():
        for path in [x for x in glob(str(data_dir) + '/*.txt') ]:
            with open(path, 'r') as file:
                text = file.read()
                text = text.lower()
            

            # Filter out non alphabetic
            for char in set(text):
                if char not in (abc + ' '):
                    text = text.replace(char, '')

            return text 

    @staticmethod 
    def generate_key():
        mixed = list(abc)
        shuffle(mixed)
        mixed = ''.join(mixed) + ' '
        return [ mixed.index(i) for i in abc]  

    @staticmethod
    def encrypt(key, plain_text):

        for idx, ele in enumerate(key):
            plain_text = plain_text.replace(abc[ele], abc[idx].upper())

        plain_text = plain_text.lower()
        return plain_text

    @staticmethod
    def decrypt(key, cypher_text):
        cypher_text = cypher_text.lower()

        for idx, ele in enumerate(key):
            plain_text = cypher_text.replace(abc[idx], abc[ele].upper())

        plain_text = plain_text.lower()
        return cypher_text

    @staticmethod
    def encode(text):
        return np.array([(abc + ' ').index(i) for i in text])

if __name__ == '__main__':
    cypher = Cypher()

