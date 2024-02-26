from cypher import Cypher
import numpy as np
from tqdm import tqdm
from pathlib import Path
from glob import glob
data_path = Path(r'/Users/work/Data/Cypher/Numpy')

def data_loader():

    size = len([i for i in  glob( str(data_path) + '/*.npy' )]) // 2

    for i in range(size):
        i = str(i)
        cypher_text = np.load( str(data_path) + '/' + i + '.npy')
        key = np.load( str(data_path) + '/' + i + '_key.npy')

        yield cypher_text, key




def make_data_set(size):
    for i in tqdm(range(size)):
        c = Cypher(1000)

        np.save(str(data_path / (str(i) + '.npy')), c.encoded_cypher_text)
        key = np.array(c.key)
        np.save(str(data_path / (str(i) + '_key.npy')), key)



if __name__ == '__main__':
    make_data_set(1000)





