from cypher import Cypher

def data_loader(text_size = 100, data_size = 100):
    for i in range(data_size):
        c = Cypher(text_size)
        
        yield c.encoded_cypher_text, c.key
