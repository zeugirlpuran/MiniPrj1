# dataset/fashion_mnist.py
import os
import gzip
import numpy as np

key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

def load_img(file_name):
    file_path = os.path.dirname(__file__) + "/" + file_name
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 1, 28, 28)

def load_label(file_name):
    file_path = os.path.dirname(__file__) + "/" + file_name
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def load_fashion_mnist(normalize=True, flatten=False):
    dataset = {}
    dataset['train_img'] = load_img(key_file['train_img'])
    dataset['train_label'] = load_label(key_file['train_label'])    
    dataset['test_img'] = load_img(key_file['test_img'])
    dataset['test_label'] = load_label(key_file['test_label'])
    
    if normalize:
        dataset['train_img'] = dataset['train_img'].astype(np.float32) / 255.0
        dataset['test_img'] = dataset['test_img'].astype(np.float32) / 255.0
        
    if flatten:
        dataset['train_img'] = dataset['train_img'].reshape(-1, 784)
        dataset['test_img'] = dataset['test_img'].reshape(-1, 784)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])