import gensim
import pickle
import os
import data
import numpy as np
import argparse
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--data_file', type=str, default='', help='a .txt file containing the corpus')
parser.add_argument('--data_path', type=str, default='data/20ng', help='directory containing data')
parser.add_argument('--emb_file', type=str, default='embeddings.txt', help='file to save the word embeddings')
parser.add_argument('--dim_rho', type=int, default=300, help='dimensionality of the word embeddings')
parser.add_argument('--min_count', type=int, default=2, help='minimum term frequency (to define the vocabulary)')
parser.add_argument('--sg', type=int, default=1, help='whether to use skip-gram')
parser.add_argument('--workers', type=int, default=25, help='number of CPU cores')
parser.add_argument('--negative_samples', type=int, default=10, help='number of negative samples')
parser.add_argument('--window_size', type=int, default=4, help='window size to determine context')
parser.add_argument('--iters', type=int, default=50, help='number of iterationst')

args = parser.parse_args()
vocab, _, _, _ = data.get_data(os.path.join(args.data_path))

# Class for a memory-friendly iterator over the dataset
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in open(self.filename):
            yield line.split()

# Memory friendly iterator over sentences
sentences = MySentences(args.data_file)

# Using Transformers to get BERT embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')
res, count = dict(), dict()

# Get contextual embeddings and average them (yes bad)
for sentence in tqdm(sentences):
    text = ' '.join(sentence)
    input_ids = tf.constant(tokenizer.encode(sentence, max_length=tokenizer.model_max_length))[None,:]
    outputs = model(input_ids)
    embeddings = outputs[0][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    for i, id in enumerate(input_ids[0]):
        word = tokens[i]
        if word in vocab:
            if word not in res.keys():
                res[word] = embeddings[i]
                count[word] = 1
            else:
                res[word] = res[word] + embeddings[i]
                count[word] += 1

# Computing average embedding
for word in res.keys():
    res[word] = res[word] / count[word]

# Write the embeddings to a file             
with open(args.emb_file, 'w') as f:
    for word, vec in res.items():
        f.write(word + ' ')
        vec_str = ['%.9f' % val for val in vec]
        vec_str = " ".join(vec_str)
        f.write(vec_str + '\n')
