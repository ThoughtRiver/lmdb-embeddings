import os
import tqdm
from gensim.models.keyedvectors import KeyedVectors
from lmdb_embeddings.writer import LmdbEmbeddingsWriter


GOOGLE_NEWS_PATH = '/home/dom/.thoughtriver/GoogleNews-vectors-negative300.w2v'
OUTPUT_DIR = os.path.abspath('GoogleNews-vectors-negative300')


print('Loading gensim model...')
gensim_model = KeyedVectors.load(GOOGLE_NEWS_PATH, mmap = 'r')


def iter_embeddings():
    for word in tqdm.tqdm(gensim_model.vocab.keys()):
        yield word, gensim_model[word]

print('Writing vectors to a LMDB database...')

writer = LmdbEmbeddingsWriter(
    iter_embeddings()
).write(OUTPUT_DIR)