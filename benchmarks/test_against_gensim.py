import timeit
import tempfile
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from lmdb_embeddings.writer import LmdbEmbeddingsWriter
from lmdb_embeddings.reader import LmdbEmbeddingsReader


NUMBER_OF_WORDS = 10000
NUMBER_OF_TIMING_ITERATIONS = 100
GOOGLE_NEWS_PATH = '/home/dom/.thoughtriver/GoogleNews-vectors-negative300.w2v'


print('Loading gensim model...')
gensim_model = KeyedVectors.load(GOOGLE_NEWS_PATH, mmap = 'r')


print('Exracting %s word vectors...' % NUMBER_OF_WORDS)
words = []
embeddings = []
for i, word in enumerate(gensim_model.vocab.keys()):

    if i > NUMBER_OF_WORDS:
        break

    words.append(word)

    embeddings.append(
        (word, gensim_model[word])
    )


print('Writing vectors to a LMDB database...')
with tempfile.TemporaryDirectory() as directory_path:

    writer = LmdbEmbeddingsWriter(
        embeddings
    ).write(directory_path)

    print('Clearing gensim model from memory...')
    del gensim_model


    print('Timing LMDB approach...')
    lmdb_time = timeit.timeit(
        '[reader.get_word_vector(word) for word in %s]' % words,
        number = NUMBER_OF_TIMING_ITERATIONS,
        setup = 'from lmdb_embeddings.reader import LmdbEmbeddingsReader; reader = LmdbEmbeddingsReader("%s")' % directory_path
    )
    print('LMDB approach took %s seconds.' % lmdb_time)


print('Timing gensim approach...')
gensim_time = timeit.timeit(
    '[gensim_model[word] for word in %s]' % words,
    number = NUMBER_OF_TIMING_ITERATIONS,
    setup = 'from gensim.models.keyedvectors import KeyedVectors; gensim_model = KeyedVectors.load("%s", mmap = "r")' % GOOGLE_NEWS_PATH
)


print('Gensim approach took %s seconds.' % gensim_time)

print('LMDB/Gensim: %s' % (lmdb_time / gensim_time))