# LMDB Embeddings
Query word vectors (embeddings) very quickly with very little querying time overhead and far less memory usage than gensim or other equivalent solutions.

Inspired by [Delft](https://github.com/kermitt2/delft). As explained in their readme, this approach permits us to have the pre-trained embeddings immediately "warm" (no load time), to free memory and to use any number of embeddings similtaneously with a very negligible impact on runtime when using SSD.

## Reading vectors

```python
from lmdb_embeddings.reader import LmdbEmbeddingsReader
from lmdb_embeddings.exceptions import MissingWordError

embeddings = LmdbEmbeddingsReader('/path/to/word/vectors/GoogleNews)

try:
  vector = embeddings.get_word_vector('google')
except MissingWordError:
  # 'google' is not in the database.
  pass
```

## Writing vectors
An example to write an LMDB vector file from the gensim. Any iterator that yeilds word and vector pairs is supported.

```python
from gensim.models.keyedvectors import KeyedVectors
from lmdb_embeddings.writer import LmdbEmbeddingsWriter

OUTPUT_DATABASE_FOLDER = 'GoogleNews-vectors-negative300'

gensim_model = KeyedVectors.load('GoogleNews-vectors-negative300.w2v', mmap = 'r')

# Define an iterator to yield the vectors.
def iter_embeddings():
    for word in tqdm.tqdm(gensim_model.vocab.keys()):
        yield word, gensim_model[word]

# Write the vectors to disk.
writer = LmdbEmbeddingsWriter(
    iter_embeddings()
).write(OUTPUT_DATABASE_FOLDER)

# These vectors can now be loaded with the LmdbEmbeddingsReader.
```

## Running tests
```
pytest
```

## Customisation
By default, LMDB Embeddings uses pickle to serialize the vectors to bytes (optimized and pickled with the highest available protocol). However, it is very easy to use an alternative approach such as [msgpack](https://msgpack.org/index.html). Simply inject the serializer and unserializer as callables into the `LmdbEmbeddingsWriter` and `LmdbEmbeddingsReader`.
