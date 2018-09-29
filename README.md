![tr_logo_cmyk_tr_logo_cmyk](https://user-images.githubusercontent.com/10864294/29792093-382146cc-8c37-11e7-9e70-6f71b3d0800b.png)

# LMDB Embeddings
Query word vectors (embeddings) very quickly with very little querying time overhead and far less memory usage than gensim or other equivalent solutions. This is made possible by [Lightning Memory-Mapped Database](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database).

Inspired by [Delft](https://github.com/kermitt2/delft). As explained in their readme, this approach permits us to have the pre-trained embeddings immediately "warm" (no load time), to free memory and to use any number of embeddings similtaneously with a very negligible impact on runtime when using SSD.

For instance, in a traditional approach `glove-840B` takes around 2 minutes to load and 4GB in memory. Managed with LMDB, `glove-840B` can be accessed immediately and takes only a couple MB in memory, for a negligible impact on runtime (around 1% slower).

## Reading vectors

```python
from lmdb_embeddings.reader import LmdbEmbeddingsReader
from lmdb_embeddings.exceptions import MissingWordError

embeddings = LmdbEmbeddingsReader('/path/to/word/vectors/eg/GoogleNews-vectors-negative300')

try:
  vector = embeddings.get_word_vector('google')
except MissingWordError:
  # 'google' is not in the database.
  pass
```

## Writing vectors
An example to write an LMDB vector file from a gensim model. As any iterator that yields word and vector pairs is supported, if you have the vectors in an alternative format then it is just a matter of altering the `iter_embeddings` method below appropriately.

I will be writing a CLI interface to convert standard formats soon.

```python
from gensim.models.keyedvectors import KeyedVectors
from lmdb_embeddings.writer import LmdbEmbeddingsWriter


GOOGLE_NEWS_PATH = 'GoogleNews-vectors-negative300.bin.gz'
OUTPUT_DATABASE_FOLDER = 'GoogleNews-vectors-negative300'


print('Loading gensim model...')
gensim_model = KeyedVectors.load_word2vec_format(GOOGLE_NEWS_PATH, binary = True)


def iter_embeddings():
    for word in gensim_model.vocab.keys():
        yield word, gensim_model[word]

print('Writing vectors to a LMDB database...')

writer = LmdbEmbeddingsWriter(
    iter_embeddings()
).write(OUTPUT_DATABASE_FOLDER)

# These vectors can now be loaded with the LmdbEmbeddingsReader.
```

## Customisation
By default, LMDB Embeddings uses pickle to serialize the vectors to bytes (optimized and pickled with the highest available protocol). However, it is very easy to use an alternative approach - simply inject the serializer and unserializer as callables into the `LmdbEmbeddingsWriter` and `LmdbEmbeddingsReader`.

A [msgpack](https://msgpack.org/index.html) serializer is included and can be used in the same way.

```python
from lmdb_embeddings.writer import LmdbEmbeddingsWriter
from lmdb_embeddings.serializers import MsgpackSerializer

writer = LmdbEmbeddingsWriter(
    iter_embeddings(),
    serializer = MsgpackSerializer.serialize
).write(OUTPUT_DATABASE_FOLDER)
```

```python
from lmdb_embeddings.reader import LmdbEmbeddingsReader
from lmdb_embeddings.serializers import MsgpackSerializer

reader = LmdbEmbeddingsReader(
    OUTPUT_DATABASE_FOLDER,
    unserializer = MsgpackSerializer.unserialize
)
```

## Running tests
```
pytest
```
