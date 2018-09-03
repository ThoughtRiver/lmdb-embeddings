import lmdb
from lmdb_embeddings import exceptions
from lmdb_embeddings.serializers import PickleSerializer


class LmdbEmbeddingsReader:

    def __init__(self, path, unserializer = PickleSerializer.unserialize):
        """ Constructor.

        :return void
        """
        self.unserializer = unserializer
        self.environment = lmdb.open(
            path,
            readonly = True,
            max_readers = 2048,
            max_spare_txns = 2
        )

    def get_word_vector(self, word):
        """ Fetch a word from the LMDB database.
        
        :raises lmdb_embeddings.exceptions.MissingWordError
        :return np.array
        """
        with self.environment.begin() as transaction:
            word_vector = transaction.get(word.encode(encoding = 'UTF-8'))

            if word_vector is None:
                raise exceptions.MissingWordError(
                    '"%s" does not exist in the database.' % word
                )

            return self.unserializer(word_vector)