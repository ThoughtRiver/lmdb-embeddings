import lmdb
from lmdb_embeddings.serializers import PickleSerializer


class LmdbEmbeddingsWriter:

    _batch_size = 1024
    _map_size = 100 * 1024 * 1024 * 1024 

    def __init__(self, embeddings_generator, serializer = PickleSerializer.serialize):
        """ Constructor.

        :return void
        """
        self.embeddings_generator = embeddings_generator
        self.serializer = serializer

    def write(self, path):
        """ Write the database of embeddings to a given
        file path.

        :return void
        """
        environment = lmdb.open(path, map_size = self._map_size)
        transaction = environment.begin(write = True)

        for i, (word, vector) in enumerate(self.embeddings_generator):

            transaction.put(word.encode(encoding = 'UTF-8'), self.serializer(vector))

            if i % self._batch_size == 0:
                transaction.commit()
                transaction = environment.begin(write = True)

        transaction.commit()