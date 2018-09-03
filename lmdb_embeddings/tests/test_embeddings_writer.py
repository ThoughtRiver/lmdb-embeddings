import os
import pytest
import numpy as np
from lmdb_embeddings import exceptions
from lmdb_embeddings.writer import LmdbEmbeddingsWriter
from lmdb_embeddings.reader import LmdbEmbeddingsReader
from lmdb_embeddings.tests.base import LmdbEmbeddingsTest


class TestEmbeddingsWriter(LmdbEmbeddingsTest):

    @LmdbEmbeddingsTest.make_temporary_folder
    def test_write_embeddings(self, folder_path):
        """ Ensure we can write embeddings to disk
        without error.

        :return void
        """
        LmdbEmbeddingsWriter([
            ('the', np.ndarray(10)),
            ('is', np.ndarray(10))
        ]).write(folder_path)

        assert os.listdir(folder_path)

    @LmdbEmbeddingsTest.make_temporary_folder
    def test_write_embeddings_generator(self, folder_path):
        """ Ensure we can a generator of embeddings to disk
        without error.

        :return void
        """
        embeddings_generator = ((str(i), np.ndarray(10)) for i in range(10))

        LmdbEmbeddingsWriter(
            embeddings_generator
        ).write(folder_path)

        assert os.listdir(folder_path)

    @LmdbEmbeddingsTest.make_temporary_folder
    def test_reading_embeddings(self, folder_path):
        """ Ensure we can retrieve embeddings from
        a database 

        :return void
        """
        the_vector = np.ndarray(10)
        LmdbEmbeddingsWriter([
            ('the', the_vector),
            ('is', np.ndarray(10))
        ]).write(folder_path)

        assert LmdbEmbeddingsReader(folder_path).get_word_vector(
            'the'
        ).tolist() == the_vector.tolist()

    @LmdbEmbeddingsTest.make_temporary_folder
    def test_missing_word_error(self, folder_path):
        """ Ensure we can retrieve embeddings from
        a database 

        :return void
        """
        LmdbEmbeddingsWriter([
            ('the', np.ndarray(10)),
            ('is', np.ndarray(10))
        ]).write(folder_path)

        reader = LmdbEmbeddingsReader(folder_path)

        with pytest.raises(exceptions.MissingWordError):
            reader.get_word_vector('unknown')
