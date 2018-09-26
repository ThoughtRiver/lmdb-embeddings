"""
LMDB Embeddings - Fast word vectors with little memory usage in Python.
dom.hudson@thoughtriver.com

Copyright (C) 2018 ThoughtRiver Limited

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import os
import pytest
import numpy as np
from lmdb_embeddings import exceptions
from lmdb_embeddings.reader import LmdbEmbeddingsReader
from lmdb_embeddings.writer import LmdbEmbeddingsWriter
from lmdb_embeddings.serializers import MsgpackSerializer
from lmdb_embeddings.tests.base import LmdbEmbeddingsTest


class TestEmbeddingsWriter(LmdbEmbeddingsTest):

    @LmdbEmbeddingsTest.make_temporary_folder
    def test_write_embeddings(self, folder_path):
        """ Ensure we can write embeddings to disk
        without error.

        :return void
        """
        LmdbEmbeddingsWriter([
            ('the', np.random.rand(10)),
            ('is', np.random.rand(10))
        ]).write(folder_path)

        assert os.listdir(folder_path)

    @LmdbEmbeddingsTest.make_temporary_folder
    def test_write_embeddings_generator(self, folder_path):
        """ Ensure we can a generator of embeddings to disk
        without error.

        :return void
        """
        embeddings_generator = ((str(i), np.random.rand(10)) for i in range(10))

        LmdbEmbeddingsWriter(
            embeddings_generator
        ).write(folder_path)

        assert os.listdir(folder_path)

    @LmdbEmbeddingsTest.make_temporary_folder
    def test_reading_embeddings(self, folder_path):
        """ Ensure we can retrieve embeddings from
        the database.

        :return void
        """
        the_vector = np.random.rand(10)
        LmdbEmbeddingsWriter([
            ('the', the_vector),
            ('is', np.random.rand(10))
        ]).write(folder_path)

        assert LmdbEmbeddingsReader(folder_path).get_word_vector(
            'the'
        ).tolist() == the_vector.tolist()

    @LmdbEmbeddingsTest.make_temporary_folder
    def test_missing_word_error(self, folder_path):
        """ Ensure a MissingWordError exception is 
        raised if the word does not exist in the 
        database.

        :return void
        """
        LmdbEmbeddingsWriter([
            ('the', np.random.rand(10)),
            ('is', np.random.rand(10))
        ]).write(folder_path)

        reader = LmdbEmbeddingsReader(folder_path)

        with pytest.raises(exceptions.MissingWordError):
            reader.get_word_vector('unknown')

    @LmdbEmbeddingsTest.make_temporary_folder
    def test_word_too_long(self, folder_path):
        """ Ensure we do not get an exception if
        attempting to write a word longer than
        LMDB's max key size,

        :return void
        """
        LmdbEmbeddingsWriter([
            ('a' * 1000, np.random.rand(10)),
        ]).write(folder_path)

    @LmdbEmbeddingsTest.make_temporary_folder
    def test_msgpack_serialization(self, folder_path):
        """ Ensure we can save and retrieve embeddings
        serialized with msgpack.

        :return void
        """
        the_vector = np.random.rand(10)

        LmdbEmbeddingsWriter([
                ('the', the_vector),
                ('is', np.random.rand(10))
            ],
            serializer = MsgpackSerializer.serialize
        ).write(folder_path)

        assert LmdbEmbeddingsReader(
            folder_path,
            unserializer = MsgpackSerializer.unserialize
        ).get_word_vector(
            'the'
        ).tolist() == the_vector.tolist()