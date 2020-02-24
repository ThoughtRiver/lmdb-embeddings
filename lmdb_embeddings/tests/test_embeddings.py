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

import numpy as np
import pytest

from lmdb_embeddings import exceptions
from lmdb_embeddings.reader import LmdbEmbeddingsReader
from lmdb_embeddings.reader import LruCachedLmdbEmbeddingsReader
from lmdb_embeddings.serializers import MsgpackSerializer
from lmdb_embeddings.writer import LmdbEmbeddingsWriter


class TestEmbeddings:

    def test_write_embeddings(self, tmp_path):
        """ Ensure we can write embeddings to disk without error.

        :param pathlib.PosixPath tmp_path:
        :return void:
        """
        directory_path = str(tmp_path)

        LmdbEmbeddingsWriter([
            ('the', np.random.rand(10)),
            ('is', np.random.rand(10))
        ]).write(directory_path)

        assert os.listdir(directory_path)

    def test_write_embeddings_generator(self, tmp_path):
        """ Ensure we can a generator of embeddings to disk without error.

        :param pathlib.PosixPath tmp_path:
        :return void:
        """
        directory_path = str(tmp_path)
        embeddings_generator = ((str(i), np.random.rand(10)) for i in range(10))

        LmdbEmbeddingsWriter(embeddings_generator).write(directory_path)

        assert os.listdir(directory_path)

    @pytest.mark.parametrize('reader_class', (LruCachedLmdbEmbeddingsReader, LmdbEmbeddingsReader))
    def test_reading_embeddings(self, tmp_path, reader_class):
        """ Ensure we can retrieve embeddings from the database.

        :param pathlib.PosixPath tmp_path:
        :return void:
        """
        directory_path = str(tmp_path)

        the_vector = np.random.rand(10)
        LmdbEmbeddingsWriter([
            ('the', the_vector),
            ('is', np.random.rand(10))
        ]).write(directory_path)

        assert reader_class(directory_path).get_word_vector('the').tolist() == the_vector.tolist()

    @pytest.mark.parametrize('reader_class', (LruCachedLmdbEmbeddingsReader, LmdbEmbeddingsReader))
    def test_missing_word_error(self, tmp_path, reader_class):
        """ Ensure a MissingWordError exception is raised if the word does not exist in the
        database.

        :param pathlib.PosixPath tmp_path:
        :return void:
        """
        directory_path = str(tmp_path)

        LmdbEmbeddingsWriter([
            ('the', np.random.rand(10)),
            ('is', np.random.rand(10))
        ]).write(directory_path)

        reader = reader_class(directory_path)

        with pytest.raises(exceptions.MissingWordError):
            reader.get_word_vector('unknown')

    def test_word_too_long(self, tmp_path):
        """ Ensure we do not get an exception if attempting to write aword longer than LMDB's
        maximum key size.

        :param pathlib.PosixPath tmp_path:
        :return void:
        """
        directory_path = str(tmp_path)

        LmdbEmbeddingsWriter([('a' * 1000, np.random.rand(10))]).write(directory_path)

    @pytest.mark.parametrize('reader_class', (LruCachedLmdbEmbeddingsReader, LmdbEmbeddingsReader))
    def test_msgpack_serialization(self, tmp_path, reader_class):
        """ Ensure we can save and retrieve embeddings serialized with msgpack.

        :param pathlib.PosixPath tmp_path:
        :return void:
        """
        directory_path = str(tmp_path)
        the_vector = np.random.rand(10)

        LmdbEmbeddingsWriter(
            [('the', the_vector), ('is', np.random.rand(10))],
            serializer = MsgpackSerializer().serialize
        ).write(directory_path)

        reader = reader_class(directory_path, unserializer = MsgpackSerializer().unserialize)
        assert reader.get_word_vector('the').tolist() == the_vector.tolist()
