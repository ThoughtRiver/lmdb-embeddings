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


import lmdb
from lmdb_embeddings import exceptions
from lmdb_embeddings.serializers import PickleSerializer


class LmdbEmbeddingsReader:

    def __init__(self, path, unserializer = PickleSerializer.unserialize, **kwargs):
        """ Constructor.

        :return void
        """
        self.unserializer = unserializer
        self.environment = lmdb.open(
            path,
            readonly = True,
            max_readers = 2048,
            max_spare_txns = 2,
            lock = kwargs.pop('lock', False),
            **kwargs
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
