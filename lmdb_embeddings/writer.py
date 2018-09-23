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

import logging
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

            encoded_word = word.encode(encoding = 'UTF-8')

            if self._word_too_long(encoded_word, environment):
                logging.getLogger(__name__).warning(
                    '[%s] is too long to use as an LMDB key.' % word
                )
                continue

            transaction.put(encoded_word, self.serializer(vector))

            if i % self._batch_size == 0:
                transaction.commit()
                transaction = environment.begin(write = True)

        transaction.commit()

    @staticmethod
    def _word_too_long(encoded_word, environment):
        """ Is a given encoded word too long for the LMDB
        environment?

        :return bool
        """
        return len(encoded_word) > environment.max_key_size()
