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

import pickle
import pickletools

import msgpack
import msgpack_numpy


class PickleSerializer:

    @staticmethod
    def serialize(vector):
        """ Serializer a vector using pickle.

        :param np.array vector:
        :return bytes:
        """
        return pickletools.optimize(pickle.dumps(vector, pickle.HIGHEST_PROTOCOL))

    @staticmethod
    def unserialize(serialized_vector):
        """ Unserialize a vector using pickle.

        :param bytes serialized_vector:
        :return np.array:
        """
        return pickle.loads(serialized_vector)


class MsgpackSerializer:

    def __init__(self, raw = False):
        """ Constructor.

        :param bool raw: If True, unpack msgpack raw to Python bytes. Otherwise, unpack to Python
            str by decoding with UTF-8 encoding (default). This is a highly confusing aspect of
            msgpack-python. They have gone through several iterations on approaches to handle both
            strings and bytes. If you are unsure what you need, leave this as False. If you
            serialized your data on an older version of msgpack than what you are currently using,
            you may need to set this to True.
        :return void:
        """
        self._raw = raw

    @staticmethod
    def serialize(vector):
        """ Serializer a vector using msgpack.

        :param np.array vector:
        :return bytes:
        """
        return msgpack.packb(vector, default = msgpack_numpy.encode)

    def unserialize(self, serialized_vector):
        """ Unserialize a vector using msgpack.

        :param bytes serialized_vector:
        :return np.array:
        """
        return msgpack.unpackb(
            serialized_vector,
            object_hook = msgpack_numpy.decode,
            raw = self._raw
        )
