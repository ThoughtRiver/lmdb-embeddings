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

        :return bytes
        """
        return pickletools.optimize(
            pickle.dumps(vector, pickle.HIGHEST_PROTOCOL)
        )

    @staticmethod
    def unserialize(serialized_vector):
        """ Unserialize a vector using pickle.

        :return np.array
        """
        return pickle.loads(serialized_vector)


class MsgpackSerializer:

    @staticmethod
    def serialize(vector):
        """ Serializer a vector using msgpack.

        :return bytes
        """
        return msgpack.packb(
            vector,
            default = msgpack_numpy.encode
        )

    @staticmethod
    def unserialize(serialized_vector):
        """ Unserialize a vector using msgpack.

        :return np.array
        """
        return msgpack.unpackb(
            serialized_vector,
            object_hook = msgpack_numpy.decode
        )