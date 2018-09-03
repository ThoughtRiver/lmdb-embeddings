import pickle
import pickletools


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