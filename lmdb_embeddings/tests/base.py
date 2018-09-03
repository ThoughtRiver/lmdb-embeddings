import tempfile


class LmdbEmbeddingsTest:

    @staticmethod
    def make_temporary_folder(function):
        """ Decorator to create a temporary file
        for the lifecycle of a test and then ensure
        that it is removed, regardless of exception.

        :return callable
        """
        def wrapper(*args, **kwargs):
            with tempfile.TemporaryDirectory() as directory_path:
                return function(*args, directory_path)

        return wrapper