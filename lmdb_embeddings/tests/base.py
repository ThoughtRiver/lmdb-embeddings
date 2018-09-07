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
