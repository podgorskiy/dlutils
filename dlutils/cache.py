# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import hashlib
import pickle
import os


class cache:
    """ Caches return value of a functions.

    Given a function with no side effects, it will compute sha256 hash of passed arguments and use that hash to retrieve
    saved pickle.

    Note:
        
        Passed arguments must be picklable.
        
        If you change function, or do any other change that invalidates previously saved caches you will need to delete
        them manually
        
        Results are saved to '.cache' folder in current directory.
    Args:
        function (function): fucntions to be called.

    Example:

        ::

            @dlutils.cache
            def expensive_function(x):
                for i in range(12):
                    x = x + x * x
                return x


    """
    def __init__(self, function):
        self.function = function
        self.pickle_name = self.function.__name__

    def __call__(self, *args, **kwargs):
        m = hashlib.sha256()
        m.update(pickle.dumps((self.function.__name__, args, frozenset(kwargs.items()))))
        output_path = os.path.join('.cache', "%s_%s" % (m.hexdigest(), self.pickle_name))
        try:
            with open(output_path, 'rb') as f:
                data = pickle.load(f)
        except (FileNotFoundError, pickle.PickleError):
            data = self.function(*args, **kwargs)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        return data


if __name__ == '__main__':

    @cache
    def expensive_function(x):
        for i in range(12):
            x = x + x * x
        return x

    print(expensive_function(1))
    print(expensive_function(2))
    print(expensive_function(5))
    print(expensive_function(1))
    print(expensive_function(2))
    print(expensive_function(5))
