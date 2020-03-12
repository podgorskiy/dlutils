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
